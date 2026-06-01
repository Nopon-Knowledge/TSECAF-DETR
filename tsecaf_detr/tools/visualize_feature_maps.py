"""Export evidence-pathway heatmaps for TSECAF-DETR paper figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig

try:
    from project_paths import DATASETS_ROOT, PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import DATASETS_ROOT, PROJECT_ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIS_PRESETS = {
    "gwhd_r50_baseline_vs_small_object_aware": {
        "description": "Compare RT-DETRv2-R50 and TSECAF-DETR-R50 evidence responses on GWHD2021 validation images.",
        "baseline_config": "configs/rtdetrv2/rtdetrv2_r50vd_180e_gwhd_baseline.yml",
        "baseline_resume": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_baseline" / "best.pth"),
        "ours_config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml",
        "ours_resume": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r50vd_180e_gwhd" / "best.pth"),
        "input": str(DATASETS_ROOT / "gwhd_2021" / "val2017"),
        "ann_file": str(DATASETS_ROOT / "gwhd_2021" / "annotations" / "instances_val2017.json"),
        "output_dir": str(PROJECT_ROOT / "vis_results" / "feature_maps" / "r50_baseline_vs_small_object_aware"),
        "device": "cuda:0",
        "max_images": 6,
        "selection": "densest",
        "input_size": None,
    },
}

VIS_PRESETS["gwhd_r50_baseline_vs_tsecaf_detr"] = {
    **VIS_PRESETS["gwhd_r50_baseline_vs_small_object_aware"],
    "description": "Compare RT-DETRv2-R50 and TSECAF-DETR-R50 evidence responses on GWHD2021 validation images.",
    "ours_config": "configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml",
    "ours_resume": str(PROJECT_ROOT / "output" / "tsecaf_detr_r50vd_180e_gwhd" / "best.pth"),
    "output_dir": str(PROJECT_ROOT / "vis_results" / "feature_maps" / "r50_baseline_vs_tsecaf_detr"),
}


def configure_torch_runtime() -> None:
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found in directory: {path}")
    return images


def get_eval_size(cfg: YAMLConfig, input_size: str | None = None) -> tuple[int, int]:
    if input_size:
        h, w = input_size.lower().replace("x", ",").split(",")
        return int(h), int(w)

    size = cfg.yaml_cfg.get("eval_spatial_size", None)
    if size is None:
        return 640, 640
    return int(size[0]), int(size[1])


def load_checkpoint(model: nn.Module, resume_path: str) -> None:
    checkpoint = torch.load(resume_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    model.load_state_dict(state)


def load_coco_boxes(ann_file: Path) -> Dict[str, List[List[float]]]:
    data = json.loads(ann_file.read_text())
    image_id_to_name = {image["id"]: image["file_name"] for image in data.get("images", [])}
    boxes_by_name: Dict[str, List[List[float]]] = {name: [] for name in image_id_to_name.values()}
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        file_name = image_id_to_name.get(ann["image_id"])
        if file_name is None:
            continue
        boxes_by_name.setdefault(file_name, []).append(ann["bbox"])
    return boxes_by_name


def select_images(
    images: Sequence[Path],
    boxes_by_name: Dict[str, List[List[float]]],
    selection: str,
    max_images: int,
) -> List[Path]:
    if selection == "densest":
        ranked = sorted(images, key=lambda p: (-len(boxes_by_name.get(p.name, [])), p.name))
    else:
        ranked = list(images)
    return ranked[:max_images]


def feature_to_map(feature: torch.Tensor) -> np.ndarray:
    if feature.dim() == 4:
        feature = feature[0]
    heat = feature.detach().float().abs().mean(dim=0).cpu().numpy()
    return heat


def normalize_maps(*maps: np.ndarray) -> List[np.ndarray]:
    stacked = np.stack(maps, axis=0)
    hi = float(stacked.max())
    lo = float(stacked.min())
    if hi <= lo:
        return [np.zeros_like(m, dtype=np.float32) for m in maps]
    return [((m - lo) / (hi - lo)).astype(np.float32) for m in maps]


def resize_map(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0), mode="L")
    image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


@dataclass
class ExtractedFeatures:
    backbone_p3: torch.Tensor
    fusion_p3: torch.Tensor
    detail_gate: torch.Tensor | None = None


class FeatureExtractor:
    def __init__(self, config_path: str, resume_path: str, device: str) -> None:
        cfg = YAMLConfig(config_path, resume=resume_path, device=device)
        load_checkpoint(cfg.model, resume_path)
        self.model = cfg.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.cfg = cfg
        self._captures: Dict[str, torch.Tensor] = {}
        self._handles = []

        encoder = self.model.encoder
        if getattr(encoder, "wheat_fusion", False) and len(getattr(encoder, "wheat_fusion_blocks", [])) > 1:
            self._handles.append(
                encoder.wheat_fusion_blocks[1].detail_gate.register_forward_hook(self._capture_tensor("detail_gate"))
            )

    def _capture_tensor(self, name: str):
        def hook(_module, _inputs, output):
            self._captures[name] = output.detach()

        return hook

    def extract(self, image_tensor: torch.Tensor) -> ExtractedFeatures:
        self._captures = {}
        with torch.no_grad():
            backbone_feats = self.model.backbone(image_tensor.to(self.device))
            encoder_feats = self.model.encoder(backbone_feats)

        return ExtractedFeatures(
            backbone_p3=backbone_feats[0].detach().cpu(),
            fusion_p3=encoder_feats[0].detach().cpu(),
            detail_gate=self._captures.get("detail_gate", None).detach().cpu() if "detail_gate" in self._captures else None,
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    x = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def make_original_panel(image_np: np.ndarray, boxes: Sequence[Sequence[float]], title: str) -> Image.Image:
    image = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
    return add_title(image, title)


def make_heatmap_panel(image_np: np.ndarray, heatmap: np.ndarray, title: str, alpha: float = 0.48) -> Image.Image:
    base = np.asarray(Image.fromarray(image_np).convert("RGB"), dtype=np.float32) / 255.0
    colored = apply_colormap(heatmap)
    mixed = np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)
    image = Image.fromarray(np.uint8(mixed * 255.0), mode="RGB")
    return add_title(image, title)


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = ImageFont.load_default()
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), color="white")
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black", font=font)
    return canvas


def save_panel(
    image_path: Path,
    image_np: np.ndarray,
    boxes: Sequence[Sequence[float]],
    baseline_p3: np.ndarray,
    ours_p3: np.ndarray,
    baseline_fusion: np.ndarray,
    ours_fusion: np.ndarray,
    ours_detail_gate: np.ndarray | None,
    output_path: Path,
) -> None:
    panels = [
        make_original_panel(image_np, boxes, f"Original ({len(boxes)} GT boxes)"),
        make_heatmap_panel(image_np, baseline_p3, "Baseline P3"),
        make_heatmap_panel(image_np, ours_p3, "TSECAF P3"),
        make_heatmap_panel(image_np, baseline_fusion, "Baseline Fused P3"),
        make_heatmap_panel(image_np, ours_fusion, "TSECAF Evidence P3"),
        make_original_panel(image_np, [], "TSECAF Detail Gate (N/A)")
        if ours_detail_gate is None
        else make_heatmap_panel(image_np, ours_detail_gate, "TSECAF Detail Gate"),
    ]

    tile_w = max(panel.width for panel in panels)
    tile_h = max(panel.height for panel in panels)
    margin = 18
    header = 28
    canvas = Image.new("RGB", (tile_w * 3 + margin * 4, tile_h * 2 + margin * 3 + header), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), image_path.name, fill="black", font=ImageFont.load_default())

    for idx, panel in enumerate(panels):
        row = idx // 3
        col = idx % 3
        x = margin + col * (tile_w + margin)
        y = header + margin + row * (tile_h + margin)
        canvas.paste(panel, (x, y))

    canvas.save(output_path)


def main(args) -> None:
    configure_torch_runtime()

    baseline_extractor = FeatureExtractor(args.baseline_config, args.baseline_resume, args.device)
    ours_extractor = FeatureExtractor(args.ours_config, args.ours_resume, args.device)

    boxes_by_name = load_coco_boxes(Path(args.ann_file)) if args.ann_file else {}
    image_paths = collect_images(Path(args.input))
    image_paths = select_images(image_paths, boxes_by_name, args.selection, args.max_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_h, eval_w = get_eval_size(ours_extractor.cfg, args.input_size)
    transforms = T.Compose([T.Resize((eval_h, eval_w)), T.ToTensor()])

    manifest = []
    try:
        for image_path in image_paths:
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.asarray(image_pil)
            image_tensor = transforms(image_pil).unsqueeze(0)

            baseline_feats = baseline_extractor.extract(image_tensor)
            ours_feats = ours_extractor.extract(image_tensor)

            baseline_p3_map, ours_p3_map = normalize_maps(
                feature_to_map(baseline_feats.backbone_p3),
                feature_to_map(ours_feats.backbone_p3),
            )
            baseline_fusion_map, ours_fusion_map = normalize_maps(
                feature_to_map(baseline_feats.fusion_p3),
                feature_to_map(ours_feats.fusion_p3),
            )

            detail_gate_map = None
            if ours_feats.detail_gate is not None:
                detail_gate_map = normalize_maps(feature_to_map(ours_feats.detail_gate))[0]

            width, height = image_pil.size
            size = (width, height)
            baseline_p3_map = resize_map(baseline_p3_map, size)
            ours_p3_map = resize_map(ours_p3_map, size)
            baseline_fusion_map = resize_map(baseline_fusion_map, size)
            ours_fusion_map = resize_map(ours_fusion_map, size)
            if detail_gate_map is not None:
                detail_gate_map = resize_map(detail_gate_map, size)

            output_path = output_dir / f"{image_path.stem}_feature_panel.png"
            boxes = boxes_by_name.get(image_path.name, [])
            save_panel(
                image_path=image_path,
                image_np=image_np,
                boxes=boxes,
                baseline_p3=baseline_p3_map,
                ours_p3=ours_p3_map,
                baseline_fusion=baseline_fusion_map,
                ours_fusion=ours_fusion_map,
                ours_detail_gate=detail_gate_map,
                output_path=output_path,
            )
            manifest.append(
                {
                    "image": str(image_path),
                    "panel": str(output_path),
                    "gt_boxes": len(boxes),
                }
            )
            print(f"saved: {output_path}")
    finally:
        baseline_extractor.close()
        ours_extractor.close()

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export evidence-pathway heatmaps for paper figures.")
    parser.add_argument("--preset", choices=sorted(VIS_PRESETS), default="gwhd_r50_baseline_vs_tsecaf_detr")
    parser.add_argument("--baseline-config", type=str, help="Baseline config path.")
    parser.add_argument("--baseline-resume", type=str, help="Baseline checkpoint path.")
    parser.add_argument("--ours-config", type=str, help="TSECAF-DETR config path.")
    parser.add_argument("--ours-resume", type=str, help="TSECAF-DETR checkpoint path.")
    parser.add_argument("-i", "--input", type=str, help="Input image or directory.")
    parser.add_argument("--ann-file", type=str, help="COCO annotation file for GT boxes.")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save panel figures.")
    parser.add_argument("-d", "--device", type=str, help="Inference device.")
    parser.add_argument("--input-size", default=None, type=str, help="Override eval size, e.g. 1024x1024.")
    parser.add_argument("--max-images", type=int, help="Number of images to export.")
    parser.add_argument(
        "--selection",
        choices=["first", "densest"],
        help="Whether to use the first sorted images or the densest annotated images.",
    )
    args = parser.parse_args()

    preset = VIS_PRESETS[args.preset]
    if args.baseline_config is None:
        args.baseline_config = preset["baseline_config"]
    if args.baseline_resume is None:
        args.baseline_resume = preset["baseline_resume"]
    if args.ours_config is None:
        args.ours_config = preset["ours_config"]
    if args.ours_resume is None:
        args.ours_resume = preset["ours_resume"]
    if args.input is None:
        args.input = preset["input"]
    if args.ann_file is None:
        args.ann_file = preset["ann_file"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]
    if args.device is None:
        args.device = preset["device"]
    if args.max_images is None:
        args.max_images = preset["max_images"]
    if args.selection is None:
        args.selection = preset["selection"]
    if args.input_size is None and preset["input_size"] is not None:
        args.input_size = preset["input_size"]

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Baseline config: {args.baseline_config}")
    print(f"Baseline checkpoint: {args.baseline_resume}")
    print(f"TSECAF-DETR config: {args.ours_config}")
    print(f"TSECAF-DETR checkpoint: {args.ours_resume}")
    print(f"Input source: {args.input}")
    print(f"Annotation file: {args.ann_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Max images: {args.max_images}")
    print(f"Selection: {args.selection}")

    main(args)
