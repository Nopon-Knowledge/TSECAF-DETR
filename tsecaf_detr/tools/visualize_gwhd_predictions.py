"""Visualize TSECAF-DETR detections on one image or a directory of images."""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig
try:
    from project_paths import DATASETS_ROOT, PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import DATASETS_ROOT, PROJECT_ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIS_PRESETS = {
    "small_object_aware_rtdetr_r18_val_best": {
        "description": "Legacy alias for TSECAF-DETR-R18 validation prediction visualization.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r18vd_120e_gwhd.yml",
        "resume": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r18vd_120e_gwhd" / "best.pth"),
        "input": str(DATASETS_ROOT / "gwhd_2021" / "val2017"),
        "output_dir": str(PROJECT_ROOT / "vis_results" / "val_best"),
        "device": "cuda:0",
        "threshold": 0.4,
        "label_name": "wheat",
        "input_size": None,
    },
    "small_object_aware_rtdetr_r50_val_best": {
        "description": "Legacy alias for TSECAF-DETR-R50 validation prediction visualization.",
        "config": "configs/rtdetrv2/small_object_aware_rtdetr_r50vd_180e_gwhd.yml",
        "resume": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r50vd_180e_gwhd" / "best.pth"),
        "input": str(DATASETS_ROOT / "gwhd_2021" / "val2017"),
        "output_dir": str(PROJECT_ROOT / "vis_results" / "r50_val_best"),
        "device": "cuda:0",
        "threshold": 0.4,
        "label_name": "wheat",
        "input_size": None,
    },
}

VIS_PRESETS.update({
    "tsecaf_detr_r18_val_best": {
        **VIS_PRESETS["small_object_aware_rtdetr_r18_val_best"],
        "description": "Visualize GWHD2021 validation predictions from the TSECAF-DETR-R18 best checkpoint.",
        "config": "configs/rtdetrv2/tsecaf_detr_r18vd_120e_gwhd.yml",
        "resume": str(PROJECT_ROOT / "output" / "tsecaf_detr_r18vd_120e_gwhd" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "vis_results" / "tsecaf_detr_r18_val_best"),
    },
    "tsecaf_detr_r50_val_best": {
        **VIS_PRESETS["small_object_aware_rtdetr_r50_val_best"],
        "description": "Visualize GWHD2021 validation predictions from the TSECAF-DETR-R50 best checkpoint.",
        "config": "configs/rtdetrv2/tsecaf_detr_r50vd_180e_gwhd.yml",
        "resume": str(PROJECT_ROOT / "output" / "tsecaf_detr_r50vd_180e_gwhd" / "best.pth"),
        "output_dir": str(PROJECT_ROOT / "vis_results" / "tsecaf_detr_r50_val_best"),
    },
})


def collect_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    images = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found in directory: {path}")
    return images


def get_eval_size(cfg: YAMLConfig, input_size: str = None) -> Tuple[int, int]:
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


def configure_torch_runtime() -> None:
    if os.environ.get("RTDETR_DISABLE_CUDNN", "1") == "1":
        torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def draw_predictions(
    image: Image.Image,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    class_names: List[str],
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    keep = scores > threshold
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    for label, box, score in zip(labels, boxes, scores):
        box = [float(v) for v in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        class_idx = int(label.item())
        class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        draw.text((box[0], max(0.0, box[1] - 12.0)), f"{class_name} {score.item():.2f}", fill="yellow")

    return image


def main(args) -> None:
    configure_torch_runtime()
    cfg = YAMLConfig(args.config, resume=args.resume, device=args.device)

    if not args.resume:
        raise ValueError("Checkpoint path is required via --resume")

    input_paths = collect_images(Path(args.input))
    if args.max_images is not None:
        input_paths = input_paths[: args.max_images]
        if not input_paths:
            raise ValueError("--max-images resolved to an empty image list.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_checkpoint(cfg.model, args.resume)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = torch.device(args.device)
    model = Model().to(device)
    model.eval()

    eval_h, eval_w = get_eval_size(cfg, args.input_size)
    transforms = T.Compose([
        T.Resize((eval_h, eval_w)),
        T.ToTensor(),
    ])

    class_names = [args.label_name] if int(cfg.yaml_cfg.get("num_classes", 1)) == 1 else [str(i) for i in range(int(cfg.yaml_cfg["num_classes"]))]

    with torch.no_grad():
        for image_path in input_paths:
            image_pil = Image.open(image_path).convert("RGB")
            w, h = image_pil.size
            orig_size = torch.tensor([[w, h]], device=device)
            image_tensor = transforms(image_pil).unsqueeze(0).to(device)

            labels, boxes, scores = model(image_tensor, orig_size)
            vis_image = draw_predictions(
                image_pil,
                labels[0].cpu(),
                boxes[0].cpu(),
                scores[0].cpu(),
                threshold=args.threshold,
                class_names=class_names,
            )

            output_path = output_dir / image_path.name
            vis_image.save(output_path)
            print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize TSECAF-DETR detections for GWHD images.")
    parser.add_argument("--preset", choices=sorted(VIS_PRESETS), default="tsecaf_detr_r18_val_best")
    parser.add_argument("-c", "--config", type=str, help="Config path. Defaults to the selected preset.")
    parser.add_argument("-r", "--resume", type=str, help="Checkpoint path. Defaults to the selected preset.")
    parser.add_argument("-i", "--input", type=str, help="Input image or directory. Defaults to the selected preset.")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save visualized images.")
    parser.add_argument("-d", "--device", type=str, help="Inference device.")
    parser.add_argument("--threshold", type=float, help="Score threshold for drawing boxes.")
    parser.add_argument("--input-size", default=None, type=str, help="Override eval size, e.g. 1024,1024 or 1024x1024.")
    parser.add_argument("--max-images", type=int, help="Only visualize the first N images after sorting the input directory.")
    parser.add_argument("--label-name", type=str, help="Class name for single-class models.")
    args = parser.parse_args()

    preset = VIS_PRESETS[args.preset]
    if args.config is None:
        args.config = preset["config"]
    if args.resume is None:
        args.resume = preset["resume"]
    if args.input is None:
        args.input = preset["input"]
    if args.output_dir is None:
        args.output_dir = preset["output_dir"]
    if args.device is None:
        args.device = preset["device"]
    if args.threshold is None:
        args.threshold = preset["threshold"]
    if args.label_name is None:
        args.label_name = preset["label_name"]
    if args.input_size is None and preset["input_size"] is not None:
        args.input_size = preset["input_size"]

    print(f"Selected preset: {args.preset} - {preset['description']}")
    print(f"Using config: {args.config}")
    print(f"Using checkpoint: {args.resume}")
    print(f"Input source: {args.input}")
    print(f"Visualization outputs: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print(f"Max images: {args.max_images}")

    main(args)
