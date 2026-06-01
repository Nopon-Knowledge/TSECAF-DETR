"""Prepare external wheat datasets into COCO-style layouts for RT-DETRv2.

This script avoids duplicating large image collections by defaulting to
symlinks in the generated train2017/ and val2017/ folders.
"""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CATEGORY = [{"id": 1, "name": "wheat_head", "supercategory": "wheat"}]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = Path(os.environ.get("WHEAT_DATASETS_ROOT", str(PROJECT_ROOT / "datasets"))).expanduser()


DATASET_PRESETS = {
    "wheat_ears": {
        "source": DATASETS_ROOT / "Wheat_Ears_Detection_Dataset",
        "output": DATASETS_ROOT / "Wheat_Ears_Detection_Dataset_rtdetr",
        "val_ratio": 0.2,
        "seed": 42,
    },
    "global_wheat_codalab": {
        "source": DATASETS_ROOT / "global-wheat-codalab-official",
        "output": DATASETS_ROOT / "global-wheat-codalab-official_rtdetr",
        "val_ratio": 0.2,
        "seed": 42,
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def split_items(items: List[str], val_ratio: float, seed: int) -> Tuple[set, set]:
    if not items:
        return set(), set()

    rng = random.Random(seed)
    ordered = sorted(items)
    rng.shuffle(ordered)

    if val_ratio <= 0:
        val_count = 0
    elif val_ratio >= 1:
        val_count = max(0, len(ordered) - 1)
    elif len(ordered) > 1:
        val_count = int(round(len(ordered) * val_ratio))
        val_count = max(1, min(len(ordered) - 1, val_count))
    else:
        val_count = 0

    val_items = set(ordered[:val_count])
    train_items = set(ordered[val_count:])
    return train_items, val_items


def build_coco_json(images: List[Dict], annotations: List[Dict]) -> Dict:
    return {
        "images": images,
        "annotations": annotations,
        "categories": DEFAULT_CATEGORY,
    }


def write_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def parse_xml_box(node: ET.Element, width: int, height: int) -> Tuple[float, float, float, float]:
    xmin = max(0.0, float(node.findtext("xmin", default="0")))
    ymin = max(0.0, float(node.findtext("ymin", default="0")))
    xmax = min(float(width), float(node.findtext("xmax", default="0")))
    ymax = min(float(height), float(node.findtext("ymax", default="0")))
    return xmin, ymin, xmax, ymax


def prepare_wheat_ears(source: Path, output: Path, val_ratio: float, seed: int) -> Dict:
    xml_files = sorted(source.glob("*.xml"))
    image_map = {p.stem: p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    stems = [xml.stem for xml in xml_files if xml.stem in image_map]
    train_stems, val_stems = split_items(stems, val_ratio, seed)

    reset_dir(output / "train2017")
    reset_dir(output / "val2017")
    ensure_dir(output / "annotations")

    split_records = {
        "train": {"images": [], "annotations": [], "stems": train_stems},
        "val": {"images": [], "annotations": [], "stems": val_stems},
    }
    next_image_id = {"train": 1, "val": 1}
    next_ann_id = {"train": 1, "val": 1}

    for xml_path in xml_files:
        stem = xml_path.stem
        if stem not in image_map:
            continue

        if stem in train_stems:
            split = "train"
            split_dir = output / "train2017"
        elif stem in val_stems:
            split = "val"
            split_dir = output / "val2017"
        else:
            continue

        image_path = image_map[stem]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")

        width = int(float(size.findtext("width", default="0")))
        height = int(float(size.findtext("height", default="0")))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}")

        safe_symlink(image_path, split_dir / image_path.name)

        image_id = next_image_id[split]
        next_image_id[split] += 1
        split_records[split]["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            bbox_node = obj.find("bndbox")
            if bbox_node is None:
                continue
            xmin, ymin, xmax, ymax = parse_xml_box(bbox_node, width, height)
            box_w = xmax - xmin
            box_h = ymax - ymin
            if box_w <= 0 or box_h <= 0:
                continue
            ann_id = next_ann_id[split]
            next_ann_id[split] += 1
            split_records[split]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(xmin, 2), round(ymin, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )

    write_json(
        output / "annotations" / "instances_train2017.json",
        build_coco_json(split_records["train"]["images"], split_records["train"]["annotations"]),
    )
    write_json(
        output / "annotations" / "instances_val2017.json",
        build_coco_json(split_records["val"]["images"], split_records["val"]["annotations"]),
    )

    return {
        "dataset": "wheat_ears",
        "source": str(source),
        "output": str(output),
        "train_images": len(split_records["train"]["images"]),
        "val_images": len(split_records["val"]["images"]),
        "train_annotations": len(split_records["train"]["annotations"]),
        "val_annotations": len(split_records["val"]["annotations"]),
        "missing_images": len(xml_files) - len(stems),
        "split_seed": seed,
        "val_ratio": val_ratio,
    }


def iter_labeled_global_subsets(source: Path) -> Iterable[Tuple[str, Path, Path]]:
    for json_path in sorted(source.glob("*.json")):
        subset_name = json_path.stem
        image_dir = source / subset_name
        if image_dir.is_dir():
            yield subset_name, image_dir, json_path


def prepare_global_wheat(source: Path, output: Path, val_ratio: float, seed: int) -> Dict:
    reset_dir(output / "train2017")
    reset_dir(output / "val2017")
    ensure_dir(output / "annotations")

    split_records = {
        "train": {"images": [], "annotations": []},
        "val": {"images": [], "annotations": []},
    }
    next_image_id = {"train": 1, "val": 1}
    next_ann_id = {"train": 1, "val": 1}
    labeled_subsets = []
    unlabeled_subsets = []

    all_dirs = {p.name for p in source.iterdir() if p.is_dir()}
    labeled_dir_names = set()

    for subset_name, image_dir, json_path in iter_labeled_global_subsets(source):
        labeled_dir_names.add(subset_name)
        labeled_subsets.append(subset_name)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        anns_by_image = defaultdict(list)
        for ann in annotations:
            anns_by_image[ann["image_id"]].append(ann)

        subset_image_ids = [str(img["id"]) for img in images]
        train_ids, val_ids = split_items(subset_image_ids, val_ratio, seed + len(labeled_subsets))
        image_lookup = {str(img["id"]): img for img in images}

        for split_name, selected_ids in (("train", train_ids), ("val", val_ids)):
            split_dir = output / f"{split_name}2017"
            for image_id_str in sorted(selected_ids):
                img = image_lookup[image_id_str]
                orig_name = img["file_name"]
                src_image = image_dir / orig_name
                if not src_image.exists():
                    path_field = img.get("path")
                    if path_field:
                        src_image = source / path_field
                if not src_image.exists():
                    raise FileNotFoundError(f"Missing image for {subset_name}: {orig_name}")

                dst_name = f"{subset_name}__{orig_name}"
                safe_symlink(src_image, split_dir / dst_name)

                new_image_id = next_image_id[split_name]
                next_image_id[split_name] += 1
                split_records[split_name]["images"].append(
                    {
                        "id": new_image_id,
                        "file_name": dst_name,
                        "width": int(img["width"]),
                        "height": int(img["height"]),
                    }
                )

                for ann in anns_by_image.get(int(image_id_str), []):
                    bbox = ann["bbox"]
                    ann_id = next_ann_id[split_name]
                    next_ann_id[split_name] += 1
                    split_records[split_name]["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": new_image_id,
                            "category_id": 1,
                            "bbox": [round(float(v), 2) for v in bbox],
                            "area": round(float(ann.get("area", bbox[2] * bbox[3])), 2),
                            "iscrowd": int(ann.get("iscrowd", 0)),
                            "segmentation": ann.get("segmentation", []),
                        }
                    )

    for dir_name in sorted(all_dirs - labeled_dir_names):
        unlabeled_subsets.append(dir_name)

    write_json(
        output / "annotations" / "instances_train2017.json",
        build_coco_json(split_records["train"]["images"], split_records["train"]["annotations"]),
    )
    write_json(
        output / "annotations" / "instances_val2017.json",
        build_coco_json(split_records["val"]["images"], split_records["val"]["annotations"]),
    )

    return {
        "dataset": "global_wheat_codalab",
        "source": str(source),
        "output": str(output),
        "train_images": len(split_records["train"]["images"]),
        "val_images": len(split_records["val"]["images"]),
        "train_annotations": len(split_records["train"]["annotations"]),
        "val_annotations": len(split_records["val"]["annotations"]),
        "labeled_subsets": labeled_subsets,
        "unlabeled_subsets": unlabeled_subsets,
        "split_seed": seed,
        "val_ratio": val_ratio,
    }


def prepare_dataset(name: str, source: Path, output: Path, val_ratio: float, seed: int) -> Dict:
    if name == "wheat_ears":
        return prepare_wheat_ears(source, output, val_ratio, seed)
    if name == "global_wheat_codalab":
        return prepare_global_wheat(source, output, val_ratio, seed)
    raise ValueError(f"Unsupported dataset preset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare wheat datasets for RT-DETRv2.")
    parser.add_argument(
        "--dataset",
        choices=["wheat_ears", "global_wheat_codalab", "all"],
        default="all",
        help="Dataset preset to prepare.",
    )
    parser.add_argument("--source", type=str, help="Override source dataset root.")
    parser.add_argument("--output", type=str, help="Override output COCO-style dataset root.")
    parser.add_argument("--val-ratio", type=float, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, help="Split seed.")
    args = parser.parse_args()

    datasets = (
        [args.dataset]
        if args.dataset != "all"
        else ["wheat_ears", "global_wheat_codalab"]
    )

    for dataset_name in datasets:
        preset = DATASET_PRESETS[dataset_name]
        source = Path(args.source) if args.source and len(datasets) == 1 else preset["source"]
        output = Path(args.output) if args.output and len(datasets) == 1 else preset["output"]
        val_ratio = args.val_ratio if args.val_ratio is not None else preset["val_ratio"]
        seed = args.seed if args.seed is not None else preset["seed"]

        summary = prepare_dataset(dataset_name, source, output, val_ratio, seed)
        write_json(output / "annotations" / "prepare_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
