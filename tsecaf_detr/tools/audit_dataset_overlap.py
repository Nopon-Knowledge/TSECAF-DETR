#!/usr/bin/env python3
"""Audit exact and perceptual image overlap between training and evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize(
            (size + 1, size), Image.Resampling.LANCZOS
        )
        pixels = (
            list(image.get_flattened_data())
            if hasattr(image, "get_flattened_data")
            else list(image.getdata())
        )
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def bands(value: int, bits: int, band_count: int) -> Iterable[tuple[int, int]]:
    base, remainder = divmod(bits, band_count)
    shift = 0
    for index in range(band_count):
        width = base + (1 if index < remainder else 0)
        mask = (1 << width) - 1
        yield index, (value >> shift) & mask
        shift += width


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find exact SHA-256 duplicates and dHash near-duplicates across two image roots."
    )
    parser.add_argument("--train", type=Path, required=True, help="training image root")
    parser.add_argument("--eval", type=Path, required=True, help="evaluation image root")
    parser.add_argument("--threshold", type=int, default=4, help="maximum dHash Hamming distance")
    parser.add_argument("--output", type=Path, default=Path("overlap_audit.json"))
    args = parser.parse_args()

    train_root = args.train.expanduser().resolve()
    eval_root = args.eval.expanduser().resolve()
    if not train_root.is_dir() or not eval_root.is_dir():
        parser.error("--train and --eval must both be existing directories")
    if not 0 <= args.threshold <= 8:
        parser.error("--threshold must be between 0 and 8")

    train_paths = image_paths(train_root)
    eval_paths = image_paths(eval_root)
    if not train_paths or not eval_paths:
        parser.error("both roots must contain at least one supported image")

    unreadable: list[dict[str, str]] = []
    train_sha: dict[str, list[Path]] = defaultdict(list)
    eval_sha: dict[str, list[Path]] = defaultdict(list)
    train_hashes: dict[Path, int] = {}
    eval_hashes: dict[Path, int] = {}

    for label, paths, sha_index, perceptual in [
        ("train", train_paths, train_sha, train_hashes),
        ("eval", eval_paths, eval_sha, eval_hashes),
    ]:
        root = train_root if label == "train" else eval_root
        for path in paths:
            sha_index[sha256(path)].append(path)
            try:
                perceptual[path] = dhash(path)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                unreadable.append({"set": label, "path": relative(path, root), "error": type(exc).__name__})

    exact_pairs: list[dict[str, str]] = []
    for digest in sorted(set(train_sha).intersection(eval_sha)):
        for train_path in train_sha[digest]:
            for eval_path in eval_sha[digest]:
                exact_pairs.append(
                    {
                        "train": relative(train_path, train_root),
                        "eval": relative(eval_path, eval_root),
                        "sha256": digest,
                    }
                )

    band_count = args.threshold + 1
    index: dict[tuple[int, int], list[tuple[Path, int]]] = defaultdict(list)
    for path, value in eval_hashes.items():
        for band in bands(value, 64, band_count):
            index[band].append((path, value))

    exact_lookup = {(row["train"], row["eval"]) for row in exact_pairs}
    near_pairs: list[dict[str, object]] = []
    seen: set[tuple[Path, Path]] = set()
    for train_path, train_value in train_hashes.items():
        candidates: set[tuple[Path, int]] = set()
        for band in bands(train_value, 64, band_count):
            candidates.update(index.get(band, []))
        for eval_path, eval_value in candidates:
            pair = (train_path, eval_path)
            if pair in seen:
                continue
            seen.add(pair)
            train_rel = relative(train_path, train_root)
            eval_rel = relative(eval_path, eval_root)
            if (train_rel, eval_rel) in exact_lookup:
                continue
            distance = (train_value ^ eval_value).bit_count()
            if distance <= args.threshold:
                near_pairs.append(
                    {
                        "train": train_rel,
                        "eval": eval_rel,
                        "dhash_hamming": distance,
                    }
                )
    near_pairs.sort(key=lambda row: (int(row["dhash_hamming"]), str(row["train"]), str(row["eval"])))

    report = {
        "schema": "tsecaf.dataset_overlap_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "exact": "SHA-256 of original image bytes",
            "near_duplicate": "64-bit horizontal dHash after EXIF normalization and grayscale resize",
            "dhash_hamming_threshold": args.threshold,
        },
        "counts": {
            "train_images": len(train_paths),
            "eval_images": len(eval_paths),
            "exact_cross_set_pairs": len(exact_pairs),
            "near_duplicate_cross_set_pairs": len(near_pairs),
            "unreadable_images": len(unreadable),
        },
        "passed": not exact_pairs and not near_pairs and not unreadable,
        "exact_pairs": exact_pairs,
        "near_duplicate_pairs": near_pairs,
        "unreadable": unreadable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"passed: {report['passed']}")
    print(f"report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
