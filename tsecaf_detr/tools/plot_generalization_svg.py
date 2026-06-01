"""Plot grouped-bar generalization figures from saved COCO eval dictionaries."""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from project_paths import PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import PROJECT_ROOT


GENERALIZATION_PRESETS = {
    "r50_baseline_vs_small_object_aware_transfer": {
        "title": "Cross-dataset Generalization from GWHD2021",
        "subtitle": "Train on GWHD2021, evaluate the same evidence-to-query pathway on target datasets",
        "output_dir": str(PROJECT_ROOT / "vis_results" / "generalization"),
        "output_name": "r50_baseline_vs_tsecaf_detr_transfer.svg",
        "datasets": [
            {
                "label": "GWHD2021",
                "baseline_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "gwhd_r50_baseline" / "eval.pth"),
                "ours_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "gwhd_r50_small_object_aware" / "eval.pth"),
            },
            {
                "label": "Wheat_Ears",
                "baseline_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "wheat_ears_r50_baseline" / "eval.pth"),
                "ours_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "wheat_ears_r50_small_object_aware" / "eval.pth"),
            },
            {
                "label": "Codalab",
                "baseline_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "global_wheat_codalab_r50_baseline" / "eval.pth"),
                "ours_eval": str(PROJECT_ROOT / "output" / "paper_eval" / "global_wheat_codalab_r50_small_object_aware" / "eval.pth"),
            },
        ],
    },
}

GENERALIZATION_PRESETS["r50_baseline_vs_tsecaf_detr_transfer"] = GENERALIZATION_PRESETS[
    "r50_baseline_vs_small_object_aware_transfer"
]


def ap_from_eval(eval_path: Path) -> float:
    data = torch.load(eval_path, map_location="cpu")
    precision = data["precision"][:, :, 0, 0, -1]
    valid = precision[precision > -1]
    size = int(valid.numel()) if hasattr(valid, "numel") else int(valid.size)
    return float(valid.mean()) if size else -1.0


def save_svg(preset: Dict[str, object], rows: List[Dict[str, float]]) -> Path:
    output_dir = Path(preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / preset["output_name"]

    width = 1120
    height = 760
    margin_left = 88
    margin_right = 56
    margin_top = 110
    margin_bottom = 96
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_ap = max(max(row["baseline"], row["ours"]) for row in rows)
    y_max = max(0.6, min(1.0, max_ap * 1.15))
    bar_width = 78
    group_gap = 120
    start_x = margin_left + 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="44" font-size="28" font-weight="700" fill="#111827">{html.escape(str(preset["title"]))}</text>',
        f'<text x="{margin_left}" y="72" font-size="14" fill="#4b5563">{html.escape(str(preset["subtitle"]))}</text>',
    ]

    for tick in range(0, 6):
        value = y_max * tick / 5
        y = margin_top + plot_h - (value / y_max) * plot_h
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_w}" y2="{y:.2f}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 14}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#6b7280">{value:.2f}</text>')

    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.4"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.4"/>')
    parts.append(f'<text x="26" y="{margin_top + plot_h / 2:.2f}" font-size="14" text-anchor="middle" fill="#111827" transform="rotate(-90 26 {margin_top + plot_h / 2:.2f})">AP50-95</text>')

    legend_x = width - margin_right - 230
    legend_y = 48
    legend = [("RT-DETRv2-R50", "#1f77b4"), ("TSECAF-DETR-R50", "#d62728")]
    for idx, (label, color) in enumerate(legend):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="18" height="18" fill="{color}" />')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-size="13" fill="#111827">{html.escape(label)}</text>')

    for idx, row in enumerate(rows):
        group_x = start_x + idx * (bar_width * 2 + group_gap)
        bars = [("baseline", "#1f77b4", group_x), ("ours", "#d62728", group_x + bar_width + 16)]
        for key, color, x in bars:
            value = row[key]
            h = (value / y_max) * plot_h
            y = margin_top + plot_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{h:.2f}" fill="{color}" rx="4" />')
            parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" font-size="12" text-anchor="middle" fill="#111827">{value:.4f}</text>')
        parts.append(
            f'<text x="{group_x + bar_width + 8:.2f}" y="{margin_top + plot_h + 28}" font-size="13" text-anchor="middle" fill="#111827">{html.escape(row["label"])}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))
    return output_path


def main(args) -> None:
    preset = GENERALIZATION_PRESETS[args.preset]
    rows = []
    for item in preset["datasets"]:
        baseline = ap_from_eval(Path(item["baseline_eval"]))
        ours = ap_from_eval(Path(item["ours_eval"]))
        rows.append({"label": item["label"], "baseline": baseline, "ours": ours})
        print(item["label"], baseline, ours)
    output_path = save_svg(preset, rows)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot grouped-bar cross-dataset generalization figures.")
    parser.add_argument("--preset", choices=sorted(GENERALIZATION_PRESETS), default="r50_baseline_vs_tsecaf_detr_transfer")
    args = parser.parse_args()
    main(args)
