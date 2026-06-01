"""Plot RT-DETR PR curves from saved COCO eval dictionaries."""

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


PR_PRESETS = {
    "gwhd_r50_baseline_vs_small_object_aware": {
        "title": "GWHD2021 PR Curve at IoU=0.50",
        "subtitle": "RT-DETRv2-R50 baseline vs TSECAF-DETR-R50",
        "output_dir": str(PROJECT_ROOT / "vis_results" / "pr_curves" / "r50_baseline_vs_tsecaf_detr"),
        "output_name": "r50_baseline_vs_tsecaf_detr_pr50.svg",
        "series": [
            {
                "name": "RT-DETRv2-R50",
                "eval_path": str(PROJECT_ROOT / "output" / "paper_eval" / "gwhd_r50_baseline" / "eval.pth"),
                "color": "#1f77b4",
            },
            {
                "name": "TSECAF-DETR-R50",
                "eval_path": str(PROJECT_ROOT / "output" / "paper_eval" / "gwhd_r50_small_object_aware" / "eval.pth"),
                "color": "#d62728",
            },
        ],
    },
}

PR_PRESETS["gwhd_r50_baseline_vs_tsecaf_detr"] = PR_PRESETS["gwhd_r50_baseline_vs_small_object_aware"]


def extract_pr(eval_path: Path) -> Dict[str, object]:
    data = torch.load(eval_path, map_location="cpu")
    precision = data["precision"]
    recalls = [i / 100.0 for i in range(precision.shape[1])]
    curve = precision[0, :, 0, 0, -1]
    curve = [float(value) for value in curve]
    valid = [value for value in curve if value > -1]
    ap50 = sum(valid) / len(valid) if valid else -1.0
    return {
        "recalls": recalls,
        "precision": curve,
        "ap50": ap50,
    }


def polyline_points(recalls: List[float], precisions: List[float], x0: float, y0: float, width: float, height: float) -> str:
    points = []
    for recall, precision in zip(recalls, precisions):
        if precision < 0:
            continue
        x = x0 + recall * width
        y = y0 + (1.0 - precision) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def save_svg(preset: Dict[str, object], series_data: List[Dict[str, object]]) -> Path:
    output_dir = Path(preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / preset["output_name"]

    width = 1180
    height = 820
    margin_left = 90
    margin_right = 60
    margin_top = 100
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="44" font-size="28" font-weight="700" fill="#111827">{html.escape(str(preset["title"]))}</text>',
        f'<text x="{margin_left}" y="72" font-size="14" fill="#4b5563">{html.escape(str(preset["subtitle"]))}</text>',
    ]

    x0 = margin_left
    y0 = margin_top
    parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#d0d7de" stroke-width="1"/>')

    for tick in range(0, 11):
        value = tick / 10.0
        x = x0 + value * plot_w
        y = y0 + (1.0 - value) * plot_h
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + plot_h}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{y0 + plot_h + 24}" font-size="12" text-anchor="middle" fill="#6b7280">{value:.1f}</text>')
        parts.append(f'<text x="{x0 - 14}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#6b7280">{value:.1f}</text>')

    parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#94a3b8" stroke-width="1.3"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#94a3b8" stroke-width="1.3"/>')
    parts.append(f'<text x="{x0 + plot_w / 2:.2f}" y="{height - 28}" font-size="14" text-anchor="middle" fill="#111827">Recall</text>')
    parts.append(f'<text x="24" y="{y0 + plot_h / 2:.2f}" font-size="14" text-anchor="middle" fill="#111827" transform="rotate(-90 24 {y0 + plot_h / 2:.2f})">Precision</text>')

    legend_x = width - margin_right - 260
    legend_y = 48
    for idx, item in enumerate(series_data):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{item["color"]}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 36}" y="{y + 4}" font-size="13" fill="#111827">{html.escape(item["label"])}</text>'
        )
        points = polyline_points(item["recalls"], item["precision"], x0, y0, plot_w, plot_h)
        parts.append(f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.8" points="{points}" />')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))
    return output_path


def main(args) -> None:
    preset = PR_PRESETS[args.preset]
    series_data = []
    for item in preset["series"]:
        data = extract_pr(Path(item["eval_path"]))
        label = f'{item["name"]} (AP50={data["ap50"]:.4f})'
        series_data.append(
            {
                "label": label,
                "color": item["color"],
                "recalls": data["recalls"],
                "precision": data["precision"],
            }
        )
        print(f'loaded {item["eval_path"]}')
    output_path = save_svg(preset, series_data)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PR curves from saved COCO eval dictionaries.")
    parser.add_argument("--preset", choices=sorted(PR_PRESETS), default="gwhd_r50_baseline_vs_tsecaf_detr")
    args = parser.parse_args()
    main(args)
