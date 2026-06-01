"""Plot TSECAF-DETR pathway ablation figures from training logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from project_paths import PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import PROJECT_ROOT


ABLATION_PRESETS = {
    "gwhd_r50_small_object_aware_ablation": {
        "title": "Pathway Ablation on GWHD2021 with RT-DETRv2-R50",
        "subtitle": "Controlled probes of evidence formation and decoder-query allocation",
        "output_dir": str(PROJECT_ROOT / "vis_results" / "ablation"),
        "output_name": "gwhd_r50_tsecaf_pathway_ablation.svg",
        "series": [
            {
                "label": "Baseline",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_baseline" / "log.txt"),
                "color": "#1f77b4",
            },
            {
                "label": "Context-detail representation",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_ablation_wheat_fusion" / "log.txt"),
                "color": "#2ca02c",
            },
            {
                "label": "Detail-branch emphasis",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_ablation_detail_enhance" / "log.txt"),
                "color": "#ff7f0e",
            },
            {
                "label": "Compact-query allocation",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_ablation_agnostic_small" / "log.txt"),
                "color": "#9467bd",
            },
            {
                "label": "TSECAF-DETR pathway",
                "log": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r50vd_180e_gwhd" / "log.txt"),
                "color": "#d62728",
            },
        ],
    },
}

ABLATION_PRESETS["gwhd_r50_tsecaf_pathway_ablation"] = ABLATION_PRESETS[
    "gwhd_r50_small_object_aware_ablation"
]


def extract_best_metrics(log_path: Path) -> Dict[str, float]:
    best_row = None
    best_ap = float("-inf")

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        named = record.get("test_coco_eval_bbox_named", {})
        ap = named.get("AP")
        ap50 = named.get("AP50")
        if ap is None or ap50 is None:
            values = record.get("test_coco_eval_bbox")
            if isinstance(values, list) and len(values) >= 2:
                ap, ap50 = values[0], values[1]

        epoch = record.get("epoch")
        if epoch is None or ap is None or ap50 is None:
            continue

        ap = float(ap)
        if ap > best_ap:
            best_ap = ap
            best_row = {
                "epoch": float(epoch),
                "ap": ap,
                "ap50": float(ap50),
            }

    if best_row is None:
        raise RuntimeError(f"No usable validation metrics found in {log_path}")
    return best_row


def save_svg(preset: Dict[str, object], rows: List[Dict[str, float]]) -> Path:
    output_dir = Path(preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / preset["output_name"]

    width = 1560
    height = 760
    margin_left = 90
    margin_right = 70
    margin_top = 124
    margin_bottom = 148
    panel_gap = 84
    panel_w = (width - margin_left - margin_right - panel_gap) / 2
    plot_h = height - margin_top - margin_bottom

    ap_values = [row["ap"] for row in rows]
    ap50_values = [row["ap50"] for row in rows]
    baseline_ap = rows[0]["ap"]

    ap_min = max(0.0, min(ap_values) - 0.01)
    ap_max = min(1.0, max(ap_values) + 0.015)
    ap50_min = max(0.0, min(ap50_values) - 0.02)
    ap50_max = min(1.0, max(ap50_values) + 0.02)

    bar_width = 78
    group_gap = 34
    start_x = margin_left + 36

    def y_map(value: float, v_min: float, v_max: float) -> float:
        span = max(v_max - v_min, 1e-6)
        return margin_top + plot_h - ((value - v_min) / span) * plot_h

    def add_axis(parts: List[str], x0: float, y_min: float, y_max: float, title: str) -> None:
        parts.append(
            f'<rect x="{x0}" y="{margin_top}" width="{panel_w}" height="{plot_h}" fill="white" stroke="#d7dee8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0}" y="{margin_top - 22}" font-size="18" font-weight="700" font-family="Times New Roman, Times, serif" fill="#111827">{title}</text>'
        )
        for tick_idx in range(6):
            value = y_min + (y_max - y_min) * tick_idx / 5
            y = y_map(value, y_min, y_max)
            parts.append(
                f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + panel_w}" y2="{y:.2f}" stroke="#eef2f7" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x0 - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end" font-family="Times New Roman, Times, serif" fill="#6b7280">{value:.2f}</text>'
            )
        parts.append(
            f'<line x1="{x0}" y1="{margin_top + plot_h}" x2="{x0 + panel_w}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.4"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{x0 + panel_w / 2:.2f}" y="{height - 36}" font-size="13" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827">Ablation Setting</text>'
        )

    bar_colors = ["#9ca3af", "#6baed6", "#74c476", "#9e9ac8", "#d62728"]
    edge_colors = ["#6b7280", "#3182bd", "#31a354", "#756bb1", "#a50f15"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="46" font-size="28" font-weight="700" font-family="Times New Roman, Times, serif" fill="#111827">{preset["title"]}</text>',
        f'<text x="{margin_left}" y="76" font-size="14" font-family="Times New Roman, Times, serif" fill="#4b5563">{preset["subtitle"]}</text>',
    ]

    left_x = margin_left
    right_x = margin_left + panel_w + panel_gap
    add_axis(parts, left_x, ap_min, ap_max, "(a) AP50-95")
    add_axis(parts, right_x, ap50_min, ap50_max, "(b) AP50")

    parts.append(
        f'<text x="{26}" y="{margin_top + plot_h / 2:.2f}" font-size="14" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827" transform="rotate(-90 26 {margin_top + plot_h / 2:.2f})">AP50-95</text>'
    )
    parts.append(
        f'<text x="{left_x + panel_w + panel_gap - 18}" y="{margin_top + plot_h / 2:.2f}" font-size="14" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827" transform="rotate(-90 {left_x + panel_w + panel_gap - 18} {margin_top + plot_h / 2:.2f})">AP50</text>'
    )

    legend_x = width - margin_right - 250
    legend_y = 52
    legend = [("Pathway probes", "#6baed6"), ("Full TSECAF-DETR", "#d62728")]
    for idx, (label, color) in enumerate(legend):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="18" height="18" fill="{color}" stroke="#4b5563" stroke-width="0.8" />')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 4}" font-size="13" font-family="Times New Roman, Times, serif" fill="#111827">{label}</text>')

    for idx, row in enumerate(rows):
        x = start_x + idx * (bar_width + group_gap)
        fill = bar_colors[idx]
        edge = edge_colors[idx]

        ap_y = y_map(row["ap"], ap_min, ap_max)
        ap_h = margin_top + plot_h - ap_y
        parts.append(
            f'<rect x="{x:.2f}" y="{ap_y:.2f}" width="{bar_width}" height="{ap_h:.2f}" fill="{fill}" fill-opacity="0.92" stroke="{edge}" stroke-width="1.0" rx="4" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{ap_y - 10:.2f}" font-size="12" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827">{row["ap"]:.4f}</text>'
        )

        delta = row["ap"] - baseline_ap
        delta_text = "ref" if idx == 0 else f"{delta:+.4f}"
        delta_fill = "#6b7280" if idx == 0 else ("#b91c1c" if delta > 0 else "#1f2937")
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + 18:.2f}" font-size="11" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="{delta_fill}">{delta_text}</text>'
        )

        ap50_y = y_map(row["ap50"], ap50_min, ap50_max)
        ap50_h = margin_top + plot_h - ap50_y
        parts.append(
            f'<rect x="{right_x + (x - left_x):.2f}" y="{ap50_y:.2f}" width="{bar_width}" height="{ap50_h:.2f}" fill="{fill}" fill-opacity="0.92" stroke="{edge}" stroke-width="1.0" rx="4" />'
        )
        parts.append(
            f'<text x="{right_x + (x - left_x) + bar_width / 2:.2f}" y="{ap50_y - 10:.2f}" font-size="12" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827">{row["ap50"]:.4f}</text>'
        )

        label_x = x + bar_width / 2
        label_lines = {
            "Baseline": ["Baseline"],
            "Context-detail representation": ["Context-detail", "representation"],
            "Detail-branch emphasis": ["Detail-branch", "emphasis"],
            "Compact-query allocation": ["Compact-query", "allocation"],
            "TSECAF-DETR pathway": ["TSECAF-DETR", "pathway"],
        }.get(row["label"], [row["label"]])
        for line_idx, line in enumerate(label_lines):
            y = margin_top + plot_h + 28 + line_idx * 16
            parts.append(
                f'<text x="{label_x:.2f}" y="{y:.2f}" font-size="12" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827">{line}</text>'
            )
            parts.append(
                f'<text x="{right_x + (label_x - left_x):.2f}" y="{y:.2f}" font-size="12" text-anchor="middle" font-family="Times New Roman, Times, serif" fill="#111827">{line}</text>'
            )

    parts.append(
        f'<text x="{margin_left}" y="{height - 16}" font-size="11" font-family="Times New Roman, Times, serif" fill="#6b7280">Text above bars shows the best validation score. Text near the top of panel (a) shows the AP50-95 change relative to the baseline.</text>'
    )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))
    return output_path


def main(args) -> None:
    preset = ABLATION_PRESETS[args.preset]
    rows = []
    for item in preset["series"]:
        metrics = extract_best_metrics(Path(item["log"]))
        metrics["label"] = item["label"]
        rows.append(metrics)
        print(item["label"], metrics["epoch"], metrics["ap"], metrics["ap50"])

    output_path = save_svg(preset, rows)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot TSECAF-DETR pathway ablation bars to an SVG file.")
    parser.add_argument("--preset", choices=sorted(ABLATION_PRESETS), default="gwhd_r50_tsecaf_pathway_ablation")
    args = parser.parse_args()
    main(args)
