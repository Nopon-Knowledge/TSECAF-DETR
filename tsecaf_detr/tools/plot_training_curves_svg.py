"""Plot RT-DETR training curves as dependency-free SVG figures."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from project_paths import PROJECT_ROOT
except ModuleNotFoundError:
    from tools.project_paths import PROJECT_ROOT


PLOTS_PRESETS = {
    "gwhd_r50_baseline_vs_small_object_aware": {
        "description": "Loss/AP/AP50 curves for RT-DETRv2-R50 baseline vs TSECAF-DETR-R50 on GWHD2021.",
        "series": [
            {
                "name": "R50 Baseline",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r50vd_180e_gwhd_baseline" / "log.txt"),
                "color": "#1f77b4",
            },
            {
                "name": "TSECAF-DETR-R50",
                "log": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r50vd_180e_gwhd" / "log.txt"),
                "color": "#d62728",
            },
        ],
        "output_dir": str(PROJECT_ROOT / "vis_results" / "training_curves" / "r50_baseline_vs_tsecaf_detr"),
        "output_name": "r50_baseline_vs_tsecaf_detr_training_curves.svg",
    },
    "gwhd_r34_baseline_vs_small_object_aware": {
        "description": "Loss/AP/AP50 curves for RT-DETRv2-R34 baseline vs TSECAF-DETR-R34 on GWHD2021.",
        "series": [
            {
                "name": "R34 Baseline",
                "log": str(PROJECT_ROOT / "output" / "rtdetrv2_r34vd_180e_gwhd_baseline" / "log.txt"),
                "color": "#1f77b4",
            },
            {
                "name": "TSECAF-DETR-R34",
                "log": str(PROJECT_ROOT / "output" / "small_object_aware_rtdetr_r34vd_180e_gwhd" / "log.txt"),
                "color": "#d62728",
            },
        ],
        "output_dir": str(PROJECT_ROOT / "vis_results" / "training_curves" / "r34_baseline_vs_tsecaf_detr"),
        "output_name": "r34_baseline_vs_tsecaf_detr_training_curves.svg",
    },
}

PLOTS_PRESETS.update({
    "gwhd_r50_baseline_vs_tsecaf_detr": PLOTS_PRESETS["gwhd_r50_baseline_vs_small_object_aware"],
    "gwhd_r34_baseline_vs_tsecaf_detr": PLOTS_PRESETS["gwhd_r34_baseline_vs_small_object_aware"],
})


def parse_log(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
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
        loss = record.get("train_loss")
        if epoch is None or loss is None or ap is None or ap50 is None:
            continue

        rows.append(
            {
                "epoch": float(epoch),
                "loss": float(loss),
                "ap": float(ap),
                "ap50": float(ap50),
            }
        )
    rows.sort(key=lambda row: row["epoch"])
    return rows


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_bounds(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - pad, hi + pad
    span = hi - lo
    return lo - span * 0.05, hi + span * 0.08


def to_polyline(rows: List[Dict[str, float]], key: str, x0: float, y0: float, width: float, height: float, x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    points = []
    for row in rows:
        x = x0 + (row["epoch"] - x_min) / (x_max - x_min) * width
        y = y0 + height - (row[key] - y_min) / (y_max - y_min) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def axis_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if count <= 1:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def draw_panel(
    parts: List[str],
    title: str,
    key: str,
    series: List[Dict[str, object]],
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> None:
    x_vals = [row["epoch"] for item in series for row in item["rows"]]
    y_vals = [row[key] for item in series for row in item["rows"]]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = nice_bounds(y_vals)

    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#d0d7de" stroke-width="1"/>')
    parts.append(f'<text x="{x0}" y="{y0 - 14}" font-size="16" font-weight="700" fill="#111827">{escape_xml(title)}</text>')

    for tick in axis_ticks(y_min, y_max):
        y = y0 + height - (tick - y_min) / (y_max - y_min) * height
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + width}" y2="{y:.2f}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 4:.2f}" font-size="11" text-anchor="end" fill="#6b7280">{tick:.2f}</text>')

    for tick in axis_ticks(x_min, x_max):
        x = x0 + (tick - x_min) / (x_max - x_min) * width
        parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + height}" stroke="#f5f7fa" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{y0 + height + 18}" font-size="11" text-anchor="middle" fill="#6b7280">{int(round(tick))}</text>')

    parts.append(f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#9ca3af" stroke-width="1.2"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#9ca3af" stroke-width="1.2"/>')

    for item in series:
        polyline = to_polyline(item["rows"], key, x0, y0, width, height, x_min, x_max, y_min, y_max)
        parts.append(
            f'<polyline fill="none" stroke="{item["color"]}" stroke-width="2.5" points="{polyline}" />'
        )


def render_svg(title: str, series: List[Dict[str, object]], output_path: Path) -> None:
    width = 1320
    height = 980
    margin = 78
    panel_w = width - margin * 2
    panel_h = 210
    panel_gap = 82
    start_y = 110

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="44" font-size="26" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
        f'<text x="{margin}" y="70" font-size="13" fill="#4b5563">Training curves exported from RT-DETR log.txt</text>',
    ]

    legend_x = width - margin - 260
    legend_y = 42
    for idx, item in enumerate(series):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{item["color"]}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 36}" y="{y + 4}" font-size="13" fill="#111827">{escape_xml(str(item["name"]))}</text>'
        )

    draw_panel(parts, "Training Loss", "loss", series, margin, start_y, panel_w, panel_h)
    draw_panel(parts, "mAP50-95", "ap", series, margin, start_y + panel_h + panel_gap, panel_w, panel_h)
    draw_panel(parts, "AP50", "ap50", series, margin, start_y + 2 * (panel_h + panel_gap), panel_w, panel_h)

    parts.append("</svg>")
    output_path.write_text("\n".join(parts))


def main(args) -> None:
    preset = PLOTS_PRESETS[args.preset]
    output_dir = Path(preset["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / preset["output_name"]

    series = []
    for item in preset["series"]:
        rows = parse_log(Path(item["log"]))
        if not rows:
            raise RuntimeError(f"No usable rows parsed from {item['log']}")
        series.append(
            {
                "name": item["name"],
                "color": item["color"],
                "rows": rows,
            }
        )
        print(f"parsed {len(rows)} epochs from {item['log']}")

    render_svg(preset["description"], series, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RT-DETR training curves to an SVG file.")
    parser.add_argument("--preset", choices=sorted(PLOTS_PRESETS), default="gwhd_r50_baseline_vs_tsecaf_detr")
    args = parser.parse_args()
    main(args)
