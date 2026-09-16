#!/usr/bin/env python3
"""Render a dependency-free SVG forest plot for the semantic transfer extension."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PAIR_LABELS = {
    "esol_to_lipophilicity": "ESOL → Lipophilicity",
    "lipophilicity_to_freesolv": "Lipophilicity → FreeSolv",
    "matbench_expt_gap_to_dielectric": "Expt. gap → Dielectric",
    "matbench_phonons_to_dielectric": "Phonons → Dielectric",
    "chemlex_to_buchwald_hartwig": "ChemLex → Buchwald–Hartwig",
}
PANELS = (
    ("final_best", "Final best delta"),
    ("best_so_far_auc", "Best-so-far AUC delta"),
)


def color_for(stats: dict[str, float]) -> str:
    if float(stats["normal_95ci_low"]) > 0.0:
        return "#12805c"
    if float(stats["normal_95ci_high"]) < 0.0:
        return "#c33d45"
    return "#b17914"


def panel_range(rows: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    values = [0.0]
    for row in rows:
        stats = row["comparisons"]["source_router_minus_target_only_llm_router"][
            "metrics"
        ][metric]
        values.extend([
            float(stats["normal_95ci_low"]),
            float(stats["normal_95ci_high"]),
        ])
    low, high = min(values), max(values)
    margin = max((high - low) * 0.10, 0.25)
    return low - margin, high + margin


def x_position(value: float, low: float, high: float, left: float, width: float) -> float:
    return left + (value - low) / (high - low) * width


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    weight: int = 400,
    fill: str = "#17202a",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["pairs"]
    width, height = 2000, 980
    label_width = 390
    panel_width = 680
    panel_gap = 95
    panel_lefts = (label_width, label_width + panel_width + panel_gap)
    plot_top = 260
    row_step = 120

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        text(90, 88, "Source evidence gain beyond target-only LLM", size=42, weight=700),
        text(
            90,
            138,
            "Frozen skill identity · 300 paired held-out seeds · mean and normal 95% CI",
            size=24,
            fill="#55606b",
        ),
        '<rect x="90" y="178" width="22" height="22" rx="3" fill="#12805c"/>',
        text(126, 197, "Significant gain", size=21, fill="#44505b"),
        '<rect x="338" y="178" width="22" height="22" rx="3" fill="#b17914"/>',
        text(374, 197, "Not significant", size=21, fill="#44505b"),
        '<rect x="606" y="178" width="22" height="22" rx="3" fill="#c33d45"/>',
        text(642, 197, "Significant harm", size=21, fill="#44505b"),
    ]

    for panel_index, (metric, title) in enumerate(PANELS):
        panel_left = panel_lefts[panel_index]
        low, high = panel_range(rows, metric)
        zero_x = x_position(0.0, low, high, panel_left, panel_width)
        svg.extend([
            text(panel_left, 224, title, size=27, weight=700),
            f'<line x1="{panel_left}" y1="{plot_top - 28}" '
            f'x2="{panel_left + panel_width}" y2="{plot_top - 28}" '
            'stroke="#d9dee3" stroke-width="2"/>',
            f'<line x1="{zero_x:.1f}" y1="{plot_top - 16}" '
            f'x2="{zero_x:.1f}" y2="{plot_top + row_step * (len(rows) - 1) + 36}" '
            'stroke="#69737d" stroke-width="2" stroke-dasharray="8 8"/>',
            text(panel_left, 878, f"{low:.1f}", size=19, fill="#68737d"),
            text(
                panel_left + panel_width,
                878,
                f"{high:.1f}",
                size=19,
                fill="#68737d",
                anchor="end",
            ),
            text(zero_x, 906, "0", size=19, fill="#68737d", anchor="middle"),
        ])
        for row_index, row in enumerate(rows):
            y = plot_top + row_step * row_index
            stats = row["comparisons"]["source_router_minus_target_only_llm_router"][
                "metrics"
            ][metric]
            mean = float(stats["mean"])
            ci_low = float(stats["normal_95ci_low"])
            ci_high = float(stats["normal_95ci_high"])
            color = color_for(stats)
            x_low = x_position(ci_low, low, high, panel_left, panel_width)
            x_high = x_position(ci_high, low, high, panel_left, panel_width)
            x_mean = x_position(mean, low, high, panel_left, panel_width)
            svg.extend([
                f'<line x1="{panel_left}" y1="{y + 46}" '
                f'x2="{panel_left + panel_width}" y2="{y + 46}" '
                'stroke="#eef1f3" stroke-width="2"/>',
                f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="8" stroke-linecap="round"/>',
                f'<line x1="{x_low:.1f}" y1="{y - 13:.1f}" x2="{x_low:.1f}" '
                f'y2="{y + 13:.1f}" stroke="{color}" stroke-width="4"/>',
                f'<line x1="{x_high:.1f}" y1="{y - 13:.1f}" x2="{x_high:.1f}" '
                f'y2="{y + 13:.1f}" stroke="{color}" stroke-width="4"/>',
                f'<circle cx="{x_mean:.1f}" cy="{y:.1f}" r="12" fill="{color}" '
                'stroke="#ffffff" stroke-width="4"/>',
                text(
                    panel_left,
                    y + 34,
                    f"{mean:+.2f} [{ci_low:+.2f}, {ci_high:+.2f}]",
                    size=18,
                    fill=color,
                ),
            ])

    for row_index, row in enumerate(rows):
        y = plot_top + row_step * row_index
        label = PAIR_LABELS.get(row["pair_id"], row["pair_id"])
        domain = str(row["domain"]).replace("_", " ")
        svg.extend([
            text(90, y - 3, label, size=24, weight=700),
            text(90, y + 29, domain, size=18, fill="#69737d"),
        ])

    svg.extend([
        text(
            90,
            944,
            "Positive values favor the source-schema router. "
            "Phonons → Dielectric improves both primary metrics but reduces top-10 hit rate.",
            size=20,
            fill="#55606b",
        ),
        "</svg>",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
