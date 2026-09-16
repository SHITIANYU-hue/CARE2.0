#!/usr/bin/env python3
"""Plot the measured source-only family-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


METHOD_LABELS = {
    "multisource_rgpe": "RGPE",
    "multisource_icm_bma": "ICM-BMA",
    "multisource_skill_prior": "Fixed skill prior",
}
ROUTE_LABELS = {
    "p01053_to_p0a9x9": "P01053→P0A9X9\nsource-only",
    "p0a9x9_to_p01053": "P0A9X9→P01053\nsource-only",
    "p01053_p0a9x9_to_p06241": "P01053+P0A9X9→P06241\nexternal target",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def plot(summary: Mapping[str, Any], path: Path) -> None:
    methods = list(summary["gate"]["candidate_methods"])
    routes = [
        "p01053_to_p0a9x9",
        "p0a9x9_to_p01053",
        "p01053_p0a9x9_to_p06241",
    ]
    matrix = np.zeros((len(methods), len(routes)), dtype=np.float64)
    for row_index, method in enumerate(methods):
        calibration = {
            item["route_id"]: item["mean_delta"]
            for item in summary["gate"]["method_diagnostics"][method][
                "route_effects"
            ]
        }
        matrix[row_index, 0] = calibration[routes[0]]
        matrix[row_index, 1] = calibration[routes[1]]
    external_values = summary["deployment"]["method_auc_deltas_vs_target_gp"]
    matrix[:, 2] = [external_values[method] for method in methods]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure = plt.figure(figsize=(10.8, 4.8))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.55, 0.75])
    figure.subplots_adjust(
        left=0.10, right=0.98, bottom=0.18, top=0.78, wspace=0.36
    )
    axis = figure.add_subplot(grid[0, 0])
    vmax = max(5.0, float(np.max(np.abs(matrix))))
    image = axis.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        aspect="auto",
    )
    axis.set_xticks(range(len(routes)), [ROUTE_LABELS[route] for route in routes])
    axis.set_yticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
    axis.axvline(1.5, color="#222222", linewidth=2.0)
    for row_index in range(len(methods)):
        for column_index in range(len(routes)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > vmax * 0.42 else "#15212B",
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    colorbar.set_label("Paired ΔAUC vs target-only GP-UCB")

    callout = figure.add_subplot(grid[0, 1])
    callout.axis("off")
    deployment = summary["deployment"]
    callout.text(0.0, 0.93, "B. Frozen gate decision", fontsize=12, fontweight="bold")
    callout.text(0.0, 0.78, "Source-only selected method", color="#5B6975")
    callout.text(0.0, 0.70, "ICM-BMA", fontsize=18, fontweight="bold")
    callout.text(0.0, 0.58, "Eligibility result", color="#5B6975")
    callout.text(0.0, 0.50, "REJECT TRANSFER", fontsize=18, fontweight="bold", color="#B23A3A")
    callout.text(0.0, 0.38, "Deployed policy", color="#5B6975")
    callout.text(0.0, 0.30, "target-only GP-UCB", fontsize=16, fontweight="bold", color="#196B4C")
    callout.text(0.0, 0.18, "P06241 AUC harm avoided", color="#5B6975")
    callout.text(
        0.0,
        0.08,
        f"+{deployment['negative_transfer_auc_avoided']:.2f}",
        fontsize=28,
        fontweight="bold",
        color="#196B4C",
    )
    figure.suptitle(
        "FLIP2 source-only gate: reject outcome transfer before using P06241 outcomes",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.5,
        0.875,
        "Retrospective mechanism audit designed after the external-family result; source-only gate computation excludes P06241 outcomes.",
        fontsize=8,
        color="#5B6975",
        ha="center",
        va="top",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        output = path.with_suffix(suffix)
        figure.savefig(output, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        write_sha256(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plot(summary, args.output)


if __name__ == "__main__":
    main()
