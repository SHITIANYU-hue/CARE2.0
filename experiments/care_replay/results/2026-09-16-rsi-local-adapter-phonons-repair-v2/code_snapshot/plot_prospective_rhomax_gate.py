#!/usr/bin/env python3
"""Plot the preregistered FLIP2 Rhomax family-gate confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "multisource_rgpe": "RGPE",
    "multisource_icm_bma": "ICM-BMA",
    "multisource_skill_prior": "Fixed skill prior",
}
METHOD_COLORS = {
    "multisource_rgpe": "#2E6F9E",
    "multisource_icm_bma": "#7A4EAB",
    "multisource_skill_prior": "#D47A22",
}
ROUTES = (
    (
        "train_to_validation",
        "Train -> validation",
        "calibration/train_to_validation/summary.json",
        False,
    ),
    (
        "validation_to_train",
        "Validation -> train",
        "calibration/validation_to_train/summary.json",
        False,
    ),
    (
        "official_test",
        "Train + validation -> official test",
        "deployment/summary.json",
        True,
    ),
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_plot_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route_id, route_label, relative_path, is_test in ROUTES:
        summary = read_json(result_root / relative_path)
        comparisons = summary["comparisons_vs_target_gp_ucb"]
        for method in METHOD_LABELS:
            effect = comparisons[method]["best_so_far_auc"]
            rows.append(
                {
                    "route_id": route_id,
                    "route_label": route_label,
                    "is_official_test": is_test,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed_count": comparisons[method]["seed_count"],
                    "mean_delta_auc": effect["mean_delta"],
                    "ci95_low": effect["normal_95ci_low"],
                    "ci95_high": effect["normal_95ci_high"],
                }
            )
    return rows


def write_plot_data(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_sha256(path)


def plot(result_root: Path, output_stem: Path) -> None:
    gate_summary = read_json(result_root / "source_only_gate_summary.json")
    rows = load_plot_rows(result_root)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    write_plot_data(rows, output_stem.with_name(output_stem.name + "_data.csv"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#23313D",
            "axes.labelcolor": "#23313D",
            "xtick.color": "#4B5964",
            "ytick.color": "#23313D",
        }
    )
    figure = plt.figure(figsize=(12.4, 6.3), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.65, 0.78))
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.74, wspace=0.34)

    axis = figure.add_subplot(grid[0, 0])
    method_offsets = {
        "multisource_rgpe": 0.23,
        "multisource_icm_bma": 0.0,
        "multisource_skill_prior": -0.23,
    }
    route_base = {
        "train_to_validation": 2.0,
        "validation_to_train": 1.0,
        "official_test": 0.0,
    }
    for row in rows:
        y = route_base[row["route_id"]] + method_offsets[row["method"]]
        mean = float(row["mean_delta_auc"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        axis.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[row["method"]],
            elinewidth=2.0,
            capsize=3.5,
            label=METHOD_LABELS[row["method"]]
            if row["route_id"] == "train_to_validation"
            else None,
            zorder=3,
        )
        axis.text(
            high + 0.12,
            y,
            f"{mean:+.2f}",
            va="center",
            fontsize=8.5,
            color="#334550",
        )

    axis.axvline(0.0, color="#1C2830", linewidth=1.25, linestyle="--", zorder=1)
    axis.axhspan(-0.47, 0.47, color="#F1F5F6", zorder=0)
    axis.axhline(0.5, color="#B9C2C8", linewidth=0.8)
    axis.set_yticks(
        [2.0, 1.0, 0.0],
        [
            "Train -> validation\n50 paired seeds",
            "Validation -> train\n50 paired seeds",
            "Official test\n100 paired seeds",
        ],
    )
    axis.set_ylim(-0.55, 2.55)
    axis.set_xlim(-5.25, 2.9)
    axis.set_xlabel("Paired ΔAUC vs target-only GP-UCB (95% CI)")
    axis.set_title(
        "A. Source-only calibration, then untouched official test",
        loc="left",
        y=1.11,
        pad=0,
    )
    axis.grid(axis="x", color="#DCE2E5", linewidth=0.75, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.4,
    )

    callout = figure.add_subplot(grid[0, 1])
    callout.axis("off")
    deployment = gate_summary["deployment"]
    selected = gate_summary["gate"]["ungated_source_only_selection"]
    avoided = float(deployment["negative_transfer_auc_avoided"])
    callout.text(0.0, 1.08, "B. Preregistered decision", fontsize=12.5, fontweight="bold")
    callout.text(0.0, 0.84, "Frozen before data download", fontsize=9, color="#596773")
    callout.text(0.0, 0.78, "Protocol + code committed", fontsize=16, fontweight="bold", color="#1D5D7B")
    callout.text(0.0, 0.63, "Eligibility rule", fontsize=9, color="#596773")
    callout.text(0.0, 0.52, "95% CI lower bound > 0\non both calibration routes", fontsize=12, fontweight="bold")
    callout.text(0.0, 0.36, "Ungated choice", fontsize=9, color="#596773")
    callout.text(0.0, 0.29, METHOD_LABELS[selected], fontsize=15, fontweight="bold", color="#D47A22")
    callout.text(0.0, 0.19, "Gate decision", fontsize=9, color="#596773")
    callout.text(0.0, 0.11, "REJECT -> target-only", fontsize=17, fontweight="bold", color="#176B4A")
    callout.text(
        0.0,
        0.0,
        f"+{avoided:.2f} AUC harm avoided",
        fontsize=19,
        fontweight="bold",
        color="#176B4A",
    )

    figure.suptitle(
        "Prospective FLIP2 Rhomax confirmation: the gate rejects harmful transfer",
        x=0.10,
        ha="left",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.10,
        0.885,
        "Official by-wild-type split. Test outcomes are used only after the frozen gate decision for evaluation.",
        fontsize=9.5,
        color="#596773",
        ha="left",
    )
    figure.text(
        0.10,
        0.035,
        "Interpretation: prospective negative-transfer prevention on an external protein family; not positive-transfer efficacy and not a wet-lab result.",
        fontsize=8.5,
        color="#596773",
        ha="left",
    )

    for suffix in (".png", ".pdf", ".svg"):
        output = output_stem.with_suffix(suffix)
        figure.savefig(output, dpi=240 if suffix == ".png" else None, bbox_inches="tight")
        write_sha256(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.result_root, args.output)


if __name__ == "__main__":
    main()
