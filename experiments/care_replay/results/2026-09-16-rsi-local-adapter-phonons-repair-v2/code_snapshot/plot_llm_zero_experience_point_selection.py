#!/usr/bin/env python3
"""Plot the recorded zero-experience LLM point-selection result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


LABELS = {
    "real_moleculenet_lipophilicity": "MoleculeNet\nLipophilicity",
    "real_matbench_expt_gap": "Matbench\nExperimental gap",
    "real_baumgartner_suzuki_minlp2": "Baumgartner\nSuzuki MINLP2",
    "real_photocatalytic_hydrogen_evolution": "Photocatalytic\nhydrogen evolution",
}
COLORS = ["#237a8e", "#3a7d44", "#bd6b2f", "#8b5ea7"]


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - spread, center + spread


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["dataset_id"]].append(row)
    return dict(grouped)


def plot(summary_path: Path, decisions_path: Path, output_base: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_rows(decisions_path)
    dataset_ids = [
        dataset_id
        for dataset_id in LABELS
        if dataset_id in summary["dataset_summaries"]
    ]
    if set(dataset_ids) != set(rows):
        raise ValueError("Summary and decisions contain different datasets")

    deltas = []
    delta_low = []
    delta_high = []
    hit_rates = []
    hit_low = []
    hit_high = []
    for dataset_id in dataset_ids:
        dataset_summary = summary["dataset_summaries"][dataset_id]
        delta = float(dataset_summary["mean_llm_minus_random_expected"])
        low, high = dataset_summary[
            "bootstrap_95ci_llm_minus_random_expected"
        ]
        deltas.append(delta)
        delta_low.append(delta - float(low))
        delta_high.append(float(high) - delta)

        dataset_rows = rows[dataset_id]
        successes = sum(
            str(row["top_quartile_hit"]).lower() == "true"
            for row in dataset_rows
        )
        rate = successes / len(dataset_rows)
        low_rate, high_rate = wilson_interval(successes, len(dataset_rows))
        hit_rates.append(rate)
        hit_low.append(rate - low_rate)
        hit_high.append(high_rate - rate)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    positions = list(range(len(dataset_ids)))
    labels = [LABELS[dataset_id] for dataset_id in dataset_ids]

    axes[0].bar(
        positions,
        deltas,
        color=COLORS[: len(dataset_ids)],
        edgecolor="#222222",
        linewidth=0.7,
        yerr=[delta_low, delta_high],
        capsize=4,
    )
    axes[0].axhline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Selected objective - menu random expectation")
    axes[0].set_title("A. Outcome-blind LLM value advantage")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_axisbelow(True)

    axes[1].bar(
        positions,
        hit_rates,
        color=COLORS[: len(dataset_ids)],
        edgecolor="#222222",
        linewidth=0.7,
        yerr=[hit_low, hit_high],
        capsize=4,
    )
    axes[1].axhline(
        0.25,
        color="#333333",
        linewidth=1.2,
        linestyle="--",
        label="Random expectation (3 of 12)",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Top-quartile hit rate")
    axes[1].set_title("B. Did the LLM select a top-quartile point?")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle(
        "LLM-only point selection without source data, RAG, target history, or GP scores",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "30 frozen random menus per dataset; 12 candidates per menu. "
        "Left: bootstrap 95% CI. Right: Wilson 95% CI.",
        ha="center",
        color="#444444",
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args(argv)
    plot(args.summary, args.decisions, args.output_base)


if __name__ == "__main__":
    main()
