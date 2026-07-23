#!/usr/bin/env python3
"""Plot held-out CARE 2.0 source-outcome transfer and round efficiency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PAIR_LABELS = {
    ("real_moleculenet_esol", "real_moleculenet_lipophilicity"):
        "ESOL -> Lipo",
    ("real_moleculenet_lipophilicity", "real_moleculenet_freesolv"):
        "Lipo -> FreeSolv",
    ("real_matbench_expt_gap", "real_matbench_dielectric"):
        "Gap -> Dielectric",
    ("real_matbench_phonons", "real_matbench_dielectric"):
        "Phonons -> Dielectric",
    ("real_chemlex_acidamine", "real_buchwald_hartwig"):
        "ChemLex -> BH",
    ("real_matbench_dielectric", "real_matbench_expt_gap"):
        "Dielectric -> Gap",
    ("real_moleculenet_freesolv", "real_moleculenet_lipophilicity"):
        "FreeSolv -> Lipo",
}


def pair_label(pair: dict[str, Any]) -> str:
    return PAIR_LABELS.get(
        (pair["source_dataset"], pair["target_dataset"]),
        f"{pair['source_dataset']} -> {pair['target_dataset']}",
    )


def load_round_reports(round_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for path in round_dir.glob("*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[(
            str(report["source_dataset"]),
            str(report["target_dataset"]),
        )] = report
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-report", type=Path, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.goal_report.read_text(encoding="utf-8"))
    round_reports = load_round_reports(args.round_dir)
    pairs = report["pairs"]
    labels = [pair_label(pair) for pair in pairs]
    y = np.arange(len(pairs))
    deployed_final = [pair["final_best"]["mean"] for pair in pairs]
    deployed_auc = [pair["best_so_far_auc"]["mean"] for pair in pairs]
    raw_composite = [
        pair["raw_source_route"]["composite"]["mean"] for pair in pairs
    ]
    raw_low = [
        pair["raw_source_route"]["composite"]["normal_95ci_low"]
        for pair in pairs
    ]
    raw_high = [
        pair["raw_source_route"]["composite"]["normal_95ci_high"]
        for pair in pairs
    ]
    selected = [pair["selected_source_outcome_transfer"] for pair in pairs]
    round_savings = []
    round_low = []
    round_high = []
    for pair in pairs:
        efficiency = round_reports[(
            pair["source_dataset"],
            pair["target_dataset"],
        )]["rounds_to_matched_llm_final"]["paired_round_saving"]
        round_savings.append(efficiency["mean"])
        round_low.append(efficiency["normal_95ci_low"])
        round_high.append(efficiency["normal_95ci_high"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.4, 5.4),
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]},
    )
    fig.patch.set_facecolor("#FAFAF8")
    for axis in axes:
        axis.set_facecolor("#FAFAF8")
        axis.axvline(0.0, color="#60646C", linewidth=0.8, zorder=0)
        axis.grid(axis="x", color="#D9D9D4", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    height = 0.34
    axes[0].barh(
        y - height / 2,
        deployed_final,
        height,
        color="#267A68",
        label="Final best",
    )
    axes[0].barh(
        y + height / 2,
        deployed_auc,
        height,
        color="#D38335",
        label="Best-so-far AUC",
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("A. Deployed gain vs matched LLM", loc="left")
    axes[0].set_xlabel("Paired held-out delta")
    axes[0].legend(frameon=False, loc="lower right")

    colors = ["#267A68" if value else "#8B8D98" for value in selected]
    raw_errors = np.array([
        np.array(raw_composite) - np.array(raw_low),
        np.array(raw_high) - np.array(raw_composite),
    ])
    for index, color in enumerate(colors):
        axes[1].errorbar(
            raw_composite[index],
            y[index],
            xerr=raw_errors[:, index].reshape(2, 1),
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
        )
    axes[1].scatter(raw_composite, y, c=colors, s=38, zorder=3)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_title("B. Raw source route", loc="left")
    axes[1].set_xlabel("Composite delta (95% CI)")
    axes[1].text(
        0.02,
        0.98,
        "Green: deployed  |  Gray: rejected by calibration",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        color="#55575E",
        fontsize=8,
        bbox={
            "facecolor": "#FAFAF8",
            "edgecolor": "none",
            "alpha": 0.9,
            "pad": 2,
        },
    )

    round_errors = np.array([
        np.array(round_savings) - np.array(round_low),
        np.array(round_high) - np.array(round_savings),
    ])
    for index, color in enumerate(colors):
        axes[2].errorbar(
            round_savings[index],
            y[index],
            xerr=round_errors[:, index].reshape(2, 1),
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
        )
    axes[2].scatter(round_savings, y, c=colors, s=38, zorder=3)
    axes[2].set_yticks(y, [])
    axes[2].invert_yaxis()
    axes[2].set_title("C. Target rounds saved", loc="left")
    axes[2].set_xlabel("Rounds to matched-LLM final (95% CI)")

    fig.suptitle(
        "CARE 2.0: Frozen Source-Outcome Transfer on Real Benchmarks",
        x=0.075,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.96,
        "50 calibration seeds select transfer or exact fallback; "
        "100 disjoint held-out seeds provide the reported estimates.",
        ha="left",
        color="#55575E",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.08, 1.0, 0.91), w_pad=2.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=320, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
