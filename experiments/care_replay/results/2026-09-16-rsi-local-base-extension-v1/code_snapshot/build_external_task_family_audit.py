#!/usr/bin/env python3
"""Build a measured-data audit for the frozen FLIP2 external task family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "multisource_rgpe": "Multisource RGPE",
    "multisource_icm_bma": "ICM-BMA",
    "multisource_skill_prior": "Fixed skill prior",
}
COLORS = {
    "multisource_rgpe": "#C94C4C",
    "multisource_icm_bma": "#D18A27",
    "multisource_skill_prior": "#7B61A8",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def build_audit(
    baseline: Mapping[str, Any],
    llm: Mapping[str, Any],
) -> dict[str, Any]:
    classical = {}
    for mode, comparison in baseline["comparisons_vs_target_gp_ucb"].items():
        effect = comparison["best_so_far_auc"]
        classical[mode] = {
            "label": LABELS[mode],
            "seed_count": comparison["seed_count"],
            "mean_auc_delta": effect["mean_delta"],
            "ci95_low": effect["normal_95ci_low"],
            "ci95_high": effect["normal_95ci_high"],
            "win_rate": effect["win_rate"],
            "non_loss_rate": effect["non_loss_rate"],
        }
    deltas = llm["deltas"]
    metrics = llm["metrics"]["online_llm_scientist"]
    return {
        "schema_version": "care.external_task_family_audit/v1",
        "dataset": "FLIP2 Hydrophobic Core official to-P06241 split",
        "source_backbones": ["P01053", "P0A9X9"],
        "target_backbone": "P06241",
        "target_candidate_count": 9972,
        "classical_transfer_vs_target_gp": classical,
        "single_online_llm_trajectory": {
            "model": llm["online_model"],
            "auc_delta_vs_same_initial_gp": deltas[
                "online_llm_increment_over_same_initial_gp"
            ]["best_so_far_auc"],
            "full_auc_delta_vs_fixed_v2": deltas[
                "full_llm_scientist_vs_fixed_v2"
            ]["best_so_far_auc"],
            "prediction_mae_after_first_reveal": metrics[
                "calibration_gate_prediction_mae"
            ],
            "later_llm_rounds_avoided": metrics[
                "calibration_gate_llm_rounds_saved"
            ],
            "nominal_llm_calls_avoided": metrics[
                "calibration_gate_nominal_llm_calls_avoided"
            ],
            "source_transfer_active_rate": metrics[
                "source_transfer_active_rate"
            ],
        },
        "interpretation": (
            "All three numerical transfer baselines show negative transfer on "
            "the frozen external protein-backbone target. The single online LLM "
            "decision matched GP rank one and then abstained, so it caused no "
            "online increment but prevented further source-conditioned actions. "
            "The LLM-generated initial design remained worse than the fixed "
            "design in this one trajectory."
        ),
        "claim_boundary": (
            "The classical effects are paired over 100 seeds. The online LLM "
            "effects are one prospective model trajectory and have no sampling "
            "confidence interval."
        ),
    }


def plot_audit(audit: Mapping[str, Any], path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 180,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    modes = list(LABELS)
    y = np.arange(len(modes))[::-1]
    means = np.asarray(
        [
            audit["classical_transfer_vs_target_gp"][mode]["mean_auc_delta"]
            for mode in modes
        ]
    )
    lows = np.asarray(
        [audit["classical_transfer_vs_target_gp"][mode]["ci95_low"] for mode in modes]
    )
    highs = np.asarray(
        [audit["classical_transfer_vs_target_gp"][mode]["ci95_high"] for mode in modes]
    )
    axes[0].axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    for index, mode in enumerate(modes):
        axes[0].errorbar(
            means[index],
            y[index],
            xerr=[[means[index] - lows[index]], [highs[index] - means[index]]],
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=2,
            color=COLORS[mode],
        )
        axes[0].text(
            means[index] - 0.5,
            y[index] + 0.18,
            f"{means[index]:+.2f}",
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLORS[mode],
            fontweight="bold",
        )
    axes[0].set_yticks(y, [LABELS[mode] for mode in modes])
    axes[0].set_xlabel("Paired $\\Delta$AUC vs target-only GP-UCB")
    axes[0].set_title("A. Frozen classical transfer (100 seeds)", loc="left")
    axes[0].grid(axis="x", color="#D8D8D8", linewidth=0.6)
    axes[0].spines[["top", "right", "left"]].set_visible(False)

    llm_record = audit["single_online_llm_trajectory"]
    llm_labels = ["Online decision\nvs same-start GP", "Full system\nvs fixed design"]
    llm_values = [
        llm_record["auc_delta_vs_same_initial_gp"],
        llm_record["full_auc_delta_vs_fixed_v2"],
    ]
    llm_colors = ["#27866F", "#C94C4C"]
    bars = axes[1].bar(
        np.arange(2),
        llm_values,
        width=0.58,
        color=llm_colors,
        edgecolor="white",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].bar_label(
        bars,
        labels=[f"{value:+.2f}" for value in llm_values],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    axes[1].set_xticks(np.arange(2), llm_labels)
    axes[1].set_ylabel("$\\Delta$AUC")
    axes[1].set_title("B. One frozen Opus trajectory", loc="left")
    axes[1].set_ylim(min(-8.5, min(llm_values) - 1.0), 2.0)
    axes[1].grid(axis="y", color="#D8D8D8", linewidth=0.6)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].text(
        0.02,
        0.03,
        "LLM selected GP rank 1, then abstained;\n11 later LLM rounds were avoided.",
        transform=axes[1].transAxes,
        fontsize=8.6,
        color="#3A3A3A",
        va="bottom",
    )

    figure.suptitle(
        "External protein-backbone transfer is a negative-transfer stress test",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.06,
        -0.01,
        "FLIP2 Hydrophobic Core, official to-P06241 split. Error bars are normal paired 95% CIs; panel B is a single model call.",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(pad=1.2)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--llm-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(
        load_json(args.baseline_summary),
        load_json(args.llm_summary),
    )
    audit_path = args.output_dir / "external_task_family_audit.json"
    figure_path = args.output_dir / "external_task_family_transfer.png"
    write_json(audit_path, audit)
    plot_audit(audit, figure_path)
    for path in (
        audit_path,
        figure_path,
        figure_path.with_suffix(".pdf"),
        figure_path.with_suffix(".svg"),
    ):
        write_sha256(path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
