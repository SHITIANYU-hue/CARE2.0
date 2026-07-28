#!/usr/bin/env python3
"""Build the 2026-07-28 baseline, iteration, and formula completion package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOAL_REPORT = ROOT / "results" / "goal-report-2026-07-24" / "goal_report.json"
DEFAULT_SOURCE_OUTCOME_ROOT = ROOT / "results" / "2026-07-24-source-outcome-transfer"
DEFAULT_OUTPUT = ROOT / "results" / "2026-07-28-evaluation-completion"

PAIR_LABELS = {
    ("real_chemlex_acidamine", "real_buchwald_hartwig"): "ChemLex -> BH",
    ("real_matbench_dielectric", "real_matbench_expt_gap"): "Dielectric -> Gap",
    ("real_matbench_expt_gap", "real_matbench_dielectric"): "Gap -> Dielectric",
    ("real_matbench_phonons", "real_matbench_dielectric"): "Phonons -> Dielectric",
    ("real_moleculenet_esol", "real_moleculenet_lipophilicity"): "ESOL -> Lipo",
    ("real_moleculenet_freesolv", "real_moleculenet_lipophilicity"): "FreeSolv -> Lipo",
    ("real_moleculenet_lipophilicity", "real_moleculenet_freesolv"): "Lipo -> FreeSolv",
}

DOMAIN_LABELS = {
    "real_chemlex_acidamine": "reaction",
    "real_matbench_dielectric": "materials",
    "real_matbench_expt_gap": "materials",
    "real_matbench_phonons": "materials",
    "real_moleculenet_esol": "molecular",
    "real_moleculenet_freesolv": "molecular",
    "real_moleculenet_lipophilicity": "molecular",
}

BASELINE_COVERAGE = [
    ("Random search", 0, 0, 0, 1, 0, 1, 1),
    ("Public incumbent", 0, 0, 0, 1, 0, 1, 0),
    ("GP-UCB", 0, 0, 1, 1, 0, 1, 0),
    ("Mixed-kernel GP-EI", 0, 0, 1, 1, 0, 1, 0),
    ("Target UCB/EI portfolio", 0, 0, 1, 1, 0, 1, 0),
    ("LLM direct prior", 0, 1, 1, 1, 0, 1, 0),
    ("LLAMBO-style warm start", 0, 1, 1, 1, 0, 1, 0),
    ("Matched target-only LLM", 0, 1, 1, 1, 1, 1, 0),
    ("Matched random rule", 0, 0, 1, 1, 0, 1, 1),
    ("CARE source-outcome router", 1, 1, 1, 1, 1, 1, 0),
]

COVERAGE_COLUMNS = [
    "Source outcomes",
    "LLM proposal",
    "Strong BO anchor",
    "Matched replay",
    "Calibration holdout",
    "Disjoint heldout",
    "Null control",
]


def label_pair(pair: dict[str, Any]) -> str:
    key = (str(pair["source_dataset"]), str(pair["target_dataset"]))
    return PAIR_LABELS.get(key, f"{key[0]} -> {key[1]}")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#777777",
            "figure.facecolor": "#FAFAF8",
            "axes.facecolor": "#FAFAF8",
            "savefig.facecolor": "#FAFAF8",
        }
    )


def load_round_reports(round_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(round_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[(str(report["source_dataset"]), str(report["target_dataset"]))] = report
    return reports


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key
        for row in rows[1:]
        for key in row
        if key not in fieldnames
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def baseline_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        strongest = str(pair["strongest_target_bo_mode"])
        bo = pair["pairwise"][strongest]
        matched = pair["pairwise"]["matched_target_only_llm"]
        rows.append(
            {
                "pair": label_pair(pair),
                "domain": DOMAIN_LABELS[str(pair["source_dataset"])],
                "source_dataset": pair["source_dataset"],
                "target_dataset": pair["target_dataset"],
                "deployed_mode": pair["selected_mode"],
                "transfer_deployed": int(pair["selected_source_outcome_transfer"]),
                "strongest_target_bo": strongest,
                "final_delta_vs_strongest_bo": bo["final_best_mean_delta"],
                "final_ci_low_vs_strongest_bo": bo["final_best_ci"][0],
                "final_ci_high_vs_strongest_bo": bo["final_best_ci"][1],
                "auc_delta_vs_strongest_bo": bo["auc_mean_delta"],
                "auc_ci_low_vs_strongest_bo": bo["auc_ci"][0],
                "auc_ci_high_vs_strongest_bo": bo["auc_ci"][1],
                "final_delta_vs_matched_llm": matched["final_best_mean_delta"],
                "final_ci_low_vs_matched_llm": matched["final_best_ci"][0],
                "final_ci_high_vs_matched_llm": matched["final_best_ci"][1],
                "auc_delta_vs_matched_llm": matched["auc_mean_delta"],
                "auc_ci_low_vs_matched_llm": matched["auc_ci"][0],
                "auc_ci_high_vs_matched_llm": matched["auc_ci"][1],
                "heldout_seeds": pair["heldout_seed_count"],
            }
        )
    return rows


def iteration_rows(
    pairs: list[dict[str, Any]],
    reports: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        key = (str(pair["source_dataset"]), str(pair["target_dataset"]))
        report = reports[key]
        final = report["rounds_to_matched_llm_final"]["paired_round_saving"]
        top10 = report["rounds_to_global_top10"]["paired_round_saving"]
        early = report["early_best_so_far_delta"]
        checkpoints = sorted(int(value) for value in early)
        row: dict[str, Any] = {
            "pair": label_pair(pair),
            "transfer_deployed": int(pair["selected_source_outcome_transfer"]),
            "replay_rounds": report["replay_rounds"],
            "rounds_saved_to_matched_final": final["mean"],
            "rounds_saved_final_ci_low": final["normal_95ci_low"],
            "rounds_saved_final_ci_high": final["normal_95ci_high"],
            "rounds_saved_to_global_top10": top10["mean"],
            "rounds_saved_top10_ci_low": top10["normal_95ci_low"],
            "rounds_saved_top10_ci_high": top10["normal_95ci_high"],
        }
        for checkpoint in checkpoints:
            row[f"best_delta_round_{checkpoint}"] = early[str(checkpoint)]["mean"]
            row[f"best_delta_round_{checkpoint}_ci_low"] = early[str(checkpoint)]["normal_95ci_low"]
            row[f"best_delta_round_{checkpoint}_ci_high"] = early[str(checkpoint)]["normal_95ci_high"]
        final_checkpoint = checkpoints[-1]
        row["best_delta_final"] = early[str(final_checkpoint)]["mean"]
        row["best_delta_final_ci_low"] = early[str(final_checkpoint)]["normal_95ci_low"]
        row["best_delta_final_ci_high"] = early[str(final_checkpoint)]["normal_95ci_high"]
        rows.append(row)
    return rows


def draw_forest_axis(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    mean_key: str,
    low_key: str,
    high_key: str,
    title: str,
) -> None:
    y = np.arange(len(rows))
    means = np.array([float(row[mean_key]) for row in rows])
    lows = np.array([float(row[low_key]) for row in rows])
    highs = np.array([float(row[high_key]) for row in rows])
    colors = ["#157A6E" if int(row["transfer_deployed"]) else "#858A8E" for row in rows]
    axis.axvline(0.0, color="#52585C", linewidth=0.9, zorder=0)
    axis.grid(axis="x", color="#D8DAD8", linewidth=0.6, alpha=0.9)
    for index, color in enumerate(colors):
        axis.errorbar(
            means[index],
            y[index],
            xerr=np.array([[means[index] - lows[index]], [highs[index] - means[index]]]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5.5,
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Paired held-out final-best delta (95% CI)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def plot_baseline_forest(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True)
    draw_forest_axis(
        axes[0],
        rows,
        "final_delta_vs_strongest_bo",
        "final_ci_low_vs_strongest_bo",
        "final_ci_high_vs_strongest_bo",
        "A. CARE deployment vs strongest target-only BO",
    )
    draw_forest_axis(
        axes[1],
        rows,
        "final_delta_vs_matched_llm",
        "final_ci_low_vs_matched_llm",
        "final_ci_high_vs_matched_llm",
        "B. Source transfer vs matched target-only LLM",
    )
    y = np.arange(len(rows))
    axes[0].set_yticks(y, [str(row["pair"]) for row in rows])
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    fig.suptitle(
        "CARE 2.0 baseline comparison on the frozen seven-pair suite",
        x=0.075,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "Teal: source-outcome transfer deployed after calibration. Gray: exact fallback to matched target-only LLM. "
        "All estimates use 100 disjoint held-out seeds.",
        color="#555B5E",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.05, 1.0, 0.88), w_pad=3.0)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_iteration_summary(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row["pair"]) for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 6.0),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]},
    )
    colors = ["#157A6E" if int(row["transfer_deployed"]) else "#858A8E" for row in rows]
    round_specs = [
        (
            "rounds_saved_to_matched_final",
            "rounds_saved_final_ci_low",
            "rounds_saved_final_ci_high",
            "A. Rounds saved to matched-LLM final",
        ),
        (
            "rounds_saved_to_global_top10",
            "rounds_saved_top10_ci_low",
            "rounds_saved_top10_ci_high",
            "B. Rounds saved to global top-10",
        ),
    ]
    for axis, (mean_key, low_key, high_key, title) in zip(axes[:2], round_specs):
        axis.axvline(0.0, color="#52585C", linewidth=0.9, zorder=0)
        axis.grid(axis="x", color="#D8DAD8", linewidth=0.6, alpha=0.9)
        for index, color in enumerate(colors):
            value = float(rows[index][mean_key])
            low = float(rows[index][low_key])
            high = float(rows[index][high_key])
            axis.errorbar(
                value,
                y[index],
                xerr=np.array([[value - low], [high - value]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=5.5,
                elinewidth=1.5,
                capsize=3,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Positive means fewer target acquisitions")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    checkpoints: list[int | str] = [1, 3, 5, "final"]
    matrix = np.array(
        [
            [
                float(
                    row.get(
                        "best_delta_final" if checkpoint == "final" else f"best_delta_round_{checkpoint}",
                        np.nan,
                    )
                )
                for checkpoint in checkpoints
            ]
            for row in rows
        ]
    )
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = axes[2].imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axes[2].set_title("C. Best-so-far delta by round", loc="left", fontweight="bold")
    axes[2].set_xticks(
        np.arange(len(checkpoints)),
        ["Final round" if value == "final" else f"Round {value}" for value in checkpoints],
    )
    axes[2].set_yticks(y, [])
    axes[2].tick_params(axis="x", rotation=35)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            axes[2].text(
                col_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#202426",
            )
    colorbar = fig.colorbar(image, ax=axes[2], fraction=0.047, pad=0.03)
    colorbar.set_label("CARE - matched target-only LLM")
    fig.suptitle(
        "Iteration efficiency under matched target budgets",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.93,
        "Initial observations are round 0; unreached thresholds are right-censored at budget + 1. "
        "Fallback paths reproduce the matched LLM exactly.",
        color="#555B5E",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.89), w_pad=2.2)
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_protocol_matrix(output: Path) -> None:
    names = [row[0] for row in BASELINE_COVERAGE]
    matrix = np.array([row[1:] for row in BASELINE_COVERAGE], dtype=float)
    fig, axis = plt.subplots(figsize=(11.8, 6.6))
    axis.imshow(matrix, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(COVERAGE_COLUMNS)), COVERAGE_COLUMNS)
    axis.set_yticks(np.arange(len(names)), names)
    axis.tick_params(axis="x", rotation=32)
    axis.tick_params(axis="both", length=0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                "yes" if matrix[row_index, col_index] else "-",
                ha="center",
                va="center",
                color="#FFFFFF" if matrix[row_index, col_index] else "#777B7D",
                fontsize=8,
                fontweight="bold" if matrix[row_index, col_index] else "normal",
            )
    axis.set_title(
        "Implemented comparator and control coverage",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=24,
    )
    fig.text(
        0.16,
        0.92,
        "The matrix distinguishes optimizer strength, LLM participation, source evidence, and evaluation controls. "
        "Not every historical comparator is rerun on every source-target pair.",
        color="#555B5E",
        fontsize=9,
    )
    axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    axis.grid(which="minor", color="#F1F1EE", linewidth=1.3)
    axis.spines[:].set_visible(False)
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.87))
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor="#62686B",
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(x + 0.02, y + height - 0.04, title, ha="left", va="top", fontsize=10, fontweight="bold")
    axis.text(x + 0.02, y + height - 0.10, body, ha="left", va="top", fontsize=8.4, linespacing=1.5)


def plot_formula_map(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(15.2, 7.2))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    add_box(
        axis,
        0.03,
        0.61,
        0.20,
        0.25,
        "1. LLM patch (frozen)",
        "scales s\nrole multipliers m_j\nbeta schedule\nsource-prior strength\ncalibration mode",
        "#E8F0EE",
    )
    add_box(
        axis,
        0.29,
        0.61,
        0.20,
        0.25,
        "2. Kernel geometry",
        "w_j = normalize(1 + s c_j m_j)\n\nGP-UCB / GP-EI ranks\nover a scale ensemble",
        "#EDF1F6",
    )
    add_box(
        axis,
        0.55,
        0.61,
        0.20,
        0.25,
        "3. Source-outcome prior",
        "neighbor + additive +\ninteraction effects\n\nLOO target calibration\nproduces bounded residuals",
        "#F4EFE5",
    )
    add_box(
        axis,
        0.81,
        0.61,
        0.16,
        0.25,
        "4. Expert evidence",
        "kernel LOO gain\nsource-prior CV gain\nGP marginal likelihood\npatch confidence",
        "#F2E9E8",
    )
    add_box(
        axis,
        0.20,
        0.19,
        0.25,
        0.22,
        "Target-only anchor",
        "a_anchor(x) = 0.5 rank(UCB)\n              + 0.5 rank(EI)",
        "#EEF1F1",
    )
    add_box(
        axis,
        0.55,
        0.19,
        0.27,
        0.22,
        "CARE acquisition",
        "a_CARE(x) = (1 - M) a_anchor(x)\n             + M sum_k pi_k a_k(x)\n\nM <= min(router cap, 0.55 quality)",
        "#E1EFE9",
    )
    arrows = [
        ((0.23, 0.735), (0.29, 0.735)),
        ((0.49, 0.735), (0.55, 0.735)),
        ((0.75, 0.735), (0.81, 0.735)),
        ((0.90, 0.61), (0.73, 0.41)),
        ((0.45, 0.30), (0.55, 0.30)),
        ((0.39, 0.61), (0.33, 0.41)),
        ((0.65, 0.61), (0.66, 0.41)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#62686B",
                connectionstyle="arc3,rad=0.0",
            )
        )
    axis.text(
        0.03,
        0.96,
        "How an LLM-generated CARE skill becomes an auditable acquisition rule",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        0.03,
        0.91,
        "The LLM proposes a bounded patch once. Target observations calibrate evidence; Python executes the equations and gate deterministically.",
        ha="left",
        va="top",
        fontsize=9,
        color="#555B5E",
    )
    axis.text(
        0.03,
        0.06,
        "If evidence is weak or acquisition loss exceeds the round-dependent risk budget, the selected candidate is the target-only anchor.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555B5E",
    )
    fig.savefig(output, dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_coverage_csv(path: Path) -> None:
    rows = []
    for item in BASELINE_COVERAGE:
        row = {"method": item[0]}
        row.update({column: value for column, value in zip(COVERAGE_COLUMNS, item[1:])})
        rows.append(row)
    write_csv(path, rows)


def write_checksums(output_dir: Path) -> None:
    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-report", type=Path, default=DEFAULT_GOAL_REPORT)
    parser.add_argument("--source-outcome-root", type=Path, default=DEFAULT_SOURCE_OUTCOME_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    goal = json.loads(args.goal_report.read_text(encoding="utf-8"))
    pairs = list(goal["source_outcome"])
    reports = load_round_reports(args.source_outcome_root / "round_efficiency")
    baseline = baseline_rows(pairs)
    iteration = iteration_rows(pairs, reports)
    write_csv(args.output_dir / "baseline_comparison.csv", baseline)
    write_csv(args.output_dir / "iteration_efficiency.csv", iteration)
    write_coverage_csv(args.output_dir / "baseline_protocol_coverage.csv")
    configure_plotting()
    plot_baseline_forest(baseline, args.output_dir / "baseline_comparison_forest.png")
    plot_iteration_summary(iteration, args.output_dir / "iteration_efficiency.png")
    plot_protocol_matrix(args.output_dir / "baseline_protocol_matrix.png")
    plot_formula_map(args.output_dir / "llm_rule_formula_map.png")
    write_checksums(args.output_dir)


if __name__ == "__main__":
    main()
