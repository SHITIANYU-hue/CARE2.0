#!/usr/bin/env python3
"""Plot the IRED source-outcome assignment falsification from saved traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


TRUE_CONDITION = "true_source_outcomes"
BASELINE_CONDITION = "target_gp_ucb"
COLORS = {
    "target": "#4A5568",
    "null": "#9AA5B1",
    "llm": "#137A68",
    "fixed": "#C97835",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n",
        encoding="utf-8",
    )


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = 5_000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = np.mean(array[indices], axis=1)
    return (
        float(np.mean(array)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    write_sha256(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reference_fixed_effect(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        item
        for item in rows
        if item["route_id"] == "official_test"
        and item["method"] == "multisource_skill_prior"
    )
    return {
        "mean": float(row["mean_delta_auc"]),
        "low": float(row["bootstrap_ci95_low"]),
        "high": float(row["bootstrap_ci95_high"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--reference-effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(
        (args.result_root / "falsification_report.json").read_text(encoding="utf-8")
    )
    effects_path = args.result_root / "assignment_effects.csv"
    with effects_path.open(newline="", encoding="utf-8") as handle:
        effects = list(csv.DictReader(handle))
    trajectories = load_jsonl(args.result_root / "trajectory_audit.jsonl")
    by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for row in trajectories:
        by_condition.setdefault(str(row["condition"]), {})[
            int(row["target_seed"])
        ] = row
    target_seeds = sorted(by_condition[BASELINE_CONDITION])
    null_conditions = sorted(
        condition
        for condition in by_condition
        if condition not in {BASELINE_CONDITION, TRUE_CONDITION}
    )
    baseline_auc = {
        seed: float(by_condition[BASELINE_CONDITION][seed]["best_so_far_auc"])
        for seed in target_seeds
    }
    null_seed_auc = {
        seed: float(
            np.mean(
                [
                    float(by_condition[condition][seed]["best_so_far_auc"])
                    for condition in null_conditions
                ]
            )
        )
        for seed in target_seeds
    }
    true_auc_deltas = [
        float(by_condition[TRUE_CONDITION][seed]["best_so_far_auc"])
        - baseline_auc[seed]
        for seed in target_seeds
    ]
    null_mean_auc_deltas = [
        null_seed_auc[seed] - baseline_auc[seed] for seed in target_seeds
    ]
    true_effect = bootstrap_mean_interval(
        true_auc_deltas, draws=50_000, seed=98301
    )
    null_mean_effect = bootstrap_mean_interval(
        null_mean_auc_deltas, draws=50_000, seed=98302
    )
    fixed_effect = reference_fixed_effect(args.reference_effects)
    null_assignment_effects = [
        float(row["auc_mean_delta"])
        for row in effects
        if row["condition"] != TRUE_CONDITION
    ]

    rounds = len(by_condition[BASELINE_CONDITION][target_seeds[0]]["best_trace"])
    trajectory_rows: list[dict[str, Any]] = []
    trajectory_series: dict[str, list[tuple[float, float, float]]] = {
        "Target-only GP-UCB": [],
        "Outcome-permuted skills": [],
        "LLM skill, true assignment": [],
    }
    for round_index in range(rounds):
        condition_values = {
            "Target-only GP-UCB": [
                float(by_condition[BASELINE_CONDITION][seed]["best_trace"][round_index])
                for seed in target_seeds
            ],
            "Outcome-permuted skills": [
                float(
                    np.mean(
                        [
                            float(
                                by_condition[condition][seed]["best_trace"][round_index]
                            )
                            for condition in null_conditions
                        ]
                    )
                )
                for seed in target_seeds
            ],
            "LLM skill, true assignment": [
                float(by_condition[TRUE_CONDITION][seed]["best_trace"][round_index])
                for seed in target_seeds
            ],
        }
        for series_index, (label, values) in enumerate(condition_values.items()):
            estimate = bootstrap_mean_interval(
                values,
                draws=20_000,
                seed=98400 + 100 * series_index + round_index,
            )
            trajectory_series[label].append(estimate)
            trajectory_rows.append(
                {
                    "series": label,
                    "round": round_index + 1,
                    "seed_count": len(values),
                    "mean_best_so_far": estimate[0],
                    "bootstrap_ci95_low": estimate[1],
                    "bootstrap_ci95_high": estimate[2],
                }
            )

    figure = plt.figure(figsize=(15.2, 5.3), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.05, 1.0, 1.45))
    axis_null = figure.add_subplot(grid[0, 0])
    axis_effect = figure.add_subplot(grid[0, 1])
    axis_trace = figure.add_subplot(grid[0, 2])

    axis_null.hist(
        null_assignment_effects,
        bins=18,
        color=COLORS["null"],
        edgecolor="white",
        linewidth=0.8,
    )
    axis_null.axvline(
        true_effect[0], color=COLORS["llm"], linewidth=2.6, label="True assignment"
    )
    axis_null.axvline(0.0, color="#2D3748", linewidth=1.0, linestyle="--")
    p_value = float(
        report["true_vs_permuted_assignment"]["one_sided_randomization_p_auc"]
    )
    axis_null.text(
        0.04,
        0.95,
        f"99 outcome permutations\none-sided p = {p_value:.3f}",
        transform=axis_null.transAxes,
        va="top",
        fontsize=10,
    )
    axis_null.set_title("A  Does the correct source binding matter?", loc="left", fontweight="bold")
    axis_null.set_xlabel("Mean paired AUC delta vs target GP")
    axis_null.set_ylabel("Outcome-permuted assignments")
    axis_null.spines[["top", "right"]].set_visible(False)

    labels = ["Permuted\nsource skills", "LLM skill\ntrue binding", "Fixed rank\nprior"]
    estimates = [null_mean_effect, true_effect, (fixed_effect["mean"], fixed_effect["low"], fixed_effect["high"])]
    colors = [COLORS["null"], COLORS["llm"], COLORS["fixed"]]
    y_positions = np.arange(len(labels))[::-1]
    for y_position, label, estimate, color in zip(y_positions, labels, estimates, colors):
        axis_effect.errorbar(
            estimate[0],
            y_position,
            xerr=[[estimate[0] - estimate[1]], [estimate[2] - estimate[0]]],
            fmt="o",
            markersize=7,
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
        )
        axis_effect.text(
            estimate[2] + 0.04,
            y_position,
            f"{estimate[0]:+.3f}",
            va="center",
            fontsize=9.5,
            color="#24313A",
        )
    axis_effect.axvline(0.0, color="#2D3748", linewidth=1.0, linestyle="--")
    axis_effect.set_yticks(y_positions, labels)
    axis_effect.set_title("B  Matched official-test effects", loc="left", fontweight="bold")
    axis_effect.set_xlabel("Paired best-so-far AUC delta (95% CI)")
    axis_effect.spines[["top", "right", "left"]].set_visible(False)
    axis_effect.tick_params(axis="y", length=0)

    x = np.arange(1, rounds + 1)
    trace_colors = {
        "Target-only GP-UCB": COLORS["target"],
        "Outcome-permuted skills": COLORS["null"],
        "LLM skill, true assignment": COLORS["llm"],
    }
    for label, estimates_by_round in trajectory_series.items():
        means = np.asarray([item[0] for item in estimates_by_round])
        lows = np.asarray([item[1] for item in estimates_by_round])
        highs = np.asarray([item[2] for item in estimates_by_round])
        axis_trace.plot(
            x,
            means,
            linewidth=2.4 if "true" in label else 1.9,
            color=trace_colors[label],
            label=label,
        )
        axis_trace.fill_between(
            x,
            lows,
            highs,
            color=trace_colors[label],
            alpha=0.12,
            linewidth=0,
        )
    axis_trace.set_title("C  Search trajectory under the same budget", loc="left", fontweight="bold")
    axis_trace.set_xlabel("Target reveal round")
    axis_trace.set_ylabel("Mean best-so-far IRED activity")
    axis_trace.set_xticks(x)
    axis_trace.grid(axis="y", color="#DDE2E7", linewidth=0.7)
    axis_trace.spines[["top", "right"]].set_visible(False)
    axis_trace.legend(frameon=False, fontsize=9, loc="lower right")

    figure.suptitle(
        "IRED mechanism audit: testing whether positive transfer depends on the measured source-outcome binding",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure.text(
        0.01,
        -0.02,
        "Official FLIP2 two-to-many test; 100 paired target seeds, 3 initial observations + 12 reveals. "
        "This audit is post-hoc to the prospective IRED result; it tests mechanism, not independent confirmation.",
        fontsize=9,
        color="#4A5568",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        path = args.output.with_suffix(suffix)
        figure.savefig(path, dpi=260 if suffix == ".png" else None, bbox_inches="tight")
        write_sha256(path)
    plt.close(figure)

    source_rows = [
        {
            "method": "outcome_permuted_skill_mean",
            "mean_auc_delta": null_mean_effect[0],
            "bootstrap_ci95_low": null_mean_effect[1],
            "bootstrap_ci95_high": null_mean_effect[2],
        },
        {
            "method": "llm_additive_skill_true_assignment",
            "mean_auc_delta": true_effect[0],
            "bootstrap_ci95_low": true_effect[1],
            "bootstrap_ci95_high": true_effect[2],
        },
        {
            "method": "fixed_skill_prior",
            "mean_auc_delta": fixed_effect["mean"],
            "bootstrap_ci95_low": fixed_effect["low"],
            "bootstrap_ci95_high": fixed_effect["high"],
        },
    ]
    write_csv(args.output.with_name(f"{args.output.name}_plot_effects.csv"), source_rows)
    write_csv(
        args.output.with_name(f"{args.output.name}_plot_trajectories.csv"),
        trajectory_rows,
    )
    stats_path = args.output.with_name(f"{args.output.name}_plot_stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": "care.ired_source_outcome_falsification_plot/v1",
                "target_seed_count": len(target_seeds),
                "permutation_count": len(null_conditions),
                "true_effect": {
                    "mean": true_effect[0],
                    "ci95_low": true_effect[1],
                    "ci95_high": true_effect[2],
                },
                "permutation_mean_effect": {
                    "mean": null_mean_effect[0],
                    "ci95_low": null_mean_effect[1],
                    "ci95_high": null_mean_effect[2],
                },
                "fixed_prior_effect": fixed_effect,
                "randomization_p": p_value,
                "source_files": {
                    "falsification_report": str(args.result_root / "falsification_report.json"),
                    "assignment_effects": str(effects_path),
                    "trajectory_audit": str(args.result_root / "trajectory_audit.jsonl"),
                    "reference_effects": str(args.reference_effects),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sha256(stats_path)


if __name__ == "__main__":
    main()
