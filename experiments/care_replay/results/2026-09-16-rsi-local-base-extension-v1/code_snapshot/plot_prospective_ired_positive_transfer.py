#!/usr/bin/env python3
"""Plot and tabulate the preregistered FLIP2 IRED positive-transfer result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "multisource_rgpe": "RGPE",
    "multisource_icm_bma": "ICM-BMA",
    "multisource_skill_prior": "Fixed skill prior",
    "source_additive_mutation_prior": "LLM additive skill",
}
METHOD_COLORS = {
    "multisource_rgpe": "#2E6F9E",
    "multisource_icm_bma": "#7A4EAB",
    "multisource_skill_prior": "#D47A22",
    "source_additive_mutation_prior": "#137A68",
}
ROUTES = (
    (
        "train_to_validation",
        "Train -> validation",
        "calibration/train_to_validation",
        97000,
    ),
    (
        "official_test",
        "Train + validation -> official test",
        "deployment",
        97200,
    ),
)
TRAJECTORY_METHODS = (
    "target_gp_ucb",
    "multisource_skill_prior",
    "source_additive_mutation_prior",
)
TRAJECTORY_LABELS = {
    "target_gp_ucb": "Target-only GP-UCB",
    "multisource_skill_prior": "Fixed skill prior",
    "source_additive_mutation_prior": "LLM additive skill",
}
TRAJECTORY_COLORS = {
    "target_gp_ucb": "#202A32",
    "multisource_skill_prior": METHOD_COLORS["multisource_skill_prior"],
    "source_additive_mutation_prior": METHOD_COLORS[
        "source_additive_mutation_prior"
    ],
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def paired_bootstrap_interval(
    deltas: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int = 200_000,
) -> tuple[float, float]:
    indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    means = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_two_sided_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    lower_tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * lower_tail / (2**n))


def summarize_effects(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route_index, (route_id, route_label, relative_dir, _) in enumerate(ROUTES):
        metrics = read_csv(result_root / relative_dir / "heldout_metrics.csv")
        by_method_seed = {
            (row["mode"], int(row["seed"])): float(row["best_so_far_auc"])
            for row in metrics
        }
        seeds = sorted(
            int(row["seed"])
            for row in metrics
            if row["mode"] == "target_gp_ucb"
        )
        baseline = np.asarray(
            [by_method_seed[("target_gp_ucb", seed)] for seed in seeds],
            dtype=float,
        )
        summary = read_json(result_root / relative_dir / "summary.json")
        comparisons = summary["comparisons_vs_target_gp_ucb"]
        for method_index, method in enumerate(METHOD_LABELS):
            values = np.asarray(
                [by_method_seed[(method, seed)] for seed in seeds], dtype=float
            )
            deltas = values - baseline
            rng = np.random.default_rng(20260825 + 100 * route_index + method_index)
            bootstrap_low, bootstrap_high = paired_bootstrap_interval(
                deltas, rng=rng
            )
            wins = int(np.sum(deltas > 1e-12))
            ties = int(np.sum(np.abs(deltas) <= 1e-12))
            losses = int(np.sum(deltas < -1e-12))
            normal_effect = comparisons[method]["best_so_far_auc"]
            rows.append(
                {
                    "route_id": route_id,
                    "route_label": route_label,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed_count": len(seeds),
                    "mean_delta_auc": float(deltas.mean()),
                    "normal_ci95_low": float(normal_effect["normal_95ci_low"]),
                    "normal_ci95_high": float(normal_effect["normal_95ci_high"]),
                    "bootstrap_ci95_low": bootstrap_low,
                    "bootstrap_ci95_high": bootstrap_high,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "exact_sign_p": exact_two_sided_sign_p(wins, losses),
                }
            )
    return rows


def summarize_trajectories(result_root: Path) -> list[dict[str, Any]]:
    deployment_dir = result_root / "deployment"
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(TRAJECTORY_METHODS):
        traces = read_jsonl(deployment_dir / f"{method}_audit.jsonl")
        by_round: dict[int, list[float]] = defaultdict(list)
        for trace in traces:
            by_round[int(trace["round_index"])].append(float(trace["best_so_far"]))
        for round_index, values in sorted(by_round.items()):
            array = np.asarray(values, dtype=float)
            rng = np.random.default_rng(
                20260825 + 1_000 * method_index + round_index
            )
            indices = rng.integers(0, len(array), size=(20_000, len(array)))
            bootstrap_means = array[indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            rows.append(
                {
                    "method": method,
                    "method_label": TRAJECTORY_LABELS[method],
                    "round": round_index + 1,
                    "seed_count": len(array),
                    "mean_best_so_far": float(array.mean()),
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                }
            )
    return rows


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)
    write_sha256(path)


def plot(result_root: Path, output_stem: Path) -> None:
    effect_rows = summarize_effects(result_root)
    trajectory_rows = summarize_trajectories(result_root)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    effect_path = output_stem.with_name(output_stem.name + "_effects.csv")
    trajectory_path = output_stem.with_name(output_stem.name + "_trajectories.csv")
    stats_path = output_stem.with_name(output_stem.name + "_stats.json")
    write_csv(effect_rows, effect_path)
    write_csv(trajectory_rows, trajectory_path)
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": "care.prospective_ired_positive_transfer_plot/v1",
                "result_root": str(result_root),
                "paired_bootstrap_draws": 200_000,
                "trajectory_bootstrap_draws": 20_000,
                "effects": effect_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sha256(stats_path)

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
    figure = plt.figure(figsize=(13.0, 6.7), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.16, 1.0))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.75, wspace=0.28)

    forest = figure.add_subplot(grid[0, 0])
    route_base = {"train_to_validation": 1.0, "official_test": 0.0}
    method_offsets = {
        "multisource_rgpe": 0.27,
        "multisource_icm_bma": 0.09,
        "multisource_skill_prior": -0.09,
        "source_additive_mutation_prior": -0.27,
    }
    for row in effect_rows:
        y = route_base[row["route_id"]] + method_offsets[row["method"]]
        mean = float(row["mean_delta_auc"])
        low = float(row["bootstrap_ci95_low"])
        high = float(row["bootstrap_ci95_high"])
        forest.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=7.3 if row["method"] == "source_additive_mutation_prior" else 6.7,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=METHOD_COLORS[row["method"]],
            elinewidth=2.1,
            capsize=3.5,
            label=METHOD_LABELS[row["method"]]
            if row["route_id"] == "train_to_validation"
            else None,
            zorder=3,
        )
    forest.axvline(0.0, color="#1C2830", linewidth=1.25, linestyle="--", zorder=1)
    forest.axhspan(-0.38, 0.38, color="#F1F5F6", zorder=0)
    forest.axhline(0.5, color="#B9C2C8", linewidth=0.8)
    forest.set_yticks(
        [1.0, 0.0],
        ["Train -> validation\n100 paired seeds", "Official test\n100 paired seeds"],
    )
    forest.set_ylim(-0.48, 1.48)
    forest.set_xlim(-0.42, 1.13)
    forest.set_xlabel("Paired ΔAUC vs target-only GP-UCB\n95% paired-bootstrap interval")
    forest.set_title("A. Positive-transfer effect and strong baselines", loc="left")
    forest.grid(axis="x", color="#DCE2E5", linewidth=0.75, zorder=0)
    forest.spines[["top", "right", "left"]].set_visible(False)
    forest.tick_params(axis="y", length=0)
    forest.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    trajectory = figure.add_subplot(grid[0, 1])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["method"]].append(row)
    for method in TRAJECTORY_METHODS:
        method_rows = sorted(grouped[method], key=lambda row: row["round"])
        rounds = np.asarray([row["round"] for row in method_rows], dtype=int)
        means = np.asarray([row["mean_best_so_far"] for row in method_rows])
        lows = np.asarray([row["bootstrap_ci95_low"] for row in method_rows])
        highs = np.asarray([row["bootstrap_ci95_high"] for row in method_rows])
        trajectory.plot(
            rounds,
            means,
            color=TRAJECTORY_COLORS[method],
            linewidth=2.6 if method == "source_additive_mutation_prior" else 2.1,
            marker="o" if method == "source_additive_mutation_prior" else None,
            markersize=3.8,
            label=TRAJECTORY_LABELS[method],
            zorder=3,
        )
        trajectory.fill_between(
            rounds,
            lows,
            highs,
            color=TRAJECTORY_COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
    trajectory.set_xlim(1, 12)
    trajectory.set_xticks([1, 3, 6, 9, 12])
    trajectory.set_xlabel("Target reveal round")
    trajectory.set_ylabel("Mean best-so-far activity")
    trajectory.set_title("B. Untouched official-test trajectories", loc="left")
    trajectory.grid(color="#DCE2E5", linewidth=0.75, zorder=0)
    trajectory.spines[["top", "right"]].set_visible(False)
    trajectory.legend(loc="upper left", frameon=False)

    figure.suptitle(
        "Prospective FLIP2 IRED confirmation: an outcome-blind LLM skill transfers positively",
        x=0.09,
        ha="left",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.09,
        0.885,
        "The LLM hypothesis, executable additive skill, gate rule, code hashes, and seed schedules were frozen before the official file was downloaded.",
        fontsize=9.4,
        color="#596773",
        ha="left",
    )
    figure.text(
        0.09,
        0.035,
        "Interpretation: the preregistered LLM additive skill improves AUC versus target-only GP, but the fixed skill prior is stronger; this is one-family finite-pool evidence, not universal transfer or wet-lab validation.",
        fontsize=8.4,
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
