#!/usr/bin/env python3
"""Plot paired early-quality gains and cumulative top-10 discovery curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import build_round_efficiency_summary as efficiency
import run_synthetic_suzuki as replay


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    average = mean(values)
    if len(values) < 2:
        return average, average, average
    se = stdev(values) / (len(values) ** 0.5)
    return average, average - 1.96 * se, average + 1.96 * se


def trace_has_top10(
    events: list[dict[str, Any]],
    initial_hit: bool,
    top10_ids: set[str],
    round_count: int,
) -> bool:
    return initial_hit or any(
        int(event["round_index"]) < round_count
        and str(event["selected_candidate"]) in top10_ids
        for event in events
    )


def build_curves(
    summary: dict[str, Any],
    audit_dir: Path,
    output_id: str,
    initial: int,
    rounds: int,
    baseline_output_id: str | None = None,
    baseline_mode: str | None = None,
) -> dict[str, Any]:
    dataset_id = str(summary["target_dataset"])
    baseline_output_id = baseline_output_id or output_id
    baseline_mode = baseline_mode or str(summary["selection"]["target_anchor_mode"])
    seed_start = int(summary["heldout_seed_start"])
    seed_count = int(summary["heldout_seed_count"])
    adapter = replay.DATASET_BUILDERS[dataset_id]()
    top10_ids = {
        candidate.candidate_id
        for candidate in sorted(
            adapter.candidates,
            key=lambda candidate: candidate.objective_value,
            reverse=True,
        )[:10]
    }
    paired_by_round: dict[int, list[float]] = {index: [] for index in range(rounds + 1)}
    llm_hits = {index: 0 for index in range(rounds + 1)}
    baseline_hits = {index: 0 for index in range(rounds + 1)}

    for seed in range(seed_start, seed_start + seed_count):
        llm_events = efficiency.load_events(
            efficiency.audit_path(audit_dir, output_id, efficiency.SELECTOR_MODE, seed)
        )
        baseline_events = efficiency.load_events(
            efficiency.audit_path(audit_dir, baseline_output_id, baseline_mode, seed)
        )
        initial_best, initial_hit = efficiency.initial_context(
            adapter,
            seed,
            initial,
            top10_ids,
        )
        for round_count in range(rounds + 1):
            paired_by_round[round_count].append(
                efficiency.best_at_round(llm_events, initial_best, round_count)
                - efficiency.best_at_round(baseline_events, initial_best, round_count)
            )
            llm_hits[round_count] += int(
                trace_has_top10(llm_events, initial_hit, top10_ids, round_count)
            )
            baseline_hits[round_count] += int(
                trace_has_top10(baseline_events, initial_hit, top10_ids, round_count)
            )

    return {
        "rounds": list(range(rounds + 1)),
        "paired_best_delta": [mean_ci(paired_by_round[index]) for index in range(rounds + 1)],
        "llm_top10_rate": [llm_hits[index] / seed_count for index in range(rounds + 1)],
        "baseline_top10_rate": [baseline_hits[index] / seed_count for index in range(rounds + 1)],
        "baseline_mode": baseline_mode,
        "seed_count": seed_count,
        "dataset": dataset_id,
    }


def plot_curves(curves: dict[str, Any], output: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to render the round-efficiency plot.") from error

    rounds = curves["rounds"]
    means = [value[0] for value in curves["paired_best_delta"]]
    lows = [value[1] for value in curves["paired_best_delta"]]
    highs = [value[2] for value in curves["paired_best_delta"]]
    green = "#007F68"
    charcoal = "#34424D"
    grid = "#DCE3E7"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": "#9AA7AE",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    figure.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].fill_between(rounds, lows, highs, color=green, alpha=0.16, linewidth=0)
    axes[0].plot(rounds, means, color=green, linewidth=2.2, marker="o", markersize=3.8)
    axes[0].axhline(0, color=charcoal, linewidth=1.0, linestyle="--")
    axes[0].set_title("Paired best-so-far improvement")
    axes[0].set_xlabel("Target acquisition round")
    axes[0].set_ylabel("LLM selector minus target baseline")

    axes[1].plot(
        rounds,
        curves["llm_top10_rate"],
        color=green,
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        label="LLM selector",
    )
    axes[1].plot(
        rounds,
        curves["baseline_top10_rate"],
        color=charcoal,
        linewidth=2.0,
        marker="s",
        markersize=3.4,
        label="Target baseline",
    )
    axes[1].set_title("Cumulative global top-10 discovery")
    axes[1].set_xlabel("Target acquisition round")
    axes[1].set_ylabel("Hit rate")
    axes[1].set_ylim(0, min(1.0, max(curves["llm_top10_rate"] + curves["baseline_top10_rate"]) + 0.12))
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_xticks(rounds)
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.text(
        0.5,
        -0.015,
        f"Held-out paired replay: n={curves['seed_count']}; baseline={curves['baseline_mode']}",
        ha="center",
        color="#5B6870",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--baseline-output-id", default="")
    parser.add_argument("--baseline-mode", default="")
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--title", default="CARE 2.0 round efficiency")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    curves = build_curves(
        summary,
        args.audit_dir,
        args.output_id,
        args.initial,
        args.rounds,
        args.baseline_output_id or None,
        args.baseline_mode or None,
    )
    plot_curves(curves, args.output, args.title)


if __name__ == "__main__":
    main()
