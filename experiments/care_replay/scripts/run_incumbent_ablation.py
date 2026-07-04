#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

AblationMode = str
DEFAULT_MODES: tuple[AblationMode, ...] = (
    "random",
    "group_only",
    "factor_only",
    "factor_ucb",
    "no_condition_prior",
    "public_incumbent",
)


def parse_modes(raw: str) -> tuple[AblationMode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(DEFAULT_MODES)}")
    return modes


def condition_prior(candidate: replay.Candidate) -> float:
    return 0.035 * candidate.x1 + 0.020 * candidate.x2 + 0.015 * candidate.x3


def public_scores_variant(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    mode: AblationMode,
) -> dict[str, float]:
    if mode == "public_incumbent":
        return replay.public_incumbent_scores(adapter, observed_ids, observed)

    group_summary = replay.group_stats(observed)
    factor_summary = replay.factor_stats(observed, adapter.decision_columns)
    global_mean = replay.observed_mean(observed)
    scores: dict[str, float] = {}

    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue

        group_count, group_mean = group_summary.get(candidate.group, (0, global_mean))
        factor_estimates: list[float] = []
        factor_counts: list[int] = []
        for key in replay.factor_values(candidate, adapter.decision_columns):
            count, value_mean = factor_summary.get(key, (0, global_mean))
            factor_estimates.append(replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0))
            factor_counts.append(count)

        if mode == "group_only":
            mean_estimate = replay.smoothed_mean(group_count, group_mean, global_mean, prior_weight=3.0)
            uncertainty = 8.0 / math.sqrt(group_count + 1.0)
            score = (mean_estimate + uncertainty) / 100.0
        elif mode == "factor_only":
            mean_estimate = mean(factor_estimates) if factor_estimates else global_mean
            score = mean_estimate / 100.0
        elif mode == "factor_ucb":
            mean_estimate = mean(factor_estimates) if factor_estimates else global_mean
            uncertainty = 8.0 / math.sqrt((max(factor_counts) if factor_counts else 0) + 1.0)
            score = (mean_estimate + uncertainty) / 100.0
        elif mode == "no_condition_prior":
            estimates = [replay.smoothed_mean(group_count, group_mean, global_mean, prior_weight=3.0)]
            estimates.extend(factor_estimates)
            support_counts = [group_count, *factor_counts]
            mean_estimate = mean(estimates)
            uncertainty = 8.0 / math.sqrt(max(support_counts) + 1.0)
            score = (mean_estimate + uncertainty) / 100.0
        else:
            raise ValueError(f"Unsupported score mode: {mode}")

        scores[candidate.candidate_id] = score
    return scores


def select_candidate(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    mode: AblationMode,
    rng: random.Random,
) -> replay.Candidate:
    if mode == "random":
        return rng.choice([candidate for candidate in adapter.candidates if candidate.candidate_id not in observed_ids])
    scores = public_scores_variant(adapter, observed_ids, observed, mode)
    selected_id = replay.top_candidate(scores)
    by_id = {candidate.candidate_id: candidate for candidate in adapter.candidates}
    return by_id[selected_id]


def run_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: AblationMode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pool = adapter.candidates
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []

    for round_index in range(task.reveal_budget):
        selected = select_candidate(adapter, observed_ids, observed, mode, rng)
        scores = None if mode == "random" else public_scores_variant(adapter, observed_ids, observed, mode)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected.candidate_id,
                "selected_score": None if scores is None else round(scores[selected.candidate_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
            }
        )

    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }
    return metrics, audit


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(values), 4),
                "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
            }
    return out


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for (mode, seed), audit_rows in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_ablation(
    dataset_id: str,
    seeds: int,
    rounds: int,
    initial: int,
    modes: tuple[AblationMode, ...],
    output_tag: str,
) -> dict[str, Any]:
    adapter = replay.DATASET_BUILDERS[dataset_id]()
    task = replay.make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed in range(seeds):
        for mode in modes:
            metrics, audit = run_policy(adapter, task, seed, mode)
            rows.append(metrics)
            audits[(mode, seed)] = audit

    output_id = f"incumbent_ablation_{dataset_id}" if not output_tag else f"incumbent_ablation_{dataset_id}_{output_tag}"
    summary = {
        "experiment": "care_incumbent_rule_ablation",
        "output_id": output_id,
        "dataset": dataset_id,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "rule_notes": {
            "public_incumbent": (
                "Current target-only incumbent: smoothed group mean, smoothed decision-factor means, "
                "UCB-like uncertainty bonus, and lightweight public continuous-feature prior."
            ),
            "no_condition_prior": "Current incumbent without the x1/x2/x3 public continuous-feature prior.",
            "factor_ucb": "Smoothed decision-factor means plus uncertainty, without group mean or condition prior.",
            "factor_only": "Smoothed decision-factor means only.",
            "group_only": "Smoothed group mean plus uncertainty only.",
            "random": "Uniform random selection from unrevealed candidates.",
        },
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, audits, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run target-only incumbent scoring ablations.")
    parser.add_argument("--dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(DEFAULT_MODES)}")
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()
    summary = run_ablation(
        dataset_id=args.dataset,
        seeds=args.seeds,
        rounds=args.rounds,
        initial=args.initial,
        modes=parse_modes(args.modes),
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
