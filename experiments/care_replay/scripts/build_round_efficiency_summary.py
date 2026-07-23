#!/usr/bin/env python3
"""Measure whether a frozen LLM selector reaches useful quality in fewer rounds."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import run_calibrated_frozen_llm_selector as selector
import run_synthetic_suzuki as replay


SELECTOR_MODE = selector.SELECTOR_MODE


def load_events(path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(events, key=lambda event: int(event["round_index"]))


def first_round_to_threshold(
    events: list[dict[str, Any]],
    initial_best: float,
    threshold: float,
    censor_round: int,
) -> tuple[int, bool]:
    if initial_best >= threshold:
        return 0, True
    for event in events:
        if float(event["best_so_far"]) >= threshold:
            return int(event["round_index"]) + 1, True
    return censor_round, False


def first_round_to_top10(
    events: list[dict[str, Any]],
    initial_hit: bool,
    top10_ids: set[str],
    censor_round: int,
) -> tuple[int, bool]:
    if initial_hit:
        return 0, True
    for event in events:
        if str(event["selected_candidate"]) in top10_ids:
            return int(event["round_index"]) + 1, True
    return censor_round, False


def best_at_round(events: list[dict[str, Any]], initial_best: float, round_count: int) -> float:
    eligible = [
        float(event["best_so_far"])
        for event in events
        if int(event["round_index"]) < round_count
    ]
    return eligible[-1] if eligible else initial_best


def initial_context(
    adapter: replay.DatasetAdapter,
    seed: int,
    initial: int,
    top10_ids: set[str],
) -> tuple[float, bool]:
    shuffled = list(adapter.candidates)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[:initial]
    return (
        max(candidate.objective_value for candidate in observed),
        any(candidate.candidate_id in top10_ids for candidate in observed),
    )


def llm_initial_context(
    adapter: replay.DatasetAdapter,
    seed: int,
    initial: int,
    top10_ids: set[str],
    events: list[dict[str, Any]],
) -> tuple[float, bool]:
    snapshot = events[0].get("hypothesis_snapshot", {}) if events else {}
    warmstart_ids = snapshot.get("warmstart_candidates")
    if isinstance(warmstart_ids, list) and warmstart_ids:
        by_id = {
            candidate.candidate_id: candidate
            for candidate in adapter.candidates
        }
        observed = [
            by_id[candidate_id]
            for candidate_id in warmstart_ids
            if candidate_id in by_id
        ]
        if observed:
            return (
                max(candidate.objective_value for candidate in observed),
                any(candidate.candidate_id in top10_ids for candidate in observed),
            )
    return initial_context(adapter, seed, initial, top10_ids)


def audit_path(audit_dir: Path, output_id: str, mode: str, seed: int) -> Path:
    return audit_dir / f"{output_id}_audit_{mode}_seed{seed}.jsonl"


def summarize_round_efficiency(
    summary: dict[str, Any],
    audit_dir: Path,
    output_id: str,
    initial: int,
    rounds: int,
    baseline_output_id: str | None = None,
    baseline_mode_override: str | None = None,
    llm_mode_override: str | None = None,
) -> dict[str, Any]:
    dataset_id = str(summary["target_dataset"])
    baseline_mode = baseline_mode_override or str(summary["selection"]["target_anchor_mode"])
    llm_mode = llm_mode_override or SELECTOR_MODE
    baseline_output_id = baseline_output_id or output_id
    seed_start = int(summary["heldout_seed_start"])
    seed_count = int(summary["heldout_seed_count"])
    seeds = range(seed_start, seed_start + seed_count)
    censor_round = rounds + 1
    checkpoints = sorted({value for value in (1, 3, 5, rounds) if value <= rounds})
    adapter = replay.DATASET_BUILDERS[dataset_id]()
    top10_ids = {
        candidate.candidate_id
        for candidate in sorted(
            adapter.candidates,
            key=lambda candidate: candidate.objective_value,
            reverse=True,
        )[:10]
    }

    baseline_to_final: list[float] = []
    llm_to_baseline_final: list[float] = []
    final_savings: list[float] = []
    baseline_top10_rounds: list[float] = []
    llm_top10_rounds: list[float] = []
    top10_savings: list[float] = []
    baseline_top10_hits = 0
    llm_top10_hits = 0
    llm_baseline_final_hits = 0
    early_deltas: dict[int, list[float]] = {checkpoint: [] for checkpoint in checkpoints}

    for seed in seeds:
        baseline_events = load_events(audit_path(audit_dir, baseline_output_id, baseline_mode, seed))
        llm_events = load_events(audit_path(audit_dir, output_id, llm_mode, seed))
        baseline_initial_best, baseline_initial_top10_hit = initial_context(
            adapter, seed, initial, top10_ids
        )
        llm_initial_best, llm_initial_top10_hit = llm_initial_context(
            adapter, seed, initial, top10_ids, llm_events
        )

        baseline_final = float(baseline_events[-1]["best_so_far"])
        baseline_round, _ = first_round_to_threshold(
            baseline_events,
            baseline_initial_best,
            baseline_final,
            censor_round,
        )
        llm_round, llm_reached = first_round_to_threshold(
            llm_events,
            llm_initial_best,
            baseline_final,
            censor_round,
        )
        baseline_to_final.append(float(baseline_round))
        llm_to_baseline_final.append(float(llm_round))
        final_savings.append(float(baseline_round - llm_round))
        llm_baseline_final_hits += int(llm_reached)

        baseline_top10_round, baseline_top10_hit = first_round_to_top10(
            baseline_events,
            baseline_initial_top10_hit,
            top10_ids,
            censor_round,
        )
        llm_top10_round, llm_top10_hit = first_round_to_top10(
            llm_events,
            llm_initial_top10_hit,
            top10_ids,
            censor_round,
        )
        baseline_top10_rounds.append(float(baseline_top10_round))
        llm_top10_rounds.append(float(llm_top10_round))
        top10_savings.append(float(baseline_top10_round - llm_top10_round))
        baseline_top10_hits += int(baseline_top10_hit)
        llm_top10_hits += int(llm_top10_hit)

        for checkpoint in checkpoints:
            early_deltas[checkpoint].append(
                best_at_round(llm_events, llm_initial_best, checkpoint)
                - best_at_round(
                    baseline_events,
                    baseline_initial_best,
                    checkpoint,
                )
            )

    return {
        "experiment": "round_efficiency",
        "dataset": dataset_id,
        "evidence_mode": summary.get("evidence_mode", "unknown"),
        "llm_mode": llm_mode,
        "llm_selected_source_mode": (
            summary.get("strategy_router", {}).get("selected_mode")
            or summary["selection"]["selected_mode"]
        ),
        "llm_skill_selected": bool(summary["selection"]["selected_llm_skill"]),
        "baseline_mode": baseline_mode,
        "llm_output_id": output_id,
        "baseline_output_id": baseline_output_id,
        "heldout_seed_count": seed_count,
        "heldout_seed_start": seed_start,
        "initial_observations": initial,
        "replay_rounds": rounds,
        "censor_round": censor_round,
        "rounds_to_baseline_final": {
            "definition": (
                "For each seed, use the frozen target baseline's end-of-budget best value as the "
                "quality threshold. A run that does not reach it is right-censored at rounds + 1."
            ),
            "baseline_mean_round": round(mean(baseline_to_final), 6),
            "llm_mean_round_censored": round(mean(llm_to_baseline_final), 6),
            "llm_reach_rate": round(llm_baseline_final_hits / seed_count, 6),
            "paired_round_saving": selector.delta_summary(final_savings),
        },
        "rounds_to_global_top10": {
            "definition": (
                "First acquisition round at which the observed set contains a globally top-10 "
                "candidate; initial-set hits are round 0 and misses are censored at rounds + 1."
            ),
            "baseline_hit_rate": round(baseline_top10_hits / seed_count, 6),
            "llm_hit_rate": round(llm_top10_hits / seed_count, 6),
            "baseline_mean_round_censored": round(mean(baseline_top10_rounds), 6),
            "llm_mean_round_censored": round(mean(llm_top10_rounds), 6),
            "paired_round_saving": selector.delta_summary(top10_savings),
        },
        "early_best_so_far_delta": {
            str(checkpoint): selector.delta_summary(values)
            for checkpoint, values in early_deltas.items()
        },
        "interpretation": (
            "Positive round saving means the frozen LLM selector used fewer target acquisitions. "
            "Positive early best-so-far deltas mean it found higher-value candidates at the same budget."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument(
        "--baseline-output-id",
        default="",
        help="Optional second run whose audits provide the comparison trajectory.",
    )
    parser.add_argument(
        "--baseline-mode",
        default="",
        help="Optional comparison mode; defaults to the calibration-selected target anchor.",
    )
    parser.add_argument(
        "--llm-mode",
        default="",
        help="Optional audit mode; defaults to llm_calibrated_selector.",
    )
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output = summarize_round_efficiency(
        summary,
        args.audit_dir,
        args.output_id,
        args.initial,
        args.rounds,
        args.baseline_output_id or None,
        args.baseline_mode or None,
        args.llm_mode or None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
