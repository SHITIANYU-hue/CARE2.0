#!/usr/bin/env python3
"""Measure target rounds saved by a frozen source-outcome deployment policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import run_calibrated_frozen_llm_selector as selector
import run_calibrated_source_outcome_transfer as calibrated
import run_synthetic_suzuki as replay


DEPLOYED_MODE = calibrated.SELECTOR_MODE
BASELINE_MODE = calibrated.MATCHED_TARGET_LLM_MODE
ROOT = Path(__file__).resolve().parents[1]


def load_events(path: Path) -> list[dict[str, Any]]:
    return sorted(
        (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        key=lambda event: int(event["round_index"]),
    )


def audit_path(audit_dir: Path, output_id: str, mode: str, seed: int) -> Path:
    return audit_dir / f"{output_id}_audit_{mode}_seed{seed}.jsonl"


def resolve_record_path(record_path: str) -> Path:
    path = Path(record_path)
    if path.exists():
        return path
    matches = list(ROOT.rglob(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve copied model record {record_path!r}."
        )
    return matches[0]


def initial_from_source_trace(
    events: list[dict[str, Any]],
) -> tuple[float, set[str]] | None:
    if not events:
        return None
    initial = (
        events[0]
        .get("hypothesis_snapshot", {})
        .get("source_initial_design", {})
        .get("revealed_initial_observations", [])
    )
    if not initial:
        return None
    values = [float(item["revealed_value"]) for item in initial]
    ids = {str(item["candidate_id"]) for item in initial}
    return max(values), ids


def first_round_to_threshold(
    initial_best: float,
    events: list[dict[str, Any]],
    threshold: float,
    censor_round: int,
) -> tuple[int, bool]:
    if initial_best >= threshold:
        return 0, True
    for event in events:
        if float(event["best_so_far"]) >= threshold:
            return int(event["round_index"]) + 1, True
    return censor_round, False


def first_round_to_candidate_set(
    initial_ids: set[str],
    events: list[dict[str, Any]],
    candidate_ids: set[str],
    censor_round: int,
) -> tuple[int, bool]:
    if initial_ids & candidate_ids:
        return 0, True
    for event in events:
        if str(event["selected_candidate"]) in candidate_ids:
            return int(event["round_index"]) + 1, True
    return censor_round, False


def best_at_round(
    initial_best: float,
    events: list[dict[str, Any]],
    round_count: int,
) -> float:
    eligible = [
        float(event["best_so_far"])
        for event in events
        if int(event["round_index"]) < round_count
    ]
    return eligible[-1] if eligible else initial_best


def summarize(
    summary: dict[str, Any],
    audit_dir: Path,
    output_id: str,
) -> dict[str, Any]:
    adapter = replay.DATASET_BUILDERS[str(summary["target_dataset"])]()
    initial_count = int(summary["initial_observations"])
    rounds = int(summary["rounds"])
    censor_round = rounds + 1
    seed_start = int(summary["heldout_seed_start"])
    seed_count = int(summary["heldout_seed_count"])
    seeds = range(seed_start, seed_start + seed_count)
    target_record = json.loads(
        resolve_record_path(
            summary["matched_target_only_llm"]["record_path"]
        ).read_text(
            encoding="utf-8"
        )
    )
    target_mode = str(summary["matched_target_only_llm"]["frozen_mode"])
    target_skill = calibrated.resolve_target_llm_skill(
        target_record,
        adapter,
        target_mode,
    )
    task = replay.make_task(
        adapter,
        initial_observations=initial_count,
        reveal_budget=rounds,
    )
    top10_ids = {
        candidate.candidate_id
        for candidate in sorted(
            adapter.candidates,
            key=lambda candidate: candidate.objective_value,
            reverse=True,
        )[:10]
    }
    checkpoints = sorted({value for value in (1, 3, 5, rounds) if value <= rounds})

    baseline_rounds: list[float] = []
    deployed_rounds: list[float] = []
    saved_rounds: list[float] = []
    baseline_top10_rounds: list[float] = []
    deployed_top10_rounds: list[float] = []
    top10_saved_rounds: list[float] = []
    deployed_reached_baseline_final = 0
    baseline_top10_hits = 0
    deployed_top10_hits = 0
    early_deltas: dict[int, list[float]] = {
        checkpoint: [] for checkpoint in checkpoints
    }

    for seed in seeds:
        baseline_events = load_events(
            audit_path(audit_dir, output_id, BASELINE_MODE, seed)
        )
        deployed_events = load_events(
            audit_path(audit_dir, output_id, DEPLOYED_MODE, seed)
        )
        baseline_initial = calibrated.target_llm_initial_observations(
            adapter,
            task,
            seed,
            target_mode,
            target_skill,
        )
        baseline_initial_best = max(
            candidate.objective_value for candidate in baseline_initial
        )
        baseline_initial_ids = {
            candidate.candidate_id for candidate in baseline_initial
        }
        source_initial = initial_from_source_trace(deployed_events)
        if source_initial is None:
            deployed_initial_best = baseline_initial_best
            deployed_initial_ids = baseline_initial_ids
        else:
            deployed_initial_best, deployed_initial_ids = source_initial

        baseline_final = float(baseline_events[-1]["best_so_far"])
        baseline_round, _ = first_round_to_threshold(
            baseline_initial_best,
            baseline_events,
            baseline_final,
            censor_round,
        )
        deployed_round, deployed_reached = first_round_to_threshold(
            deployed_initial_best,
            deployed_events,
            baseline_final,
            censor_round,
        )
        baseline_rounds.append(float(baseline_round))
        deployed_rounds.append(float(deployed_round))
        saved_rounds.append(float(baseline_round - deployed_round))
        deployed_reached_baseline_final += int(deployed_reached)

        baseline_top10_round, baseline_hit = first_round_to_candidate_set(
            baseline_initial_ids,
            baseline_events,
            top10_ids,
            censor_round,
        )
        deployed_top10_round, deployed_hit = first_round_to_candidate_set(
            deployed_initial_ids,
            deployed_events,
            top10_ids,
            censor_round,
        )
        baseline_top10_rounds.append(float(baseline_top10_round))
        deployed_top10_rounds.append(float(deployed_top10_round))
        top10_saved_rounds.append(
            float(baseline_top10_round - deployed_top10_round)
        )
        baseline_top10_hits += int(baseline_hit)
        deployed_top10_hits += int(deployed_hit)

        for checkpoint in checkpoints:
            early_deltas[checkpoint].append(
                best_at_round(
                    deployed_initial_best,
                    deployed_events,
                    checkpoint,
                )
                - best_at_round(
                    baseline_initial_best,
                    baseline_events,
                    checkpoint,
                )
            )

    return {
        "experiment": "care2_source_outcome_round_efficiency",
        "source_dataset": summary["source_dataset"],
        "target_dataset": summary["target_dataset"],
        "deployed_mode": DEPLOYED_MODE,
        "selected_source_outcome_transfer": bool(
            summary["selection"]["selected_source_outcome_transfer"]
        ),
        "baseline_mode": BASELINE_MODE,
        "heldout_seed_start": seed_start,
        "heldout_seed_count": seed_count,
        "initial_observations": initial_count,
        "replay_rounds": rounds,
        "censor_round": censor_round,
        "rounds_to_matched_llm_final": {
            "definition": (
                "For each seed, the matched target-only LLM's end-of-budget best "
                "value is the threshold. Misses are right-censored at rounds + 1."
            ),
            "baseline_mean_round": round(mean(baseline_rounds), 6),
            "deployed_mean_round_censored": round(mean(deployed_rounds), 6),
            "deployed_reach_rate": round(
                deployed_reached_baseline_final / seed_count,
                6,
            ),
            "paired_round_saving": selector.delta_summary(saved_rounds),
        },
        "rounds_to_global_top10": {
            "baseline_hit_rate": round(baseline_top10_hits / seed_count, 6),
            "deployed_hit_rate": round(deployed_top10_hits / seed_count, 6),
            "baseline_mean_round_censored": round(
                mean(baseline_top10_rounds),
                6,
            ),
            "deployed_mean_round_censored": round(
                mean(deployed_top10_rounds),
                6,
            ),
            "paired_round_saving": selector.delta_summary(
                top10_saved_rounds
            ),
        },
        "early_best_so_far_delta": {
            str(checkpoint): selector.delta_summary(values)
            for checkpoint, values in early_deltas.items()
        },
        "interpretation": (
            "Positive paired round saving means the frozen deployed policy used "
            "fewer target acquisitions. Fallback pairs should reproduce the matched "
            "target-only LLM exactly and therefore report zero."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    suffix = "_summary.json"
    output_id = (
        summary_path.name[:-len(suffix)]
        if summary_path.name.endswith(suffix)
        else summary_path.stem
    )
    report = summarize(
        json.loads(summary_path.read_text(encoding="utf-8")),
        args.audit_dir.resolve(),
        output_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
