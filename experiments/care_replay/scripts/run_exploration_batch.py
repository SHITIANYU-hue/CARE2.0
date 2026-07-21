#!/usr/bin/env python3
"""Batch-diverse exploration ablation for CARE 2.0 finite-pool replay."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
MODES = (
    "incumbent_batch",
    "target_diverse_batch",
    "target_diverse_batch_gate",
    "llm_explore_batch_gate",
)
DEFAULT_MODES = (
    "incumbent_batch",
    "target_diverse_batch",
    "target_diverse_batch_gate",
)


def candidate_factor_distance(
    adapter: replay.DatasetAdapter,
    left: replay.Candidate,
    right: replay.Candidate,
) -> float:
    fields = tuple(dict.fromkeys((adapter.group_column, *adapter.decision_columns)))
    mismatches: list[float] = []
    for field_name in fields:
        left_value = left.group if field_name == adapter.group_column else left.metadata.get(field_name)
        right_value = right.group if field_name == adapter.group_column else right.metadata.get(field_name)
        if left_value is None and right_value is None:
            continue
        mismatches.append(float(str(left_value) != str(right_value)))
    return mean(mismatches) if mismatches else float(left.candidate_id != right.candidate_id)


def historical_novelty_scores(
    adapter: replay.DatasetAdapter,
    candidate_ids: set[str],
    observed: list[replay.Candidate],
) -> dict[str, float]:
    by_id = {candidate.candidate_id: candidate for candidate in adapter.candidates}
    if not observed:
        return {candidate_id: 1.0 for candidate_id in candidate_ids}
    return {
        candidate_id: min(
            candidate_factor_distance(adapter, by_id[candidate_id], reference)
            for reference in observed
        )
        for candidate_id in candidate_ids
    }


def diversity_eligible_ids(
    adapter: replay.DatasetAdapter,
    candidate_ids: set[str],
    pending_batch: list[replay.Candidate],
    min_distance: float,
) -> set[str]:
    if not pending_batch:
        return set(candidate_ids)
    by_id = {candidate.candidate_id: candidate for candidate in adapter.candidates}
    eligible = {
        candidate_id
        for candidate_id in candidate_ids
        if min(
            candidate_factor_distance(adapter, by_id[candidate_id], pending)
            for pending in pending_batch
        ) >= min_distance
    }
    return eligible or set(candidate_ids)


def target_diversity_gate_decision(
    base_scores: dict[str, float],
    novelty_scores: dict[str, float],
    challenger_id: str,
    max_acquisition_loss: float,
) -> replay.GateCertificate:
    incumbent_id = replay.top_candidate(base_scores)
    acquisition_loss = max(0.0, base_scores[incumbent_id] - base_scores[challenger_id])
    novelty_gain = novelty_scores[challenger_id] - novelty_scores[incumbent_id]
    authorized = (
        challenger_id != incumbent_id
        and acquisition_loss <= max_acquisition_loss
        and novelty_gain > 0.0
    )
    if challenger_id == incumbent_id:
        reason = "challenger_matches_incumbent"
    elif acquisition_loss > max_acquisition_loss:
        reason = "rejected_by_acquisition_loss"
    elif novelty_gain <= 0.0:
        reason = "rejected_without_novelty_gain"
    else:
        reason = "authorized_low_cost_diversity"
    return replay.GateCertificate(
        gate_version="target_diverse_batch_gate_v1",
        incumbent_candidate=incumbent_id,
        challenger_candidate=challenger_id,
        selected_candidate=challenger_id if authorized else incumbent_id,
        authorized=authorized,
        gate_margin=round(novelty_gain, 6),
        acquisition_loss=round(acquisition_loss, 6),
        row_order_stable=True,
        applied_skill_ids=("batch_diversity_constraint",) if authorized else (),
        reason=reason,
    )


def aggregate(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in modes:
        items = [row for row in rows if row["mode"] == mode]
        if not items:
            continue
        result[mode] = {}
        for field_name in (
            "final_best",
            "best_so_far_auc",
            "simple_regret",
            "top10_hit",
            "mean_historical_novelty",
            "mean_within_batch_distance",
            "challenger_change_rate",
            "gate_authorization_rate",
            "llm_call_count",
        ):
            values = [float(item[field_name]) for item in items]
            result[mode][field_name] = {
                "mean": round(mean(values), 5),
                "std": round(pstdev(values), 5) if len(values) > 1 else 0.0,
            }
    return result


def paired_comparison(
    rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    baseline: str = "incumbent_batch",
) -> list[dict[str, Any]]:
    by_key = {(str(row["mode"]), int(row["seed"])): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for mode in modes:
        if mode == baseline:
            continue
        final_deltas: list[float] = []
        auc_deltas: list[float] = []
        for row in rows:
            if row["mode"] != mode:
                continue
            baseline_row = by_key.get((baseline, int(row["seed"])))
            if baseline_row is None:
                continue
            final_deltas.append(float(row["final_best"]) - float(baseline_row["final_best"]))
            auc_deltas.append(float(row["best_so_far_auc"]) - float(baseline_row["best_so_far_auc"]))
        if final_deltas:
            comparisons.append(
                {
                    "method": mode,
                    "baseline": baseline,
                    "n": len(final_deltas),
                    "final_best_delta_mean": round(mean(final_deltas), 5),
                    "final_best_win_rate": round(sum(delta > 0 for delta in final_deltas) / len(final_deltas), 5),
                    "auc_delta_mean": round(mean(auc_deltas), 5),
                }
            )
    return comparisons


def run_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: str,
    batch_size: int,
    novelty_weight: float,
    min_batch_distance: float,
    max_acquisition_loss: float,
    llm_config: replay.LLMConfig | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode.startswith("llm_") and llm_config is None:
        raise RuntimeError("LLM mode requires an API configuration.")

    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {
        candidate.candidate_id
        for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]
    }
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    selected_novelties: list[float] = []
    within_batch_distances: list[float] = []
    challenger_changes = 0
    authorizations = 0
    llm_calls = 0
    exploration_interventions = 0
    revealed_count = 0
    batch_index = 0

    while revealed_count < task.reveal_budget:
        current_batch_size = min(batch_size, task.reveal_budget - revealed_count)
        public_observed = list(observed)
        public_observed_ids = set(observed_ids)
        base_scores = replay.public_incumbent_scores(adapter, public_observed_ids, public_observed)
        novelty_scores = historical_novelty_scores(adapter, set(base_scores), public_observed)
        adjustments = {candidate_id: 0.0 for candidate_id in base_scores}
        llm_record: dict[str, Any] | None = None
        confidence = 0.0
        if mode == "llm_explore_batch_gate":
            adjustments, _skill_cert, llm_record = replay.llm_skill_adjustments(
                adapter,
                pool,
                public_observed_ids,
                public_observed,
                seed,
                batch_index,
                "llm_explore_gate_v1",
                llm_config,
            )
            llm_calls += int(bool(llm_record.get("called")))
            confidence = float(
                llm_record.get("confidence_calibration", {}).get("calibrated", 0.0)
            )

        pending_batch: list[replay.Candidate] = []
        pending_ids: set[str] = set()
        batch_events: list[dict[str, Any]] = []
        for batch_slot in range(current_batch_size):
            available = set(base_scores) - pending_ids
            available_base_scores = {
                candidate_id: base_scores[candidate_id] for candidate_id in available
            }
            use_diversity = mode in {
                "target_diverse_batch",
                "target_diverse_batch_gate",
                "llm_explore_batch_gate",
            }
            eligible = (
                diversity_eligible_ids(adapter, available, pending_batch, min_batch_distance)
                if use_diversity
                else available
            )
            score_ids = available if mode == "llm_explore_batch_gate" else eligible
            slot_base = {candidate_id: base_scores[candidate_id] for candidate_id in score_ids}
            slot_adjustments = {
                candidate_id: (
                    adjustments.get(candidate_id, 0.0)
                    if candidate_id in eligible
                    else 0.0
                )
                for candidate_id in score_ids
            }
            adjusted_scores = {
                candidate_id: slot_base[candidate_id]
                + slot_adjustments[candidate_id]
                + (
                    novelty_weight * novelty_scores[candidate_id]
                    if use_diversity and candidate_id in eligible
                    else 0.0
                )
                for candidate_id in score_ids
            }

            incumbent_id = replay.top_candidate(available_base_scores)
            challenger_id = replay.top_candidate(adjusted_scores)
            if mode == "llm_explore_batch_gate":
                gate = replay.exploration_gate_decision(
                    "llm_explore_batch_gate_v1",
                    available_base_scores,
                    adjusted_scores,
                    {
                        candidate_id: adjusted_scores[candidate_id] - available_base_scores[candidate_id]
                        for candidate_id in available
                    },
                    {candidate_id: novelty_scores[candidate_id] for candidate_id in available},
                    True,
                    ("llm_exploration_policy", "batch_diversity_constraint"),
                    seed,
                    revealed_count + batch_slot,
                    confidence,
                    exploration_interventions,
                    max(candidate.objective_value for candidate in public_observed),
                )
                selected_id = gate.selected_candidate
                authorizations += int(gate.authorized)
                exploration_interventions += int(gate.authorized)
            elif mode == "target_diverse_batch_gate":
                gate = target_diversity_gate_decision(
                    available_base_scores,
                    novelty_scores,
                    challenger_id,
                    max_acquisition_loss,
                )
                selected_id = gate.selected_candidate
                authorizations += int(gate.authorized)
            else:
                selected_id = challenger_id
                gate = replay.GateCertificate(
                    gate_version="none",
                    incumbent_candidate=incumbent_id,
                    challenger_candidate=challenger_id,
                    selected_candidate=selected_id,
                    authorized=False,
                    gate_margin=round(
                        adjusted_scores[challenger_id] - base_scores[incumbent_id],
                        6,
                    ),
                    acquisition_loss=round(
                        max(0.0, base_scores[incumbent_id] - base_scores[challenger_id]),
                        6,
                    ),
                    row_order_stable=True,
                    applied_skill_ids=("batch_diversity_constraint",) if use_diversity else (),
                    reason="target_diverse_batch" if use_diversity else "target_incumbent_batch",
                )

            challenger_changes += int(challenger_id != incumbent_id)
            selected = by_id[selected_id]
            selected_novelties.append(novelty_scores[selected_id])
            if pending_batch:
                within_batch_distances.append(
                    min(candidate_factor_distance(adapter, selected, prior) for prior in pending_batch)
                )
            pending_batch.append(selected)
            pending_ids.add(selected_id)
            batch_events.append(
                {
                    "dataset_id": adapter.dataset_id,
                    "seed": seed,
                    "batch_index": batch_index,
                    "batch_slot": batch_slot,
                    "public_observed_count": len(public_observed),
                    "mode": mode,
                    "incumbent_candidate": incumbent_id,
                    "challenger_candidate": challenger_id,
                    "selected_candidate": selected_id,
                    "selected_score": round(
                        adjusted_scores.get(selected_id, base_scores[selected_id]),
                        6,
                    ),
                    "historical_novelty": round(novelty_scores[selected_id], 6),
                    "min_within_batch_distance": None if batch_slot == 0 else round(within_batch_distances[-1], 6),
                    "gate": asdict(gate),
                    "llm_policy": llm_record if batch_slot == 0 else {"reused_batch_decision": True},
                    "evidence_boundary": (
                        "All candidates in this batch are selected from the same pre-batch public observations. "
                        "Hidden target values are revealed only after the batch is complete."
                    ),
                }
            )

        observed.extend(pending_batch)
        observed_ids.update(pending_ids)
        selected_top10 = selected_top10 or any(candidate.candidate_id in top10 for candidate in pending_batch)
        best_so_far = max(candidate.objective_value for candidate in observed)
        for event in batch_events:
            selected = by_id[event["selected_candidate"]]
            event["revealed_value"] = selected.objective_value
            event["best_so_far_after_batch"] = best_so_far
            audit.append(event)
            best_trace.append(best_so_far)
        revealed_count += current_batch_size
        batch_index += 1

    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
        "mean_historical_novelty": round(mean(selected_novelties), 5),
        "mean_within_batch_distance": round(mean(within_batch_distances), 5) if within_batch_distances else 0.0,
        "challenger_change_rate": round(challenger_changes / max(1, task.reveal_budget), 5),
        "gate_authorization_rate": round(authorizations / max(1, task.reveal_budget), 5),
        "llm_call_count": llm_calls,
    }, audit


def run_dataset(
    adapter: replay.DatasetAdapter,
    seeds: int,
    seed_start: int,
    rounds: int,
    initial: int,
    batch_size: int,
    novelty_weight: float,
    min_batch_distance: float,
    max_acquisition_loss: float,
    modes: tuple[str, ...],
    llm_config: replay.LLMConfig | None,
    output_tag: str,
) -> dict[str, Any]:
    task = replay.make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for mode in modes:
        for seed in range(seed_start, seed_start + seeds):
            metrics, trace = run_policy(
                adapter,
                task,
                seed,
                mode,
                batch_size,
                novelty_weight,
                min_batch_distance,
                max_acquisition_loss,
                llm_config,
            )
            rows.append(metrics)
            audits[(mode, seed)] = trace

    output_id = f"{adapter.dataset_id}_exploration_batch_{output_tag}".rstrip("_")
    summary = {
        "experiment": "care2_batch_diverse_exploration",
        "output_id": output_id,
        "dataset": adapter.dataset_id,
        "task": asdict(task),
        "modes": list(modes),
        "seeds": seeds,
        "seed_start": seed_start,
        "batch_size": batch_size,
        "novelty_weight": novelty_weight,
        "min_batch_distance": min_batch_distance,
        "max_acquisition_loss": max_acquisition_loss,
        "llm": None if llm_config is None else {
            "base_url": llm_config.base_url,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
        },
        "aggregate": aggregate(rows, modes),
        "paired_comparison": paired_comparison(rows, modes),
    }
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for (mode, seed), events in audits.items():
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE 2.0 batch-diverse exploration ablations.")
    parser.add_argument("--dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=6, help="Total reveal budget, not number of batches.")
    parser.add_argument("--initial", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--novelty-weight", type=float, default=0.04)
    parser.add_argument("--min-batch-distance", type=float, default=0.34)
    parser.add_argument("--max-acquisition-loss", type=float, default=0.02)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--llm-base-url", default="https://api.commonstack.ai/v1")
    parser.add_argument("--llm-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--llm-api-key-env", default="CARE_LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1000)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    unknown_modes = sorted(set(modes) - set(MODES))
    if not modes or unknown_modes:
        parser.error(f"Unknown modes: {unknown_modes}; available: {', '.join(MODES)}")
    llm_config = (
        replay.llm_config_from_args(args, ("llm_explore_gate_v1",))
        if "llm_explore_batch_gate" in modes
        else None
    )
    summary = run_dataset(
        replay.DATASET_BUILDERS[args.dataset](),
        args.seeds,
        args.seed_start,
        args.rounds,
        args.initial,
        max(1, args.batch_size),
        max(0.0, args.novelty_weight),
        max(0.0, min(1.0, args.min_batch_distance)),
        max(0.0, args.max_acquisition_loss),
        modes,
        llm_config,
        args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
