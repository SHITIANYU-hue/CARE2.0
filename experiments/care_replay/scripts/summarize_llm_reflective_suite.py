#!/usr/bin/env python3
"""Summarize the multi-target LLM scientist reflection study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REFLECTIVE_MODE = "llm_hypothesis_reflective_design"
FIXED_MODE = "fixed_v2_initial_design"


def normal_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    value_mean = float(np.mean(array))
    if len(array) <= 1:
        return value_mean, value_mean, value_mean
    half_width = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return value_mean, value_mean - half_width, value_mean + half_width


def resolve_record(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_audit(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    model_names: set[str] = set()
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    status_counts: Counter[str] = Counter()
    total_gate_proposals = 0
    total_gate_acceptances = 0
    for summary_path in args.summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reflection_path = resolve_record(summary["frozen_reflection_record"])
        reflection_record = json.loads(reflection_path.read_text(encoding="utf-8"))
        hypothesis_path = resolve_record(summary["frozen_hypothesis_record"])
        hypothesis_record = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        reflection = reflection_record["frozen_reflection"]
        metrics = summary["summary_by_mode"]
        reflective = metrics[REFLECTIVE_MODE]
        fixed = metrics[FIXED_MODE]
        audit = read_audit(summary_path.parent / "audits.jsonl")
        gate_events = [
            event
            for event in audit
            if event.get("event") == "llm_reflection_reveal"
        ]
        gate_acceptances = sum(
            bool(event.get("reflection_gate", {}).get("authorized"))
            for event in gate_events
        )
        total_gate_proposals += len(gate_events)
        total_gate_acceptances += gate_acceptances
        status_counts[str(reflection["hypothesis_status"])] += 1
        initial_design = reflection_record.get("initial_design", {})
        initial_design_mode = initial_design.get("mode") or (
            "llm_hypothesis_compiled_initial_design"
            if len(summary["source_tasks"]) == 1
            else "llm_hypothesis_initial_design"
        )
        rows.append(
            {
                "target_task": summary["target_task"],
                "source_count": len(summary["source_tasks"]),
                "initial_llm_model": hypothesis_record["model"],
                "reflection_llm_model": reflection_record["model"],
                "initial_design_mode": initial_design_mode,
                "hypothesis_status": reflection["hypothesis_status"],
                "stop_transfer": bool(reflection["stop_transfer"]),
                "reflection_confidence": reflection["confidence"],
                "gate_proposal_count": len(gate_events),
                "gate_acceptance_count": gate_acceptances,
                "reflective_auc": reflective["best_so_far_auc"],
                "fixed_v2_auc": fixed["best_so_far_auc"],
                "auc_delta_vs_fixed": round(
                    reflective["best_so_far_auc"] - fixed["best_so_far_auc"], 6
                ),
                "reflective_final_best": reflective["final_best"],
                "fixed_v2_final_best": fixed["final_best"],
                "final_delta_vs_fixed": round(
                    reflective["final_best"] - fixed["final_best"], 6
                ),
                "revised_hypothesis": reflection["revised_hypothesis"],
                "evidence_interpretation": reflection["evidence_interpretation"],
                "reflection_record": portable_path(reflection_path),
            }
        )
        for record in (hypothesis_record, reflection_record):
            model_names.add(str(record["model"]))
            usage = record.get("usage", {})
            for field in token_usage:
                token_usage[field] += int(usage.get(field, 0) or 0)

    auc_deltas = [float(row["auc_delta_vs_fixed"]) for row in rows]
    final_deltas = [float(row["final_delta_vs_fixed"]) for row in rows]
    auc_mean, auc_low, auc_high = normal_interval(auc_deltas)
    final_mean, final_low, final_high = normal_interval(final_deltas)
    aggregate = {
        "schema_version": "care.llm_reflective_scientist_suite/v1",
        "evidence_class": "retrospective_multi_target_component_ablation",
        "task_count": len(rows),
        "scientist_loop": [
            "outcome_blind_hypothesis_generation",
            "initial_experiment_design",
            "initial_target_observation",
            "llm_evidence_interpretation_and_hypothesis_revision",
            "llm_followup_or_stop_decision",
            "acquisition_risk_gate",
            "target_only_gp_completion",
            "audited_memory_update",
        ],
        "evidence_boundary": {
            "initial_llm": "source outcomes and public target conditions only",
            "reflection_llm": "source evidence plus executed initial target observations only",
            "unrevealed_target_outcomes": "never exposed to either LLM stage",
            "status": "exploratory_candidate_not_externally_confirmed",
        },
        "llm_models": sorted(model_names),
        "llm_call_count": 2 * len(rows),
        "token_usage": token_usage,
        "reflection_hypothesis_status_counts": dict(sorted(status_counts.items())),
        "reflection_stop_rate": round(
            float(np.mean([bool(row["stop_transfer"]) for row in rows])), 6
        ),
        "gate": {
            "max_acquisition_loss": 0.0,
            "proposal_count": total_gate_proposals,
            "acceptance_count": total_gate_acceptances,
            "acceptance_rate": round(
                total_gate_acceptances / max(1, total_gate_proposals), 6
            ),
            "interpretation": (
                "The LLM may revise or stop a transfer hypothesis, but a follow-up "
                "candidate spends budget only when it is acquisition-equivalent to the "
                "target GP incumbent under the frozen zero-loss gate."
            ),
        },
        "auc_delta_vs_fixed_v2": {
            "task_mean": round(auc_mean, 6),
            "task_95ci_low": round(auc_low, 6),
            "task_95ci_high": round(auc_high, 6),
            "task_win_rate": round(
                float(np.mean(np.asarray(auc_deltas) > 0.0)), 6
            ),
            "task_nonloss_rate": round(
                float(np.mean(np.asarray(auc_deltas) >= 0.0)), 6
            ),
        },
        "final_best_delta_vs_fixed_v2": {
            "task_mean": round(final_mean, 6),
            "task_95ci_low": round(final_low, 6),
            "task_95ci_high": round(final_high, 6),
            "task_win_rate": round(
                float(np.mean(np.asarray(final_deltas) > 0.0)), 6
            ),
            "task_nonloss_rate": round(
                float(np.mean(np.asarray(final_deltas) >= 0.0)), 6
            ),
        },
        "per_task": rows,
        "interpretation": (
            "The LLM now participates in hypothesis generation, experiment design, "
            "evidence interpretation, hypothesis revision, counterexample planning, and "
            "stop decisions. The strict gate preserved the earlier routed performance; "
            "the new reflection stage is mechanistically richer but does not yet add an "
            "independent aggregate performance gain on this retrospective suite."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "suite_summary.json", aggregate)
    with (args.output_dir / "suite_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
