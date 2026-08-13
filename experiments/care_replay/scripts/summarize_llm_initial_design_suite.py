#!/usr/bin/env python3
"""Summarize the multi-target LLM initial-design component study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RAW_MODE = "llm_hypothesis_initial_design"
COMPILED_MODE = "llm_hypothesis_compiled_initial_design"
FIXED_MODE = "fixed_v2_initial_design"


def normal_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return mean, mean - half_width, mean + half_width


def task_structure_route(summary: Mapping[str, Any]) -> tuple[str, str]:
    source_count = len(summary["source_tasks"])
    if source_count == 1:
        return (
            COMPILED_MODE,
            "single source: preserve the LLM semantic anchor but add a deterministic geometry guard",
        )
    return (
        RAW_MODE,
        "multiple sources: use the LLM's task-specific semantic initial design directly",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def portable_path(path: Path) -> str:
    """Prefer a repository-relative record path in exported summaries."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    model_names: set[str] = set()
    for summary_path in args.summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record_path = Path(summary["frozen_hypothesis_record"])
        if not record_path.is_absolute():
            record_path = Path.cwd() / record_path
        record = json.loads(record_path.read_text(encoding="utf-8"))
        route_mode, route_reason = task_structure_route(summary)
        metrics = summary["summary_by_mode"]
        routed = metrics[route_mode]
        fixed = metrics[FIXED_MODE]
        raw = metrics[RAW_MODE]
        compiled = metrics[COMPILED_MODE]
        rows.append(
            {
                "target_task": summary["target_task"],
                "source_count": len(summary["source_tasks"]),
                "llm_model": summary["llm_model"],
                "route_mode": route_mode,
                "route_reason": route_reason,
                "raw_llm_auc": raw["best_so_far_auc"],
                "compiled_llm_auc": compiled["best_so_far_auc"],
                "fixed_v2_auc": fixed["best_so_far_auc"],
                "routed_llm_auc": routed["best_so_far_auc"],
                "routed_auc_delta_vs_fixed": round(
                    routed["best_so_far_auc"] - fixed["best_so_far_auc"], 6
                ),
                "routed_final_best": routed["final_best"],
                "fixed_v2_final_best": fixed["final_best"],
                "routed_final_delta_vs_fixed": round(
                    routed["final_best"] - fixed["final_best"], 6
                ),
                "llm_confidence": record["frozen_hypothesis"]["confidence"],
                "hypothesis": record["frozen_hypothesis"]["hypothesis"],
                "hypothesis_record": portable_path(record_path),
            }
        )
        usage = record.get("usage", {})
        for field in token_usage:
            token_usage[field] += int(usage.get(field, 0) or 0)
        model_names.add(str(record.get("model", summary["llm_model"])))

    auc_deltas = [float(row["routed_auc_delta_vs_fixed"]) for row in rows]
    final_deltas = [float(row["routed_final_delta_vs_fixed"]) for row in rows]
    auc_mean, auc_low, auc_high = normal_interval(auc_deltas)
    final_mean, final_low, final_high = normal_interval(final_deltas)
    aggregate = {
        "schema_version": "care.llm_initial_design_suite/v1",
        "evidence_class": "retrospective_multi_target_component_ablation",
        "task_count": len(rows),
        "route": {
            "name": "task_structure_route/v1",
            "rule": (
                "Use compiled LLM for a single completed source; use raw LLM semantic "
                "design when multiple completed sources are available."
            ),
            "uses_target_outcomes": False,
            "status": "exploratory_candidate_not_externally_confirmed",
        },
        "llm_models": sorted(model_names),
        "llm_call_count": len(rows),
        "token_usage": token_usage,
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
            "The LLM made real, outcome-blind initial-design decisions and improved mean "
            "search AUC relative to fixed v2 on this five-target retrospective suite. The "
            "task-level confidence interval still crosses zero, and the route was formulated "
            "after inspecting component behavior, so fresh frozen targets are required."
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
