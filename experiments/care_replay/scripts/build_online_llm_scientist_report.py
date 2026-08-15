#!/usr/bin/env python3
"""Aggregate frozen online-LLM-scientist task results and trace coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "care.online_llm_scientist_suite/v1"
COMPARISONS = (
    "online_llm_increment_over_same_initial_gp",
    "full_llm_scientist_vs_fixed_v2",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def trace_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = str(json.loads(line)["event"])
        counts[event] = counts.get(event, 0) + 1
    return counts


def exact_one_sided_sign_p(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    return sum(
        math.comb(trials, successes) for successes in range(wins, trials + 1)
    ) / (2**trials)


def comparison_summary(values: Sequence[float]) -> dict[str, Any]:
    tolerance = 1e-9
    wins = sum(value > tolerance for value in values)
    ties = sum(abs(value) <= tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    return {
        "mean_delta": round(sum(values) / len(values), 6),
        "minimum_delta": round(min(values), 6),
        "maximum_delta": round(max(values), 6),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "exact_one_sided_sign_test_p_excluding_ties": exact_one_sided_sign_p(
            wins, losses
        ),
    }


def build_report(result_dirs: Sequence[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    seen_tasks: set[str] = set()
    for result_dir in result_dirs:
        summary = read_json(result_dir / "summary.json")
        if summary.get("schema_version") != "care.online_llm_scientist/v1":
            raise ValueError(f"Unexpected result schema in {result_dir}")
        task = str(summary["target_task"])
        if task in seen_tasks:
            raise ValueError(f"Duplicate target task: {task}")
        seen_tasks.add(task)
        online = summary["metrics"]["online_llm_scientist"]
        counts = trace_counts(result_dir / "llm_trace.jsonl")
        actual_rounds = int(summary["actual_reveal_rounds"])
        if counts.get("target_reveal", 0) != actual_rounds:
            raise ValueError(f"Incomplete reveal trace for {task}")
        if counts.get("llm_round_response", 0) != actual_rounds:
            raise ValueError(f"Incomplete LLM response trace for {task}")
        initial = read_json(result_dir / "initial_record.json")
        row = {
            "task": task,
            "initial_model": summary["initial_model"],
            "online_model": summary["online_model"],
            "initial_design_mode": summary["initial_design_mode"],
            "requested_rounds": int(summary["requested_reveal_rounds"]),
            "actual_rounds": actual_rounds,
            "best_so_far_auc": float(online["best_so_far_auc"]),
            "final_best": float(online["final_best"]),
            "simple_regret": float(online["simple_regret"]),
            "llm_participation_rate": float(online["llm_participation_rate"]),
            "gp_rank_one_choice_rate": float(online["gp_rank_one_choice_rate"]),
            "consensus_rank_one_choice_rate": float(
                online["consensus_rank_one_choice_rate"]
            ),
            "source_transfer_active_rate": float(
                online["source_transfer_active_rate"]
            ),
            "online_llm_auc_delta_vs_same_initial_gp": float(
                summary["deltas"][COMPARISONS[0]]["best_so_far_auc"]
            ),
            "online_llm_final_delta_vs_same_initial_gp": float(
                summary["deltas"][COMPARISONS[0]]["final_best"]
            ),
            "full_llm_auc_delta_vs_fixed_v2": float(
                summary["deltas"][COMPARISONS[1]]["best_so_far_auc"]
            ),
            "full_llm_final_delta_vs_fixed_v2": float(
                summary["deltas"][COMPARISONS[1]]["final_best"]
            ),
            "initial_tokens": int(initial["usage"]["total_tokens"]),
            "online_tokens": int(summary["usage"]["total_tokens"]),
            "llm_errors": counts.get("llm_round_error", 0),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["task"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "claude_opus_5_high_participation_online_scientist",
        "evidence_class": "retrospective_development_suite",
        "task_count": len(rows),
        "executed_rounds": sum(row["actual_rounds"] for row in rows),
        "models": sorted({row["online_model"] for row in rows}),
        "mean_llm_participation_rate": round(
            sum(row["llm_participation_rate"] for row in rows) / len(rows), 6
        ),
        "total_initial_tokens": sum(row["initial_tokens"] for row in rows),
        "total_online_tokens": sum(row["online_tokens"] for row in rows),
        "total_llm_errors": sum(row["llm_errors"] for row in rows),
        "comparisons": {
            "online_llm_increment_over_same_initial_gp": {
                "best_so_far_auc": comparison_summary(
                    [
                        row["online_llm_auc_delta_vs_same_initial_gp"]
                        for row in rows
                    ]
                ),
                "final_best": comparison_summary(
                    [
                        row["online_llm_final_delta_vs_same_initial_gp"]
                        for row in rows
                    ]
                ),
            },
            "full_llm_scientist_vs_fixed_v2": {
                "best_so_far_auc": comparison_summary(
                    [row["full_llm_auc_delta_vs_fixed_v2"] for row in rows]
                ),
                "final_best": comparison_summary(
                    [row["full_llm_final_delta_vs_fixed_v2"] for row in rows]
                ),
            },
        },
        "interpretation": (
            "The five targets were used during retrospective method development. "
            "The exact sign test is descriptive and excludes ties; independent new "
            "campaigns are required for a confirmatory generalization claim."
        ),
        "tasks": rows,
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = build_report(args.result_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "suite_summary.json", report)
    with (args.output_dir / "suite_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
