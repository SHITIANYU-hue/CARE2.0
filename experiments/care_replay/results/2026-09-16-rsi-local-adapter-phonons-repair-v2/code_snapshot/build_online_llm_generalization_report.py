#!/usr/bin/env python3
"""Aggregate the staged Opus online-scientist generalization study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "care.online_llm_generalization_suite/v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def exact_one_sided_sign_p(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    return sum(
        math.comb(trials, successes)
        for successes in range(wins, trials + 1)
    ) / (2**trials)


def comparison(values: Sequence[float]) -> dict[str, Any]:
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
            wins,
            losses,
        ),
    }


def target_domain(target: str) -> str:
    if target.startswith("real_moleculenet_"):
        return "molecular_property"
    if target.startswith("real_matbench_"):
        return "materials_property"
    return "other"


def trace_diagnostics(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    authority_rounds = 0
    eligibility_modes: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event_name = str(event["event"])
        counts[event_name] += 1
        if event_name != "llm_round_request":
            continue
        diagnostics = event.get("menu_diagnostics", {})
        eligibility_modes[str(diagnostics.get("eligibility_mode", "legacy"))] += 1
        candidate_menu = event.get("prompt", {}).get("candidate_menu", {})
        columns = list(candidate_menu.get("columns", []))
        if "decision_eligible" not in columns:
            continue
        eligible_index = columns.index("decision_eligible")
        eligible_count = sum(
            bool(row[eligible_index])
            for row in candidate_menu.get("rows", [])
        )
        if eligible_count > 1:
            authority_rounds += 1
    return {
        "counts": dict(counts),
        "authority_rounds": authority_rounds,
        "eligibility_modes": dict(eligibility_modes),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "executed_rounds": sum(int(row["actual_rounds"]) for row in rows),
        "mean_llm_participation_rate": round(
            sum(float(row["llm_participation_rate"]) for row in rows) / len(rows),
            6,
        ),
        "mean_llm_decision_authority_rate": round(
            sum(float(row["llm_decision_authority_rate"]) for row in rows)
            / len(rows),
            6,
        ),
        "online_llm_increment_over_same_initial_gp": {
            "best_so_far_auc": comparison(
                [float(row["online_auc_delta_vs_same_initial_gp"]) for row in rows]
            ),
            "final_best": comparison(
                [float(row["online_final_delta_vs_same_initial_gp"]) for row in rows]
            ),
        },
        "full_llm_scientist_vs_frozen_source_diverse_control": {
            "best_so_far_auc": comparison(
                [float(row["full_auc_delta_vs_frozen_control"]) for row in rows]
            ),
            "final_best": comparison(
                [float(row["full_final_delta_vs_frozen_control"]) for row in rows]
            ),
        },
        "llm_initial_design_increment_over_frozen_control": {
            "best_so_far_auc": comparison(
                [float(row["initial_auc_delta_vs_frozen_control"]) for row in rows]
            ),
            "final_best": comparison(
                [float(row["initial_final_delta_vs_frozen_control"]) for row in rows]
            ),
        },
    }


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("v*/*/summary.json")):
        result_dir = summary_path.parent
        protocol_version = result_dir.parent.name
        summary = read_json(summary_path)
        initial = read_json(result_dir / "initial_record.json")
        trace = trace_diagnostics(result_dir / "llm_trace.jsonl")
        rounds = int(summary["actual_reveal_rounds"])
        counts = trace["counts"]
        if counts.get("llm_round_request", 0) != rounds:
            raise ValueError(f"Incomplete request trace: {result_dir}")
        if counts.get("llm_round_response", 0) != rounds:
            raise ValueError(f"Incomplete response trace: {result_dir}")
        if counts.get("target_reveal", 0) != rounds:
            raise ValueError(f"Incomplete reveal trace: {result_dir}")
        online = summary["metrics"]["online_llm_scientist"]
        online_delta = summary["deltas"][
            "online_llm_increment_over_same_initial_gp"
        ]
        full_delta = summary["deltas"]["full_llm_scientist_vs_fixed_v2"]
        initial_auc_delta = (
            float(full_delta["best_so_far_auc"])
            - float(online_delta["best_so_far_auc"])
        )
        initial_final_delta = (
            float(full_delta["final_best"])
            - float(online_delta["final_best"])
        )
        rows.append(
            {
                "protocol_version": protocol_version,
                "case": result_dir.name,
                "domain": target_domain(str(summary["target_task"])),
                "source_tasks": ";".join(summary["source_tasks"]),
                "target_task": summary["target_task"],
                "actual_rounds": rounds,
                "best_so_far_auc": float(online["best_so_far_auc"]),
                "final_best": float(online["final_best"]),
                "llm_participation_rate": round(
                    counts.get("llm_round_response", 0) / rounds,
                    6,
                ),
                "llm_decision_authority_rate": round(
                    int(trace["authority_rounds"]) / rounds,
                    6,
                ),
                "eligibility_modes": json.dumps(
                    trace["eligibility_modes"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "online_auc_delta_vs_same_initial_gp": float(
                    online_delta["best_so_far_auc"]
                ),
                "online_final_delta_vs_same_initial_gp": float(
                    online_delta["final_best"]
                ),
                "full_auc_delta_vs_frozen_control": float(
                    full_delta["best_so_far_auc"]
                ),
                "full_final_delta_vs_frozen_control": float(
                    full_delta["final_best"]
                ),
                "initial_auc_delta_vs_frozen_control": round(
                    initial_auc_delta,
                    6,
                ),
                "initial_final_delta_vs_frozen_control": round(
                    initial_final_delta,
                    6,
                ),
                "initial_tokens": int(initial["usage"]["total_tokens"]),
                "online_tokens": int(summary["usage"]["total_tokens"]),
                "llm_errors": counts.get("llm_round_error", 0),
            }
        )
    if not rows:
        raise ValueError(f"No staged results found under {root}")
    groups: dict[str, dict[str, Any]] = {}
    for field in ("protocol_version", "domain"):
        for value in sorted({str(row[field]) for row in rows}):
            subset = [row for row in rows if row[field] == value]
            groups[f"{field}:{value}"] = aggregate_rows(subset)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "claude_opus_5_cross_domain_generalization_study",
        "evidence_class": "sequential_development_with_new_target_extensions",
        "model": "claude-opus-5",
        "overall": aggregate_rows(rows),
        "total_initial_tokens": sum(int(row["initial_tokens"]) for row in rows),
        "total_online_tokens": sum(int(row["online_tokens"]) for row in rows),
        "total_llm_errors": sum(int(row["llm_errors"]) for row in rows),
        "grouped_results": groups,
        "tasks": rows,
        "interpretation": (
            "The extensions cover real molecular-property and materials-property "
            "targets. They show route-specific positive transfer but do not support "
            "a universal claim that online LLM reranking beats same-initial target-only "
            "GP-UCB. Later protocol versions were designed after diagnosing earlier "
            "versions, so only each version's newly introduced targets are valid for "
            "that version's extension claim."
        ),
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = build_report(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "generalization_summary.json", report)
    with (args.output_dir / "generalization_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
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
