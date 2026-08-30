#!/usr/bin/env python3
"""Summarize paired AstaBench generation logs when the official scorer is absent."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, EvalSample, read_eval_log

EXPECTED_KEYS = {"hypothesis", "workflow"}


def parse_strict_submission(completion: str) -> bool:
    try:
        parsed = json.loads(completion)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(parsed, dict)
        and set(parsed) == EXPECTED_KEYS
        and all(isinstance(parsed[key], str) and parsed[key].strip() for key in EXPECTED_KEYS)
    )


def tool_call_names(sample: EvalSample) -> list[str]:
    names: list[str] = []
    for message in sample.messages:
        for call in getattr(message, "tool_calls", None) or []:
            names.append(call.function)
    return names


def sample_usage(sample: EvalSample) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in sample.model_usage.values():
        for key in total:
            total[key] += int(getattr(usage, key) or 0)
    return total


def sample_record(sample: EvalSample) -> dict[str, Any]:
    completion = sample.output.completion if sample.output else ""
    calls = tool_call_names(sample)
    usage = sample_usage(sample)
    return {
        "sample_id": str(sample.id),
        "strict_required_json": parse_strict_submission(completion),
        "submit_calls": calls.count("submit"),
        "python_calls": calls.count("python_session"),
        "model_calls": sum(message.role == "assistant" for message in sample.messages),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "completion_chars": len(completion),
        "sample_error": bool(sample.error),
    }


def elapsed_seconds(log: EvalLog) -> float:
    if not log.stats:
        return 0.0
    started = datetime.fromisoformat(log.stats.started_at)
    completed = datetime.fromisoformat(log.stats.completed_at)
    return (completed - started).total_seconds()


def summarize_arm(log: EvalLog, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(records),
        "strict_required_json_count": sum(row["strict_required_json"] for row in records),
        "submit_call_count": sum(row["submit_calls"] for row in records),
        "python_call_count": sum(row["python_calls"] for row in records),
        "model_call_count": sum(row["model_calls"] for row in records),
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
        "total_tokens": sum(row["total_tokens"] for row in records),
        "sample_error_count": sum(row["sample_error"] for row in records),
        "elapsed_seconds": elapsed_seconds(log),
        "inspect_status": log.status,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--care-log", type=Path, required=True)
    parser.add_argument("--react-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    care_log = read_eval_log(args.care_log.as_posix())
    react_log = read_eval_log(args.react_log.as_posix())
    care_records = {str(sample.id): sample_record(sample) for sample in care_log.samples or []}
    react_records = {str(sample.id): sample_record(sample) for sample in react_log.samples or []}
    if not care_records or set(care_records) != set(react_records):
        raise ValueError("CARE and ReAct logs must contain the same non-empty sample set")

    sample_order = [str(sample_id) for sample_id in care_log.eval.dataset.sample_ids]
    paired_rows: list[dict[str, Any]] = []
    for sample_id in sample_order:
        care = care_records[sample_id]
        react = react_records[sample_id]
        paired_rows.append(
            {
                "sample_id": sample_id,
                **{f"care_{key}": value for key, value in care.items() if key != "sample_id"},
                **{f"react_{key}": value for key, value in react.items() if key != "sample_id"},
                "care_minus_react_total_tokens": care["total_tokens"] - react["total_tokens"],
            }
        )

    care_summary = summarize_arm(care_log, list(care_records.values()))
    react_summary = summarize_arm(react_log, list(react_records.values()))
    token_reduction = 1.0 - care_summary["total_tokens"] / react_summary["total_tokens"]
    summary = {
        "schema_version": "care.astabench.unscored_paired_generation/v1",
        "benchmark": "AstaBench 0.5.3 DiscoveryBench validation",
        "sample_count": len(paired_rows),
        "model": care_log.eval.model,
        "official_scientific_quality_score_available": False,
        "arms": {"care": care_summary, "react": react_summary},
        "care_token_reduction_fraction_vs_react": token_reduction,
        "paired_strict_json_counts": {
            "care_only": sum(
                row["care_strict_required_json"] and not row["react_strict_required_json"]
                for row in paired_rows
            ),
            "react_only": sum(
                row["react_strict_required_json"] and not row["care_strict_required_json"]
                for row in paired_rows
            ),
            "both": sum(
                row["care_strict_required_json"] and row["react_strict_required_json"]
                for row in paired_rows
            ),
            "neither": sum(
                not row["care_strict_required_json"] and not row["react_strict_required_json"]
                for row in paired_rows
            ),
        },
        "claim_boundary": (
            "This no-score run measures structural completion and resource use only. "
            "It does not establish scientific-answer quality because the official "
            "DiscoveryBench scorer model was unavailable."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


if __name__ == "__main__":
    main()
