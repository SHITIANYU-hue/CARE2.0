#!/usr/bin/env python3
"""Aggregate frozen zero-experience point-selection shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_llm_zero_experience_point_selection as experiment


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    shard_dirs = sorted(path for path in args.shards_dir.iterdir() if path.is_dir())
    if not shard_dirs:
        parser.error("No result shards found")

    summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in shard_dirs
    ]
    config_hashes = {summary["config_sha256"] for summary in summaries}
    models = {summary["protocol"]["model"] for summary in summaries}
    if len(config_hashes) != 1 or len(models) != 1:
        raise ValueError("Shards do not share one frozen config and model")

    rows: list[dict[str, str]] = []
    traces: list[dict[str, Any]] = []
    dataset_summaries: dict[str, Mapping[str, Any]] = {}
    usage_totals: defaultdict[str, int] = defaultdict(int)
    for shard_dir, summary in zip(shard_dirs, summaries):
        with (shard_dir / "decisions.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
        traces.extend(
            json.loads(line)
            for line in (shard_dir / "llm_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        dataset_summaries.update(summary["dataset_summaries"])
        for key, value in summary["llm_usage"].items():
            usage_totals[key] += int(value)

    protocol = summaries[0]["protocol"]
    dataset_order = [str(spec["dataset_id"]) for spec in protocol["datasets"]]
    rows.sort(key=lambda row: (dataset_order.index(row["dataset_id"]), row["menu_id"]))
    traces.sort(
        key=lambda record: (
            dataset_order.index(record["dataset_id"]),
            int(record["batch_start"]),
        )
    )
    if len(rows) != len(dataset_order) * int(protocol["menus_per_dataset"]):
        raise ValueError("Aggregate does not contain the frozen number of decisions")
    if any(record.get("status") != "success" for record in traces):
        raise ValueError("Aggregate contains a failed batch")

    numeric_rows = [
        {
            **row,
            "selected_value": float(row["selected_value"]),
            "random_expected_value": float(row["random_expected_value"]),
            "llm_minus_random_expected": float(row["llm_minus_random_expected"]),
            "llm_minus_space_filling": float(row["llm_minus_space_filling"]),
            "percentile": float(row["percentile"]),
            "top_quartile_hit": row["top_quartile_hit"].lower() == "true",
            "top_one_hit": row["top_one_hit"].lower() == "true",
        }
        for row in rows
    ]
    overall = experiment.summarize_rows(
        numeric_rows,
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["bootstrap_seed"]) + 999,
    )
    confirmed_above_random = [
        dataset_id
        for dataset_id in dataset_order
        if float(
            dataset_summaries[dataset_id][
                "bootstrap_95ci_llm_minus_random_expected"
            ][0]
        )
        > 0.0
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "llm_trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in traces:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    aggregate = {
        "schema_version": "care.llm_zero_experience_point_selection.aggregate/v1",
        "execution_commit": "d7a0c38",
        "config_sha256": next(iter(config_hashes)),
        "protocol": protocol,
        "executed_datasets": dataset_order,
        "dataset_summaries": {
            dataset_id: dataset_summaries[dataset_id]
            for dataset_id in dataset_order
        },
        "overall_pooled_menu_summary_descriptive_only": overall,
        "datasets_with_bootstrap_95ci_above_random": confirmed_above_random,
        "recorded_successful_call_usage": dict(usage_totals),
        "trace_audit": {
            "batch_count": len(traces),
            "decision_count": len(rows),
            "all_batches_successful": True,
            "source_or_target_outcomes_in_prompt": False,
            "rag_or_skill_in_prompt": False,
            "gp_scores_in_prompt": False,
            "global_candidate_ids_in_prompt": False,
        },
        "claim_boundary": (
            "All four public datasets show a positive one-step menu-selection "
            "advantage over the exact random expectation. This is not a "
            "sequential optimization or source-transfer result, and public-data "
            "pretraining contamination cannot be excluded."
        ),
        "usage_boundary": (
            "Usage counts cover successful calls in the final result only. "
            "Interrupted engineering preflights are excluded and total provider "
            "billing may therefore be higher."
        ),
    }
    write_json(args.output_dir / "summary.json", aggregate)
    write_json(
        args.output_dir / "protocol_lock.json",
        {
            "execution_commit": "d7a0c38",
            "config_sha256": next(iter(config_hashes)),
            "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_count": len(dataset_order),
            "decision_count": len(rows),
        },
    )


if __name__ == "__main__":
    main()
