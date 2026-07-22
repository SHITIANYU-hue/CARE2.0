#!/usr/bin/env python3
"""Compare full, schema-only, and target-only LLM skill libraries on paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_calibrated_frozen_llm_selector as selector


METRICS = ("final_best", "best_so_far_auc", "top10_hit")


def load_mode(path: Path, mode: str, split: str) -> tuple[str, dict[int, dict[str, Any]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("mode") == mode and row.get("split") == split
        ]
    if not rows:
        raise ValueError(f"No {mode!r} rows with split={split!r} in {path}")
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset in {path}, found {sorted(datasets)}")
    return datasets.pop(), {int(row["seed"]): row for row in rows}


def paired_summary(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("The compared runs have no paired seeds")
    return {
        "paired_seed_count": len(seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "metrics": {
            metric: selector.delta_summary([
                float(left[seed][metric]) - float(right[seed][metric])
                for seed in seeds
            ])
            for metric in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--source-schema-metrics", type=Path, required=True)
    parser.add_argument("--target-only-metrics", type=Path, required=True)
    parser.add_argument("--mode", default=selector.SELECTOR_MODE)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "full": args.full_metrics,
        "source_schema_only": args.source_schema_metrics,
        "target_only": args.target_only_metrics,
    }
    datasets: set[str] = set()
    rows_by_evidence: dict[str, dict[int, dict[str, Any]]] = {}
    for evidence_mode, path in paths.items():
        dataset, rows = load_mode(path, args.mode, args.split)
        datasets.add(dataset)
        rows_by_evidence[evidence_mode] = rows
    if len(datasets) != 1:
        raise ValueError(f"All evidence modes must evaluate one target dataset: {sorted(datasets)}")

    comparisons = {
        "full_minus_source_schema_only": paired_summary(
            rows_by_evidence["full"], rows_by_evidence["source_schema_only"]
        ),
        "full_minus_target_only": paired_summary(
            rows_by_evidence["full"], rows_by_evidence["target_only"]
        ),
        "source_schema_only_minus_target_only": paired_summary(
            rows_by_evidence["source_schema_only"], rows_by_evidence["target_only"]
        ),
    }
    output = {
        "experiment": "source_evidence_causality_ablation",
        "dataset": datasets.pop(),
        "mode": args.mode,
        "split": args.split,
        "inputs": {key: str(value) for key, value in paths.items()},
        "comparisons": comparisons,
        "interpretation": (
            "Positive full-minus-target-only intervals isolate the incremental contribution of the "
            "supplied source task under a fixed target schema and paired replay seeds. Positive "
            "full-minus-source-schema-only intervals further isolate source outcome statistics from "
            "source identity and field alignment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
