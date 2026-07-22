#!/usr/bin/env python3
"""Compare a frozen semantic skill with no-prior and schedule-only ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_source_evidence_ablation as paired
import run_calibrated_frozen_llm_selector as selector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--no-prior-metrics", type=Path, required=True)
    parser.add_argument("--schedule-only-metrics", type=Path, required=True)
    parser.add_argument("--mode", default=selector.SELECTOR_MODE)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "full_semantic_skill": args.full_metrics,
        "no_rule_prior": args.no_prior_metrics,
        "schedule_only": args.schedule_only_metrics,
    }
    datasets: set[str] = set()
    rows: dict[str, dict[int, dict[str, str]]] = {}
    for label, path in paths.items():
        dataset, mode_rows = paired.load_mode(path, args.mode, args.split)
        datasets.add(dataset)
        rows[label] = mode_rows
    if len(datasets) != 1:
        raise ValueError(f"All ablations must use one target dataset: {sorted(datasets)}")

    output = {
        "experiment": "semantic_skill_component_ablation",
        "dataset": datasets.pop(),
        "mode": args.mode,
        "split": args.split,
        "inputs": {label: str(path) for label, path in paths.items()},
        "comparisons": {
            "full_minus_no_rule_prior": paired.paired_summary(
                rows["full_semantic_skill"], rows["no_rule_prior"]
            ),
            "full_minus_schedule_only": paired.paired_summary(
                rows["full_semantic_skill"], rows["schedule_only"]
            ),
            "no_rule_prior_minus_schedule_only": paired.paired_summary(
                rows["no_rule_prior"], rows["schedule_only"]
            ),
        },
        "interpretation": (
            "Full-minus-no-prior isolates the LLM-proposed coefficient direction. "
            "No-prior-minus-schedule isolates the executable rule partition learned online from "
            "target observations. Full-minus-schedule measures their combined contribution."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
