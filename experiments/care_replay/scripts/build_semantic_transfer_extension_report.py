#!/usr/bin/env python3
"""Build paired source-evidence and target-only LLM comparisons for an extension study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = ("final_best", "best_so_far_auc", "top10_hit")
DEFAULT_MODE = "care_strategy_router"


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_rows(path: Path, mode: str, split: str) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("mode") == mode and row.get("split") == split
        ]
    if not rows:
        raise ValueError(f"No mode={mode!r}, split={split!r} rows in {path}")
    return {int(row["seed"]): row for row in rows}


def delta_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty paired comparison")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "normal_95ci_low": mean - half_width,
        "normal_95ci_high": mean + half_width,
        "positive_fraction": sum(value > 0.0 for value in values) / len(values),
        "zero_fraction": sum(value == 0.0 for value in values) / len(values),
    }


def compare(
    left: dict[int, dict[str, str]],
    right: dict[int, dict[str, str]],
) -> dict[str, Any]:
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("The compared conditions have no paired seeds")
    return {
        "paired_seed_count": len(seeds),
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "metrics": {
            metric: delta_summary([
                float(left[seed][metric]) - float(right[seed][metric])
                for seed in seeds
            ])
            for metric in METRICS
        },
    }


def has_primary_gain(comparison: dict[str, Any]) -> bool:
    metrics = comparison["metrics"]
    return any(
        metrics[metric]["normal_95ci_low"] > 0.0
        for metric in ("final_best", "best_so_far_auc")
    )


def has_primary_harm(comparison: dict[str, Any]) -> bool:
    metrics = comparison["metrics"]
    return any(
        metrics[metric]["normal_95ci_high"] < 0.0
        for metric in ("final_best", "best_so_far_auc")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("split", "heldout"))
    mode = str(manifest.get("mode", DEFAULT_MODE))

    pair_reports: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        source_metrics = resolve_path(manifest_path, pair["source_metrics"])
        target_metrics = resolve_path(manifest_path, pair["target_metrics"])
        source_summary_path = resolve_path(manifest_path, pair["source_summary"])
        target_summary_path = resolve_path(manifest_path, pair["target_summary"])
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        target_summary = json.loads(target_summary_path.read_text(encoding="utf-8"))

        source_router = load_rows(source_metrics, mode, split)
        target_router = load_rows(target_metrics, mode, split)
        source_anchor_mode = str(source_summary["strategy_router_strongest_target_mode"])
        target_anchor_mode = str(target_summary["strategy_router_strongest_target_mode"])
        source_anchor = load_rows(source_metrics, source_anchor_mode, split)
        target_anchor = load_rows(target_metrics, target_anchor_mode, split)

        comparisons = {
            "source_router_minus_target_only_llm_router": compare(
                source_router, target_router
            ),
            "source_router_minus_strongest_bo": compare(
                source_router, source_anchor
            ),
            "target_only_llm_router_minus_strongest_bo": compare(
                target_router, target_anchor
            ),
        }
        source_increment = comparisons["source_router_minus_target_only_llm_router"]
        pair_report = {
            "pair_id": pair["pair_id"],
            "domain": pair["domain"],
            "source_dataset": source_summary.get("source_dataset"),
            "target_dataset": source_summary.get("target_dataset"),
            "source_evidence_mode": source_summary.get("evidence_mode"),
            "source_selected_route": source_summary["strategy_router"]["selected_mode"],
            "target_only_selected_route": target_summary["strategy_router"]["selected_mode"],
            "source_strongest_bo": source_anchor_mode,
            "target_only_strongest_bo": target_anchor_mode,
            "comparisons": comparisons,
            "source_evidence_primary_gain": has_primary_gain(source_increment),
            "source_evidence_primary_harm": has_primary_harm(source_increment),
        }
        pair_reports.append(pair_report)

        for comparison_name, comparison in comparisons.items():
            for metric, summary in comparison["metrics"].items():
                long_rows.append({
                    "pair_id": pair["pair_id"],
                    "domain": pair["domain"],
                    "comparison": comparison_name,
                    "metric": metric,
                    **summary,
                })

    target_reports = {
        str(report["target_dataset"]): report
        for report in pair_reports
    }
    aggregate = {
        "pair_count": len(pair_reports),
        "source_evidence_gain_count": sum(
            report["source_evidence_primary_gain"] for report in pair_reports
        ),
        "source_evidence_harm_count": sum(
            report["source_evidence_primary_harm"] for report in pair_reports
        ),
        "source_router_beats_bo_count": sum(
            has_primary_gain(report["comparisons"]["source_router_minus_strongest_bo"])
            for report in pair_reports
        ),
        "target_only_llm_beats_bo_count": sum(
            has_primary_gain(
                report["comparisons"]["target_only_llm_router_minus_strongest_bo"]
            )
            for report in pair_reports
        ),
        "unique_target_count": len(target_reports),
        "target_only_llm_beats_bo_unique_target_count": sum(
            has_primary_gain(
                report["comparisons"]["target_only_llm_router_minus_strongest_bo"]
            )
            for report in target_reports.values()
        ),
    }
    report = {
        "study": manifest.get("study"),
        "phase": manifest.get("phase"),
        "mode": mode,
        "split": split,
        "classification_rule": (
            "A primary gain requires the paired normal 95% confidence interval for "
            "final_best or best_so_far_auc to be strictly above zero. Primary harm uses "
            "the symmetric strictly-below-zero rule. Top-10 hit is reported but does not "
            "alone determine the classification."
        ),
        "aggregate": aggregate,
        "pairs": pair_reports,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "extension_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "extension_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(long_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(long_rows)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
