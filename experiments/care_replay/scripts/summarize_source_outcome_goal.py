#!/usr/bin/env python3
"""Verify the frozen multi-pair source-outcome transfer objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


TRANSFER_MODE = "care_source_outcome_router"
RAW_TRANSFER_MODE = "llm_transfer_router"
BASELINE_MODE = "matched_target_only_llm"


def paired_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty paired sample.")
    average = mean(values)
    standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": float(len(values)),
        "mean": round(average, 6),
        "se": round(standard_error, 6),
        "normal_95ci_low": round(average - 1.96 * standard_error, 6),
        "normal_95ci_high": round(average + 1.96 * standard_error, 6),
        "win_rate": round(sum(value > 0.0 for value in values) / len(values), 6),
        "non_loss_rate": round(sum(value >= 0.0 for value in values) / len(values), 6),
    }


def load_pair(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suffix = "_summary.json"
    output_id = (
        summary_path.name[:-len(suffix)]
        if summary_path.name.endswith(suffix)
        else summary_path.stem
    )
    metrics_path = (
        summary_path.parents[1]
        / "tables"
        / f"{output_id}_metrics.csv"
    )
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    by_key = {
        (str(row.get("split", "")), str(row["mode"]), int(row["seed"])): row
        for row in rows
    }
    heldout_seeds = sorted({
        seed
        for split, mode, seed in by_key
        if split == "heldout" and mode == TRANSFER_MODE
    })
    def mode_diagnostics(mode: str) -> dict[str, Any]:
        final_deltas: list[float] = []
        auc_deltas: list[float] = []
        for seed in heldout_seeds:
            route_row = by_key[("heldout", mode, seed)]
            baseline_row = by_key[("heldout", BASELINE_MODE, seed)]
            final_deltas.append(
                float(route_row["final_best"])
                - float(baseline_row["final_best"])
            )
            auc_deltas.append(
                float(route_row["best_so_far_auc"])
                - float(baseline_row["best_so_far_auc"])
            )
        composite_deltas = [
            final + auc
            for final, auc in zip(final_deltas, auc_deltas)
        ]
        final_stats = paired_summary(final_deltas)
        auc_stats = paired_summary(auc_deltas)
        composite_stats = paired_summary(composite_deltas)
        return {
            "final_best": final_stats,
            "best_so_far_auc": auc_stats,
            "composite": composite_stats,
            "non_negative": (
                final_stats["mean"] >= 0.0
                and auc_stats["mean"] >= 0.0
            ),
            "statistically_positive": (
                composite_stats["normal_95ci_low"] > 0.0
            ),
        }

    deployed = mode_diagnostics(TRANSFER_MODE)
    raw = mode_diagnostics(RAW_TRANSFER_MODE)
    return {
        "source_dataset": summary["source_dataset"],
        "target_dataset": summary["target_dataset"],
        "selected_source_outcome_transfer": bool(
            summary["selection"]["selected_source_outcome_transfer"]
        ),
        "heldout_seed_count": len(heldout_seeds),
        "final_best": deployed["final_best"],
        "best_so_far_auc": deployed["best_so_far_auc"],
        "composite": deployed["composite"],
        "non_negative": deployed["non_negative"],
        "statistically_positive": deployed["statistically_positive"],
        "raw_source_route": raw,
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CARE 2.0 source-outcome transfer held-out report",
        "",
        "| Source -> target | Route | Final delta | AUC delta | Deployed composite 95% CI | Raw route composite | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for pair in report["pairs"]:
        route = "transfer" if pair["selected_source_outcome_transfer"] else "fallback"
        result = (
            "significant positive"
            if pair["statistically_positive"]
            else "non-negative"
            if pair["non_negative"]
            else "negative"
        )
        lines.append(
            "| {source} -> {target} | {route} | {final:+.3f} | {auc:+.3f} | "
            "[{low:+.3f}, {high:+.3f}] | {raw:+.3f} | {result} |".format(
                source=pair["source_dataset"],
                target=pair["target_dataset"],
                route=route,
                final=pair["final_best"]["mean"],
                auc=pair["best_so_far_auc"]["mean"],
                low=pair["composite"]["normal_95ci_low"],
                high=pair["composite"]["normal_95ci_high"],
                raw=pair.get("raw_source_route", {})
                .get("composite", {})
                .get("mean", 0.0),
                result=result,
            )
        )
    lines.extend([
        "",
        f"- All pairs non-negative: **{report['goal']['all_pairs_non_negative']}**",
        (
            "- Statistically positive pairs: "
            f"**{report['goal']['statistically_positive_pair_count']}/"
            f"{report['goal']['pair_count']}**"
        ),
        f"- Majority statistically positive: **{report['goal']['majority_statistically_positive']}**",
        f"- Goal achieved: **{report['goal']['achieved']}**",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    pairs = [load_pair(path.resolve()) for path in args.summaries]
    positive_count = sum(pair["statistically_positive"] for pair in pairs)
    all_non_negative = all(pair["non_negative"] for pair in pairs)
    majority_positive = positive_count > len(pairs) / 2
    report = {
        "experiment": "care2_source_outcome_transfer_goal_verification",
        "comparison": BASELINE_MODE,
        "criteria": {
            "all_pairs": "mean final delta >= 0 and mean AUC delta >= 0",
            "majority": "paired composite normal 95% CI lower bound > 0",
            "selection_boundary": (
                "Each pair's source route is frozen on calibration seeds. "
                "Only held-out seeds enter this report."
            ),
        },
        "pairs": pairs,
        "goal": {
            "pair_count": len(pairs),
            "all_pairs_non_negative": all_non_negative,
            "statistically_positive_pair_count": positive_count,
            "majority_statistically_positive": majority_positive,
            "achieved": all_non_negative and majority_positive,
        },
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_prefix.with_suffix(".md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(json.dumps(report["goal"], ensure_ascii=False))


if __name__ == "__main__":
    main()
