#!/usr/bin/env python3
"""Apply the calibration-only CARE strategy router to archived semantic runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_calibrated_frozen_llm_selector as selector
import run_calibrated_llm_semantic_selector as semantic_selector


METRICS = ("final_best", "best_so_far_auc", "top10_hit")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def apply_router(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        row for row in rows
        if str(row["mode"]) != semantic_selector.CARE_STRATEGY_ROUTER_MODE
    ]
    calibration_seeds = {
        int(row["seed"]) for row in rows if str(row["split"]) == "calibration"
    }
    heldout_seeds = {
        int(row["seed"]) for row in rows if str(row["split"]) == "heldout"
    }
    target_anchor = str(summary["selection"]["target_anchor_mode"])
    candidates = [target_anchor, str(summary["selection"]["selected_mode"])]
    for family in summary.get("external_llm_baselines", {}).values():
        selected = str(family.get("selected_mode", ""))
        if selected:
            candidates.append(selected)
    thresholds = summary["selection"]["thresholds"]
    selected_mode, route = semantic_selector.select_strategy_route(
        rows,
        calibration_seeds,
        tuple(candidates),
        target_anchor,
        float(thresholds["min_risk_adjusted_composite_gain"]),
        float(thresholds["min_positive_fold_rate"]),
    )
    for source_row in list(rows):
        if (
            str(source_row["mode"]) == selected_mode
            and int(source_row["seed"]) in heldout_seeds
        ):
            row = dict(source_row)
            row["mode"] = semantic_selector.CARE_STRATEGY_ROUTER_MODE
            row["selected_source_mode"] = selected_mode
            rows.append(row)

    all_modes = tuple(sorted({str(row["mode"]) for row in rows}))
    summary["calibration"] = selector.mode_means(rows, all_modes, calibration_seeds)
    summary["heldout"] = selector.mode_means(rows, all_modes, heldout_seeds)
    strongest_target = max(
        selector.TARGET_MODES,
        key=lambda mode: (summary["heldout"][mode]["composite"], mode),
    )
    router_pairwise = {
        baseline: {
            metric: selector.delta_summary(selector.paired_deltas(
                rows,
                semantic_selector.CARE_STRATEGY_ROUTER_MODE,
                baseline,
                heldout_seeds,
                metric,
            ))
            for metric in METRICS
        }
        for baseline in selector.TARGET_MODES
    }
    external_aliases = (
        semantic_selector.LLM_DIRECT_SELECTOR_MODE,
        semantic_selector.LLAMBO_WARMSTART_SELECTOR_MODE,
    )
    router_vs_external = {
        alias: {
            metric: selector.delta_summary(selector.paired_deltas(
                rows,
                semantic_selector.CARE_STRATEGY_ROUTER_MODE,
                alias,
                heldout_seeds,
                metric,
            ))
            for metric in METRICS
        }
        for alias in external_aliases
        if any(str(row["mode"]) == alias for row in rows)
    }
    strongest_stats = router_pairwise[strongest_target]
    summary["strategy_router"] = route
    summary["strategy_router_strongest_target_mode"] = strongest_target
    summary["strategy_router_pairwise"] = router_pairwise
    summary["strategy_router_vs_external_llm"] = router_vs_external
    summary["strategy_router_validation"] = {
        "selected_llm_strategy": route["selected_llm_strategy"],
        "strongest_target_mode": strongest_target,
        "final_95ci_positive": (
            strongest_stats["final_best"]["normal_95ci_low"] > 0.0
        ),
        "auc_95ci_positive": (
            strongest_stats["best_so_far_auc"]["normal_95ci_low"] > 0.0
        ),
        "top10_95ci_positive": (
            strongest_stats["top10_hit"]["normal_95ci_low"] > 0.0
        ),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    routed_summary, routed_rows = apply_router(summary, load_rows(args.metrics))
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(routed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    routed_rows.sort(
        key=lambda row: (str(row["split"]), int(row["seed"]), str(row["mode"]))
    )
    with args.output_metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(routed_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(routed_rows)
    print(json.dumps(routed_summary["strategy_router"], ensure_ascii=False))


if __name__ == "__main__":
    main()
