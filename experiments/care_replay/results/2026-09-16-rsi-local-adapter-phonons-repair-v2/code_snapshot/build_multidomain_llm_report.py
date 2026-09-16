#!/usr/bin/env python3
"""Aggregate frozen semantic-skill confirmations across predeclared domains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("final_best", "best_so_far_auc", "top10_hit")


def domain_for_target(target: str) -> str:
    if target.startswith("real_matbench_"):
        return "materials_property"
    if target.startswith("real_moleculenet_"):
        return "molecular_property"
    if target == "real_chemlex_acidamine":
        return "wetlab_reaction"
    if target in {"real_buchwald_hartwig", "real_suzuki_miyaura"}:
        return "reaction_hte"
    return "unknown"


def metric_stats(summary: dict[str, Any], baseline: str, metric: str) -> dict[str, Any]:
    return dict(summary.get("heldout_pairwise", {}).get(baseline, {}).get(metric, {}))


def summarize_run(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    target = str(summary.get("target_dataset", ""))
    heldout = summary.get("heldout", {})
    baseline = str(summary.get("heldout_strongest_target_mode_descriptive_only", ""))
    route = summary.get("strategy_router", {})
    routed = bool(route)
    if routed:
        baseline = str(summary.get("strategy_router_strongest_target_mode", baseline))
    selector_mode = "care_strategy_router" if routed else "llm_calibrated_selector"
    selector_metrics = heldout.get(selector_mode, {})
    pairwise_key = "strategy_router_pairwise" if routed else "heldout_pairwise"
    stats = {
        metric: dict(summary.get(pairwise_key, {}).get(baseline, {}).get(metric, {}))
        for metric in METRICS
    }
    positive_ci_metrics = [
        metric
        for metric in METRICS
        if float(stats[metric].get("normal_95ci_low", 0.0)) > 0.0
    ]
    external = summary.get(
        "strategy_router_vs_external_llm" if routed else "care_vs_external_llm",
        {},
    )
    care_beats_external = {
        mode: [
            metric
            for metric in METRICS
            if float(values.get(metric, {}).get("normal_95ci_low", 0.0)) > 0.0
        ]
        for mode, values in external.items()
    }
    selected_llm = (
        bool(route.get("selected_llm_strategy"))
        if routed
        else bool(summary.get("selection", {}).get("selected_llm_skill"))
    )
    heldout_seeds = int(summary.get("heldout_seed_count") or 0)
    preselected = bool(summary.get("selection", {}).get("preselected_skill_id"))
    if not selected_llm:
        evaluation_stage = "fallback"
    elif preselected or heldout_seeds >= 500:
        evaluation_stage = "confirmation"
    else:
        evaluation_stage = "development"
    return {
        "source": summary.get("source_dataset"),
        "target": target,
        "domain": summary.get("domain") or domain_for_target(target),
        "evidence_mode": summary.get("evidence_mode"),
        "llm_model": summary.get("llm_model"),
        "selected_mode": (
            route.get("selected_mode")
            if routed
            else summary.get("selection", {}).get("selected_mode")
        ),
        "strategy_routed": routed,
        "selected_llm_skill": selected_llm,
        "strongest_target_baseline": baseline,
        "heldout_seeds": heldout_seeds,
        "evaluation_stage": evaluation_stage,
        "selector_final_best": selector_metrics.get("final_best"),
        "selector_auc": selector_metrics.get("best_so_far_auc"),
        "delta_final_best": stats["final_best"].get("mean", 0.0),
        "final_ci_low": stats["final_best"].get("normal_95ci_low", 0.0),
        "final_ci_high": stats["final_best"].get("normal_95ci_high", 0.0),
        "delta_auc": stats["best_so_far_auc"].get("mean", 0.0),
        "auc_ci_low": stats["best_so_far_auc"].get("normal_95ci_low", 0.0),
        "auc_ci_high": stats["best_so_far_auc"].get("normal_95ci_high", 0.0),
        "delta_top10_hit": stats["top10_hit"].get("mean", 0.0),
        "top10_ci_low": stats["top10_hit"].get("normal_95ci_low", 0.0),
        "top10_ci_high": stats["top10_hit"].get("normal_95ci_high", 0.0),
        "positive_ci_metrics": positive_ci_metrics,
        "confirmed_any_metric_gain": bool(
            evaluation_stage == "confirmation" and positive_ci_metrics
        ),
        "development_any_metric_signal": bool(
            evaluation_stage == "development" and positive_ci_metrics
        ),
        "care_beats_external_llm": care_beats_external,
        "summary_path": str(path),
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    runs = [summarize_run(json.loads(path.read_text(encoding="utf-8")), path) for path in paths]
    confirmed = [run for run in runs if run["confirmed_any_metric_gain"]]
    fallback = [run for run in runs if not run["selected_llm_skill"]]
    development = [run for run in runs if run["evaluation_stage"] == "development"]
    evaluated = [
        run for run in runs
        if run["evaluation_stage"] in {"confirmation", "fallback"}
    ]
    domains = sorted({str(run["domain"]) for run in runs if run["domain"] != "unknown"})
    confirmed_domains = sorted({
        str(run["domain"])
        for run in confirmed
        if run["domain"] != "unknown"
    })
    return {
        "experiment": "care2_multidomain_llm_confirmation",
        "predeclared_run_count": len(runs),
        "evaluated_run_count": len(evaluated),
        "confirmed_gain_count": len(confirmed),
        "confirmed_gain_rate": (
            round(len(confirmed) / len(evaluated), 6) if evaluated else 0.0
        ),
        "fallback_count": len(fallback),
        "development_signal_count": sum(
            bool(run["development_any_metric_signal"]) for run in development
        ),
        "development_run_count": len(development),
        "domains": domains,
        "confirmed_domains": confirmed_domains,
        "multidomain_gain": len(confirmed_domains) >= 2,
        "majority_gain": (
            len(confirmed) > len(evaluated) / 2 if evaluated else False
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = report["runs"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            flattened = [
                {
                    **row,
                    "positive_ci_metrics": ",".join(row["positive_ci_metrics"]),
                    "care_beats_external_llm": json.dumps(row["care_beats_external_llm"], sort_keys=True),
                }
                for row in rows
            ]
            writer = csv.DictWriter(
                handle,
                fieldnames=list(flattened[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(flattened)
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
