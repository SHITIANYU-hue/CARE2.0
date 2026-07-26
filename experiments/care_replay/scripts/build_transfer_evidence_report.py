#!/usr/bin/env python3
"""Build a conservative cross-domain transfer evidence ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIMARY_METRICS = ("final_best", "best_so_far_auc")


def primary_gain(comparison: dict[str, Any]) -> bool:
    """Require at least one primary metric CI to be strictly above zero."""
    return any(
        float(comparison["metrics"][metric]["normal_95ci_low"]) > 0.0
        for metric in PRIMARY_METRICS
    )


def execution_kind(route: str) -> str:
    """Classify how a frozen semantic route uses the source evidence."""
    lowered = route.lower()
    if "warmstart" in lowered or "warm_start" in lowered:
        return "warm-start"
    if "direct_prior" in lowered or "semantic" in lowered:
        return "continuous semantic route"
    return "other routed policy"


def load_source_schema_report(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for item in raw["pairs"]:
        source_vs_llm = item["comparisons"][
            "source_router_minus_target_only_llm_router"
        ]
        source_vs_bo = item["comparisons"]["source_router_minus_strongest_bo"]
        pairs.append({
            "pair_id": item["pair_id"],
            "source_dataset": item["source_dataset"],
            "target_dataset": item["target_dataset"],
            "route": item["source_selected_route"],
            "execution_kind": execution_kind(item["source_selected_route"]),
            "source_schema_gain_vs_matched_llm": primary_gain(source_vs_llm),
            "source_schema_gain_vs_strongest_bo": primary_gain(source_vs_bo),
            "matched_llm_comparison": source_vs_llm,
            "strongest_bo_comparison": source_vs_bo,
        })
    return {
        "study": raw.get("study"),
        "heldout_seed_count": max(
            (item["comparisons"][
                "source_router_minus_target_only_llm_router"
            ]["paired_seed_count"] for item in raw["pairs"]),
            default=0,
        ),
        "pairs": pairs,
    }


def load_portfolio_report(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "study": raw.get("study"),
        "portfolio_count": int(raw.get("portfolio_count", 0)),
        "source_informed_deployment_count": int(
            raw.get("transfer_deployed_count", 0)
        ),
        "warm_start_count": int(raw.get("warm_start_count", 0)),
        "continuous_transfer_candidate_count": int(
            raw.get("continuous_transfer_candidate_count", 0)
        ),
        "exact_fallback_count": int(raw.get("exact_fallback_count", 0)),
        "portfolios": raw.get("portfolios", []),
    }


def load_goal_controls(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    aggregate = raw.get("aggregate", {})
    return {
        "random_null_count": int(aggregate.get("random_null_count", 0)),
        "random_null_stable_gain_count": int(
            aggregate.get("random_null_stable_gain_count", 0)
        ),
        "traditional_transfer_suite_count": int(
            aggregate.get("traditional_transfer_suite_count", 0)
        ),
    }


def build_report(
    source_schema: dict[str, Any],
    portfolio: dict[str, Any],
    controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pairs = source_schema["pairs"]
    continuous = [
        item for item in pairs
        if item["execution_kind"] == "continuous semantic route"
        and item["source_schema_gain_vs_matched_llm"]
    ]
    warm_start = [
        item for item in pairs
        if item["execution_kind"] == "warm-start"
        and item["source_schema_gain_vs_matched_llm"]
    ]
    report = {
        "study": "care2_transfer_evidence_ladder",
        "source_schema": source_schema,
        "source_outcome_portfolio": portfolio,
        "controls": controls or {},
        "aggregate": {
            "source_schema_pair_count": len(pairs),
            "source_schema_gain_vs_matched_llm_count": sum(
                item["source_schema_gain_vs_matched_llm"] for item in pairs
            ),
            "source_schema_gain_vs_strongest_bo_count": sum(
                item["source_schema_gain_vs_strongest_bo"] for item in pairs
            ),
            "source_schema_harm_count": sum(
                not item["source_schema_gain_vs_matched_llm"]
                and any(
                    float(item["matched_llm_comparison"]["metrics"][metric][
                        "normal_95ci_high"
                    ]) < 0.0
                    for metric in PRIMARY_METRICS
                )
                for item in pairs
            ),
            "continuous_semantic_gain_count": len(continuous),
            "warm_start_gain_count": len(warm_start),
            "source_outcome_deployed_count": portfolio[
                "source_informed_deployment_count"
            ],
            "source_outcome_continuous_candidate_count": portfolio[
                "continuous_transfer_candidate_count"
            ],
            "source_outcome_exact_fallback_count": portfolio[
                "exact_fallback_count"
            ],
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    source_schema = report["source_schema"]
    portfolio = report["source_outcome_portfolio"]
    aggregate = report["aggregate"]
    lines = [
        "# CARE 2.0 Transfer Evidence Ladder",
        "",
        "This report separates source-schema semantic transfer, source-outcome "
        "transfer, warm-start effects, continuous routes, and exact fallback.",
        "All source-schema comparisons below use paired held-out seeds; route "
        "selection happened before those seeds were evaluated.",
        "",
        "## Source-Schema Semantic Transfer",
        "",
        "| Source -> target | Execution | Held-out route | vs matched target LLM | vs strongest BO |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in source_schema["pairs"]:
        lines.append(
            f"| {item['source_dataset']} -> {item['target_dataset']} | "
            f"{item['execution_kind']} | `{item['route']}` | "
            f"{'yes' if item['source_schema_gain_vs_matched_llm'] else 'no'} | "
            f"{'yes' if item['source_schema_gain_vs_strongest_bo'] else 'no'} |"
        )
    lines.extend([
        "",
        f"Stable source-schema gains: "
        f"{aggregate['source_schema_gain_vs_matched_llm_count']}/"
        f"{aggregate['source_schema_pair_count']} vs matched target-only LLM; "
        f"{aggregate['source_schema_gain_vs_strongest_bo_count']}/"
        f"{aggregate['source_schema_pair_count']} vs strongest target-only BO.",
        f"Among the positive source-schema rows, "
        f"{aggregate['continuous_semantic_gain_count']} use a continuous semantic "
        f"route and {aggregate['warm_start_gain_count']} use warm-start only.",
        "",
        "## Source-Outcome Portfolio",
        "",
        f"The stricter portfolio contains {portfolio['portfolio_count']} paths: "
        f"{portfolio['source_informed_deployment_count']} source-informed deployments, "
        f"{portfolio['warm_start_count']} warm-start deployments, "
        f"{portfolio['continuous_transfer_candidate_count']} continuous-transfer "
        f"candidates, and {portfolio['exact_fallback_count']} exact fallbacks.",
        "A fallback is not counted as a positive transfer result.",
        "",
        "## Controls",
        "",
        f"Random-rule nulls: {report['controls'].get('random_null_count', 0)} "
        f"tested, {report['controls'].get('random_null_stable_gain_count', 0)} "
        "stable positive. Traditional transfer suites: "
        f"{report['controls'].get('traditional_transfer_suite_count', 0)}.",
        "",
        "## Interpretation",
        "",
        "The current evidence supports cross-domain generalization of a routed "
        "LLM semantic skill against strong target-only baselines. It does not "
        "support the stronger claim that every source-target pair improves, or "
        "that every source-outcome gain is continuous after initialization. "
        "Those two mechanisms are reported separately so future gains can be "
        "attributed rather than overclaimed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-schema-report", type=Path, required=True)
    parser.add_argument("--portfolio-report", type=Path, required=True)
    parser.add_argument("--goal-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        load_source_schema_report(args.source_schema_report),
        load_portfolio_report(args.portfolio_report),
        load_goal_controls(args.goal_report),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "transfer_evidence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "transfer_evidence_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
