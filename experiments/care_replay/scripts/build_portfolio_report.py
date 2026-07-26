#!/usr/bin/env python3
"""Build a compact audit report for frozen transfer portfolios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PORTFOLIO_MODE = "care_source_outcome_portfolio"
MATCHED_MODE = "matched_target_only_llm"
BO_MODES = ("gp_ucb", "mixed_kernel_gp_ei", "target_acquisition_portfolio")


def _scalar(metrics: dict[str, Any], name: str) -> float:
    value = metrics[name]
    if isinstance(value, dict):
        return float(value["mean"])
    return float(value)


def _strongest_bo(summary: dict[str, Any]) -> str:
    heldout = summary["heldout"]
    available = [mode for mode in BO_MODES if mode in heldout]
    return max(
        available,
        key=lambda mode: _scalar(heldout[mode], "composite"),
    )


def load_portfolio_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    selection = summary["selection"]
    heldout = summary["heldout"]
    candidate_by_id = {
        item["candidate_id"]: item
        for item in summary.get("portfolio_candidates", [])
    }
    selected_candidate = candidate_by_id.get(selection.get("selected_candidate_id"), {})
    initial_strategy = selection.get(
        "selected_initial_strategy",
        selected_candidate.get("source_initial_strategy", "matched"),
    )
    max_transfer_mass = float(selection.get(
        "selected_router_max_transfer_mass",
        selected_candidate.get("router_max_transfer_mass", 0.0),
    ))
    deployed_mode = (
        PORTFOLIO_MODE
        if selection["selected_source_outcome_transfer"]
        else MATCHED_MODE
    )
    anchor_mode = selection.get("target_anchor_mode") or _strongest_bo(summary)
    deployed = heldout[deployed_mode]
    matched = heldout[MATCHED_MODE]
    anchor = heldout[anchor_mode]
    if not selection["selected_source_outcome_transfer"]:
        deployment_kind = "exact fallback"
    elif initial_strategy != "matched":
        deployment_kind = "source-informed warm start"
    elif max_transfer_mass > 0.0:
        deployment_kind = "continuous transfer candidate"
    else:
        deployment_kind = "source-informed deployment"
    return {
        "source_dataset": summary["source_dataset"],
        "target_dataset": summary["target_dataset"],
        "calibration_seed_count": int(summary["calibration_seed_count"]),
        "heldout_seed_count": int(summary["heldout_seed_count"]),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "selected_initial_strategy": initial_strategy,
        "selected_router_max_transfer_mass": max_transfer_mass,
        "deployed_mode": deployed_mode,
        "transfer_deployed": bool(selection["selected_source_outcome_transfer"]),
        "deployment_kind": deployment_kind,
        "strongest_bo_mode": anchor_mode,
        "final_delta_vs_matched": _scalar(deployed, "final_best") - _scalar(matched, "final_best"),
        "auc_delta_vs_matched": _scalar(deployed, "best_so_far_auc") - _scalar(matched, "best_so_far_auc"),
        "final_delta_vs_strongest_bo": _scalar(deployed, "final_best") - _scalar(anchor, "final_best"),
        "auc_delta_vs_strongest_bo": _scalar(deployed, "best_so_far_auc") - _scalar(anchor, "best_so_far_auc"),
        "archive": str(path.parent.parent),
    }


def render_markdown(items: list[dict[str, Any]]) -> str:
    deployed = sum(item["transfer_deployed"] for item in items)
    warm_start = sum(
        item["deployment_kind"] == "source-informed warm start"
        for item in items
    )
    continuous = sum(
        item["deployment_kind"] == "continuous transfer candidate"
        for item in items
    )
    exact_fallback = len(items) - deployed
    lines = [
        "# CARE 2.0 Frozen Transfer Portfolio Report",
        "",
        "This report summarizes frozen candidate portfolios. Candidate selection "
        "uses calibration seeds only; held-out seeds execute the selected route or "
        "the exact matched target-only fallback.",
        "",
        "The deltas below are held-out point estimates. Inferential claims should "
        "use the per-archive paired summaries and audit files; no fallback is "
        "classified as a positive transfer gain.",
        "",
        "| Source -> target | Cal / held-out | Deployed | Candidate | "
        "Final vs matched | AUC vs matched | Final vs strongest BO | AUC vs strongest BO |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in items:
        candidate = item["selected_candidate_id"] or "-"
        lines.append(
            f"| {item['source_dataset']} -> {item['target_dataset']} | "
            f"{item['calibration_seed_count']} / {item['heldout_seed_count']} | "
            f"{item['deployment_kind']} | {candidate} | "
            f"{item['final_delta_vs_matched']:+.4f} | "
            f"{item['auc_delta_vs_matched']:+.4f} | "
            f"{item['final_delta_vs_strongest_bo']:+.4f} | "
            f"{item['auc_delta_vs_strongest_bo']:+.4f} |"
        )
    lines.extend([
        "",
        f"Source-informed deployments: {deployed}/{len(items)}; warm-start "
        f"deployments: {warm_start}/{len(items)}; continuous-transfer candidates: "
        f"{continuous}/{len(items)}; exact fallbacks: "
        f"{exact_fallback}/{len(items)}.",
        "",
        "The current evidence supports selective source-informed deployment: the "
        "router can deploy a strong warm-start route on a reaction pair and refuse "
        "unstable molecular/materials routes. Warm-start and continuous-transfer "
        "mechanisms are reported separately; it does not yet support the stronger "
        "claim that every domain or every source-target pair improves.",
        "",
        "## Archives",
        "",
    ])
    lines.extend(f"- `{item['archive']}`" for item in items)
    lines.append("")
    return "\n".join(lines)


def build_report(archive_dirs: list[Path]) -> dict[str, Any]:
    items = []
    for archive in sorted(archive_dirs):
        summaries = sorted((archive / "raw_summaries").glob("*_summary.json"))
        if len(summaries) != 1:
            raise ValueError(f"Expected one summary in {archive}, found {len(summaries)}")
        items.append(load_portfolio_summary(summaries[0]))
    return {
        "study": "care2_frozen_transfer_portfolio",
        "portfolio_count": len(items),
        "transfer_deployed_count": sum(item["transfer_deployed"] for item in items),
        "warm_start_count": sum(
            item["deployment_kind"] == "source-informed warm start"
            for item in items
        ),
        "continuous_transfer_candidate_count": sum(
            item["deployment_kind"] == "continuous transfer candidate"
            for item in items
        ),
        "exact_fallback_count": sum(not item["transfer_deployed"] for item in items),
        "portfolios": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "portfolio_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "portfolio_report.md").write_text(
        render_markdown(report["portfolios"]), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "portfolio_count", "transfer_deployed_count", "exact_fallback_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
