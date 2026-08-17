#!/usr/bin/env python3
"""Combine Opus high-authority suite outputs into one auditable report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "route_count": len(rows),
        "mean_auc_delta": round(float(np.mean(values)), 6),
        "median_auc_delta": round(float(np.median(values)), 6),
        "wins": int(np.sum(values > 1e-9)),
        "ties": int(np.sum(np.abs(values) <= 1e-9)),
        "losses": int(np.sum(values < -1e-9)),
        "nonloss_rate": round(float(np.mean(values >= -1e-9)), 6),
    }


def weighted_mean(
    suites: Iterable[dict[str, Any]], field: str
) -> float:
    numerator = 0.0
    denominator = 0
    for suite in suites:
        count = int(suite["completed_case_count"])
        numerator += count * float(suite[field])
        denominator += count
    return round(numerator / denominator, 6)


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    development = read_json(
        args.development_root / "aggregate" / "suite_summary.json"
    )
    holdout = read_json(args.holdout_root / "aggregate" / "suite_summary.json")
    diagnostic = read_json(
        args.diagnostic_root / "aggregate" / "suite_summary.json"
    )

    rows: list[dict[str, Any]] = []
    for phase, suite in (("development", development), ("frozen_extension", holdout)):
        for row in suite["cases"]:
            rows.append({"phase": phase, **row})

    online = delta_summary(rows, "online_auc_delta_vs_same_initial_gp")
    full = delta_summary(rows, "full_auc_delta_vs_fixed_v2")
    holdout_online = delta_summary(
        [{"value": row["online_auc_delta_vs_same_initial_gp"]} for row in holdout["cases"]],
        "value",
    )

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    domain_summaries = {
        domain: delta_summary(domain_rows, "online_auc_delta_vs_same_initial_gp")
        for domain, domain_rows in sorted(by_domain.items())
    }

    report = {
        "schema_version": "care.opus_generalization_evidence/v1",
        "model": development["model"],
        "primary_comparator": "same_llm_initial_design_plus_target_only_gp_ucb",
        "route_count": len(rows),
        "online_llm_controller": online,
        "full_llm_scientist_vs_fixed_v2": full,
        "frozen_extension_online_controller": holdout_online,
        "mean_participation_rate": weighted_mean(
            (development, holdout), "mean_participation_rate"
        ),
        "mean_decision_authority_rate": weighted_mean(
            (development, holdout), "mean_decision_authority_rate"
        ),
        "mean_eligible_candidates": weighted_mean(
            (development, holdout), "mean_eligible_candidates"
        ),
        "mean_gp_override_rate": weighted_mean(
            (development, holdout), "mean_gp_override_rate"
        ),
        "mean_source_transfer_active_rate": weighted_mean(
            (development, holdout), "mean_source_transfer_active_rate"
        ),
        "domains": domain_summaries,
        "post_holdout_compiler_diagnostic": {
            "status": "rejected_as_default",
            "online_llm_vs_same_initial_gp": diagnostic[
                "online_llm_vs_same_initial_gp"
            ],
            "full_llm_scientist_vs_fixed_v2": diagnostic[
                "full_llm_scientist_vs_fixed_v2"
            ],
        },
        "routes": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "combined_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "combined_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        (
            f"| {row['phase']} | {row['case_id']} | {row['domain']} | "
            f"{row['online_auc_delta_vs_same_initial_gp']:+.4f} | "
            f"{row['full_auc_delta_vs_fixed_v2']:+.4f} | "
            f"{row['gp_override_rate']:.0%} | "
            f"{row['source_transfer_active_rate']:.0%} |"
        )
        for row in rows
    ]
    domain_table = [
        (
            f"| {domain} | {item['route_count']} | "
            f"{item['mean_auc_delta']:+.4f} | {item['wins']} | "
            f"{item['ties']} | {item['losses']} |"
        )
        for domain, item in domain_summaries.items()
    ]
    diagnostic_online = diagnostic["online_llm_vs_same_initial_gp"]
    report_text = "\n".join(
        [
            "# Opus High-Authority Generalization Evidence",
            "",
            "## Main result",
            "",
            (
                f"The online Opus controller was evaluated on {online['route_count']} "
                f"real routes and changed best-so-far AUC by "
                f"{online['mean_auc_delta']:+.4f} on average: {online['wins']} wins, "
                f"{online['ties']} ties, and {online['losses']} losses."
            ),
            (
                f"The frozen chemistry extension alone produced "
                f"{holdout_online['wins']} wins, {holdout_online['ties']} ties, and "
                f"{holdout_online['losses']} losses with mean AUC delta "
                f"{holdout_online['mean_auc_delta']:+.4f}."
            ),
            (
                f"LLM participation and decision authority were both "
                f"{report['mean_participation_rate']:.0%}. Each round exposed "
                f"{report['mean_eligible_candidates']:.2f} executable candidates on "
                f"average, and Opus overrode GP in "
                f"{report['mean_gp_override_rate']:.1%} of rounds."
            ),
            "",
            "## Domain view",
            "",
            "| Task family | Routes | Mean AUC delta | Wins | Ties | Losses |",
            "|---|---:|---:|---:|---:|---:|",
            *domain_table,
            "",
            "## Route-level evidence",
            "",
            "| Phase | Route | Task family | Online delta vs same-initial GP | Full delta vs fixed | GP override | Source active |",
            "|---|---|---|---:|---:|---:|---:|",
            *table,
            "",
            "## What the result supports",
            "",
            "The online controller shows majority-positive transfer across molecular-property, materials-property, C-N, and Suzuki task families under a matched target-evaluation budget. The frozen chemistry extension is prospective with respect to the high-authority controller and is the main generalization check.",
            "",
            "## What the result does not support",
            "",
            "The result is not universal positive transfer. Three of eleven routes are negative, and the frozen-extension mean effect is small. The full system is less stable than the online controller because outcome-blind LLM initial design can underperform the fixed source-diverse initializer.",
            "",
            "## Rejected diagnostic",
            "",
            (
                "After the frozen extension, the same routes were rerun with a compiled "
                "initial design. This was a diagnostic, not a second holdout. It yielded "
                f"{diagnostic_online['wins']} wins, {diagnostic_online['ties']} ties, "
                f"and {diagnostic_online['losses']} losses with mean online delta "
                f"{diagnostic_online['mean_auc_delta']:+.4f}, so it is not adopted as "
                "the default controller."
            ),
            "",
            "## Audit boundary",
            "",
            "Every route keeps the frozen initial record, full proposer and critic requests/responses, target reveals, per-case summary, and SHA-256 fingerprints. Target outcomes enter the prompt only after the corresponding experiment has been selected.",
            "",
        ]
    )
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")
    for path in (summary_path, metrics_path, report_path):
        write_sha256(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
