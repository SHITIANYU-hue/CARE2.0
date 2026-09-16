#!/usr/bin/env python3
"""Run a frozen collection of online-LLM scientist cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_online_llm_scientist as online


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ensure_initial_record(
    case: dict[str, Any],
    suite: dict[str, Any],
    output_dir: Path,
) -> Path:
    configured_record = case.get("initial_record")
    if configured_record:
        return resolve(str(configured_record))

    target_task = str(case.get("target_task", ""))
    source_tasks = [str(item) for item in case.get("source_tasks", [])]
    if not target_task or not source_tasks:
        raise ValueError(
            "A suite case must provide either initial_record or both "
            "target_task and source_tasks."
        )

    record_path = output_dir / "initial_record.json"
    if record_path.exists():
        return record_path

    generate_args = argparse.Namespace(
        config=resolve(str(case["config"])),
        target_task=target_task,
        source_tasks=source_tasks,
        output=record_path,
        per_view_limit=int(suite.get("initial_per_view_limit", 10)),
        llm_base_url=str(suite["base_url"]),
        llm_model=str(suite["model"]),
        llm_api_key_env=str(suite["api_key_env"]),
        llm_api_mode=str(suite["api_mode"]),
        llm_temperature=float(suite["temperature"]),
        llm_max_tokens=int(suite.get("initial_max_tokens", 2200)),
        llm_repair_attempts=int(suite["repair_attempts"]),
    )
    online.generate_initial(generate_args)
    return record_path


def aggregate(output_root: Path, suite: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in suite["cases"]:
        summary_path = output_root / str(case["case_id"]) / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        metrics = summary["metrics"][online.MODE]
        deltas = summary["deltas"]
        rows.append(
            {
                "case_id": case["case_id"],
                "domain": case["domain"],
                "target_task": summary["target_task"],
                "online_auc_delta_vs_same_initial_gp": deltas[
                    "online_llm_increment_over_same_initial_gp"
                ]["best_so_far_auc"],
                "online_final_delta_vs_same_initial_gp": deltas[
                    "online_llm_increment_over_same_initial_gp"
                ]["final_best"],
                "full_auc_delta_vs_fixed_v2": deltas[
                    "full_llm_scientist_vs_fixed_v2"
                ]["best_so_far_auc"],
                "full_final_delta_vs_fixed_v2": deltas[
                    "full_llm_scientist_vs_fixed_v2"
                ]["final_best"],
                "participation_rate": metrics["llm_participation_rate"],
                "decision_authority_rate": metrics[
                    "llm_decision_authority_rate"
                ],
                "mean_eligible_candidates": metrics[
                    "llm_mean_eligible_candidate_count"
                ],
                "gp_override_rate": metrics["llm_gp_override_rate"],
                "critic_revision_rate": metrics["llm_critic_revision_rate"],
                "source_transfer_active_rate": metrics[
                    "source_transfer_active_rate"
                ],
                "total_tokens": summary["usage"]["total_tokens"],
            }
        )
    if not rows:
        raise ValueError("No completed case summaries were found.")

    online_deltas = np.asarray(
        [row["online_auc_delta_vs_same_initial_gp"] for row in rows],
        dtype=np.float64,
    )
    online_final_deltas = np.asarray(
        [row["online_final_delta_vs_same_initial_gp"] for row in rows],
        dtype=np.float64,
    )
    full_deltas = np.asarray(
        [row["full_auc_delta_vs_fixed_v2"] for row in rows],
        dtype=np.float64,
    )
    full_final_deltas = np.asarray(
        [row["full_final_delta_vs_fixed_v2"] for row in rows],
        dtype=np.float64,
    )

    def delta_summary(
        auc_deltas: np.ndarray, final_deltas: np.ndarray
    ) -> dict[str, Any]:
        return {
            "mean_auc_delta": round(float(np.mean(auc_deltas)), 6),
            "mean_final_delta": round(float(np.mean(final_deltas)), 6),
            "wins": int(np.sum(auc_deltas > 1e-9)),
            "ties": int(np.sum(np.abs(auc_deltas) <= 1e-9)),
            "losses": int(np.sum(auc_deltas < -1e-9)),
        }

    report = {
        "schema_version": "care.opus_high_authority_suite/v1",
        "suite": suite["suite"],
        "evidence_class": suite["evidence_class"],
        "model": suite["model"],
        "completed_case_count": len(rows),
        "online_llm_vs_same_initial_gp": delta_summary(
            online_deltas, online_final_deltas
        ),
        "full_llm_scientist_vs_fixed_v2": delta_summary(
            full_deltas, full_final_deltas
        ),
        "mean_participation_rate": round(
            float(np.mean([row["participation_rate"] for row in rows])), 6
        ),
        "mean_decision_authority_rate": round(
            float(np.mean([row["decision_authority_rate"] for row in rows])), 6
        ),
        "mean_eligible_candidates": round(
            float(np.mean([row["mean_eligible_candidates"] for row in rows])), 6
        ),
        "mean_gp_override_rate": round(
            float(np.mean([row["gp_override_rate"] for row in rows])), 6
        ),
        "mean_critic_revision_rate": round(
            float(np.mean([row["critic_revision_rate"] for row in rows])), 6
        ),
        "mean_source_transfer_active_rate": round(
            float(np.mean([row["source_transfer_active_rate"] for row in rows])),
            6,
        ),
        "cases": rows,
    }
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "suite_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (aggregate_dir / "suite_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = [
        (
            f"| {row['case_id']} | {row['domain']} | "
            f"{row['online_auc_delta_vs_same_initial_gp']:+.4f} | "
            f"{row['online_final_delta_vs_same_initial_gp']:+.4f} | "
            f"{row['gp_override_rate']:.0%} | "
            f"{row['source_transfer_active_rate']:.0%} |"
        )
        for row in rows
    ]
    comparison = report["online_llm_vs_same_initial_gp"]
    report_lines = [
        f"# {suite['suite']}",
        "",
        f"- Evidence class: `{suite['evidence_class']}`",
        f"- Model: `{suite['model']}`",
        "- Primary comparator: the same LLM-selected initial observations followed by target-only GP-UCB.",
        "- Target outcomes are revealed only after each candidate is selected.",
        "",
        "## Aggregate",
        "",
        (
            f"Across {len(rows)} completed routes, the online LLM controller changed "
            f"best-so-far AUC by {comparison['mean_auc_delta']:+.4f} on average "
            f"({comparison['wins']} wins, {comparison['ties']} ties, "
            f"{comparison['losses']} losses)."
        ),
        (
            f"Mean LLM participation was {report['mean_participation_rate']:.0%}; "
            f"mean decision authority was {report['mean_decision_authority_rate']:.0%}; "
            f"the GP default was overridden in {report['mean_gp_override_rate']:.0%} "
            "of rounds."
        ),
        "",
        "## Routes",
        "",
        "| Route | Domain | AUC delta vs GP | Final delta vs GP | GP override | Source active |",
        "|---|---|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Interpretation boundary",
        "",
        (
            "A positive delta isolates the online LLM controller because both methods "
            "start from the same initial observations and use the same target budget. "
            "It does not by itself prove universal cross-domain transfer."
        ),
        (
            "`Source active` reports whether the LLM continued to rely on source-task "
            "evidence. A route can improve after source transfer is stopped; that is a "
            "useful routing result, but it is not positive source-outcome transfer."
        ),
        "",
        "Each case directory contains the frozen initial record, full proposal/critic trace, summary, and SHA-256 fingerprints.",
        "",
    ]
    report_path = aggregate_dir / "RESULTS.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    for path in (
        aggregate_dir / "suite_summary.json",
        aggregate_dir / "suite_metrics.csv",
        report_path,
    ):
        online.write_fingerprint(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    suite = load_json(args.suite_config)
    selected_cases = set(args.case)
    for case in suite["cases"]:
        case_id = str(case["case_id"])
        if selected_cases and case_id not in selected_cases:
            continue
        output_dir = args.output_root / case_id
        if args.skip_existing and (output_dir / "summary.json").exists():
            continue
        initial_record = ensure_initial_record(case, suite, output_dir)
        run_args = argparse.Namespace(
            config=resolve(str(case["config"])),
            initial_record=initial_record,
            output_dir=output_dir,
            rounds=int(suite["rounds"]),
            initial_design_mode=str(suite["initial_design_mode"]),
            menu_gp_count=int(suite["menu"]["gp_count"]),
            menu_source_count=int(suite["menu"]["source_count"]),
            menu_consensus_count=int(suite["menu"]["consensus_count"]),
            max_transfer_gp_rank=int(suite["menu"]["max_transfer_gp_rank"]),
            safety_fallback_gp_count=int(
                suite["menu"]["safety_fallback_gp_count"]
            ),
            force_first_consensus=bool(suite["force_first_consensus"]),
            menu_diversity_count=int(suite["menu"]["diversity_count"]),
            decision_policy=str(suite["decision_policy"]),
            deliberation_mode=str(suite["deliberation_mode"]),
            fail_on_llm_error=True,
            llm_base_url=str(suite["base_url"]),
            llm_model=str(suite["model"]),
            llm_api_key_env=str(suite["api_key_env"]),
            llm_api_mode=str(suite["api_mode"]),
            llm_temperature=float(suite["temperature"]),
            llm_max_tokens=int(suite["max_tokens"]),
            llm_repair_attempts=int(suite["repair_attempts"]),
            calibration_gate_round=suite.get("calibration_gate", {}).get(
                "round"
            ),
            calibration_gate_mae_threshold=suite.get(
                "calibration_gate", {}
            ).get("mae_threshold"),
            calibration_gate_hard_abstention=bool(
                suite.get("calibration_gate", {}).get(
                    "hard_abstention", True
                )
            ),
            calibration_gate_force_fallback=bool(
                suite.get("calibration_gate", {}).get(
                    "force_fallback", False
                )
            ),
        )
        online.run_online(run_args)

    report = aggregate(args.output_root, suite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
