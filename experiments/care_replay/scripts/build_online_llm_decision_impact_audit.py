#!/usr/bin/env python3
"""Audit whether bounded online LLM calls actually change the search path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


SCHEMA_VERSION = "care.online_llm_decision_impact_audit/v1"
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def repository_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def final_llm_decision(
    events: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    for event in events:
        if event.get("event") != "llm_round_response":
            continue
        decision = event.get("normalized_decision")
        if not isinstance(decision, Mapping):
            continue
        return dict(decision), str(event.get("round_index", 0))
    raise ValueError("Trace has no executable llm_round_response event.")


def gp_default_candidate(events: Iterable[Mapping[str, Any]]) -> str:
    for event in events:
        if event.get("event") != "llm_round_request":
            continue
        prompt = event.get("prompt", {})
        context = prompt.get("decision_context", {}) if isinstance(prompt, Mapping) else {}
        candidate = context.get("gp_default_candidate")
        if candidate:
            return str(candidate)
    raise ValueError("Trace has no GP default candidate in the LLM request.")


def route_record(summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    trace_path = summary_path.with_name("llm_trace.jsonl")
    events = read_trace(trace_path)
    decision, decision_round = final_llm_decision(events)
    gp_default = gp_default_candidate(events)
    selected = str(decision["selected_candidate_id"])
    metrics = summary["metrics"]["online_llm_scientist"]
    online_delta = summary["deltas"][
        "online_llm_increment_over_same_initial_gp"
    ]["best_so_far_auc"]
    full_delta = summary["deltas"]["full_llm_scientist_vs_fixed_v2"][
        "best_so_far_auc"
    ]
    return {
        "case_id": summary_path.parents[1].name,
        "replicate_id": int(summary_path.parent.name.split("_")[-1]),
        "source_tasks": list(summary.get("source_tasks", [])),
        "target_task": str(summary.get("target_task", "")),
        "online_model": str(summary.get("online_model", "")),
        "decision_round": int(decision_round or 0),
        "gp_default_candidate": gp_default,
        "selected_candidate_id": selected,
        "changed_gp_rank_one": selected != gp_default,
        "decision_verdict": str(decision.get("decision_verdict", "")),
        "hypothesis_status": str(decision.get("hypothesis_status", "")),
        "continue_source_transfer": bool(
            decision.get("continue_source_transfer", False)
        ),
        "critic_revised_choice": float(metrics["llm_critic_revision_rate"]) > 0.0,
        "online_auc_delta_vs_same_initial_gp": float(online_delta),
        "complete_system_auc_delta_vs_fixed_v2": float(full_delta),
        "total_tokens": int(summary["usage"]["total_tokens"]),
        "fallback_rounds": int(metrics["calibration_gate_fallback_rounds"]),
        "nominal_llm_calls_avoided": int(
            metrics["calibration_gate_nominal_llm_calls_avoided"]
        ),
        "summary_path": repository_path(summary_path),
        "trace_path": repository_path(trace_path),
    }


def collect_records(input_root: Path) -> list[dict[str, Any]]:
    records = [
        route_record(path)
        for path in sorted(input_root.glob("*/trajectory_*/summary.json"))
    ]
    if not records:
        raise ValueError(f"No completed trajectories found below {input_root}.")
    return records


def sign_counts(values: Iterable[float], tolerance: float = 1e-12) -> dict[str, int]:
    values = list(values)
    return {
        "wins": sum(value > tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
        "losses": sum(value < -tolerance for value in values),
    }


def build_audit(
    records: list[dict[str, Any]], input_root: Path
) -> dict[str, Any]:
    online_deltas = [
        float(row["online_auc_delta_vs_same_initial_gp"]) for row in records
    ]
    full_deltas = [
        float(row["complete_system_auc_delta_vs_fixed_v2"]) for row in records
    ]
    failure_path = input_root / "aggregate" / "repeated_confirmation.json"
    failures = []
    if failure_path.exists():
        failures = list(load_json(failure_path).get("failures", []))
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "partial_operational_pilot_not_confirmatory",
        "input_root": repository_path(input_root),
        "completed_trajectory_count": len(records),
        "failed_trajectory_count": len(failures),
        "route_count": len({row["case_id"] for row in records}),
        "decision_impact": {
            "gp_rank_one_choice_count": sum(
                not bool(row["changed_gp_rank_one"]) for row in records
            ),
            "gp_override_count": sum(
                bool(row["changed_gp_rank_one"]) for row in records
            ),
            "gp_override_rate": float(
                np.mean([bool(row["changed_gp_rank_one"]) for row in records])
            ),
            "source_transfer_continue_count": sum(
                bool(row["continue_source_transfer"]) for row in records
            ),
            "critic_choice_revision_count": sum(
                bool(row["critic_revised_choice"]) for row in records
            ),
        },
        "performance": {
            "mean_online_auc_delta_vs_same_initial_gp": float(np.mean(online_deltas)),
            "online_auc_sign_counts": sign_counts(online_deltas),
            "mean_complete_system_auc_delta_vs_fixed_v2": float(
                np.mean(full_deltas)
            ),
            "complete_system_auc_sign_counts": sign_counts(full_deltas),
        },
        "cost": {
            "total_tokens": sum(int(row["total_tokens"]) for row in records),
            "mean_tokens_per_trajectory": float(
                np.mean([int(row["total_tokens"]) for row in records])
            ),
            "total_fallback_rounds": sum(
                int(row["fallback_rounds"]) for row in records
            ),
            "total_nominal_llm_calls_avoided": sum(
                int(row["nominal_llm_calls_avoided"]) for row in records
            ),
        },
        "claim_boundary": (
            "One completed trajectory per route is an operational pilot only. "
            "The frozen protocol disables confirmatory inference until all declared "
            "replicates finish; no confidence interval or superiority claim is made."
        ),
        "failures": failures,
        "routes": records,
    }


def route_label(case_id: str) -> str:
    replacements = {
        "molecular_lipophilicity_to_freesolv": "Lipo → FreeSolv",
        "materials_bulk_to_shear_modulus": "Bulk → Shear",
        "materials_phonons_to_perovskites": "Phonons → Perovskites",
        "materials_expt_gap_to_steels": "Expt gap → Steels",
        "molecular_freesolv_to_bace": "FreeSolv → BACE",
        "materials_perovskites_to_mp_e_form": "Perovskites → Formation E",
    }
    return replacements.get(case_id, case_id.replace("_", " "))


def plot_audit(records: list[dict[str, Any]], output: Path) -> None:
    labels = [route_label(str(row["case_id"])) for row in records]
    online = [float(row["online_auc_delta_vs_same_initial_gp"]) for row in records]
    complete = [
        float(row["complete_system_auc_delta_vs_fixed_v2"]) for row in records
    ]
    y = np.arange(len(records))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.8),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    ax = axes[0]
    ax.axvline(0.0, color="#30343b", linewidth=1.0)
    ax.barh(
        y + 0.17,
        complete,
        height=0.3,
        color="#3e7cb1",
        label="Complete system vs fixed",
    )
    ax.scatter(
        online,
        y - 0.17,
        marker="|",
        s=260,
        linewidths=3,
        color="#ef8354",
        label="Online LLM vs same-start GP",
        zorder=4,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Best-so-far AUC delta")
    ax.set_title("Observed decision outcome by route")
    ax.grid(axis="x", alpha=0.2)
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    behavior_labels = [
        "GP rank-1 selected",
        "Non-GP override",
        "Source transfer kept",
        "Critic changed choice",
    ]
    behavior_values = [
        sum(not bool(row["changed_gp_rank_one"]) for row in records),
        sum(bool(row["changed_gp_rank_one"]) for row in records),
        sum(bool(row["continue_source_transfer"]) for row in records),
        sum(bool(row["critic_revised_choice"]) for row in records),
    ]
    ax = axes[1]
    colors = ["#5b8e7d", "#c44536", "#f2b134", "#6c5b7b"]
    bars = ax.barh(behavior_labels, behavior_values, color=colors, height=0.58)
    ax.set_xlim(0, max(6, len(records)) + 0.7)
    ax.set_xlabel("Completed trajectories")
    ax.set_title("Did the LLM change the action?")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, behavior_values):
        ax.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=10,
        )
    fig.suptitle(
        "Bounded online-LLM pilot: structured decisions, zero GP override",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
    )
    fig.text(
        0.5,
        0.015,
        "Six real online traces; one trajectory per route. Partial operational evidence only.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)
    svg_path = output.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def write_route_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "replicate_id",
        "target_task",
        "gp_default_candidate",
        "selected_candidate_id",
        "changed_gp_rank_one",
        "decision_verdict",
        "hypothesis_status",
        "continue_source_transfer",
        "critic_revised_choice",
        "online_auc_delta_vs_same_initial_gp",
        "complete_system_auc_delta_vs_fixed_v2",
        "total_tokens",
        "fallback_rounds",
        "nominal_llm_calls_avoided",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({field: row[field] for field in fields})


def write_results(path: Path, audit: Mapping[str, Any]) -> None:
    impact = audit["decision_impact"]
    performance = audit["performance"]
    cost = audit["cost"]
    online_counts = performance["online_auc_sign_counts"]
    full_counts = performance["complete_system_auc_sign_counts"]
    path.write_text(
        "\n".join(
            [
                "# Bounded online-LLM decision-impact pilot",
                "",
                (
                    f"The pilot completed {audit['completed_trajectory_count']} real "
                    f"online trajectories across {audit['route_count']} routes. "
                    "It is not a confirmatory analysis."
                ),
                "",
                "| Diagnostic | Result |",
                "|---|---:|",
                f"| LLM selected GP rank one | {impact['gp_rank_one_choice_count']} / {audit['completed_trajectory_count']} |",
                f"| LLM overrode GP rank one | {impact['gp_override_count']} / {audit['completed_trajectory_count']} |",
                f"| LLM kept source transfer active | {impact['source_transfer_continue_count']} / {audit['completed_trajectory_count']} |",
                f"| Critic changed proposer choice | {impact['critic_choice_revision_count']} / {audit['completed_trajectory_count']} |",
                f"| Online LLM mean AUC delta vs same-start GP | {performance['mean_online_auc_delta_vs_same_initial_gp']:+.4f} |",
                f"| Online LLM win / tie / loss | {online_counts['wins']} / {online_counts['ties']} / {online_counts['losses']} |",
                f"| Complete-system mean AUC delta vs fixed initial design | {performance['mean_complete_system_auc_delta_vs_fixed_v2']:+.4f} |",
                f"| Complete-system win / tie / loss | {full_counts['wins']} / {full_counts['ties']} / {full_counts['losses']} |",
                f"| Total tokens | {cost['total_tokens']:,} |",
                f"| Mean tokens per trajectory | {cost['mean_tokens_per_trajectory']:,.1f} |",
                f"| Nominal later LLM calls avoided by bounded authority | {cost['total_nominal_llm_calls_avoided']} |",
                "",
                "## Interpretation",
                "",
                (
                    "Every valid LLM call produced an auditable hypothesis and critic "
                    "decision, but every executable decision selected the target-only "
                    "GP-UCB rank-one candidate. The online module therefore had zero "
                    "functional decision impact in this pilot."
                ),
                "",
                (
                    "This result separates model participation from causal contribution. "
                    "The next method-development question is not whether the LLM can emit "
                    "reasoning, but whether a frozen semantic skill or challenger policy "
                    "changes actions and improves held-out outcomes."
                ),
                "",
                "## Claim boundary",
                "",
                str(audit["claim_boundary"]),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root / "pilot_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_records(input_root)
    audit = build_audit(records, input_root)

    json_path = output_dir / "decision_impact_audit.json"
    csv_path = output_dir / "route_decision_metrics.csv"
    results_path = output_dir / "PILOT_RESULTS.md"
    figure_path = output_dir / "decision_impact.png"
    write_json(json_path, audit)
    write_route_csv(csv_path, records)
    write_results(results_path, audit)
    plot_audit(records, figure_path)
    for path in (
        json_path,
        csv_path,
        results_path,
        figure_path,
        figure_path.with_suffix(".svg"),
        figure_path.with_suffix(".pdf"),
    ):
        write_sha256(path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
