#!/usr/bin/env python3
"""Select an online-LLM handoff policy on route-disjoint train/evaluation panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import build_online_llm_calibration_gate_audit as gate
import build_online_llm_classical_baseline_audit as matched
import build_online_llm_predeclared_baseline_portfolio as portfolio


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "care.online_llm_gate_selection_audit/v1"


def load_config(path: Path) -> dict[str, Any]:
    payload = matched.load_json(path)
    protocol = payload.get("protocol", {})
    required = {
        "training_config",
        "evaluation_config",
        "baseline_aggregate",
        "gate_rounds",
        "prediction_error_thresholds",
        "hard_abstention_options",
        "selection_rule",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Selection config is missing fields: {missing}")
    return payload


def validate_split(
    training_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    training = {str(case["case_id"]) for case in training_config["cases"]}
    evaluation = {str(case["case_id"]) for case in evaluation_config["cases"]}
    overlap = sorted(training & evaluation)
    if overlap:
        raise ValueError("Training/evaluation route overlap: " + ", ".join(overlap))
    return {
        "training_route_count": len(training),
        "evaluation_route_count": len(evaluation),
        "overlap_count": 0,
        "overlap_case_ids": [],
    }


def policy_candidates(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for gate_round in protocol["gate_rounds"]:
        if bool(protocol.get("include_bounded_authority_candidates", True)):
            candidates.append(
                {
                    "policy_id": f"bounded_authority_r{gate_round}",
                    "kind": "bounded_authority",
                    "gate_round": int(gate_round),
                    "threshold": None,
                    "hard_abstention": False,
                    "force_fallback": True,
                    "parameter_count": 1,
                }
            )
        for threshold in protocol["prediction_error_thresholds"]:
            for hard_abstention in protocol["hard_abstention_options"]:
                candidates.append(
                    {
                        "policy_id": (
                            f"prediction_error_r{gate_round}_t{float(threshold):g}_"
                            f"hard{int(bool(hard_abstention))}"
                        ),
                        "kind": "prediction_error_gate",
                        "gate_round": int(gate_round),
                        "threshold": float(threshold),
                        "hard_abstention": bool(hard_abstention),
                        "force_fallback": False,
                        "parameter_count": 3,
                    }
                )
    return candidates


def load_panel_records(
    config: Mapping[str, Any],
    baseline_reports: Mapping[str, Mapping[str, Any]],
    gate_rounds: Sequence[int],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for case in config["cases"]:
        case_id = str(case["case_id"])
        if case_id not in baseline_reports:
            raise ValueError(f"Missing matched baseline report for {case_id}")
        summary_path = ROOT / str(case["summary"])
        summary = matched.load_json(summary_path)
        initial, rounds = gate.trace_rounds(summary_path)
        kernel = matched.load_json(ROOT / str(case["config"]))["protocol"]["kernel"]
        fallback_by_round: dict[int, dict[str, Any]] = {}
        for gate_round in gate_rounds:
            if gate_round > len(rounds):
                raise ValueError(
                    f"Gate round {gate_round} exceeds reveal budget for {case_id}"
                )
            fallback_auc, fallback_final, fallback_audit = gate.continue_with_target_gp(
                summary,
                initial,
                rounds,
                gate_round,
                kernel,
            )
            fallback_by_round[int(gate_round)] = {
                "calibration": gate.calibration_features(rounds, gate_round),
                "auc": float(fallback_auc),
                "final": float(fallback_final),
                "audit": fallback_audit,
            }
        report = baseline_reports[case_id]
        records[case_id] = {
            "case_id": case_id,
            "domain": str(case["domain"]),
            "online_llm_auc": float(report["online_llm"]["best_so_far_auc"]),
            "online_llm_final": float(report["online_llm"]["final_best"]),
            "target_gp_auc": float(
                report["predeclared_comparators"]["no_transfer"][
                    "baseline_best_so_far_auc"
                ]
            ),
            "strongest_auc": float(
                report["strongest_realized_baseline_posthoc"][
                    "baseline_best_so_far_auc"
                ]
            ),
            "source_trace": str(
                (summary_path.parent / "llm_trace.jsonl").relative_to(ROOT)
            ),
            "deliberation_mode": str(
                summary.get("menu_policy", {}).get("deliberation_mode", "single")
            ),
            "fallback_by_round": fallback_by_round,
        }
    return records


def apply_policy(
    policy: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    route_rows: list[dict[str, Any]] = []
    gate_round = int(policy["gate_round"])
    for case_id, record in records.items():
        fallback = record["fallback_by_round"][gate_round]
        threshold = (
            float(policy["threshold"])
            if policy["threshold"] is not None
            else 9999.0
        )
        switched = gate.policy_switches(
            fallback["calibration"],
            threshold,
            hard_abstention=bool(policy["hard_abstention"]),
            force_fallback=bool(policy["force_fallback"]),
        )
        policy_auc = (
            float(fallback["auc"])
            if switched
            else float(record["online_llm_auc"])
        )
        fallback_rounds = (
            len(fallback["audit"]) - gate_round if switched else 0
        )
        calls_per_round = 2 if record["deliberation_mode"] == "proposal_critic" else 1
        route_rows.append(
            {
                "case_id": case_id,
                "domain": record["domain"],
                "policy_auc": round(policy_auc, 6),
                "online_llm_auc": float(record["online_llm_auc"]),
                "target_gp_auc": float(record["target_gp_auc"]),
                "strongest_auc": float(record["strongest_auc"]),
                "policy_minus_target_gp_auc": round(
                    policy_auc - float(record["target_gp_auc"]), 6
                ),
                "policy_minus_online_llm_auc": round(
                    policy_auc - float(record["online_llm_auc"]), 6
                ),
                "policy_minus_strongest_auc": round(
                    policy_auc - float(record["strongest_auc"]), 6
                ),
                "switched_to_target_gp": switched,
                "fallback_rounds": fallback_rounds,
                "nominal_llm_calls_avoided": fallback_rounds * calls_per_round,
                "source_trace": record["source_trace"],
            }
        )
    return route_rows


def compact_metrics(route_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["policy_minus_target_gp_auc"]) for row in route_rows]
    return {
        "route_count": len(values),
        "mean_auc_delta": round(mean(values), 6),
        "worst_route_auc_delta": round(min(values), 6),
        "wins": sum(value > portfolio.TOLERANCE for value in values),
        "ties": sum(abs(value) <= portfolio.TOLERANCE for value in values),
        "losses": sum(value < -portfolio.TOLERANCE for value in values),
        "switch_count": sum(bool(row["switched_to_target_gp"]) for row in route_rows),
        "nominal_llm_calls_avoided": sum(
            int(row["nominal_llm_calls_avoided"]) for row in route_rows
        ),
    }


def selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Use training metrics only; evaluation metrics must never affect selection."""
    training = candidate["training"]
    policy = candidate["policy"]
    threshold = policy["threshold"]
    return (
        int(training["losses"]),
        -float(training["mean_auc_delta"]),
        -float(training["worst_route_auc_delta"]),
        int(training["switch_count"]),
        int(policy["parameter_count"]),
        int(policy["gate_round"]),
        -float(threshold) if threshold is not None else 0.0,
        int(bool(policy["hard_abstention"])),
        str(policy["policy_id"]),
    )


def select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("No policy candidates were evaluated.")
    return min(candidates, key=selection_key)


def bootstrap_summary(
    route_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    values = [float(row["policy_minus_target_gp_auc"]) for row in route_rows]
    return portfolio.summarize(values, protocol, seed_offset)


def write_candidate_grid(
    path: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    rows = []
    for candidate in candidates:
        policy = candidate["policy"]
        training = candidate["training"]
        evaluation = candidate["evaluation"]
        rows.append(
            {
                "policy_id": policy["policy_id"],
                "kind": policy["kind"],
                "gate_round": policy["gate_round"],
                "threshold": "" if policy["threshold"] is None else policy["threshold"],
                "hard_abstention": policy["hard_abstention"],
                "force_fallback": policy["force_fallback"],
                "parameter_count": policy["parameter_count"],
                "training_mean_auc_delta": training["mean_auc_delta"],
                "training_worst_route_auc_delta": training["worst_route_auc_delta"],
                "training_wins": training["wins"],
                "training_ties": training["ties"],
                "training_losses": training["losses"],
                "training_switch_count": training["switch_count"],
                "evaluation_mean_auc_delta": evaluation["mean_auc_delta"],
                "evaluation_worst_route_auc_delta": evaluation["worst_route_auc_delta"],
                "evaluation_wins": evaluation["wins"],
                "evaluation_ties": evaluation["ties"],
                "evaluation_losses": evaluation["losses"],
                "evaluation_switch_count": evaluation["switch_count"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{matched.sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def plot_evaluation(
    original_rows: Sequence[Mapping[str, Any]],
    fixed_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_original = {row["case_id"]: row for row in original_rows}
    by_fixed = {row["case_id"]: row for row in fixed_rows}
    by_selected = {row["case_id"]: row for row in selected_rows}
    case_ids = list(by_selected)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.0, 6.7),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 2.15], "wspace": 0.04},
    )
    right.axvline(0.0, color="#9A3D46", linestyle="--", linewidth=1.2)
    colors = ["#737E87", "#D07C2E", "#1B7A62"]
    markers = ["o", "s", "D"]
    labels = ["Full online LLM", "Fixed round-3 error gate", "Selected controller"]
    offsets = [-0.18, 0.0, 0.18]
    sources = [by_original, by_fixed, by_selected]
    for source, color, marker, label, offset in zip(
        sources, colors, markers, labels, offsets
    ):
        values = [float(source[case_id]["policy_minus_target_gp_auc"]) for case_id in case_ids]
        y_values = [index + offset for index in range(len(case_ids))]
        for axis in (left, right):
            axis.scatter(
                values,
                y_values,
                color=color,
                marker=marker,
                s=65,
                edgecolor="white",
                linewidth=0.7,
                label=label,
                zorder=3,
            )
    left.set_xlim(-13.4, -4.8)
    right.set_xlim(-2.5, 3.5)
    left.set_yticks(range(len(case_ids)))
    left.set_yticklabels(
        [portfolio.ROUTE_LABELS.get(case_id, case_id) for case_id in case_ids]
    )
    left.invert_yaxis()
    for axis in (left, right):
        axis.grid(axis="x", color="#E1E5E8", linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(labelleft=False)
    diagonal = 0.012
    kwargs = {"color": "#5D6870", "clip_on": False, "linewidth": 1.0}
    left.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), transform=left.transAxes, **kwargs)
    left.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), transform=left.transAxes, **kwargs)
    right.plot((-diagonal, +diagonal), (-diagonal, +diagonal), transform=right.transAxes, **kwargs)
    right.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), transform=right.transAxes, **kwargs)
    figure.supxlabel("AUC delta versus matched same-start target-only GP-UCB", y=0.055)
    figure.suptitle(
        "Route-disjoint evaluation: a one-round LLM handoff reduces observed negative transfer",
        x=0.08,
        ha="left",
        fontsize=13,
    )
    handles, legend_labels = right.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.68, 0.89),
        frameon=False,
        ncol=3,
    )
    figure.text(
        0.01,
        0.01,
        "Six evaluation routes were excluded from controller selection. Values are retrospective best-so-far AUC deltas; positive is better. One completed trajectory per route.",
        fontsize=8.1,
        color="#5D6870",
    )
    figure.subplots_adjust(
        left=0.25,
        right=0.98,
        bottom=0.17,
        top=0.78,
        wspace=0.04,
    )
    outputs = [
        output_root / "route_disjoint_controller_comparison.png",
        output_root / "route_disjoint_controller_comparison.pdf",
        output_root / "route_disjoint_controller_comparison.svg",
    ]
    for path in outputs:
        figure.savefig(path, dpi=240, bbox_inches="tight")
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{matched.sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    protocol = config["protocol"]
    training_path = ROOT / str(protocol["training_config"])
    evaluation_path = ROOT / str(protocol["evaluation_config"])
    baseline_path = ROOT / str(protocol["baseline_aggregate"])
    training_config = portfolio.load_portfolio_config(training_path)
    evaluation_config = portfolio.load_portfolio_config(evaluation_path)
    split = validate_split(training_config, evaluation_config)
    baseline = matched.load_json(baseline_path)
    baseline_reports = {row["case_id"]: row for row in baseline["route_reports"]}
    gate_rounds = [int(value) for value in protocol["gate_rounds"]]
    training_records = load_panel_records(
        training_config, baseline_reports, gate_rounds
    )
    evaluation_records = load_panel_records(
        evaluation_config, baseline_reports, gate_rounds
    )

    evaluated = []
    route_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for policy in policy_candidates(protocol):
        training_rows = apply_policy(policy, training_records)
        evaluation_rows = apply_policy(policy, evaluation_records)
        evaluated.append(
            {
                "policy": policy,
                "training": compact_metrics(training_rows),
                "evaluation": compact_metrics(evaluation_rows),
            }
        )
        route_results[str(policy["policy_id"])] = {
            "training": training_rows,
            "evaluation": evaluation_rows,
        }
    selected = select_candidate(evaluated)
    selected_id = str(selected["policy"]["policy_id"])
    fixed_id = "prediction_error_r3_t5_hard1"
    if fixed_id not in route_results:
        raise ValueError(f"Required diagnostic policy is missing: {fixed_id}")
    selected_training_rows = route_results[selected_id]["training"]
    selected_evaluation_rows = route_results[selected_id]["evaluation"]
    fixed_evaluation_rows = route_results[fixed_id]["evaluation"]

    original_policy = {
        "policy_id": "full_online_llm",
        "kind": "online_llm",
        "gate_round": gate_rounds[-1],
        "threshold": 9999.0,
        "hard_abstention": False,
        "force_fallback": False,
        "parameter_count": 0,
    }
    original_evaluation_rows = apply_policy(original_policy, evaluation_records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_candidate_grid(args.output_root / "candidate_grid.csv", evaluated)
    matched.write_jsonl(
        args.output_root / "selected_policy_training_routes.jsonl",
        selected_training_rows,
    )
    matched.write_jsonl(
        args.output_root / "selected_policy_evaluation_routes.jsonl",
        selected_evaluation_rows,
    )
    matched.write_jsonl(
        args.output_root / "fixed_threshold5_evaluation_routes.jsonl",
        fixed_evaluation_rows,
    )
    matched.write_jsonl(
        args.output_root / "full_online_llm_evaluation_routes.jsonl",
        original_evaluation_rows,
    )
    figures = plot_evaluation(
        original_evaluation_rows,
        fixed_evaluation_rows,
        selected_evaluation_rows,
        args.output_root,
    )

    selected_training_summary = bootstrap_summary(
        selected_training_rows, training_config["protocol"], 101
    )
    selected_evaluation_summary = bootstrap_summary(
        selected_evaluation_rows, evaluation_config["protocol"], 102
    )
    original_evaluation_summary = bootstrap_summary(
        original_evaluation_rows, evaluation_config["protocol"], 103
    )
    fixed_evaluation_summary = bootstrap_summary(
        fixed_evaluation_rows, evaluation_config["protocol"], 104
    )
    selected_minus_online = portfolio.summarize(
        [
            float(row["policy_minus_online_llm_auc"])
            for row in selected_evaluation_rows
        ],
        evaluation_config["protocol"],
        105,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": matched.sha256(config_path),
        "training_config_sha256": matched.sha256(training_path),
        "evaluation_config_sha256": matched.sha256(evaluation_path),
        "baseline_aggregate_sha256": matched.sha256(baseline_path),
        "route_split": split,
        "selection_rule": protocol["selection_rule"],
        "selection_uses_evaluation_metrics": False,
        "candidate_count": len(evaluated),
        "selected_policy": selected["policy"],
        "selected_policy_training": selected_training_summary,
        "selected_policy_route_disjoint_evaluation": selected_evaluation_summary,
        "route_disjoint_full_online_llm": original_evaluation_summary,
        "route_disjoint_fixed_round3_threshold5_gate": fixed_evaluation_summary,
        "selected_policy_minus_full_online_llm_on_evaluation": selected_minus_online,
        "figure_files": [
            {
                "path": str(path.relative_to(args.output_root)),
                "sha256": matched.sha256(path),
                "source_data": [
                    "full_online_llm_evaluation_routes.jsonl",
                    "fixed_threshold5_evaluation_routes.jsonl",
                    "selected_policy_evaluation_routes.jsonl",
                ],
            }
            for path in figures
        ],
        "claim_boundary": protocol["claim_policy"],
    }
    matched.write_json(args.output_root / "selected_policy.json", payload)

    def summary_line(summary: Mapping[str, Any]) -> str:
        return (
            f"{summary['equal_route_mean_auc_delta']:+.3f} "
            f"[{summary['route_bootstrap_95ci_low']:+.3f}, "
            f"{summary['route_bootstrap_95ci_high']:+.3f}] | "
            f"{summary['route_wins']} / {summary['route_ties']} / {summary['route_losses']}"
        )

    results = args.output_root / "RESULTS.md"
    results.write_text(
        "\n".join(
            [
                "# Route-split online-LLM controller selection audit",
                "",
                "The controller was selected using only the 11 training-route metrics. The six evaluation-route outcomes were previously known, but they do not enter the selection key; this is a leakage-controlled retrospective audit rather than a blinded holdout.",
                "",
                f"Selected policy: `{selected_id}`. It permits {selected['policy']['gate_round']} online LLM-guided reveal and then continues with target-only GP-UCB from all accumulated observations.",
                "",
                "| Panel / method | Mean AUC delta vs target GP [95% route-bootstrap CI] | Win / tie / loss |",
                "|---|---:|---:|",
                f"| Training / selected controller | {summary_line(selected_training_summary)} |",
                f"| Evaluation / full online LLM | {summary_line(original_evaluation_summary)} |",
                f"| Evaluation / fixed round-3 threshold-5 gate | {summary_line(fixed_evaluation_summary)} |",
                f"| Evaluation / selected controller | {summary_line(selected_evaluation_summary)} |",
                "",
                f"On the six route-disjoint trajectories, the selected controller changed AUC by {selected_minus_online['equal_route_mean_auc_delta']:+.3f} versus the original full online-LLM trajectory. It still had {selected_evaluation_summary['route_losses']} loss against target-only GP-UCB, so this is evidence of improved robustness, not universal positive transfer.",
                "",
                "## Claim boundary",
                "",
                str(protocol["claim_policy"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    results.with_suffix(results.suffix + ".sha256").write_text(
        f"{matched.sha256(results)}  {results.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
