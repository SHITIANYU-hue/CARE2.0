#!/usr/bin/env python3
"""Audit a prediction-error gate that hands an online LLM trajectory back to GP."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import build_online_llm_classical_baseline_audit as matched
import build_online_llm_predeclared_baseline_portfolio as portfolio
import run_classical_transfer_baselines as classical
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "care.online_llm_calibration_gate_audit/v2"
ROUTE_LABELS = portfolio.ROUTE_LABELS


def trace_rounds(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = [
        json.loads(line)
        for line in (summary_path.parent / "llm_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    initial = next(event for event in events if event["event"] == "initial_design_revealed")
    responses = {
        int(event["round_index"]): event
        for event in events
        if event["event"] == "llm_round_response"
    }
    gate_decisions = {
        int(event["round_index"]): event
        for event in events
        if event["event"] == "calibration_gate_target_gp_decision"
    }
    reveals = {
        int(event["round_index"]): event
        for event in events
        if event["event"] == "target_reveal"
    }
    rounds = []
    for round_index in sorted(reveals):
        response = responses.get(round_index) or gate_decisions.get(round_index)
        if response is None:
            raise ValueError(f"Missing decision event for reveal round {round_index}")
        reveal = reveals[round_index]
        decision = response.get("normalized_decision", {})
        expected = decision.get("expected_outcome")
        prediction_is_scored = (
            expected is not None
            and decision.get("decision_verdict") != "revise_to_gp"
        )
        rounds.append(
            {
                "round_index": round_index,
                "candidate_id": str(reveal["selected_candidate"]),
                "revealed_value": float(reveal["revealed_value"]),
                "expected_outcome": float(expected) if prediction_is_scored else None,
                "hard_abstention": (
                    decision.get("hypothesis_status") == "falsified"
                    or decision.get("continue_source_transfer") is False
                ),
                "decision_verdict": decision.get("decision_verdict"),
            }
        )
    return initial, rounds


def calibration_features(rounds: Sequence[Mapping[str, Any]], gate_round: int) -> dict[str, Any]:
    prefix = list(rounds[:gate_round])
    errors = [
        abs(float(row["expected_outcome"]) - float(row["revealed_value"]))
        for row in prefix
        if row["expected_outcome"] is not None
    ]
    return {
        "gate_round": gate_round,
        "scored_prediction_count": len(errors),
        "mean_absolute_prediction_error": mean(errors) if errors else None,
        "hard_abstention": any(bool(row["hard_abstention"]) for row in prefix),
    }


def gate_switches(features: Mapping[str, Any], threshold: float) -> bool:
    error = features["mean_absolute_prediction_error"]
    return bool(features["hard_abstention"]) or (
        error is not None and float(error) > threshold
    )


def continue_with_target_gp(
    summary: Mapping[str, Any],
    initial: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    gate_round: int,
    kernel: Mapping[str, Any],
) -> tuple[float, float, list[dict[str, Any]]]:
    target = replay.DATASET_BUILDERS[str(summary["target_task"])]()
    pool = target.candidates
    candidate_index = {candidate.candidate_id: index for index, candidate in enumerate(pool)}
    initial_ids = [str(value) for value in initial["candidate_ids"]]
    prefix_ids = [str(row["candidate_id"]) for row in rounds[:gate_round]]
    observed_indices = [candidate_index[value] for value in [*initial_ids, *prefix_ids]]
    observed_set = set(observed_indices)
    features = classical.feature_arrays(target, pool)
    initial_best = max(pool[candidate_index[value]].objective_value for value in initial_ids)
    best_so_far = initial_best
    best_trace: list[float] = []
    audit = []
    for row in rounds[:gate_round]:
        best_so_far = max(best_so_far, float(row["revealed_value"]))
        best_trace.append(best_so_far)
        audit.append({**row, "selected_by": "online_llm", "best_so_far": best_so_far})

    remaining_rounds = len(rounds) - gate_round
    for fallback_round in range(remaining_rounds):
        observed_candidates = [pool[index] for index in observed_indices]
        observed_y, center, scale = classical.normalized_outcomes(observed_candidates)
        posterior_mean, posterior_variance, diagnostics = classical.target_gp_posterior(
            features,
            observed_indices,
            observed_y,
            float(kernel["numeric_length_scale"]),
            float(kernel["categorical_length_scale"]),
            float(kernel["gp_noise"]),
        )
        scores = posterior_mean + float(kernel["gp_beta"]) * posterior_variance**0.5
        selected_index = classical.top_unobserved(scores, observed_set)
        selected = pool[selected_index]
        observed_indices.append(selected_index)
        observed_set.add(selected_index)
        best_so_far = max(best_so_far, selected.objective_value)
        best_trace.append(best_so_far)
        audit.append(
            {
                "round_index": gate_round + fallback_round,
                "candidate_id": selected.candidate_id,
                "revealed_value": selected.objective_value,
                "selected_by": "calibration_gate_target_gp",
                "best_so_far": best_so_far,
                "diagnostics": {
                    "target_center": round(center, 6),
                    "target_scale": round(scale, 6),
                    "target_gp_jitter": float(diagnostics["jitter"]),
                },
            }
        )
    return mean(best_trace), best_so_far, audit


def select_threshold(
    held_out: str,
    thresholds: Sequence[float],
    records: Mapping[str, Mapping[str, Any]],
) -> float:
    training = [case_id for case_id in records if case_id != held_out]

    def score(threshold: float) -> tuple[float, float, float]:
        deltas = []
        for case_id in training:
            record = records[case_id]
            auc = (
                float(record["fallback_auc"])
                if gate_switches(record["calibration"], threshold)
                else float(record["online_llm_auc"])
            )
            deltas.append(auc - float(record["target_gp_auc"]))
        return (
            mean(deltas),
            sum(value >= -portfolio.TOLERANCE for value in deltas) / len(deltas),
            threshold,
        )

    return max(thresholds, key=score)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{matched.sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def plot(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    threshold_selection: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    figure, axis = plt.subplots(figsize=(10.8, 6.7))
    axis.axvline(0.0, color="#A23B45", linestyle="--", linewidth=1.2)
    for index, row in enumerate(rows):
        original = float(row["online_llm_minus_target_gp_auc"])
        gated = float(row["gated_minus_target_gp_auc"])
        axis.plot([original, gated], [index, index], color="#C9CFD3", linewidth=1.5)
        axis.scatter(original, index, s=58, color="#7A8790", edgecolor="white", linewidth=0.7, zorder=3)
        axis.scatter(gated, index, s=70, marker="D", color="#1E7D62", edgecolor="white", linewidth=0.8, zorder=3)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([ROUTE_LABELS.get(str(row["case_id"]), str(row["case_id"])) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Δ best-so-far AUC versus same-start target-only GP-UCB")
    axis.set_title("Prediction-error gate reduces negative transfer in the frozen 11-route portfolio", loc="left", fontsize=13)
    axis.grid(axis="x", color="#E1E5E8", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.scatter([], [], s=58, color="#7A8790", label="Original online LLM")
    gate_label = (
        "Fixed global calibration gate"
        if threshold_selection == "fixed"
        else "Leave-one-route-out calibration gate"
    )
    axis.scatter([], [], s=70, marker="D", color="#1E7D62", label=gate_label)
    axis.legend(loc="lower right", frameon=False, ncol=2)
    figure.text(
        0.01,
        0.01,
        (
            "Gate evaluated after three LLM-guided reveals. One fixed global threshold is applied to every route; "
            "a triggered gate continues with target-only GP from the accumulated observations. Retrospective replay."
            if threshold_selection == "fixed"
            else "Gate evaluated after three LLM-guided reveals. Threshold for each route is selected on the other ten routes only; a triggered gate continues with target-only GP from the accumulated observations. Retrospective leave-one-route-out audit."
        ),
        fontsize=8.1,
        color="#5D6870",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    outputs = [
        output_root / "calibration_gate_route_effects.png",
        output_root / "calibration_gate_route_effects.pdf",
        output_root / "calibration_gate_route_effects.svg",
    ]
    for path in outputs:
        figure.savefig(path, dpi=220, bbox_inches="tight")
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{matched.sha256(path)}  {path.name}\n",
            encoding="utf-8",
        )
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--portfolio-aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-round", type=int, default=3)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30, 40, 60, 9999],
    )
    parser.add_argument(
        "--threshold-selection",
        choices=("leave_one_route_out", "fixed"),
        default="leave_one_route_out",
    )
    args = parser.parse_args()
    if args.threshold_selection == "fixed" and len(args.thresholds) != 1:
        parser.error("Fixed threshold selection requires exactly one threshold.")
    config = portfolio.load_portfolio_config(args.portfolio_config)
    baseline_audit = matched.load_json(args.portfolio_aggregate)
    baseline_reports = {row["case_id"]: row for row in baseline_audit["route_reports"]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for case in config["cases"]:
        case_id = str(case["case_id"])
        summary_path = ROOT / str(case["summary"])
        summary = matched.load_json(summary_path)
        initial, rounds = trace_rounds(summary_path)
        if len(rounds) != int(summary["requested_reveal_rounds"]):
            raise ValueError(f"{case_id} trace does not contain the declared reveal budget")
        kernel = matched.load_json(ROOT / str(case["config"]))["protocol"]["kernel"]
        fallback_auc, fallback_final, fallback_audit = continue_with_target_gp(
            summary,
            initial,
            rounds,
            args.gate_round,
            kernel,
        )
        report = baseline_reports[case_id]
        records[case_id] = {
            "case_id": case_id,
            "domain": case["domain"],
            "calibration": calibration_features(rounds, args.gate_round),
            "online_llm_auc": float(report["online_llm"]["best_so_far_auc"]),
            "online_llm_final": float(report["online_llm"]["final_best"]),
            "target_gp_auc": float(report["predeclared_comparators"]["no_transfer"]["baseline_best_so_far_auc"]),
            "target_gp_final": float(report["predeclared_comparators"]["no_transfer"]["baseline_final_best"]),
            "strongest_auc": float(report["strongest_realized_baseline_posthoc"]["baseline_best_so_far_auc"]),
            "fallback_auc": fallback_auc,
            "fallback_final": fallback_final,
            "fallback_audit": fallback_audit,
            "source_trace": str(
                (summary_path.parent / "llm_trace.jsonl").relative_to(ROOT)
            ),
            "deliberation_mode": str(
                summary.get("menu_policy", {}).get(
                    "deliberation_mode", "single"
                )
            ),
        }

    rows = []
    selected_thresholds = []
    for case_id, record in records.items():
        threshold = (
            float(args.thresholds[0])
            if args.threshold_selection == "fixed"
            else select_threshold(case_id, args.thresholds, records)
        )
        selected_thresholds.append(threshold)
        switched = gate_switches(record["calibration"], threshold)
        fallback_rounds = (
            len(record["fallback_audit"]) - args.gate_round
            if switched
            else 0
        )
        logical_calls_per_round = (
            2 if record["deliberation_mode"] == "proposal_critic" else 1
        )
        gated_auc = float(record["fallback_auc"] if switched else record["online_llm_auc"])
        gated_final = float(record["fallback_final"] if switched else record["online_llm_final"])
        row = {
            "case_id": case_id,
            "domain": record["domain"],
            "selected_threshold": threshold,
            "mean_absolute_prediction_error": record["calibration"]["mean_absolute_prediction_error"],
            "hard_abstention": record["calibration"]["hard_abstention"],
            "gate_switched_to_target_gp": switched,
            "fallback_rounds": fallback_rounds,
            "nominal_llm_calls_avoided": (
                fallback_rounds * logical_calls_per_round
            ),
            "online_llm_auc": record["online_llm_auc"],
            "gated_auc": round(gated_auc, 6),
            "target_gp_auc": record["target_gp_auc"],
            "strongest_realized_baseline_auc": record["strongest_auc"],
            "online_llm_minus_target_gp_auc": round(float(record["online_llm_auc"]) - float(record["target_gp_auc"]), 6),
            "gated_minus_target_gp_auc": round(gated_auc - float(record["target_gp_auc"]), 6),
            "gated_minus_online_llm_auc": round(gated_auc - float(record["online_llm_auc"]), 6),
            "gated_minus_strongest_realized_baseline_auc": round(gated_auc - float(record["strongest_auc"]), 6),
            "gated_final": round(gated_final, 6),
        }
        rows.append(row)
        matched.write_jsonl(
            args.output_root / case_id / "gated_trace.jsonl",
            (
                record["fallback_audit"]
                if switched
                else [
                    {
                        "selected_by": "online_llm_full_trajectory",
                        "source_trace": record["source_trace"],
                    }
                ]
            ),
        )

    original_values = [float(row["online_llm_minus_target_gp_auc"]) for row in rows]
    gated_values = [float(row["gated_minus_target_gp_auc"]) for row in rows]
    gate_gain_values = [float(row["gated_minus_online_llm_auc"]) for row in rows]
    strongest_values = [float(row["gated_minus_strongest_realized_baseline_auc"]) for row in rows]
    write_csv(args.output_root / "route_results.csv", rows)
    figures = plot(rows, args.output_root, args.threshold_selection)
    if args.threshold_selection == "fixed":
        threshold_phrase = "the fixed global threshold"
        gate_method_label = "Fixed global calibration gate"
        evaluation_phrase = "retrospective route replays"
        selection_description = (
            "One global threshold was supplied before this replay and applied to "
            "every route without route-specific selection."
        )
        claim_boundary = (
            "This is a retrospective replay of one previously observed trajectory per "
            "route under a single fixed threshold. It validates executable semantics and "
            "motivates the frozen repeated protocol, but it is not prospective or external validation."
        )
    else:
        threshold_phrase = "a leave-one-route-out threshold"
        gate_method_label = "Leave-one-route-out calibration gate"
        evaluation_phrase = "held-out route evaluations"
        selection_description = (
            "Leave one route out; maximize mean AUC delta versus target-only GP on the "
            "other routes, then maximize non-loss rate and prefer the less aggressive "
            "larger threshold on ties."
        )
        claim_boundary = (
            "This is a retrospective leave-one-route-out audit on one trajectory per route. "
            "The held-out route never contributes to its threshold selection, and the fallback "
            "uses only observations available by the gate round, but prospective repeated "
            "confirmation is still required."
        )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "gate_round": args.gate_round,
            "threshold_grid": args.thresholds,
            "threshold_selection": args.threshold_selection,
            "selection": selection_description,
            "prediction_error": "Mean absolute error of scored LLM expected outcomes over the first three online reveals. revise_to_gp responses are excluded because their expected_outcome field is not an asserted prediction.",
            "fallback": "Continue target-only GP-UCB from all initial and first-three LLM observations; do not restart or splice a separate GP trajectory.",
        },
        "route_count": len(rows),
        "switch_count": sum(bool(row["gate_switched_to_target_gp"]) for row in rows),
        "total_fallback_rounds": sum(int(row["fallback_rounds"]) for row in rows),
        "total_nominal_llm_calls_avoided": sum(
            int(row["nominal_llm_calls_avoided"]) for row in rows
        ),
        "mean_nominal_llm_calls_avoided_per_route": round(
            mean(float(row["nominal_llm_calls_avoided"]) for row in rows), 6
        ),
        "selected_threshold_counts": {str(key): value for key, value in Counter(selected_thresholds).items()},
        "original_online_llm_vs_target_gp": portfolio.summarize(original_values, config["protocol"], 20),
        "calibration_gate_vs_target_gp": portfolio.summarize(gated_values, config["protocol"], 21),
        "calibration_gate_minus_online_llm": portfolio.summarize(gate_gain_values, config["protocol"], 22),
        "calibration_gate_vs_strongest_realized_baseline_posthoc": portfolio.summarize(strongest_values, config["protocol"], 23),
        "route_results": rows,
        "figure_files": [{"path": str(path), "sha256": matched.sha256(path), "source_data": str(args.output_root / "route_results.csv")} for path in figures],
        "claim_boundary": claim_boundary,
    }
    matched.write_json(args.output_root / "aggregate.json", aggregate)
    original_summary = aggregate["original_online_llm_vs_target_gp"]
    gated_summary = aggregate["calibration_gate_vs_target_gp"]
    strongest_summary = aggregate["calibration_gate_vs_strongest_realized_baseline_posthoc"]
    results_path = args.output_root / "RESULTS.md"
    results_path.write_text(
        "\n".join(
            [
                "# Prediction-error calibration gate audit",
                "",
                f"The gate is evaluated after three LLM-guided target reveals. If the first-three mean absolute prediction error exceeds {threshold_phrase}, or the LLM explicitly falsifies or abandons the transfer hypothesis, target-only GP-UCB continues from the accumulated observations.",
                "",
                "| Method | Mean AUC delta vs target GP | Route-bootstrap 95% CI | Win / tie / loss |",
                "|---|---:|---:|---:|",
                (
                    f"| Original online LLM | {original_summary['equal_route_mean_auc_delta']:+.3f} | "
                    f"[{original_summary['route_bootstrap_95ci_low']:+.3f}, {original_summary['route_bootstrap_95ci_high']:+.3f}] | "
                    f"{original_summary['route_wins']} / {original_summary['route_ties']} / {original_summary['route_losses']} |"
                ),
                (
                    f"| {gate_method_label} | {gated_summary['equal_route_mean_auc_delta']:+.3f} | "
                    f"[{gated_summary['route_bootstrap_95ci_low']:+.3f}, {gated_summary['route_bootstrap_95ci_high']:+.3f}] | "
                    f"{gated_summary['route_wins']} / {gated_summary['route_ties']} / {gated_summary['route_losses']} |"
                ),
                "",
                f"The gate switched on {aggregate['switch_count']} of {aggregate['route_count']} {evaluation_phrase}. It reduced the negative-transfer count against target GP from {original_summary['route_losses']} to {gated_summary['route_losses']} and changed the equal-route mean by {aggregate['calibration_gate_minus_online_llm']['equal_route_mean_auc_delta']:+.3f} AUC.",
                "",
                f"In this counterfactual replay, switching removed {aggregate['total_fallback_rounds']} later LLM-guided rounds, corresponding to {aggregate['total_nominal_llm_calls_avoided']} nominal proposer/critic calls. These are calls that the executable gate would avoid; they are not claimed as already realized API savings.",
                "",
                f"Against the strongest realized baseline selected post hoc, the gated controller remained {strongest_summary['equal_route_mean_auc_delta']:+.3f} AUC on average with {strongest_summary['route_wins']} wins, {strongest_summary['route_ties']} ties and {strongest_summary['route_losses']} losses. The gate is therefore a negative-transfer control, not evidence of universal baseline superiority.",
                "",
                "## Claim boundary",
                "",
                str(aggregate["claim_boundary"]),
                "",
            ]
        ),
        encoding="utf-8",
    )
    results_path.with_suffix(results_path.suffix + ".sha256").write_text(
        f"{matched.sha256(results_path)}  {results_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
