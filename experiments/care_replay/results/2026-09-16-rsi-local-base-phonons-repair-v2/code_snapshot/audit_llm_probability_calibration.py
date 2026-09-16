#!/usr/bin/env python3
"""Audit LLM improvement probabilities on the complete archived route portfolio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import audit_hidden_target_noninterference as portfolio


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
SCHEMA_VERSION = "care.llm_probability_calibration_audit/v1"
EPSILON = 1e-9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)


def event_index(events: Sequence[Mapping[str, Any]], event_name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != event_name:
            continue
        round_index = int(event["round_index"])
        if round_index in indexed:
            raise ValueError(f"Duplicate {event_name} for round {round_index}")
        indexed[round_index] = dict(event)
    return indexed


def candidate_menu_record(prompt: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    menu = prompt["candidate_menu"]
    columns = [str(value) for value in menu["columns"]]
    for raw_row in menu["rows"]:
        row = dict(zip(columns, raw_row))
        if str(row["candidate_id"]) == candidate_id:
            return row
    raise ValueError(f"Selected candidate {candidate_id!r} is absent from the candidate menu")


def gp_probability_for_candidate(
    menu_record: Mapping[str, Any],
    current_best: float,
) -> tuple[float, str]:
    recorded = menu_record.get("gp_probability_improvement")
    if recorded is not None:
        return min(1.0, max(0.0, float(recorded))), "recorded"
    mean = float(menu_record["gp_posterior_mean"])
    standard_deviation = float(menu_record["gp_posterior_std"])
    if standard_deviation <= EPSILON:
        return (1.0 if mean > current_best else 0.0), "reconstructed"
    z_score = (mean - current_best) / standard_deviation
    probability = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    return min(1.0, max(0.0, probability)), "reconstructed"


def calibration_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    if len(probabilities) != len(outcomes):
        raise ValueError("Probabilities and outcomes must have equal length")
    if len(edges) < 2 or list(edges) != sorted(edges):
        raise ValueError("Calibration edges must be sorted and contain at least two values")
    rows: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            position
            for position, probability in enumerate(probabilities)
            if probability >= low and (
                probability < high or (index == len(edges) - 2 and probability <= high)
            )
        ]
        count = len(selected)
        rows.append({
            "bin_index": index,
            "bin_low": float(low),
            "bin_high": float(high),
            "count": count,
            "mean_probability": (
                float(np.mean([probabilities[position] for position in selected]))
                if selected
                else None
            ),
            "observed_improvement_rate": (
                float(np.mean([outcomes[position] for position in selected]))
                if selected
                else None
            ),
        })
    return rows


def roc_auc(probabilities: Sequence[float], outcomes: Sequence[int]) -> float | None:
    positives = [probability for probability, outcome in zip(probabilities, outcomes) if outcome]
    negatives = [probability for probability, outcome in zip(probabilities, outcomes) if not outcome]
    if not positives or not negatives:
        return None
    score = 0.0
    for positive in positives:
        for negative in negatives:
            score += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return score / (len(positives) * len(negatives))


def binary_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    edges: Sequence[float],
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("A non-empty, aligned probability/outcome sample is required")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError("Probabilities must lie in [0, 1]")
    clipped = np.clip(probability_array, 1e-6, 1.0 - 1e-6)
    bins = calibration_bins(probabilities, outcomes, edges)
    total = len(probabilities)
    expected_calibration_error = sum(
        (row["count"] / total)
        * abs(float(row["mean_probability"]) - float(row["observed_improvement_rate"]))
        for row in bins
        if row["count"]
    )
    return {
        "count": total,
        "improvements": int(np.sum(outcome_array)),
        "observed_improvement_rate": float(np.mean(outcome_array)),
        "mean_probability": float(np.mean(probability_array)),
        "mean_probability_bias": float(np.mean(probability_array - outcome_array)),
        "brier_score": float(np.mean((probability_array - outcome_array) ** 2)),
        "log_loss": float(np.mean(-(
            outcome_array * np.log(clipped)
            + (1.0 - outcome_array) * np.log(1.0 - clipped)
        ))),
        "expected_calibration_error": float(expected_calibration_error),
        "roc_auc": roc_auc(probabilities, outcomes),
        "bins": bins,
    }


def route_bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("At least one route value is required")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def load_decisions(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    portfolio_path = portfolio.resolve_path(str(config["portfolio_config"]))
    cases = portfolio.portfolio_cases(portfolio_path)
    decisions: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for case in cases:
        summary_path = portfolio.resolve_path(str(case["summary"]))
        summary = load_json(summary_path)
        trace_path = portfolio.resolve_trace_path(summary_path, str(summary["trace_file"]))
        events = portfolio.load_trace(trace_path)
        requests = event_index(events, "llm_round_request")
        responses = event_index(events, "llm_round_response")
        reveals = event_index(events, "target_reveal")
        round_indices = sorted(set(requests) & set(responses) & set(reveals))
        if not round_indices or not (
            len(round_indices) == len(requests) == len(responses) == len(reveals)
        ):
            raise ValueError(
                f"Incomplete request/response/reveal alignment for {case['case_id']}: "
                f"{len(requests)}/{len(responses)}/{len(reveals)}"
            )
        for round_index in round_indices:
            request = requests[round_index]
            response = responses[round_index]
            reveal = reveals[round_index]
            decision = response["normalized_decision"]
            selected_candidate = str(decision["selected_candidate_id"])
            if selected_candidate != str(reveal["selected_candidate"]):
                raise ValueError(
                    f"Executed candidate mismatch for {case['case_id']} round {round_index}"
                )
            menu_record = candidate_menu_record(request["prompt"], selected_candidate)
            current_best = float(request["prompt"]["current_best"])
            revealed_value = float(reveal["revealed_value"])
            improvement = int(revealed_value > current_best + EPSILON)
            llm_probability = min(
                1.0,
                max(0.0, float(decision["probability_of_improving_current_best"])),
            )
            gp_probability, gp_probability_source = gp_probability_for_candidate(
                menu_record,
                current_best,
            )
            expected_outcome = float(decision["expected_outcome"])
            decisions.append({
                "case_id": str(case["case_id"]),
                "domain": str(case["domain"]),
                "evidence_tier": str(case["evidence_tier"]),
                "round_index": round_index,
                "model": str(response.get("model", "unknown")),
                "selected_candidate": selected_candidate,
                "selected_by": str(response.get("selected_by", "unknown")),
                "gp_rank_at_selection": int(reveal["gp_rank_at_selection"]),
                "llm_probability": llm_probability,
                "gp_probability": gp_probability,
                "gp_probability_source": gp_probability_source,
                "current_best_before_reveal": current_best,
                "expected_outcome": expected_outcome,
                "revealed_value": revealed_value,
                "realized_improvement": improvement,
                "realized_gain": max(0.0, revealed_value - current_best),
                "absolute_expected_outcome_error": abs(expected_outcome - revealed_value),
                "llm_brier": (llm_probability - improvement) ** 2,
                "gp_brier": (gp_probability - improvement) ** 2,
                "llm_minus_gp_brier": (
                    (llm_probability - improvement) ** 2
                    - (gp_probability - improvement) ** 2
                ),
            })
        inputs.append({
            "case_id": str(case["case_id"]),
            "summary": str(summary_path.relative_to(REPOSITORY_ROOT)),
            "summary_sha256": file_sha256(summary_path),
            "trace": str(trace_path.relative_to(REPOSITORY_ROOT)),
            "trace_sha256": file_sha256(trace_path),
        })
    return decisions, inputs


def summarize_group(rows: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> dict[str, Any]:
    outcomes = [int(row["realized_improvement"]) for row in rows]
    return {
        "decision_count": len(rows),
        "route_count": len({str(row["case_id"]) for row in rows}),
        "llm": binary_metrics(
            [float(row["llm_probability"]) for row in rows], outcomes, edges
        ),
        "gp_same_candidate": binary_metrics(
            [float(row["gp_probability"]) for row in rows], outcomes, edges
        ),
        "mean_absolute_expected_outcome_error": float(np.mean([
            float(row["absolute_expected_outcome_error"]) for row in rows
        ])),
        "override_count": sum(int(row["gp_rank_at_selection"]) > 1 for row in rows),
        "gp_probability_sources": dict(Counter(
            str(row["gp_probability_source"]) for row in rows
        )),
    }


def route_summaries(
    decisions: Sequence[Mapping[str, Any]],
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[str(row["case_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for case_id, rows in grouped.items():
        summary = summarize_group(rows, edges)
        summaries.append({
            "case_id": case_id,
            "domain": str(rows[0]["domain"]),
            "evidence_tier": str(rows[0]["evidence_tier"]),
            "model": str(rows[0]["model"]),
            "decision_count": len(rows),
            "improvement_count": summary["llm"]["improvements"],
            "observed_improvement_rate": summary["llm"]["observed_improvement_rate"],
            "llm_mean_probability": summary["llm"]["mean_probability"],
            "gp_mean_probability": summary["gp_same_candidate"]["mean_probability"],
            "llm_brier_score": summary["llm"]["brier_score"],
            "gp_brier_score": summary["gp_same_candidate"]["brier_score"],
            "llm_minus_gp_brier": (
                summary["llm"]["brier_score"]
                - summary["gp_same_candidate"]["brier_score"]
            ),
            "llm_expected_calibration_error": summary["llm"]["expected_calibration_error"],
            "gp_expected_calibration_error": summary["gp_same_candidate"]["expected_calibration_error"],
            "llm_roc_auc": summary["llm"]["roc_auc"],
            "gp_roc_auc": summary["gp_same_candidate"]["roc_auc"],
            "llm_minus_gp_roc_auc": (
                None
                if summary["llm"]["roc_auc"] is None
                or summary["gp_same_candidate"]["roc_auc"] is None
                else summary["llm"]["roc_auc"]
                - summary["gp_same_candidate"]["roc_auc"]
            ),
            "mean_absolute_expected_outcome_error": summary["mean_absolute_expected_outcome_error"],
            "override_count": summary["override_count"],
        })
    return summaries


def grouped_summaries(
    decisions: Sequence[Mapping[str, Any]],
    key: str,
    edges: Sequence[float],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[str(row[key])].append(row)
    return {name: summarize_group(rows, edges) for name, rows in sorted(grouped.items())}


def write_results(path: Path, report: Mapping[str, Any]) -> None:
    pooled = report["pooled"]
    comparison = report["route_level_comparison"]
    llm = pooled["llm"]
    gp = pooled["gp_same_candidate"]
    low = comparison["route_bootstrap_95ci_low"]
    high = comparison["route_bootstrap_95ci_high"]
    if high < 0:
        conclusion = "The archived LLM probabilities are better calibrated than the GP probabilities for the same executed candidates."
    elif low > 0:
        conclusion = "The archived LLM probabilities are less calibrated than the GP probabilities for the same executed candidates."
    else:
        conclusion = "The route-level interval crosses zero, so the audit does not establish a calibration advantage for either probability source."
    lines = [
        "# Archived LLM probability-calibration audit",
        "",
        (
            f"The complete retrospective portfolio contains {pooled['decision_count']} "
            f"decisions across {pooled['route_count']} routes. "
            f"{llm['improvements']} decisions improved the pre-round best "
            f"({llm['observed_improvement_rate']:.3f})."
        ),
        "",
        "| Probability source | Mean probability | Observed rate | Bias | Brier | Log loss | ECE | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Final LLM decision | {llm['mean_probability']:.3f} | "
            f"{llm['observed_improvement_rate']:.3f} | {llm['mean_probability_bias']:+.3f} | "
            f"{llm['brier_score']:.3f} | {llm['log_loss']:.3f} | "
            f"{llm['expected_calibration_error']:.3f} | {llm['roc_auc']:.3f} |"
        ),
        (
            f"| Production GP, same candidate | {gp['mean_probability']:.3f} | "
            f"{gp['observed_improvement_rate']:.3f} | {gp['mean_probability_bias']:+.3f} | "
            f"{gp['brier_score']:.3f} | {gp['log_loss']:.3f} | "
            f"{gp['expected_calibration_error']:.3f} | {gp['roc_auc']:.3f} |"
        ),
        "",
        (
            "Equal-route mean LLM-minus-GP Brier difference: "
            f"{comparison['equal_route_mean_llm_minus_gp_brier']:+.4f} "
            f"(route-bootstrap 95% CI [{low:+.4f}, {high:+.4f}]). "
            "Negative values favor the LLM."
        ),
        (
            "Equal-route mean LLM-minus-GP ROC-AUC difference: "
            f"{comparison['equal_route_mean_llm_minus_gp_roc_auc']:+.4f} "
            f"(route-bootstrap 95% CI "
            f"[{comparison['route_roc_auc_bootstrap_95ci_low']:+.4f}, "
            f"{comparison['route_roc_auc_bootstrap_95ci_high']:+.4f}]; "
            f"{comparison['route_roc_auc_count']} routes with both outcomes)."
        ),
        "",
        conclusion,
        "",
        (
            "GP probability provenance: "
            f"{pooled['gp_probability_sources'].get('recorded', 0)} recorded in the "
            "archived menu and "
            f"{pooled['gp_probability_sources'].get('reconstructed', 0)} reconstructed "
            "from the archived GP posterior mean, standard deviation, and current best."
        ),
        "",
        (
            "This comparison scores the LLM and GP probability assigned to the same "
            "executed candidate, so it tests probability estimation rather than action "
            "selection. The audit is retrospective, pools multiple model/controller "
            "versions, and does not validate a deployment gate."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_json(config_path)
    protocol = config["protocol"]
    edges = [float(value) for value in protocol["calibration_bin_edges"]]
    decisions, inputs = load_decisions(config)
    routes = route_summaries(decisions, edges)
    route_differences = [float(row["llm_minus_gp_brier"]) for row in routes]
    low, high = route_bootstrap_interval(
        route_differences,
        samples=int(protocol["route_bootstrap_samples"]),
        seed=int(protocol["route_bootstrap_seed"]),
    )
    route_auc_differences = [
        float(row["llm_minus_gp_roc_auc"])
        for row in routes
        if row["llm_minus_gp_roc_auc"] is not None
    ]
    auc_low, auc_high = route_bootstrap_interval(
        route_auc_differences,
        samples=int(protocol["route_bootstrap_samples"]),
        seed=int(protocol["route_bootstrap_seed"]) + 1,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "config_sha256": file_sha256(config_path),
        "inputs": inputs,
        "pooled": summarize_group(decisions, edges),
        "route_level_comparison": {
            "route_count": len(routes),
            "equal_route_mean_llm_minus_gp_brier": float(np.mean(route_differences)),
            "route_bootstrap_95ci_low": low,
            "route_bootstrap_95ci_high": high,
            "llm_better_routes": sum(value < -EPSILON for value in route_differences),
            "equal_routes": sum(abs(value) <= EPSILON for value in route_differences),
            "gp_better_routes": sum(value > EPSILON for value in route_differences),
            "route_roc_auc_count": len(route_auc_differences),
            "equal_route_mean_llm_minus_gp_roc_auc": float(np.mean(route_auc_differences)),
            "route_roc_auc_bootstrap_95ci_low": auc_low,
            "route_roc_auc_bootstrap_95ci_high": auc_high,
        },
        "by_domain": grouped_summaries(decisions, "domain", edges),
        "by_model": grouped_summaries(decisions, "model", edges),
        "routes": routes,
        "decisions": decisions,
        "claim_boundary": protocol["claim_boundary"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "probability_calibration_report.json"
    decisions_path = args.output_dir / "decision_calibration.csv"
    routes_path = args.output_dir / "route_calibration.csv"
    results_path = args.output_dir / "RESULTS.md"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(decisions_path, decisions)
    write_csv(routes_path, routes)
    write_results(results_path, report)
    for path in (report_path, decisions_path, routes_path, results_path):
        write_sha256(path)
    print(json.dumps({
        "decisions": len(decisions),
        "routes": len(routes),
        "llm_brier": report["pooled"]["llm"]["brier_score"],
        "gp_brier": report["pooled"]["gp_same_candidate"]["brier_score"],
        "route_mean_difference": report["route_level_comparison"]["equal_route_mean_llm_minus_gp_brier"],
        "route_bootstrap_95ci": [low, high],
    }, indent=2))


if __name__ == "__main__":
    main()
