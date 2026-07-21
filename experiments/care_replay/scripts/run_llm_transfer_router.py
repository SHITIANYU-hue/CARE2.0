#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_frozen_llm_kernel_patch as frozen
import run_llm_kernel_online_bma as bma
import run_llm_kernel_skill_evolution as evolution
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

MOLECULAR_DATASETS = (
    "real_moleculenet_esol",
    "real_moleculenet_freesolv",
    "real_moleculenet_lipophilicity",
)
EXACT_IDENTITY_FIELDS: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
for _source_dataset in MOLECULAR_DATASETS:
    for _target_dataset in MOLECULAR_DATASETS:
        if _source_dataset != _target_dataset:
            EXACT_IDENTITY_FIELDS[(_source_dataset, _target_dataset)] = (("smiles", "smiles"),)
for _source_dataset in transfer.MATERIAL_DATASETS:
    for _target_dataset in transfer.MATERIAL_DATASETS:
        if _source_dataset != _target_dataset:
            EXACT_IDENTITY_FIELDS[(_source_dataset, _target_dataset)] = (("composition", "composition"),)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss <= 1e-12 or y_ss <= 1e-12:
        return 0.0
    return numerator / math.sqrt(x_ss * y_ss)


def linear_fit(xs: list[float], ys: list[float], ridge: float = 0.05) -> tuple[float, float]:
    x_mean = mean(xs)
    y_mean = mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = covariance / (variance + ridge)
    return y_mean - slope * x_mean, slope


def leave_one_out_gain(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 5:
        return 0.0
    model_errors: list[float] = []
    baseline_errors: list[float] = []
    for heldout in range(len(xs)):
        train_x = [value for index, value in enumerate(xs) if index != heldout]
        train_y = [value for index, value in enumerate(ys) if index != heldout]
        intercept, slope = linear_fit(train_x, train_y)
        prediction = intercept + slope * xs[heldout]
        baseline = mean(train_y)
        model_errors.append((ys[heldout] - prediction) ** 2)
        baseline_errors.append((ys[heldout] - baseline) ** 2)
    baseline_mse = mean(baseline_errors)
    if baseline_mse <= 1e-12:
        return 0.0
    return max(-2.0, min(1.0, 1.0 - mean(model_errors) / baseline_mse))


def weighted_kernel_loo_mae(
    observed: list[replay.Candidate],
    features_by_id: dict[str, surrogate.FeatureRecord],
    weight_ensemble: list[tuple[float, ...]],
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> float:
    if len(observed) < 5 or not weight_ensemble:
        return float("inf")
    errors: list[float] = []
    for heldout_index, heldout in enumerate(observed):
        training = [
            candidate
            for index, candidate in enumerate(observed)
            if index != heldout_index
        ]
        y_values = [candidate.objective_value / 100.0 for candidate in training]
        y_mean = mean(y_values)
        y_scale = max(pstdev(y_values), 0.05)
        y_norm = [(value - y_mean) / y_scale for value in y_values]
        predictions: list[float] = []
        for categorical_weights in weight_ensemble:
            size = len(training)
            kernel_matrix = [[0.0] * size for _ in range(size)]
            for i in range(size):
                for j in range(i + 1):
                    value = weighted.mixed_kernel_weighted(
                        features_by_id[training[i].candidate_id],
                        features_by_id[training[j].candidate_id],
                        categorical_weights,
                        numeric_length_scale,
                        categorical_length_scale,
                    )
                    kernel_matrix[i][j] = value
                    kernel_matrix[j][i] = value
            for i in range(size):
                kernel_matrix[i][i] += gp_noise
            lower = surrogate.cholesky_spd(kernel_matrix)
            alpha = surrogate.solve_cholesky(lower, y_norm)
            heldout_feature = features_by_id[heldout.candidate_id]
            k_vec = [
                weighted.mixed_kernel_weighted(
                    heldout_feature,
                    features_by_id[candidate.candidate_id],
                    categorical_weights,
                    numeric_length_scale,
                    categorical_length_scale,
                )
                for candidate in training
            ]
            mean_norm = sum(value * coefficient for value, coefficient in zip(k_vec, alpha))
            predictions.append(y_mean + y_scale * mean_norm)
        prediction = mean(predictions)
        errors.append(abs(heldout.objective_value / 100.0 - prediction))
    return mean(errors)


def relative_loo_gain(baseline_mae: float, challenger_mae: float) -> float:
    if not math.isfinite(baseline_mae) or baseline_mae <= 1e-12:
        return 0.0
    return max(-2.0, min(1.0, 1.0 - challenger_mae / baseline_mae))


def aligned_source_prior(
    source_observed: list[replay.Candidate],
    target_adapter: replay.DatasetAdapter,
    card: transfer.TransferCard,
    patch: evolution.KernelSkillPatch,
) -> tuple[dict[str, float], dict[str, Any]]:
    identity_fields = EXACT_IDENTITY_FIELDS.get(
        (card.source_dataset, card.target_dataset),
        (),
    )
    active_roles = [
        role
        for role in card.roles
        if role.transfer_weight > 0.0
        and patch.role_multipliers.get(role.target_field, 1.0) > 0.0
    ]
    if patch.source_prior_strength <= 0.0 or (not active_roles and not identity_fields):
        return {}, {"active": False, "reason": "source_prior_disabled_or_no_roles"}

    source_values = [candidate.objective_value / 100.0 for candidate in source_observed]
    source_mean = mean(source_values)
    source_scale = max(pstdev(source_values), 0.05)
    source_z = {
        candidate.candidate_id: (candidate.objective_value / 100.0 - source_mean) / source_scale
        for candidate in source_observed
    }
    role_weights = {
        role.target_field: max(
            0.02,
            role.transfer_weight
            * patch.role_multipliers.get(role.target_field, 1.0),
        )
        for role in active_roles
    }
    total_role_weight = sum(role_weights.values())
    neighbor_source_observed = source_observed[: min(256, len(source_observed))]
    neighbor_count = min(patch.source_neighbor_count, len(neighbor_source_observed))
    exact_values: dict[tuple[str, str], list[float]] = {}
    for source_field, target_field in identity_fields:
        for source_candidate in source_observed:
            value = str(source_candidate.metadata.get(source_field, "")).strip()
            if value:
                exact_values.setdefault((target_field, value), []).append(
                    source_z[source_candidate.candidate_id]
                )
    prior_by_id: dict[str, float] = {}
    nearest_distance_values: list[float] = []
    exact_match_count = 0
    exact_match_candidate_ids: list[str] = []

    for target_candidate in target_adapter.candidates:
        identity_predictions = [
            mean(exact_values[(target_field, target_value)])
            for _source_field, target_field in identity_fields
            if (target_value := str(target_candidate.metadata.get(target_field, "")).strip())
            and (target_field, target_value) in exact_values
        ]
        if identity_predictions:
            prior_by_id[target_candidate.candidate_id] = mean(identity_predictions)
            exact_match_count += 1
            exact_match_candidate_ids.append(target_candidate.candidate_id)
            continue
        if not active_roles:
            prior_by_id[target_candidate.candidate_id] = 0.0
            continue
        neighbors: list[tuple[float, float]] = []
        for source_candidate in neighbor_source_observed:
            mismatch = 0.0
            for role in active_roles:
                source_value = str(source_candidate.metadata.get(role.source_field, ""))
                target_value = str(target_candidate.metadata.get(role.target_field, ""))
                if not source_value or not target_value:
                    mismatch += 0.5 * role_weights[role.target_field]
                elif source_value != target_value:
                    mismatch += role_weights[role.target_field]
            distance = mismatch / max(total_role_weight, 1e-9)
            neighbors.append((distance, source_z[source_candidate.candidate_id]))
        neighbors.sort(key=lambda item: item[0])
        nearest = neighbors[:neighbor_count]
        similarity_weights = [
            math.exp(-distance / patch.source_similarity_temperature)
            for distance, _value in nearest
        ]
        weight_sum = sum(similarity_weights)
        prior_by_id[target_candidate.candidate_id] = sum(
            weight * value
            for weight, (_distance, value) in zip(similarity_weights, nearest)
        ) / max(weight_sum, 1e-12)
        nearest_distance_values.append(nearest[0][0])

    return prior_by_id, {
        "active": True,
        "role_count": len(active_roles),
        "roles": [
            {
                "source_field": role.source_field,
                "target_field": role.target_field,
                "weight": round(role_weights[role.target_field], 6),
            }
            for role in active_roles
        ],
        "source_observation_count": len(source_observed),
        "neighbor_source_observation_count": len(neighbor_source_observed),
        "neighbor_count": neighbor_count,
        "temperature": patch.source_similarity_temperature,
        "nearest_distance_mean": (
            round(mean(nearest_distance_values), 6)
            if nearest_distance_values
            else None
        ),
        "exact_identity_fields": [list(pair) for pair in identity_fields],
        "exact_match_count": exact_match_count,
        "exact_match_fraction": round(
            exact_match_count / max(1, len(target_adapter.candidates)),
            6,
        ),
        "_exact_match_candidate_ids": exact_match_candidate_ids,
    }


def exact_identity_cold_start_adjustments(
    prior_by_id: dict[str, float],
    exact_match_candidate_ids: set[str],
    observed: list[replay.Candidate],
    patch: evolution.KernelSkillPatch,
) -> tuple[dict[str, float], dict[str, Any]]:
    if (
        patch.calibration_mode != "positive_only"
        or not exact_match_candidate_ids
        or len(observed) >= 10
    ):
        return {}, {"active": False, "reason": "cold_start_not_authorized_or_complete"}
    cap = min(0.08, 0.04 * patch.source_prior_strength * patch.confidence)
    if cap <= 0.0:
        return {}, {"active": False, "reason": "zero_cold_start_cap"}
    observed_ids = {candidate.candidate_id for candidate in observed}
    adjustments = {
        candidate_id: cap * max(-1.5, min(1.5, prior_by_id[candidate_id])) / 1.5
        for candidate_id in exact_match_candidate_ids
        if candidate_id not in observed_ids and candidate_id in prior_by_id
    }
    return adjustments, {
        "active": bool(adjustments),
        "mode": "llm_authorized_exact_identity_cold_start",
        "adjustment_cap": round(cap, 6),
        "eligible_candidate_count": len(adjustments),
        "observed_count": len(observed),
        "evidence_boundary": (
            "The frozen LLM patch authorizes a positive source direction for exact public "
            "identity matches only. The cold-start cap expires after 10 target observations."
        ),
    }


def calibrated_prior_adjustments(
    prior_by_id: dict[str, float],
    observed: list[replay.Candidate],
    patch: evolution.KernelSkillPatch,
) -> tuple[dict[str, float], dict[str, Any]]:
    if patch.calibration_mode == "off" or not prior_by_id:
        return {}, {"active": False, "reason": "calibration_off_or_prior_missing"}
    xs = [prior_by_id[candidate.candidate_id] for candidate in observed]
    ys = [candidate.objective_value / 100.0 for candidate in observed]
    correlation = pearson(xs, ys)
    cv_gain = leave_one_out_gain(xs, ys)
    intercept, slope = linear_fit(xs, ys)
    if patch.calibration_mode == "positive_only" and slope <= 0.0:
        return {}, {
            "active": False,
            "reason": "negative_slope_blocked",
            "correlation": round(correlation, 6),
            "cv_gain": round(cv_gain, 6),
        }
    if cv_gain < patch.min_cv_gain:
        return {}, {
            "active": False,
            "reason": "leave_one_out_gain_below_threshold",
            "correlation": round(correlation, 6),
            "cv_gain": round(cv_gain, 6),
            "min_cv_gain": patch.min_cv_gain,
        }

    target_mean = mean(ys)
    target_scale = max(pstdev(ys), 0.05)
    evidence_gate = min(1.0, len(observed) / 10.0)
    gain_gate = min(1.0, max(0.0, (cv_gain - patch.min_cv_gain) / max(0.15, 1.0 - patch.min_cv_gain)))
    reliability = evidence_gate * math.sqrt(gain_gate)
    adjustment_cap = min(0.35, 0.12 + 0.10 * math.log1p(len(observed)))
    adjustments: dict[str, float] = {}
    for candidate_id, source_prior in prior_by_id.items():
        predicted = intercept + slope * source_prior
        centered = max(-2.5 * target_scale, min(2.5 * target_scale, predicted - target_mean))
        adjustments[candidate_id] = max(
            -adjustment_cap,
            min(adjustment_cap, patch.source_prior_strength * reliability * centered),
        )
    return adjustments, {
        "active": True,
        "correlation": round(correlation, 6),
        "cv_gain": round(cv_gain, 6),
        "intercept": round(intercept, 6),
        "slope": round(slope, 6),
        "reliability": round(reliability, 6),
        "adjustment_cap": round(adjustment_cap, 6),
        "max_abs_adjustment": round(max(abs(value) for value in adjustments.values()), 6),
    }


def target_anchor_scores(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, surrogate.FeatureRecord],
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    gp_ucb, ucb_diagnostics = surrogate.gp_scores(
        adapter,
        observed_ids,
        observed,
        features_by_id,
        "mixed_kernel_gp_ucb",
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    gp_ei, ei_diagnostics = surrogate.gp_scores(
        adapter,
        observed_ids,
        observed,
        features_by_id,
        "mixed_kernel_gp_ei",
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    ucb_rank = weighted.rank_normalized(gp_ucb)
    ei_rank = weighted.rank_normalized(gp_ei)
    portfolio = {
        candidate_id: 0.5 * ucb_rank[candidate_id] + 0.5 * ei_rank[candidate_id]
        for candidate_id in ucb_rank
    }
    return {
        "gp_ucb": gp_ucb,
        "gp_ei": gp_ei,
        "target_acquisition_portfolio": portfolio,
    }, {"gp_ucb": ucb_diagnostics, "gp_ei": ei_diagnostics}


def run_target_portfolio(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
        for candidate in pool
    }
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        anchors, diagnostics = target_anchor_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        scores = anchors["target_acquisition_portfolio"]
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": "target_acquisition_portfolio",
                "selected_candidate": selected_id,
                "selected_score": round(scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "diagnostics": diagnostics,
            }
        )
    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": adapter.dataset_id,
        "mode": "target_acquisition_portfolio",
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit


def softmax_weights(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    peak = max(values.values())
    unnormalized = {key: math.exp(value - peak) for key, value in values.items()}
    total = sum(unnormalized.values())
    return {key: value / total for key, value in unnormalized.items()}


def router_gate_decision(
    anchor_candidate: str,
    router_candidate: str,
    observed_count: int,
    max_quality: float,
    anchor_loss: float,
    risk_budget: float,
    transfer_mass: float,
    min_observations: int = 10,
    min_quality: float = 0.15,
) -> tuple[bool, str]:
    if router_candidate == anchor_candidate:
        return True, "router_matches_target_anchor"
    if observed_count < min_observations:
        return False, "target_warmup_incomplete"
    if max_quality < min_quality:
        return False, "online_quality_below_threshold"
    if anchor_loss > risk_budget:
        return False, "anchor_acquisition_loss_above_budget"
    if transfer_mass > 0.45:
        return False, "transfer_mass_above_cap"
    return True, "online_evidence_authorized_transfer"


def run_router_policy(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    card: transfer.TransferCard,
    source_observed: list[replay.Candidate],
    patches: tuple[evolution.KernelSkillPatch, ...],
    normalize: bool,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = target_adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(target_adapter, candidate)
        for candidate in pool
    }
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    source_priors: dict[str, dict[str, float]] = {}
    source_prior_diagnostics: dict[str, Any] = {}
    for patch in patches:
        mode = evolution.patch_mode(patch)
        source_priors[mode], source_prior_diagnostics[mode] = aligned_source_prior(
            source_observed,
            target_adapter,
            card,
            patch,
        )

    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        anchors, anchor_diagnostics = target_anchor_scores(
            target_adapter,
            observed_ids,
            observed,
            features_by_id,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        anchor_scores = anchors["target_acquisition_portfolio"]
        zero_weights, _ = weighted.transfer_categorical_weights(target_adapter, card, 0.0, normalize)
        baseline_loo_mae = weighted_kernel_loo_mae(
            observed,
            features_by_id,
            [zero_weights],
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        _gp_scores, gp_kernel_diagnostics = weighted.weighted_gp_scores(
            target_adapter,
            observed_ids,
            observed,
            features_by_id,
            zero_weights,
            gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        gp_evidence = float(gp_kernel_diagnostics["log_marginal_likelihood"])
        expert_scores: dict[str, dict[str, float]] = {}
        route_values: dict[str, float] = {}
        route_diagnostics: dict[str, Any] = {}
        for patch in patches:
            mode = evolution.patch_mode(patch)
            current_patch_beta = weighted.scheduled_gp_beta(
                patch.gp_beta,
                patch.gp_beta_end,
                round_index,
                task.reveal_budget,
            )
            base_scores, evidence, kernel_diagnostics = bma.score_scale_ensemble(
                target_adapter,
                observed_ids,
                observed,
                features_by_id,
                card,
                patch.scales,
                patch.role_multipliers,
                current_patch_beta,
                normalize,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            adjustments, calibration = calibrated_prior_adjustments(
                source_priors[mode], observed, patch
            )
            patch_weight_ensemble = [
                weighted.transfer_categorical_weights(
                    target_adapter,
                    card,
                    scale,
                    normalize,
                    patch.role_multipliers,
                )[0]
                for scale in patch.scales
            ]
            patch_loo_mae = weighted_kernel_loo_mae(
                observed,
                features_by_id,
                patch_weight_ensemble,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            kernel_loo_gain = relative_loo_gain(baseline_loo_mae, patch_loo_mae)
            adjusted_scores = {
                candidate_id: score + adjustments.get(candidate_id, 0.0)
                for candidate_id, score in base_scores.items()
            }
            evidence_delta = evidence - gp_evidence
            cv_gain = float(calibration.get("cv_gain", -2.0))
            prior_active = bool(calibration.get("active")) and cv_gain >= 0.30
            kernel_active = kernel_loo_gain >= 0.05 and evidence_delta >= 0.0
            if prior_active or kernel_active:
                quality = max(
                    0.0,
                    cv_gain if prior_active else 0.0,
                    kernel_loo_gain if kernel_active else 0.0,
                )
                route_values[mode] = math.log(max(patch.confidence, 0.05)) + 3.0 * quality
                expert_scores[mode] = weighted.rank_normalized(adjusted_scores)
            route_diagnostics[mode] = {
                "patch": asdict(patch),
                "source_prior": source_prior_diagnostics[mode],
                "calibration": calibration,
                "kernel_log_marginal_likelihood": round(evidence, 6),
                "kernel_evidence_delta_vs_gp": round(evidence_delta, 6),
                "target_loo": {
                    "baseline_mae": round(baseline_loo_mae, 6),
                    "patch_mae": round(patch_loo_mae, 6),
                    "relative_gain": round(kernel_loo_gain, 6),
                },
                "gp_beta_schedule": {
                    "start": patch.gp_beta,
                    "end": patch.gp_beta_end,
                    "current": round(current_patch_beta, 6),
                },
                "active": prior_active or kernel_active,
            }

        expert_weights = softmax_weights(route_values)
        max_quality = max(
            (
                max(
                    0.0,
                    float(route_diagnostics[mode]["calibration"].get("cv_gain", 0.0)),
                    float(route_diagnostics[mode]["target_loo"]["relative_gain"]),
                )
                for mode in expert_weights
            ),
            default=0.0,
        )
        # Weak evidence should converge to the target-only anchor, not receive a
        # fixed minimum transfer weight.
        transfer_mass = min(0.45, 0.55 * max_quality) if expert_weights else 0.0
        combined_scores = {
            candidate_id: (1.0 - transfer_mass) * anchor_scores[candidate_id]
            + transfer_mass
            * sum(
                expert_weights[mode] * expert_scores[mode][candidate_id]
                for mode in expert_weights
            )
            for candidate_id in anchor_scores
        }
        anchor_selected_id = replay.top_candidate(anchor_scores)
        router_selected_id = replay.top_candidate(combined_scores)
        anchor_loss = max(
            0.0,
            anchor_scores[anchor_selected_id] - anchor_scores[router_selected_id],
        )
        router_risk_budget = max(0.025, 0.080 * math.exp(-0.22 * round_index))
        router_authorized, router_reason = router_gate_decision(
            anchor_selected_id,
            router_selected_id,
            len(observed),
            max_quality,
            anchor_loss,
            router_risk_budget,
            transfer_mass,
        )
        selected_id = router_selected_id if router_authorized else anchor_selected_id
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": target_adapter.dataset_id,
                "source_dataset_id": source_adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": "llm_transfer_router",
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected_id,
                "selected_score": round(combined_scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "hypothesis_snapshot": {
                    "target_anchor": "equal_rank_gp_ucb_gp_ei",
                    "transfer_mass": round(transfer_mass, 6),
                    "router_gate": {
                        "anchor_candidate": anchor_selected_id,
                        "router_candidate": router_selected_id,
                        "selected_candidate": selected_id,
                        "authorized": router_authorized,
                        "reason": router_reason,
                        "anchor_acquisition_loss": round(anchor_loss, 6),
                        "risk_budget": round(router_risk_budget, 6),
                        "max_quality": round(max_quality, 6),
                        "min_observations": 10,
                        "min_quality": 0.15,
                    },
                    "expert_weights": expert_weights,
                    "route_diagnostics": route_diagnostics,
                    "anchor_diagnostics": anchor_diagnostics,
                    "evidence_boundary": (
                        "The LLM patches are frozen before replay. Routing and source-prior calibration "
                        "use source observations plus target outcomes revealed before this selection only."
                    ),
                },
            }
        )

    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": target_adapter.dataset_id,
        "mode": "llm_transfer_router",
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit


def run_seed(
    seed: int,
    source_dataset: str,
    target_dataset: str,
    patches: tuple[evolution.KernelSkillPatch, ...],
    fixed_scales: tuple[float, ...],
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    initial: int,
    rounds: int,
    normalize: bool,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    source_observed = transfer.source_observations(source_adapter, seed, source_observation_count)
    card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        source_observed,
        transfer.role_map_for(source_dataset, target_dataset),
        discount,
        min_source_support,
    )
    rows, audits = evolution.run_seed_evaluation(
        seed,
        source_dataset,
        target_dataset,
        source_observation_count,
        discount,
        min_source_support,
        (),
        fixed_scales,
        normalize,
        initial,
        rounds,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    ei_metrics, ei_audit = surrogate.run_policy(
        target_adapter,
        task,
        seed,
        "mixed_kernel_gp_ei",
        gp_beta,
        gp_xi,
        5,
        0.35,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows.append(ei_metrics)
    audits[(ei_metrics["mode"], seed)] = ei_audit
    portfolio_metrics, portfolio_audit = run_target_portfolio(
        target_adapter,
        task,
        seed,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows.append(portfolio_metrics)
    audits[(portfolio_metrics["mode"], seed)] = portfolio_audit
    router_metrics, router_audit = run_router_policy(
        source_adapter,
        target_adapter,
        task,
        seed,
        card,
        source_observed,
        patches,
        normalize,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows.append(router_metrics)
    audits[(router_metrics["mode"], seed)] = router_audit
    return rows, audits


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for (mode, seed), audit_rows in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as file:
            for row in audit_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen LLM transfer skills with target-only prequential routing."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seed-start", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=96)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--fixed-ensemble-scales", default="0.5,1,1.5,2,3,4")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--gp-xi", type=float, default=0.01)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    target_adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    patches = evolution.normalize_patches(
        {"patches": record.get("normalized_patches", [])},
        target_adapter.decision_columns,
        max_patches=100,
    )
    if not patches:
        raise RuntimeError("The LLM record contains no usable frozen patches.")
    fixed_scales = weighted.parse_scales(args.fixed_ensemble_scales)
    worker_args = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "patches": patches,
        "fixed_scales": fixed_scales,
        "source_observation_count": args.source_observations,
        "discount": args.discount,
        "min_source_support": args.min_source_support,
        "initial": args.initial,
        "rounds": args.rounds,
        "normalize": not args.no_normalize,
        "gp_beta": args.gp_beta,
        "gp_xi": args.gp_xi,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
    }
    seed_values = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.workers == 1:
        seed_results = [run_seed(seed=seed, **worker_args) for seed in seed_values]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_seed, seed=seed, **worker_args) for seed in seed_values]
            seed_results = [future.result() for future in futures]
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed_rows, seed_audits in seed_results:
        rows.extend(seed_rows)
        audits.update(seed_audits)
    rows.sort(key=lambda row: (int(row["seed"]), str(row["mode"])))
    fixed_mode = f"fixed_scale_ensemble_{weighted.scales_label(fixed_scales)}"
    aggregate = evolution.summarize(rows, set(seed_values))
    target_modes = ("gp_ucb", "mixed_kernel_gp_ei", "target_acquisition_portfolio")
    strongest_target_mode = max(
        target_modes,
        key=lambda mode: aggregate[mode]["final_best_mean"] + aggregate[mode]["best_so_far_auc_mean"],
    )
    summary = {
        "experiment": "care_llm_transfer_skill_router",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "initial_observations": args.initial,
        "rounds": args.rounds,
        "llm_model": record.get("model"),
        "llm_call_count_for_skill_generation": 1,
        "patches": [asdict(patch) for patch in patches],
        "strongest_target_mode_by_aggregate": strongest_target_mode,
        "evidence_boundary": (
            "Frozen LLM skills see source evidence and public target schema. During replay, the router "
            "uses only target outcomes revealed before each selection. No hidden target row is used for routing."
        ),
        "aggregate": aggregate,
        "paired_vs_gp_ucb": frozen.paired_comparison(rows, "llm_transfer_router", "gp_ucb"),
        "paired_vs_gp_ei": frozen.paired_comparison(rows, "llm_transfer_router", "mixed_kernel_gp_ei"),
        "paired_vs_target_portfolio": frozen.paired_comparison(
            rows, "llm_transfer_router", "target_acquisition_portfolio"
        ),
        "paired_vs_fixed_transfer": frozen.paired_comparison(
            rows, "llm_transfer_router", fixed_mode
        ),
        "paired_vs_strongest_target": frozen.paired_comparison(
            rows, "llm_transfer_router", strongest_target_mode
        ),
    }
    output_id = f"llm_transfer_router_{args.source_dataset}_to_{args.target_dataset}"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    write_outputs(output_id, rows, audits, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
