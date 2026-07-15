#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

FeatureRecord = surrogate.FeatureRecord


def parse_scales(raw: str) -> tuple[float, ...]:
    scales = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not scales:
        raise ValueError("At least one transfer weight scale is required.")
    if any(scale < 0 for scale in scales):
        raise ValueError("Transfer weight scales must be non-negative.")
    return scales


def scale_label(scale: float) -> str:
    text = f"{scale:g}".replace(".", "p")
    return text.replace("-", "m")


def scales_label(scales: tuple[float, ...]) -> str:
    return "_".join(scale_label(scale) for scale in scales)


def mixed_kernel_weighted(
    a: FeatureRecord,
    b: FeatureRecord,
    categorical_weights: tuple[float, ...],
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> float:
    a_num, a_cat = a
    b_num, b_cat = b
    numeric_sq = sum((x - y) ** 2 for x, y in zip(a_num, b_num))
    categorical_mismatch = sum(
        weight for x, y, weight in zip(a_cat, b_cat, categorical_weights) if x != y
    )
    return math.exp(
        -(numeric_sq / (2.0 * numeric_length_scale * numeric_length_scale))
        - (categorical_mismatch / categorical_length_scale)
    )


def transfer_categorical_weights(
    adapter: replay.DatasetAdapter,
    card: transfer.TransferCard,
    scale: float,
    normalize: bool,
    role_multipliers: dict[str, float] | None = None,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    role_multipliers = role_multipliers or {}
    confidence_by_target = {
        role.target_field: role.confidence
        for role in card.roles
        if role.transfer_weight > 0.0
    }
    raw_weights = [1.0]
    weight_rows = [
        {
            "categorical_position": 0,
            "field": adapter.group_column,
            "source_field": "",
            "role_confidence": 0.0,
            "raw_weight": 1.0,
        }
    ]
    role_by_target = {
        role.target_field: role
        for role in card.roles
        if role.transfer_weight > 0.0
    }
    for index, field_name in enumerate(adapter.decision_columns, start=1):
        role = role_by_target.get(field_name)
        confidence = confidence_by_target.get(field_name, 0.0)
        role_multiplier = role_multipliers.get(field_name, 1.0)
        weight = 1.0 + scale * confidence * role_multiplier
        raw_weights.append(weight)
        weight_rows.append(
            {
                "categorical_position": index,
                "field": field_name,
                "source_field": "" if role is None else role.source_field,
                "role_confidence": confidence,
                "role_multiplier": round(role_multiplier, 6),
                "raw_weight": round(weight, 6),
            }
        )

    weights = raw_weights
    normalization_factor = 1.0
    if normalize:
        normalization_factor = sum(raw_weights) / len(raw_weights)
        weights = [weight / normalization_factor for weight in raw_weights]
    for row, weight in zip(weight_rows, weights):
        row["normalized_weight"] = round(weight, 6)
    diagnostics = {
        "scale": scale,
        "normalize": normalize,
        "normalization_factor": round(normalization_factor, 6),
        "weights": weight_rows,
        "evidence_boundary": (
            "Weights are derived from source transfer-role confidence only. "
            "They change the GP categorical kernel; they do not read target "
            "hidden outcomes or directly select candidate ids."
        ),
    }
    return tuple(weights), diagnostics


def weighted_gp_scores(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, FeatureRecord],
    categorical_weights: tuple[float, ...],
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    include_posterior_maps: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    observed_features = [features_by_id[candidate.candidate_id] for candidate in observed]
    y_values = [candidate.objective_value / 100.0 for candidate in observed]
    y_mean = mean(y_values)
    y_scale = max(pstdev(y_values), 0.05)
    y_norm = [(value - y_mean) / y_scale for value in y_values]
    size = len(observed)
    kernel_matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = mixed_kernel_weighted(
                observed_features[i],
                observed_features[j],
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
    quadratic = sum(value * coefficient for value, coefficient in zip(y_norm, alpha))
    log_marginal_likelihood = (
        -0.5 * quadratic
        - sum(math.log(max(lower[index][index], 1e-12)) for index in range(size))
        - 0.5 * size * math.log(2.0 * math.pi)
    )

    scores: dict[str, float] = {}
    posterior_means: list[float] = []
    posterior_stds: list[float] = []
    posterior_mean_by_id: dict[str, float] = {}
    posterior_std_by_id: dict[str, float] = {}
    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue
        feature = features_by_id[candidate.candidate_id]
        k_vec = [
            mixed_kernel_weighted(
                feature,
                observed_feature,
                categorical_weights,
                numeric_length_scale,
                categorical_length_scale,
            )
            for observed_feature in observed_features
        ]
        mean_norm = sum(k * a for k, a in zip(k_vec, alpha))
        solved = surrogate.solve_cholesky(lower, k_vec)
        variance = max(1e-9, 1.0 - sum(k * v for k, v in zip(k_vec, solved)))
        posterior_mean = y_mean + y_scale * mean_norm
        posterior_std = y_scale * math.sqrt(variance)
        scores[candidate.candidate_id] = posterior_mean + gp_beta * posterior_std
        posterior_means.append(posterior_mean)
        posterior_stds.append(posterior_std)
        if include_posterior_maps:
            posterior_mean_by_id[candidate.candidate_id] = posterior_mean
            posterior_std_by_id[candidate.candidate_id] = posterior_std

    diagnostics = {
        "candidate_count": len(scores),
        "posterior_mean_min": round(min(posterior_means), 6) if posterior_means else 0.0,
        "posterior_mean_max": round(max(posterior_means), 6) if posterior_means else 0.0,
        "posterior_std_mean": round(mean(posterior_stds), 6) if posterior_stds else 0.0,
        "y_mean": round(y_mean, 6),
        "y_scale": round(y_scale, 6),
        "log_marginal_likelihood": round(log_marginal_likelihood, 6),
    }
    if include_posterior_maps:
        diagnostics["posterior_mean_by_id"] = posterior_mean_by_id
        diagnostics["posterior_std_by_id"] = posterior_std_by_id
    return scores, diagnostics


def rank_normalized(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    denominator = float(len(ordered) - 1)
    return {
        candidate_id: 1.0 - (rank / denominator)
        for rank, (candidate_id, _score) in enumerate(ordered)
    }


def ensemble_weighted_gp_scores(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, FeatureRecord],
    scale_weights: list[tuple[float, tuple[float, ...], dict[str, Any]]],
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    include_posterior_maps: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    rank_scores: dict[str, list[float]] = {}
    scale_diagnostics: list[dict[str, Any]] = []
    for scale, categorical_weights, weight_diagnostics in scale_weights:
        scores, gp_diagnostics = weighted_gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            categorical_weights,
            gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            include_posterior_maps,
        )
        normalized = rank_normalized(scores)
        for candidate_id, score in normalized.items():
            rank_scores.setdefault(candidate_id, []).append(score)
        scale_diagnostics.append(
            {
                "scale": scale,
                "weight_diagnostics": weight_diagnostics,
                "gp_diagnostics": gp_diagnostics,
            }
        )
    ensemble_scores = {
        candidate_id: mean(values)
        for candidate_id, values in rank_scores.items()
    }
    diagnostics = {
        "candidate_count": len(ensemble_scores),
        "ensemble_scales": [scale for scale, _weights, _diagnostics in scale_weights],
        "score_aggregation": "mean_normalized_rank",
        "scale_diagnostics": scale_diagnostics,
    }
    if include_posterior_maps:
        candidate_ids = tuple(ensemble_scores)
        diagnostics["posterior_mean_by_id"] = {
            candidate_id: mean(
                item["gp_diagnostics"]["posterior_mean_by_id"][candidate_id]
                for item in scale_diagnostics
            )
            for candidate_id in candidate_ids
        }
        diagnostics["posterior_std_by_id"] = {
            candidate_id: math.sqrt(
                mean(
                    item["gp_diagnostics"]["posterior_std_by_id"][candidate_id] ** 2
                    for item in scale_diagnostics
                )
            )
            for candidate_id in candidate_ids
        }
    return ensemble_scores, diagnostics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(values), 4),
                "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
            }
    return out


def run_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: str,
    categorical_weights: tuple[float, ...],
    weight_diagnostics: dict[str, Any],
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {candidate.candidate_id: surrogate.candidate_features(adapter, candidate) for candidate in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []

    for round_index in range(task.reveal_budget):
        scores, gp_diagnostics = weighted_gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            categorical_weights,
            gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected.candidate_id,
                "selected_score": round(scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "hypothesis_snapshot": {
                    "base_acquisition": "mixed_kernel_gp_ucb",
                    "skill_optimization": "transfer_weighted_categorical_kernel",
                    "weight_diagnostics": weight_diagnostics,
                    "gp_diagnostics": gp_diagnostics,
                },
            }
        )

    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }
    return metrics, audit


def run_ensemble_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: str,
    scale_weights: list[tuple[float, tuple[float, ...], dict[str, Any]]],
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {candidate.candidate_id: surrogate.candidate_features(adapter, candidate) for candidate in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []

    for round_index in range(task.reveal_budget):
        scores, ensemble_diagnostics = ensemble_weighted_gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            scale_weights,
            gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected.candidate_id,
                "selected_score": round(scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "hypothesis_snapshot": {
                    "base_acquisition": "mixed_kernel_gp_ucb",
                    "skill_optimization": "transfer_weighted_categorical_kernel_ensemble",
                    "ensemble_diagnostics": ensemble_diagnostics,
                },
            }
        )

    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }
    return metrics, audit


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    cards: dict[int, transfer.TransferCard],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for seed, card in sorted(cards.items()):
        (OUTPUT_RUNS / f"{output_id}_transfer_card_seed{seed}.json").write_text(
            json.dumps(asdict(card), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for (mode, seed), audit_rows in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_transfer_weighted_kernel(
    source_dataset: str,
    target_dataset: str,
    seeds: int,
    rounds: int,
    initial: int,
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    scales: tuple[float, ...],
    normalize: bool,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    ensemble_scales: tuple[float, ...],
    output_tag: str,
) -> dict[str, Any]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    role_map = transfer.role_map_for(source_dataset, target_dataset)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    cards: dict[int, transfer.TransferCard] = {}
    weight_examples: dict[str, Any] = {}

    for seed in range(seeds):
        source_observed = transfer.source_observations(source_adapter, seed, source_observation_count)
        card = transfer.compile_transfer_card(
            source_adapter,
            target_adapter,
            source_observed,
            role_map,
            discount,
            min_source_support,
        )
        cards[seed] = card
        for scale in scales:
            mode = "gp_ucb" if scale == 0 else f"transfer_weighted_gp_ucb_scale_{scale_label(scale)}"
            categorical_weights, weight_diagnostics = transfer_categorical_weights(
                target_adapter,
                card,
                scale,
                normalize,
            )
            if seed == 0:
                weight_examples[mode] = weight_diagnostics
            metrics, audit = run_policy(
                target_adapter,
                task,
                seed,
                mode,
                categorical_weights,
                weight_diagnostics,
                gp_beta,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit
        if ensemble_scales:
            scale_weights: list[tuple[float, tuple[float, ...], dict[str, Any]]] = []
            for scale in ensemble_scales:
                categorical_weights, weight_diagnostics = transfer_categorical_weights(
                    target_adapter,
                    card,
                    scale,
                    normalize,
                )
                scale_weights.append((scale, categorical_weights, weight_diagnostics))
            mode = f"transfer_weighted_gp_ucb_scale_ensemble_{scales_label(ensemble_scales)}"
            if seed == 0:
                weight_examples[mode] = {
                    "ensemble_scales": list(ensemble_scales),
                    "score_aggregation": "mean_normalized_rank",
                    "scale_weight_examples": [item[2] for item in scale_weights],
                }
            metrics, audit = run_ensemble_policy(
                target_adapter,
                task,
                seed,
                mode,
                scale_weights,
                gp_beta,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit

    output_id = f"transfer_weighted_kernel_{source_dataset}_to_{target_dataset}"
    if output_tag:
        output_id = f"{output_id}_{output_tag}"
    summary = {
        "experiment": "care_transfer_weighted_gp_kernel",
        "output_id": output_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "role_map": role_map,
        "base_acquisition": "mixed_kernel_gp_ucb",
        "skill_optimization": (
            "CARE transfer-card role confidence is used to reweight categorical "
            "fields in the GP kernel. This optimizes the acquisition geometry "
            "rather than adding a post-hoc candidate bonus."
        ),
        "source_observation_count": source_observation_count,
        "discount": discount,
        "min_source_support": min_source_support,
        "normalize_weights": normalize,
        "gp_beta": gp_beta,
        "numeric_length_scale": numeric_length_scale,
        "categorical_length_scale": categorical_length_scale,
        "gp_noise": gp_noise,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "scales": list(scales),
        "ensemble_scales": list(ensemble_scales),
        "weight_examples_seed0": weight_examples,
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, summary, audits, cards)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE transfer-weighted GP-kernel skill optimization.")
    parser.add_argument("--source-dataset", default="real_suzuki_miyaura", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=96)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--scales", default="0,0.5,1,1.5,2,4")
    parser.add_argument(
        "--ensemble-scales",
        default="",
        help="Optional comma-separated transfer scales to aggregate by normalized-rank ensemble.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Do not normalize categorical weights to mean 1.")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()
    summary = run_transfer_weighted_kernel(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        seeds=args.seeds,
        rounds=args.rounds,
        initial=args.initial,
        source_observation_count=args.source_observations,
        discount=args.discount,
        min_source_support=args.min_source_support,
        scales=parse_scales(args.scales),
        normalize=not args.no_normalize,
        gp_beta=args.gp_beta,
        numeric_length_scale=args.numeric_length_scale,
        categorical_length_scale=args.categorical_length_scale,
        gp_noise=args.gp_noise,
        ensemble_scales=parse_scales(args.ensemble_scales) if args.ensemble_scales else (),
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
