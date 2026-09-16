#!/usr/bin/env python3
"""Run matched-initial classical transfer-BO baselines on compatible tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import numpy as np

import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
MODES = ("target_gp_ucb", "rgpe", "multitask_gp_icm")


@dataclass(frozen=True)
class FeatureArrays:
    numeric: np.ndarray
    categorical: np.ndarray


@dataclass(frozen=True)
class SourcePosterior:
    mean: np.ndarray
    variance: np.ndarray
    projection: np.ndarray
    inducing_count: int
    source_observation_count: int


def feature_arrays(
    adapter: replay.DatasetAdapter,
    candidates: Iterable[replay.Candidate],
) -> FeatureArrays:
    numeric_rows: list[tuple[float, ...]] = []
    categorical_rows: list[tuple[str, ...]] = []
    for candidate in candidates:
        numeric_rows.append(
            candidate.numeric_features
            or (float(candidate.x1), float(candidate.x2), float(candidate.x3))
        )
        categorical_rows.append(
            (
                str(candidate.group),
                *(str(candidate.metadata.get(field, "")) for field in adapter.decision_columns),
            )
        )
    return FeatureArrays(
        numeric=np.asarray(numeric_rows, dtype=np.float64),
        categorical=np.asarray(categorical_rows, dtype=str),
    )


def validate_compatible_spaces(
    source: replay.DatasetAdapter,
    target: replay.DatasetAdapter,
) -> None:
    if source.decision_columns != target.decision_columns:
        raise ValueError(
            "Classical transfer baseline requires identical ordered decision columns: "
            f"{source.dataset_id} has {source.decision_columns}, while "
            f"{target.dataset_id} has {target.decision_columns}."
        )
    source_features = feature_arrays(source, source.candidates[:1])
    target_features = feature_arrays(target, target.candidates[:1])
    if source_features.numeric.shape[1] != target_features.numeric.shape[1]:
        raise ValueError("Source and target numeric feature dimensions differ.")
    if source_features.categorical.shape[1] != target_features.categorical.shape[1]:
        raise ValueError("Source and target categorical feature dimensions differ.")


def mixed_kernel_matrix(
    left: FeatureArrays,
    right: FeatureArrays,
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> np.ndarray:
    if numeric_length_scale <= 0 or categorical_length_scale <= 0:
        raise ValueError("Kernel length scales must be positive.")
    left_sq = np.sum(left.numeric * left.numeric, axis=1)[:, None]
    right_sq = np.sum(right.numeric * right.numeric, axis=1)[None, :]
    numeric_sq = np.maximum(
        0.0,
        left_sq + right_sq - 2.0 * left.numeric @ right.numeric.T,
    )
    mismatch = np.zeros((len(left.numeric), len(right.numeric)), dtype=np.float64)
    for column in range(left.categorical.shape[1]):
        mismatch += (
            left.categorical[:, column, None]
            != right.categorical[None, :, column]
        )
    return np.exp(
        -numeric_sq / (2.0 * numeric_length_scale * numeric_length_scale)
        -mismatch / categorical_length_scale
    )


def stable_cholesky(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    jitter = 1e-10
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity), jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Unable to stabilize covariance matrix.")


def normalized_outcomes(candidates: Iterable[replay.Candidate]) -> tuple[np.ndarray, float, float]:
    values = np.asarray([candidate.objective_value / 100.0 for candidate in candidates], dtype=np.float64)
    center = float(np.mean(values))
    scale = max(float(np.std(values)), 0.05)
    return (values - center) / scale, center, scale


def mixed_distance_to_point(features: FeatureArrays, index: int) -> np.ndarray:
    numeric = np.sqrt(np.sum((features.numeric - features.numeric[index]) ** 2, axis=1))
    categorical = np.mean(features.categorical != features.categorical[index], axis=1)
    return numeric + categorical


def select_inducing_indices(features: FeatureArrays, limit: int) -> np.ndarray:
    count = len(features.numeric)
    if limit <= 0:
        raise ValueError("source_inducing_limit must be positive.")
    if count <= limit:
        return np.arange(count, dtype=int)
    selected = [0]
    min_distance = mixed_distance_to_point(features, 0)
    min_distance[0] = -1.0
    while len(selected) < limit:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(
            min_distance,
            mixed_distance_to_point(features, next_index),
        )
        min_distance[np.asarray(selected, dtype=int)] = -1.0
    return np.asarray(selected, dtype=int)


def build_source_posterior(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    source_observed: list[replay.Candidate],
    source_inducing_limit: int,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> SourcePosterior:
    source_all = feature_arrays(source_adapter, source_observed)
    inducing_indices = select_inducing_indices(source_all, source_inducing_limit)
    inducing = [source_observed[index] for index in inducing_indices]
    source_features = feature_arrays(source_adapter, inducing)
    target_features = feature_arrays(target_adapter, target_adapter.candidates)
    source_y, _center, _scale = normalized_outcomes(inducing)
    kernel_source = mixed_kernel_matrix(
        source_features,
        source_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    kernel_source += gp_noise * np.eye(len(inducing), dtype=np.float64)
    lower, _jitter = stable_cholesky(kernel_source)
    alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, source_y))
    kernel_target_source = mixed_kernel_matrix(
        target_features,
        source_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    projection = np.linalg.solve(lower, kernel_target_source.T).T
    source_mean = kernel_target_source @ alpha
    source_variance = np.maximum(1e-9, 1.0 - np.sum(projection * projection, axis=1))
    return SourcePosterior(
        mean=source_mean,
        variance=source_variance,
        projection=projection,
        inducing_count=len(inducing),
        source_observation_count=len(source_observed),
    )


def target_gp_posterior(
    target_features: FeatureArrays,
    observed_indices: list[int],
    observed_y: np.ndarray,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    observed_features = FeatureArrays(
        target_features.numeric[observed_indices],
        target_features.categorical[observed_indices],
    )
    kernel_observed = mixed_kernel_matrix(
        observed_features,
        observed_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    kernel_observed += gp_noise * np.eye(len(observed_indices), dtype=np.float64)
    lower, jitter = stable_cholesky(kernel_observed)
    alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, observed_y))
    kernel_pool_observed = mixed_kernel_matrix(
        target_features,
        observed_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    posterior_mean = kernel_pool_observed @ alpha
    solved = np.linalg.solve(lower, kernel_pool_observed.T)
    posterior_variance = np.maximum(1e-9, 1.0 - np.sum(solved * solved, axis=0))
    inverse = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(len(observed_indices))))
    diagonal = np.maximum(np.diag(inverse), 1e-9)
    loo_mean = observed_y - alpha / diagonal
    loo_variance = 1.0 / diagonal
    return posterior_mean, posterior_variance, {
        "lower": lower,
        "alpha": alpha,
        "loo_mean": loo_mean,
        "loo_variance": loo_variance,
        "jitter": jitter,
    }


def ranking_loss(prediction: np.ndarray, observed_y: np.ndarray) -> int:
    loss = 0
    for left in range(len(observed_y)):
        for right in range(left + 1, len(observed_y)):
            target_order = observed_y[left] > observed_y[right]
            prediction_order = prediction[left] > prediction[right]
            loss += int(target_order != prediction_order)
    return loss


def rgpe_weights(
    source: SourcePosterior,
    target_features: FeatureArrays,
    observed_indices: list[int],
    observed_y: np.ndarray,
    target_diagnostics: dict[str, Any],
    draws: int,
    seed: int,
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> tuple[float, float, dict[str, Any]]:
    if draws <= 0:
        raise ValueError("RGPE draws must be positive.")
    rng = np.random.default_rng(seed)
    source_mean = source.mean[observed_indices]
    source_projection = source.projection[observed_indices]
    observed_features = FeatureArrays(
        target_features.numeric[observed_indices],
        target_features.categorical[observed_indices],
    )
    source_covariance = (
        mixed_kernel_matrix(
            observed_features,
            observed_features,
            numeric_length_scale,
            categorical_length_scale,
        )
        - source_projection @ source_projection.T
    )
    source_lower, _jitter = stable_cholesky(source_covariance)
    target_mean = np.asarray(target_diagnostics["loo_mean"], dtype=np.float64)
    target_std = np.sqrt(np.maximum(target_diagnostics["loo_variance"], 1e-9))
    source_wins = 0.0
    target_wins = 0.0
    source_losses: list[int] = []
    target_losses: list[int] = []
    for _ in range(draws):
        source_sample = source_mean + source_lower @ rng.standard_normal(len(observed_indices))
        target_sample = target_mean + target_std * rng.standard_normal(len(observed_indices))
        source_loss = ranking_loss(source_sample, observed_y)
        target_loss = ranking_loss(target_sample, observed_y)
        source_losses.append(source_loss)
        target_losses.append(target_loss)
        if source_loss < target_loss:
            source_wins += 1.0
        elif target_loss < source_loss:
            target_wins += 1.0
        else:
            source_wins += 0.5
            target_wins += 0.5
    return source_wins / draws, target_wins / draws, {
        "draws": draws,
        "source_weight": round(source_wins / draws, 6),
        "target_weight": round(target_wins / draws, 6),
        "source_ranking_loss_mean": round(float(np.mean(source_losses)), 6),
        "target_ranking_loss_mean": round(float(np.mean(target_losses)), 6),
    }


def multitask_gp_posterior(
    target_features: FeatureArrays,
    source: SourcePosterior,
    observed_indices: list[int],
    observed_y: np.ndarray,
    rho_grid: tuple[float, ...],
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    observed_features = FeatureArrays(
        target_features.numeric[observed_indices],
        target_features.categorical[observed_indices],
    )
    kernel_observed = mixed_kernel_matrix(
        observed_features,
        observed_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    kernel_pool_observed = mixed_kernel_matrix(
        target_features,
        observed_features,
        numeric_length_scale,
        categorical_length_scale,
    )
    projection_observed = source.projection[observed_indices]
    candidates: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for rho in rho_grid:
        if not -1.0 < rho < 1.0:
            raise ValueError("ICM rho values must lie in (-1, 1).")
        prior_mean = rho * source.mean
        covariance_observed = (
            kernel_observed
            - rho * rho * (projection_observed @ projection_observed.T)
        )
        covariance_observed += gp_noise * np.eye(len(observed_indices), dtype=np.float64)
        lower, jitter = stable_cholesky(covariance_observed)
        residual = observed_y - prior_mean[observed_indices]
        alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, residual))
        log_marginal_likelihood = (
            -0.5 * float(residual @ alpha)
            - float(np.sum(np.log(np.diag(lower))))
            - 0.5 * len(observed_indices) * math.log(2.0 * math.pi)
        )
        candidates.append((log_marginal_likelihood, rho, lower, alpha, prior_mean, jitter))
    log_likelihood, rho, lower, alpha, prior_mean, jitter = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    cross_covariance = (
        kernel_pool_observed
        - rho * rho * (source.projection @ projection_observed.T)
    )
    posterior_mean = prior_mean + cross_covariance @ alpha
    solved = np.linalg.solve(lower, cross_covariance.T)
    prior_variance = np.maximum(
        1e-9,
        1.0 - rho * rho * np.sum(source.projection * source.projection, axis=1),
    )
    posterior_variance = np.maximum(
        1e-9,
        prior_variance - np.sum(solved * solved, axis=0),
    )
    return posterior_mean, posterior_variance, {
        "selected_rho": rho,
        "log_marginal_likelihood": round(log_likelihood, 6),
        "rho_grid": list(rho_grid),
        "jitter": jitter,
    }


def top_unobserved(scores: np.ndarray, observed_indices: set[int]) -> int:
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[list(observed_indices)] = -np.inf
    return int(np.argmax(masked))


def run_seed(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    source_posterior: SourcePosterior,
    seed: int,
    initial: int,
    rounds: int,
    mode: str,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    rgpe_draws: int,
    rho_grid: tuple[float, ...],
    fixed_initial_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in MODES:
        raise ValueError(f"Unknown classical baseline mode: {mode}")
    pool = target_adapter.candidates
    target_features = feature_arrays(target_adapter, pool)
    if fixed_initial_indices is None:
        shuffled_indices = list(range(len(pool)))
        random.Random(seed).shuffle(shuffled_indices)
        observed_indices = shuffled_indices[:initial]
    else:
        observed_indices = [int(index) for index in fixed_initial_indices]
        if len(observed_indices) != initial:
            raise ValueError("fixed_initial_indices must match initial observations.")
        if len(set(observed_indices)) != len(observed_indices):
            raise ValueError("fixed_initial_indices must be unique.")
        if any(index < 0 or index >= len(pool) for index in observed_indices):
            raise ValueError("fixed_initial_indices contains an out-of-range index.")
        observed_set_for_order = set(observed_indices)
        shuffled_indices = [
            *observed_indices,
            *(index for index in range(len(pool)) if index not in observed_set_for_order),
        ]
    observed_set = set(observed_indices)
    top10 = {
        index
        for index, _candidate in sorted(
            enumerate(pool),
            key=lambda item: item[1].objective_value,
            reverse=True,
        )[:10]
    }
    selected_top10 = bool(observed_set & top10)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(rounds):
        observed_candidates = [pool[index] for index in observed_indices]
        observed_y, target_center, target_scale = normalized_outcomes(observed_candidates)
        target_mean, target_variance, target_diagnostics = target_gp_posterior(
            target_features,
            observed_indices,
            observed_y,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        diagnostics: dict[str, Any] = {
            "target_center": round(target_center, 6),
            "target_scale": round(target_scale, 6),
        }
        if mode == "target_gp_ucb":
            posterior_mean = target_mean
            posterior_variance = target_variance
        elif mode == "rgpe":
            source_weight, target_weight, weight_diagnostics = rgpe_weights(
                source_posterior,
                target_features,
                observed_indices,
                observed_y,
                target_diagnostics,
                rgpe_draws,
                seed * 10_000 + round_index,
                numeric_length_scale,
                categorical_length_scale,
            )
            posterior_mean = source_weight * source_posterior.mean + target_weight * target_mean
            posterior_variance = (
                source_weight * source_weight * source_posterior.variance
                + target_weight * target_weight * target_variance
            )
            diagnostics["rgpe"] = weight_diagnostics
        else:
            posterior_mean, posterior_variance, multitask_diagnostics = multitask_gp_posterior(
                target_features,
                source_posterior,
                observed_indices,
                observed_y,
                rho_grid,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            diagnostics["multitask_gp_icm"] = multitask_diagnostics
        scores = posterior_mean + gp_beta * np.sqrt(posterior_variance)
        selected_index = top_unobserved(scores, observed_set)
        selected = pool[selected_index]
        observed_indices.append(selected_index)
        observed_set.add(selected_index)
        selected_top10 = selected_top10 or selected_index in top10
        best_so_far = max(pool[index].objective_value for index in observed_indices)
        best_trace.append(best_so_far)
        audit.append(
            {
                "source_dataset": source_adapter.dataset_id,
                "target_dataset": target_adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "initial_candidate_ids": [pool[index].candidate_id for index in shuffled_indices[:initial]],
                "selected_candidate": selected.candidate_id,
                "selected_score": round(float(scores[selected_index]), 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "diagnostics": diagnostics,
            }
        )
    final_best = max(pool[index].objective_value for index in observed_indices)
    oracle = max(candidate.objective_value for candidate in pool)
    return {
        "source_dataset": source_adapter.dataset_id,
        "target_dataset": target_adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "initial_observations": initial,
        "reveal_rounds": rounds,
        "final_best": round(final_best, 6),
        "best_so_far_auc": round(mean(best_trace), 6),
        "simple_regret": round(oracle - final_best, 6),
        "top10_hit": int(selected_top10),
        "source_observations": source_posterior.source_observation_count,
        "source_inducing_points": source_posterior.inducing_count,
    }, audit


def paired_comparison(
    rows: list[dict[str, Any]],
    challenger: str,
    baseline: str,
) -> dict[str, Any]:
    indexed = {(str(row["mode"]), int(row["seed"])): row for row in rows}
    seeds = sorted(
        seed
        for mode, seed in indexed
        if mode == challenger and (baseline, seed) in indexed
    )
    output: dict[str, Any] = {"seed_count": len(seeds)}
    for field in ("final_best", "best_so_far_auc", "top10_hit"):
        deltas = np.asarray(
            [
                float(indexed[(challenger, seed)][field])
                - float(indexed[(baseline, seed)][field])
                for seed in seeds
            ],
            dtype=np.float64,
        )
        delta_mean = float(np.mean(deltas)) if len(deltas) else 0.0
        delta_std = float(np.std(deltas)) if len(deltas) > 1 else 0.0
        half_width = 1.96 * delta_std / math.sqrt(len(deltas)) if len(deltas) else 0.0
        output[field] = {
            "mean_delta": round(delta_mean, 6),
            "normal_95ci_low": round(delta_mean - half_width, 6),
            "normal_95ci_high": round(delta_mean + half_width, 6),
            "win_rate": round(float(np.mean(deltas > 0.0)), 6) if len(deltas) else 0.0,
            "non_loss_rate": round(float(np.mean(deltas >= 0.0)), 6) if len(deltas) else 0.0,
        }
    return output


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        items = [row for row in rows if row["mode"] == mode]
        if not items:
            continue
        output[mode] = {}
        for field in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = [float(row[field]) for row in items]
            output[mode][field] = {
                "mean": round(mean(values), 6),
                "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
            }
    return output


def run_pair(
    config: dict[str, Any],
    pair: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    protocol = config["protocol"]
    kernel = protocol["kernel"]
    source = replay.DATASET_BUILDERS[pair["source_dataset"]]()
    target = replay.DATASET_BUILDERS[pair["target_dataset"]]()
    validate_compatible_spaces(source, target)
    source_observed = transfer.source_observations(
        source,
        int(protocol["source_seed"]),
        int(pair["source_observations"]),
    )
    source_posterior = build_source_posterior(
        source,
        target,
        source_observed,
        int(protocol["source_inducing_limit"]),
        float(kernel["numeric_length_scale"]),
        float(kernel["categorical_length_scale"]),
        float(kernel["gp_noise"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audits" / pair["pair_id"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in ("calibration", "heldout"):
        start = int(protocol[f"{split}_seed_start"])
        count = int(protocol[f"{split}_seed_count"])
        split_rows: list[dict[str, Any]] = []
        for seed in range(start, start + count):
            for mode in MODES:
                metrics, audit = run_seed(
                    source,
                    target,
                    source_posterior,
                    seed,
                    int(pair["initial_observations"]),
                    int(pair["reveal_rounds"]),
                    mode,
                    float(kernel["gp_beta"]),
                    float(kernel["numeric_length_scale"]),
                    float(kernel["categorical_length_scale"]),
                    float(kernel["gp_noise"]),
                    int(protocol["rgpe_draws"]),
                    tuple(float(value) for value in protocol["multitask_rho_grid"]),
                )
                metrics["pair_id"] = pair["pair_id"]
                metrics["split"] = split
                split_rows.append(metrics)
                with (audit_dir / f"{split}_{mode}_seed{seed}.jsonl").open(
                    "w", encoding="utf-8"
                ) as handle:
                    for event in audit:
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        all_rows.extend(split_rows)
        split_summaries[split] = {
            "aggregate": aggregate(split_rows),
            "rgpe_vs_target_gp": paired_comparison(split_rows, "rgpe", "target_gp_ucb"),
            "multitask_gp_vs_target_gp": paired_comparison(
                split_rows,
                "multitask_gp_icm",
                "target_gp_ucb",
            ),
        }
    metrics_path = output_dir / f"{pair['pair_id']}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "experiment": "care2_classical_transfer_bo_confirmation",
        "protocol_version": protocol["version"],
        "pair": pair,
        "matched_initial_observations": True,
        "source_model": {
            "source_observation_count": source_posterior.source_observation_count,
            "inducing_count": source_posterior.inducing_count,
            "approximation": "deterministic farthest-point sparse GP",
        },
        "methods": {
            "target_gp_ucb": "Target-only GP-UCB using the same initial target observations.",
            "rgpe": "Ranking-weighted source/target GP ensemble with Monte Carlo ranking weights.",
            "multitask_gp_icm": "Two-task intrinsic-coregionalization GP conditioned on the source task; rho is selected online by target marginal likelihood, including rho=0 fallback.",
        },
        "splits": split_summaries,
    }
    (output_dir / f"{pair['pair_id']}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "classical_transfer_benchmark_v1.json",
    )
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pair = next(
        (item for item in config["pairs"] if item["pair_id"] == args.pair_id),
        None,
    )
    if pair is None:
        raise SystemExit(f"Unknown pair id: {args.pair_id}")
    summary = run_pair(config, pair, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
