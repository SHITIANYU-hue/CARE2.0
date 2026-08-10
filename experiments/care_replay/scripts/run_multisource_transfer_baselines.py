#!/usr/bin/env python3
"""Evaluate A,B,C -> E transfer without target-task calibration.

The runner extends the same-space classical baselines to multiple completed
source tasks.  Method identities and hyperparameters are frozen in a config;
the unseen target task is used only for sequential ask/tell replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

import run_classical_transfer_baselines as classical
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
POLICY_MODES = (
    "target_gp_ucb",
    "multisource_rgpe",
    "multisource_icm_bma",
    "multisource_skill_prior",
)
SELECTED_ROUTE_MODE = "development_selected_route"
MODES = POLICY_MODES + (SELECTED_ROUTE_MODE,)


def validate_new_task_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    development = set(protocol["development_task_ids"])
    evaluation = set(protocol["evaluation_task_ids"])
    overlap = development & evaluation
    if overlap:
        raise ValueError(f"Development and evaluation tasks overlap: {sorted(overlap)}")
    if protocol.get("target_task_calibration", True):
        raise ValueError("New-task confirmation forbids target-task calibration.")
    for experiment in config["experiments"]:
        sources = set(experiment["source_datasets"])
        target = str(experiment["target_dataset"])
        if target in sources:
            raise ValueError(f"Target task {target} is also listed as a source.")
        if not sources <= development:
            raise ValueError("Every source dataset must be a declared development task.")
        if target not in evaluation:
            raise ValueError("Every target dataset must be a declared evaluation task.")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        output[mode] = {}
        for field in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
            output[mode][field] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
    return output


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scheduled_mass(start: float, end: float, round_index: int, rounds: int) -> float:
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("Skill-prior masses must lie in [0, 1].")
    if rounds <= 1:
        return end
    fraction = round_index / float(rounds - 1)
    return start + fraction * (end - start)


def consensus_prior(source_posteriors: Sequence[classical.SourcePosterior]) -> np.ndarray:
    if not source_posteriors:
        raise ValueError("Consensus prior requires at least one source.")
    ranked = np.stack(
        [
            np.asarray(
                list(classical_rank_normalized(source.mean).values()),
                dtype=np.float64,
            )
            for source in source_posteriors
        ]
    )
    return np.median(ranked, axis=0)


def classical_rank_normalized(values: np.ndarray) -> dict[int, float]:
    order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort"))
    denominator = max(1, len(order) - 1)
    return {index: float(rank / denominator) for index, rank in enumerate(order)}


def source_draw(
    source: classical.SourcePosterior,
    target_features: classical.FeatureArrays,
    observed_indices: list[int],
    rng: np.random.Generator,
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> np.ndarray:
    observed_features = classical.FeatureArrays(
        target_features.numeric[observed_indices],
        target_features.categorical[observed_indices],
    )
    source_projection = source.projection[observed_indices]
    covariance = (
        classical.mixed_kernel_matrix(
            observed_features,
            observed_features,
            numeric_length_scale,
            categorical_length_scale,
        )
        - source_projection @ source_projection.T
    )
    lower, _jitter = classical.stable_cholesky(covariance)
    return source.mean[observed_indices] + lower @ rng.standard_normal(
        len(observed_indices)
    )


def multisource_rgpe_weights(
    sources: Sequence[classical.SourcePosterior],
    target_features: classical.FeatureArrays,
    observed_indices: list[int],
    observed_y: np.ndarray,
    target_diagnostics: dict[str, Any],
    draws: int,
    seed: int,
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not sources:
        raise ValueError("At least one source posterior is required.")
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(sources) + 1, dtype=np.float64)
    loss_sums = np.zeros(len(sources) + 1, dtype=np.float64)
    target_mean = np.asarray(target_diagnostics["loo_mean"], dtype=np.float64)
    target_std = np.sqrt(
        np.maximum(np.asarray(target_diagnostics["loo_variance"]), 1e-9)
    )
    for _ in range(draws):
        predictions = [
            source_draw(
                source,
                target_features,
                observed_indices,
                rng,
                numeric_length_scale,
                categorical_length_scale,
            )
            for source in sources
        ]
        predictions.append(
            target_mean + target_std * rng.standard_normal(len(observed_indices))
        )
        losses = np.asarray(
            [classical.ranking_loss(prediction, observed_y) for prediction in predictions],
            dtype=np.float64,
        )
        loss_sums += losses
        winners = np.flatnonzero(losses == np.min(losses))
        wins[winners] += 1.0 / len(winners)
    weights = wins / float(draws)
    return weights, {
        "draws": draws,
        "source_weights": [round(float(value), 6) for value in weights[:-1]],
        "target_weight": round(float(weights[-1]), 6),
        "ranking_loss_means": [
            round(float(value / draws), 6) for value in loss_sums
        ],
    }


def target_log_marginal_likelihood(
    observed_y: np.ndarray,
    target_diagnostics: dict[str, Any],
) -> float:
    lower = np.asarray(target_diagnostics["lower"], dtype=np.float64)
    alpha = np.asarray(target_diagnostics["alpha"], dtype=np.float64)
    return (
        -0.5 * float(observed_y @ alpha)
        - float(np.sum(np.log(np.diag(lower))))
        - 0.5 * len(observed_y) * math.log(2.0 * math.pi)
    )


def multisource_icm_bma(
    sources: Sequence[classical.SourcePosterior],
    target_features: classical.FeatureArrays,
    observed_indices: list[int],
    observed_y: np.ndarray,
    target_mean: np.ndarray,
    target_variance: np.ndarray,
    target_diagnostics: dict[str, Any],
    rho_grid: tuple[float, ...],
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if temperature <= 0.0:
        raise ValueError("BMA temperature must be positive.")
    means = [target_mean]
    variances = [target_variance]
    log_likelihoods = [target_log_marginal_likelihood(observed_y, target_diagnostics)]
    source_diagnostics: list[dict[str, Any]] = []
    for source in sources:
        posterior_mean, posterior_variance, diagnostics = (
            classical.multitask_gp_posterior(
                target_features,
                source,
                observed_indices,
                observed_y,
                rho_grid,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
        )
        means.append(posterior_mean)
        variances.append(posterior_variance)
        log_likelihoods.append(float(diagnostics["log_marginal_likelihood"]))
        source_diagnostics.append(diagnostics)
    logits = np.asarray(log_likelihoods, dtype=np.float64) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    stacked_means = np.stack(means)
    stacked_variances = np.stack(variances)
    mixture_mean = np.sum(weights[:, None] * stacked_means, axis=0)
    second_moment = np.sum(
        weights[:, None] * (stacked_variances + stacked_means * stacked_means),
        axis=0,
    )
    mixture_variance = np.maximum(1e-9, second_moment - mixture_mean * mixture_mean)
    return mixture_mean, mixture_variance, {
        "target_weight": round(float(weights[0]), 6),
        "source_weights": [round(float(value), 6) for value in weights[1:]],
        "log_marginal_likelihoods": [round(value, 6) for value in log_likelihoods],
        "source_models": source_diagnostics,
        "temperature": temperature,
    }


def run_seed(
    source_adapters: Sequence[replay.DatasetAdapter],
    target_adapter: replay.DatasetAdapter,
    source_posteriors: Sequence[classical.SourcePosterior],
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
    bma_temperature: float,
    skill_prior_mass_start: float = 0.0,
    skill_prior_mass_end: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in POLICY_MODES:
        raise ValueError(f"Unknown multi-source mode: {mode}")
    pool = target_adapter.candidates
    target_features = classical.feature_arrays(target_adapter, pool)
    shuffled_indices = list(range(len(pool)))
    random.Random(seed).shuffle(shuffled_indices)
    observed_indices = shuffled_indices[:initial]
    observed_set = set(observed_indices)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    source_consensus = consensus_prior(source_posteriors)
    for round_index in range(rounds):
        observed_candidates = [pool[index] for index in observed_indices]
        observed_y, target_center, target_scale = classical.normalized_outcomes(
            observed_candidates
        )
        target_mean, target_variance, target_diagnostics = (
            classical.target_gp_posterior(
                target_features,
                observed_indices,
                observed_y,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
        )
        diagnostics: dict[str, Any] = {
            "target_center": round(target_center, 6),
            "target_scale": round(target_scale, 6),
        }
        if mode == "target_gp_ucb":
            posterior_mean = target_mean
            posterior_variance = target_variance
        elif mode == "multisource_rgpe":
            weights, weight_diagnostics = multisource_rgpe_weights(
                source_posteriors,
                target_features,
                observed_indices,
                observed_y,
                target_diagnostics,
                rgpe_draws,
                seed * 10_000 + round_index,
                numeric_length_scale,
                categorical_length_scale,
            )
            models_mean = [source.mean for source in source_posteriors] + [target_mean]
            models_variance = [source.variance for source in source_posteriors] + [
                target_variance
            ]
            posterior_mean = np.sum(weights[:, None] * np.stack(models_mean), axis=0)
            posterior_variance = np.sum(
                (weights * weights)[:, None] * np.stack(models_variance), axis=0
            )
            diagnostics["multisource_rgpe"] = weight_diagnostics
        elif mode == "multisource_icm_bma":
            posterior_mean, posterior_variance, bma_diagnostics = multisource_icm_bma(
                source_posteriors,
                target_features,
                observed_indices,
                observed_y,
                target_mean,
                target_variance,
                target_diagnostics,
                rho_grid,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
                bma_temperature,
            )
            diagnostics["multisource_icm_bma"] = bma_diagnostics
        else:
            target_scores = target_mean + gp_beta * np.sqrt(target_variance)
            target_rank = np.asarray(
                list(classical_rank_normalized(target_scores).values()),
                dtype=np.float64,
            )
            mass = scheduled_mass(
                skill_prior_mass_start,
                skill_prior_mass_end,
                round_index,
                rounds,
            )
            scores = (1.0 - mass) * target_rank + mass * source_consensus
            diagnostics["multisource_skill_prior"] = {
                "mass": round(mass, 6),
                "mass_start": skill_prior_mass_start,
                "mass_end": skill_prior_mass_end,
                "source_aggregation": "median_of_per_source_rank",
            }
        if mode != "multisource_skill_prior":
            scores = posterior_mean + gp_beta * np.sqrt(posterior_variance)
        selected_index = classical.top_unobserved(scores, observed_set)
        selected = pool[selected_index]
        observed_indices.append(selected_index)
        observed_set.add(selected_index)
        best_so_far = max(pool[index].objective_value for index in observed_indices)
        best_trace.append(best_so_far)
        audit.append(
            {
                "source_datasets": [adapter.dataset_id for adapter in source_adapters],
                "target_dataset": target_adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "initial_candidate_ids": [
                    pool[index].candidate_id for index in shuffled_indices[:initial]
                ],
                "selected_candidate": selected.candidate_id,
                "public_conditions": {
                    "catalyst": selected.metadata.get("catalyst"),
                    "residence_time_seconds": selected.metadata.get(
                        "residence_time_seconds"
                    ),
                    "temperature_celsius": selected.metadata.get(
                        "temperature_celsius"
                    ),
                    "catalyst_loading_mol_percent": selected.metadata.get(
                        "catalyst_loading_mol_percent"
                    ),
                },
                "selected_score": round(float(scores[selected_index]), 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "diagnostics": diagnostics,
            }
        )
    final_best = max(pool[index].objective_value for index in observed_indices)
    oracle = max(candidate.objective_value for candidate in pool)
    top10 = {
        index
        for index, _candidate in sorted(
            enumerate(pool),
            key=lambda item: item[1].objective_value,
            reverse=True,
        )[:10]
    }
    selected_top10 = bool(set(observed_indices) & top10)
    return {
        "source_datasets": ";".join(
            adapter.dataset_id for adapter in source_adapters
        ),
        "target_dataset": target_adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "initial_observations": initial,
        "reveal_rounds": rounds,
        "final_best": round(final_best, 6),
        "best_so_far_auc": round(mean(best_trace), 6),
        "simple_regret": round(oracle - final_best, 6),
        "top10_hit": int(selected_top10),
        "source_task_count": len(source_adapters),
        "source_observations": sum(
            source.source_observation_count for source in source_posteriors
        ),
    }, audit


def run_experiment(
    config: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = config["protocol"]
    kernel = protocol["kernel"]
    if selection.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Selection record does not match the frozen benchmark config.")
    selected_policy = selection["selected_skill_prior"]
    selected_route = selection["selected_route"]
    sources = [
        replay.DATASET_BUILDERS[dataset_id]()
        for dataset_id in experiment["source_datasets"]
    ]
    target = replay.DATASET_BUILDERS[experiment["target_dataset"]]()
    for source in sources:
        classical.validate_compatible_spaces(source, target)
    observation_counts = list(experiment["source_observations"])
    if len(sources) != len(observation_counts):
        raise ValueError(
            "source_observations must contain one count per source dataset."
        )
    source_posteriors = []
    for source, observation_count in zip(
        sources, observation_counts
    ):
        observed = transfer.source_observations(
            source,
            int(protocol["source_seed"]),
            int(observation_count),
        )
        source_posteriors.append(
            classical.build_source_posterior(
                source,
                target,
                observed,
                int(protocol["source_inducing_limit"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audits" / experiment["experiment_id"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    start = int(protocol["evaluation_seed_start"])
    count = int(protocol["evaluation_seed_count"])
    for seed in range(start, start + count):
        for mode in MODES:
            execution_mode = (
                str(selected_route["mode"])
                if mode == SELECTED_ROUTE_MODE
                else mode
            )
            metrics, audit = run_seed(
                sources,
                target,
                source_posteriors,
                seed,
                int(experiment["initial_observations"]),
                int(experiment["reveal_rounds"]),
                execution_mode,
                float(kernel["gp_beta"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
                int(protocol["rgpe_draws"]),
                tuple(float(value) for value in protocol["multitask_rho_grid"]),
                float(protocol["bma_temperature"]),
                float(selected_policy["mass_start"]),
                float(selected_policy["mass_end"]),
            )
            if mode == SELECTED_ROUTE_MODE:
                metrics["mode"] = SELECTED_ROUTE_MODE
                for event in audit:
                    event["policy_mode"] = execution_mode
                    event["mode"] = SELECTED_ROUTE_MODE
                    event["development_selected_route"] = dict(selected_route)
            metrics["experiment_id"] = experiment["experiment_id"]
            rows.append(metrics)
            with (audit_dir / f"{mode}_seed{seed}.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for event in audit:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    metrics_path = output_dir / f"{experiment['experiment_id']}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "care2_multisource_new_task_confirmation",
        "protocol_version": protocol["version"],
        "task": experiment,
        "target_task_calibration": False,
        "development_selection": dict(selection),
        "aggregate": aggregate(rows),
        "comparisons": {
            mode: classical.paired_comparison(rows, mode, "target_gp_ucb")
            for mode in MODES
            if mode != "target_gp_ucb"
        },
    }
    (output_dir / f"{experiment['experiment_id']}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "multisource_new_task_benchmark_v1.json",
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_new_task_protocol(config)
    experiment = next(
        (
            item
            for item in config["experiments"]
            if item["experiment_id"] == args.experiment_id
        ),
        None,
    )
    if experiment is None:
        raise SystemExit(f"Unknown experiment id: {args.experiment_id}")
    selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
    summary = run_experiment(config, experiment, args.output_dir, selection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
