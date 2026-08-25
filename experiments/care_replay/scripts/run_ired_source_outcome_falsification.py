#!/usr/bin/env python3
"""Falsify the IRED additive skill with matched source-outcome permutations."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

import run_classical_transfer_baselines as classical
import run_multisource_transfer_baselines as multisource
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
TRUE_CONDITION = "true_source_outcomes"
BASELINE_CONDITION = "target_gp_ucb"
SCHEMA_VERSION = "care.ired_source_outcome_falsification/v1"
METRICS = ("final_best", "best_so_far_auc", "simple_regret", "top10_hit")

_WORKER: dict[str, Any] = {}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_sha256(path)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol.get("status") != "retrospective_mechanism_falsification":
        raise ValueError("The IRED falsification must retain its post-hoc status.")
    if int(protocol["permutation_count"]) < 19:
        raise ValueError("At least 19 source-outcome permutations are required.")
    if int(protocol["target_seed_count"]) < 30:
        raise ValueError("At least 30 paired target seeds are required.")
    if protocol["source_task_id"] not in replay.DATASET_BUILDERS:
        raise ValueError("Unknown source task.")
    if protocol["target_task_id"] not in replay.DATASET_BUILDERS:
        raise ValueError("Unknown target task.")


def permuted_source_observations(
    source_observed: Sequence[replay.Candidate],
    permutation_seed: int,
) -> list[replay.Candidate]:
    outcomes = [candidate.objective_value for candidate in source_observed]
    shuffled = list(outcomes)
    random.Random(permutation_seed).shuffle(shuffled)
    if shuffled == outcomes:
        shuffled = shuffled[1:] + shuffled[:1]
    return [
        replace(candidate, objective_value=value)
        for candidate, value in zip(source_observed, shuffled)
    ]


def normal_interval(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize an empty sample.")
    mean_value = float(np.mean(array))
    standard_error = (
        float(np.std(array, ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return {
        "n": int(len(array)),
        "mean": mean_value,
        "ci95_low": mean_value - 1.96 * standard_error,
        "ci95_high": mean_value + 1.96 * standard_error,
    }


def bootstrap_interval(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = np.random.default_rng(seed)
    sample_means = np.empty(draws, dtype=np.float64)
    chunk = 10_000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        sample_means[start:stop] = np.mean(array[indices], axis=1)
    return {
        "n": int(len(array)),
        "draws": int(draws),
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(sample_means, 0.025)),
        "ci95_high": float(np.quantile(sample_means, 0.975)),
    }


def randomization_p_value(
    observed: float,
    null_values: Sequence[float],
) -> float:
    return (1 + sum(value >= observed for value in null_values)) / (
        1 + len(null_values)
    )


def _init_worker(config_path: str, priors_path: str) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    protocol = config["protocol"]
    target = replay.DATASET_BUILDERS[str(protocol["target_task_id"])]()
    archive = np.load(priors_path, allow_pickle=False)
    _WORKER.clear()
    _WORKER.update(
        {
            "config": config,
            "target": target,
            "features": classical.feature_arrays(target, target.candidates),
            "prior_names": [str(value) for value in archive["condition_names"]],
            "priors": np.asarray(archive["priors"], dtype=np.float64),
        }
    )


def _run_policy(
    *,
    seed: int,
    source_prior: np.ndarray | None,
) -> dict[str, Any]:
    protocol = _WORKER["config"]["protocol"]
    target = _WORKER["target"]
    target_features = _WORKER["features"]
    pool = target.candidates
    initial = int(protocol["initial_observations"])
    rounds = int(protocol["reveal_rounds"])
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    observed_indices = list(order[:initial])
    observed_set = set(observed_indices)
    selected_ids: list[str] = []
    best_trace: list[float] = []
    for round_index in range(rounds):
        observed_candidates = [pool[index] for index in observed_indices]
        observed_y, _center, _scale = classical.normalized_outcomes(
            observed_candidates
        )
        target_mean, target_variance, _diagnostics = classical.target_gp_posterior(
            target_features,
            observed_indices,
            observed_y,
            float(protocol["kernel"]["numeric_length_scale"]),
            float(protocol["kernel"]["categorical_length_scale"]),
            float(protocol["kernel"]["gp_noise"]),
        )
        target_scores = target_mean + float(protocol["kernel"]["gp_beta"]) * np.sqrt(
            target_variance
        )
        if source_prior is None:
            scores = target_scores
        else:
            target_rank = np.asarray(
                list(multisource.classical_rank_normalized(target_scores).values()),
                dtype=np.float64,
            )
            mass = multisource.scheduled_mass(
                float(protocol["skill_prior_mass_start"]),
                float(protocol["skill_prior_mass_end"]),
                round_index,
                rounds,
            )
            scores = (1.0 - mass) * target_rank + mass * source_prior
        selected_index = classical.top_unobserved(scores, observed_set)
        observed_indices.append(selected_index)
        observed_set.add(selected_index)
        selected_ids.append(pool[selected_index].candidate_id)
        best_trace.append(
            max(pool[index].objective_value for index in observed_indices)
        )
    final_best = max(pool[index].objective_value for index in observed_indices)
    oracle = max(candidate.objective_value for candidate in pool)
    top10_indices = {
        index
        for index, _candidate in sorted(
            enumerate(pool),
            key=lambda item: item[1].objective_value,
            reverse=True,
        )[:10]
    }
    return {
        "target_seed": seed,
        "initial_candidate_ids": [pool[index].candidate_id for index in order[:initial]],
        "selected_candidate_ids": selected_ids,
        "best_trace": best_trace,
        "final_best": float(final_best),
        "best_so_far_auc": float(mean(best_trace)),
        "simple_regret": float(oracle - final_best),
        "top10_hit": int(bool(set(observed_indices) & top10_indices)),
    }


def _run_target_seed(seed: int) -> list[dict[str, Any]]:
    rows = [
        {
            "condition": BASELINE_CONDITION,
            "permutation_seed": None,
            **_run_policy(seed=seed, source_prior=None),
        }
    ]
    for index, name in enumerate(_WORKER["prior_names"]):
        permutation_seed = None
        if name.startswith("permuted_source_outcomes_"):
            permutation_seed = int(name.rsplit("_", 1)[-1])
        rows.append(
            {
                "condition": name,
                "permutation_seed": permutation_seed,
                **_run_policy(seed=seed, source_prior=_WORKER["priors"][index]),
            }
        )
    return rows


def build_priors(
    config: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    protocol = config["protocol"]
    source = replay.DATASET_BUILDERS[str(protocol["source_task_id"])]()
    target = replay.DATASET_BUILDERS[str(protocol["target_task_id"])]()
    source_observed = transfer.source_observations(
        source,
        int(protocol["source_seed"]),
        int(protocol["source_observation_count"]),
    )
    names = [TRUE_CONDITION]
    true_prior, true_diagnostics = multisource.additive_mutation_prior(
        [source_observed], target.candidates
    )
    priors = [true_prior]
    permutation_diagnostics: list[dict[str, Any]] = []
    start = int(protocol["permutation_seed_start"])
    for permutation_seed in range(start, start + int(protocol["permutation_count"])):
        permuted = permuted_source_observations(source_observed, permutation_seed)
        prior, diagnostics = multisource.additive_mutation_prior(
            [permuted], target.candidates
        )
        names.append(f"permuted_source_outcomes_{permutation_seed}")
        priors.append(prior)
        permutation_diagnostics.append(
            {
                "permutation_seed": permutation_seed,
                "prior_sha256": hashlib.sha256(prior.tobytes()).hexdigest(),
                "diagnostics": diagnostics,
            }
        )
    diagnostics = {
        "source_dataset": source.dataset_id,
        "target_dataset": target.dataset_id,
        "source_observation_count": len(source_observed),
        "source_candidate_ids_sha256": hashlib.sha256(
            "\n".join(candidate.candidate_id for candidate in source_observed).encode(
                "utf-8"
            )
        ).hexdigest(),
        "source_outcomes_sha256": hashlib.sha256(
            np.asarray(
                [candidate.objective_value for candidate in source_observed],
                dtype=np.float64,
            ).tobytes()
        ).hexdigest(),
        "true_prior_sha256": hashlib.sha256(true_prior.tobytes()).hexdigest(),
        "true_diagnostics": true_diagnostics,
        "permutations": permutation_diagnostics,
    }
    return names, np.stack(priors), diagnostics


def build_report(
    config: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = config["protocol"]
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in trajectories:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    baseline = {
        int(row["target_seed"]): row
        for row in by_condition[BASELINE_CONDITION]
    }
    condition_summaries: list[dict[str, Any]] = []
    condition_seed_values: dict[str, dict[int, Mapping[str, Any]]] = {}
    for condition, rows in sorted(by_condition.items()):
        if condition == BASELINE_CONDITION:
            continue
        ordered = sorted(rows, key=lambda row: int(row["target_seed"]))
        condition_seed_values[condition] = {
            int(row["target_seed"]): row for row in ordered
        }
        auc_deltas = [
            float(row["best_so_far_auc"])
            - float(baseline[int(row["target_seed"])]["best_so_far_auc"])
            for row in ordered
        ]
        final_deltas = [
            float(row["final_best"])
            - float(baseline[int(row["target_seed"])]["final_best"])
            for row in ordered
        ]
        top10_deltas = [
            float(row["top10_hit"])
            - float(baseline[int(row["target_seed"])]["top10_hit"])
            for row in ordered
        ]
        condition_summaries.append(
            {
                "condition": condition,
                "permutation_seed": ordered[0].get("permutation_seed"),
                "auc_delta_vs_target_gp": normal_interval(auc_deltas),
                "final_delta_vs_target_gp": normal_interval(final_deltas),
                "top10_delta_vs_target_gp": normal_interval(top10_deltas),
                "wins": sum(value > 1e-12 for value in auc_deltas),
                "ties": sum(abs(value) <= 1e-12 for value in auc_deltas),
                "losses": sum(value < -1e-12 for value in auc_deltas),
            }
        )
    true_summary = next(
        row for row in condition_summaries if row["condition"] == TRUE_CONDITION
    )
    null_summaries = [
        row for row in condition_summaries if row["condition"] != TRUE_CONDITION
    ]
    true_by_seed = condition_seed_values[TRUE_CONDITION]
    null_mean_by_seed: dict[int, dict[str, float]] = {}
    for target_seed in sorted(baseline):
        null_rows = [
            values[target_seed]
            for condition, values in condition_seed_values.items()
            if condition != TRUE_CONDITION
        ]
        null_mean_by_seed[target_seed] = {
            metric: float(np.mean([float(row[metric]) for row in null_rows]))
            for metric in METRICS
        }
    true_minus_null_auc = [
        float(true_by_seed[seed]["best_so_far_auc"])
        - null_mean_by_seed[seed]["best_so_far_auc"]
        for seed in sorted(baseline)
    ]
    true_minus_null_final = [
        float(true_by_seed[seed]["final_best"])
        - null_mean_by_seed[seed]["final_best"]
        for seed in sorted(baseline)
    ]
    true_minus_null_top10 = [
        float(true_by_seed[seed]["top10_hit"])
        - null_mean_by_seed[seed]["top10_hit"]
        for seed in sorted(baseline)
    ]
    true_auc_effect = float(true_summary["auc_delta_vs_target_gp"]["mean"])
    null_auc_effects = [
        float(row["auc_delta_vs_target_gp"]["mean"]) for row in null_summaries
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": protocol["version"],
        "evidence_class": protocol["evidence_class"],
        "claim_boundary": protocol["claim_boundary"],
        "target_seed_count": int(protocol["target_seed_count"]),
        "permutation_count": int(protocol["permutation_count"]),
        "true_assignment": true_summary,
        "null_assignment_distribution": {
            "auc_delta_vs_target_gp": {
                "mean": float(np.mean(null_auc_effects)),
                "median": float(np.median(null_auc_effects)),
                "min": float(np.min(null_auc_effects)),
                "max": float(np.max(null_auc_effects)),
                "positive_count": int(sum(value > 0.0 for value in null_auc_effects)),
                "stable_positive_count": int(
                    sum(
                        float(row["auc_delta_vs_target_gp"]["ci95_low"]) > 0.0
                        for row in null_summaries
                    )
                ),
            }
        },
        "true_vs_permuted_assignment": {
            "auc": bootstrap_interval(
                true_minus_null_auc,
                draws=int(protocol["bootstrap_draws"]),
                seed=98201,
            ),
            "final_best": bootstrap_interval(
                true_minus_null_final,
                draws=int(protocol["bootstrap_draws"]),
                seed=98202,
            ),
            "top10_hit": bootstrap_interval(
                true_minus_null_top10,
                draws=int(protocol["bootstrap_draws"]),
                seed=98203,
            ),
            "one_sided_randomization_p_auc": randomization_p_value(
                true_auc_effect, null_auc_effects
            ),
            "null_assignments_at_least_as_good_as_true": int(
                sum(value >= true_auc_effect for value in null_auc_effects)
            ),
        },
    }
    return report, condition_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "flip2_ired_source_outcome_falsification_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    protocol = config["protocol"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data" / "raw" / str(protocol["dataset"]["local_filename"])
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    implementation_files = [
        Path(__file__).resolve(),
        Path(classical.__file__).resolve(),
        Path(multisource.__file__).resolve(),
        Path(surrogate.__file__).resolve(),
        Path(replay.__file__).resolve(),
        Path(transfer.__file__).resolve(),
    ]
    lock = {
        "schema_version": f"{SCHEMA_VERSION}.lock",
        "config_sha256": canonical_sha256(config),
        "raw_data_sha256": sha256_path(raw_path),
        "implementation_sha256": {
            path.name: sha256_path(path) for path in implementation_files
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gp_backend": surrogate.gp_backend_name(),
        },
        "posthoc_status_retained": True,
    }
    lock_path = args.output_dir / "protocol_lock.json"
    write_json(lock_path, lock)
    names, priors, prior_diagnostics = build_priors(config)
    priors_path = args.output_dir / "source_priors.npz"
    np.savez_compressed(
        priors_path,
        condition_names=np.asarray(names),
        priors=priors,
    )
    write_sha256(priors_path)
    write_json(args.output_dir / "source_prior_diagnostics.json", prior_diagnostics)
    seeds = range(
        int(protocol["target_seed_start"]),
        int(protocol["target_seed_start"]) + int(protocol["target_seed_count"]),
    )
    trajectories: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.config.resolve()), str(priors_path.resolve())),
    ) as executor:
        for rows in executor.map(_run_target_seed, seeds):
            trajectories.extend(rows)
    trajectories.sort(
        key=lambda row: (int(row["target_seed"]), str(row["condition"]))
    )
    trajectory_path = args.output_dir / "trajectory_audit.jsonl"
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for row in trajectories:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_sha256(trajectory_path)
    metrics_path = args.output_dir / "trajectory_metrics.csv"
    metric_fields = [
        "condition",
        "permutation_seed",
        "target_seed",
        *METRICS,
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=metric_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in trajectories:
            writer.writerow({field: row.get(field) for field in metric_fields})
    write_sha256(metrics_path)
    report, condition_summaries = build_report(config, trajectories)
    summary_path = args.output_dir / "assignment_effects.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "condition",
            "permutation_seed",
            "auc_mean_delta",
            "auc_ci95_low",
            "auc_ci95_high",
            "final_mean_delta",
            "final_ci95_low",
            "final_ci95_high",
            "top10_mean_delta",
            "top10_ci95_low",
            "top10_ci95_high",
            "wins",
            "ties",
            "losses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in condition_summaries:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "permutation_seed": row["permutation_seed"],
                    "auc_mean_delta": row["auc_delta_vs_target_gp"]["mean"],
                    "auc_ci95_low": row["auc_delta_vs_target_gp"]["ci95_low"],
                    "auc_ci95_high": row["auc_delta_vs_target_gp"]["ci95_high"],
                    "final_mean_delta": row["final_delta_vs_target_gp"]["mean"],
                    "final_ci95_low": row["final_delta_vs_target_gp"]["ci95_low"],
                    "final_ci95_high": row["final_delta_vs_target_gp"]["ci95_high"],
                    "top10_mean_delta": row["top10_delta_vs_target_gp"]["mean"],
                    "top10_ci95_low": row["top10_delta_vs_target_gp"]["ci95_low"],
                    "top10_ci95_high": row["top10_delta_vs_target_gp"]["ci95_high"],
                    "wins": row["wins"],
                    "ties": row["ties"],
                    "losses": row["losses"],
                }
            )
    write_sha256(summary_path)
    write_json(args.output_dir / "falsification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
