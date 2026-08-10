#!/usr/bin/env python3
"""Select a multi-source skill prior by leave-one-development-task-out replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import run_classical_transfer_baselines as classical
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]


def policy_id(policy: dict[str, float]) -> str:
    return f"skill_prior_{policy['mass_start']:.2f}_{policy['mass_end']:.2f}"


def paired_development_comparison(
    rows: list[dict[str, Any]],
    challenger: str,
) -> dict[str, Any]:
    indexed = {
        (str(row["development_target"]), str(row["mode"]), int(row["seed"])): row
        for row in rows
    }
    keys = sorted(
        (task, seed)
        for task, mode, seed in indexed
        if mode == challenger and (task, "target_gp_ucb", seed) in indexed
    )
    output: dict[str, Any] = {"fold_seed_count": len(keys)}
    for field in ("final_best", "best_so_far_auc", "top10_hit"):
        deltas = np.asarray(
            [
                float(indexed[(task, challenger, seed)][field])
                - float(indexed[(task, "target_gp_ucb", seed)][field])
                for task, seed in keys
            ],
            dtype=np.float64,
        )
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        half_width = 1.96 * delta_std / math.sqrt(len(deltas))
        output[field] = {
            "mean_delta": round(delta_mean, 6),
            "normal_95ci_low": round(delta_mean - half_width, 6),
            "normal_95ci_high": round(delta_mean + half_width, 6),
            "win_rate": round(float(np.mean(deltas > 0.0)), 6),
            "non_loss_rate": round(float(np.mean(deltas >= 0.0)), 6),
        }
    return output


def select_route(
    comparisons: dict[str, Any],
    metric: str,
    required_ci_low: float,
) -> str:
    """Admit source transfer only when development evidence clears the gate."""

    eligible = [
        mode
        for mode in comparisons
        if mode != "target_gp_ucb"
        and comparisons[mode][metric]["normal_95ci_low"] > required_ci_low
    ]
    if not eligible:
        return "target_gp_ucb"
    return max(
        eligible,
        key=lambda mode: (
            comparisons[mode][metric]["normal_95ci_low"],
            comparisons[mode][metric]["mean_delta"],
            comparisons[mode]["final_best"]["mean_delta"],
        ),
    )


def route_record(mode: str, policies: list[dict[str, float]]) -> dict[str, Any]:
    if mode.startswith("skill_prior_"):
        policy = next(policy for policy in policies if policy_id(policy) == mode)
        return {"mode": "multisource_skill_prior", "skill_prior": policy}
    return {"mode": mode}


def calibrate(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    multisource.validate_new_task_protocol(config)
    protocol = config["protocol"]
    kernel = protocol["kernel"]
    development_tasks = tuple(protocol["development_task_ids"])
    policies = [dict(item) for item in protocol["skill_prior_candidates"]]
    initial = int(protocol["development_initial_observations"])
    rounds = int(protocol["development_reveal_rounds"])
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for development_target in development_tasks:
        source_ids = tuple(
            dataset_id
            for dataset_id in development_tasks
            if dataset_id != development_target
        )
        sources = [replay.DATASET_BUILDERS[dataset_id]() for dataset_id in source_ids]
        target = replay.DATASET_BUILDERS[development_target]()
        source_posteriors = []
        for source in sources:
            classical.validate_compatible_spaces(source, target)
            observations = transfer.source_observations(
                source,
                int(protocol["source_seed"]),
                len(source.candidates),
            )
            source_posteriors.append(
                classical.build_source_posterior(
                    source,
                    target,
                    observations,
                    int(protocol["source_inducing_limit"]),
                    float(kernel["numeric_length_scale"]),
                    float(kernel["categorical_length_scale"]),
                    float(kernel["gp_noise"]),
                )
            )
        start = int(protocol["development_seed_start"])
        count = int(protocol["development_seed_count"])
        for seed in range(start, start + count):
            baseline, _audit = multisource.run_seed(
                sources,
                target,
                source_posteriors,
                seed,
                initial,
                rounds,
                "target_gp_ucb",
                float(kernel["gp_beta"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
                int(protocol["rgpe_draws"]),
                tuple(float(value) for value in protocol["multitask_rho_grid"]),
                float(protocol["bma_temperature"]),
            )
            baseline["development_target"] = development_target
            rows.append(baseline)
            for mode in ("multisource_rgpe", "multisource_icm_bma"):
                metrics, _audit = multisource.run_seed(
                    sources,
                    target,
                    source_posteriors,
                    seed,
                    initial,
                    rounds,
                    mode,
                    float(kernel["gp_beta"]),
                    float(kernel["numeric_length_scale"]),
                    float(kernel["categorical_length_scale"]),
                    float(kernel["gp_noise"]),
                    int(protocol["rgpe_draws"]),
                    tuple(float(value) for value in protocol["multitask_rho_grid"]),
                    float(protocol["bma_temperature"]),
                )
                metrics["development_target"] = development_target
                rows.append(metrics)
            for policy in policies:
                metrics, _audit = multisource.run_seed(
                    sources,
                    target,
                    source_posteriors,
                    seed,
                    initial,
                    rounds,
                    "multisource_skill_prior",
                    float(kernel["gp_beta"]),
                    float(kernel["numeric_length_scale"]),
                    float(kernel["categorical_length_scale"]),
                    float(kernel["gp_noise"]),
                    int(protocol["rgpe_draws"]),
                    tuple(float(value) for value in protocol["multitask_rho_grid"]),
                    float(protocol["bma_temperature"]),
                    float(policy["mass_start"]),
                    float(policy["mass_end"]),
                )
                metrics["mode"] = policy_id(policy)
                metrics["development_target"] = development_target
                rows.append(metrics)
    candidate_modes = [
        "target_gp_ucb",
        "multisource_rgpe",
        "multisource_icm_bma",
        *(policy_id(policy) for policy in policies),
    ]
    comparisons = {
        mode: paired_development_comparison(rows, mode) for mode in candidate_modes
    }
    selection_metric = str(protocol["selection_metric"])
    required_ci_low = float(protocol["selection_required_ci_low"])
    selected_mode = select_route(comparisons, selection_metric, required_ci_low)
    skill_modes = [policy_id(policy) for policy in policies]
    selected_skill_mode = max(
        skill_modes,
        key=lambda mode: (
            comparisons[mode][selection_metric]["normal_95ci_low"],
            comparisons[mode][selection_metric]["mean_delta"],
        ),
    )
    selected_policy = next(
        policy for policy in policies if policy_id(policy) == selected_skill_mode
    )
    record = {
        "schema_version": "care.multisource_skill_selection/v1",
        "selection_scope": "leave_one_development_task_out_only",
        "config_fingerprint": multisource.config_fingerprint(config),
        "development_task_ids": list(development_tasks),
        "evaluation_task_ids_not_loaded": list(protocol["evaluation_task_ids"]),
        "candidate_comparisons": comparisons,
        "selection_metric": selection_metric,
        "selection_required_ci_low": required_ci_low,
        "selected_mode": selected_mode,
        "selected_route": route_record(selected_mode, policies),
        "selected_skill_mode": selected_skill_mode,
        "selected_skill_prior": selected_policy,
        "target_task_calibration": False,
    }
    with (output_dir / "development_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "selection_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "multisource_new_task_benchmark_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(calibrate(config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
