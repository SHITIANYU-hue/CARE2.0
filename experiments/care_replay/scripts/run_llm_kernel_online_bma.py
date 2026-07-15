#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import run_frozen_llm_kernel_patch as frozen
import run_llm_kernel_skill_evolution as evolution
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


def log_mean_exp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values) / len(values))


def normalized_weights(log_weights: dict[str, float]) -> dict[str, float]:
    peak = max(log_weights.values())
    unnormalized = {
        name: math.exp(value - peak)
        for name, value in log_weights.items()
    }
    total = sum(unnormalized.values())
    return {name: value / total for name, value in unnormalized.items()}


def normal_logpdf(value: float, mean_value: float, std_value: float) -> float:
    std = max(float(std_value), 0.05)
    z = (float(value) - float(mean_value)) / std
    return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


def score_scale_ensemble(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, surrogate.FeatureRecord],
    card: transfer.TransferCard,
    scales: tuple[float, ...],
    role_multipliers: dict[str, float],
    gp_beta: float,
    normalize: bool,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    include_posterior_maps: bool = False,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    scale_weights = []
    for scale in scales:
        categorical_weights, weight_diagnostics = weighted.transfer_categorical_weights(
            adapter,
            card,
            scale,
            normalize,
            role_multipliers,
        )
        scale_weights.append((scale, categorical_weights, weight_diagnostics))
    if len(scale_weights) == 1:
        scores, gp_diagnostics = weighted.weighted_gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            scale_weights[0][1],
            gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            include_posterior_maps,
        )
        diagnostics = {
            "scales": list(scales),
            "scale_diagnostics": [
                {
                    "scale": scales[0],
                    "weight_diagnostics": scale_weights[0][2],
                    "gp_diagnostics": gp_diagnostics,
                }
            ],
        }
        if include_posterior_maps:
            diagnostics["posterior_mean_by_id"] = gp_diagnostics["posterior_mean_by_id"]
            diagnostics["posterior_std_by_id"] = gp_diagnostics["posterior_std_by_id"]
        return scores, float(gp_diagnostics["log_marginal_likelihood"]), diagnostics
    scores, diagnostics = weighted.ensemble_weighted_gp_scores(
        adapter,
        observed_ids,
        observed,
        features_by_id,
        scale_weights,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
        include_posterior_maps,
    )
    component_evidence = [
        float(item["gp_diagnostics"]["log_marginal_likelihood"])
        for item in diagnostics["scale_diagnostics"]
    ]
    return scores, log_mean_exp(component_evidence), diagnostics


def expert_priors(
    patches: tuple[evolution.KernelSkillPatch, ...],
    gp_prior_mass: float,
    fixed_prior_mass: float,
) -> dict[str, float]:
    llm_mass = max(0.0, 1.0 - gp_prior_mass - fixed_prior_mass)
    confidence_total = sum(max(patch.confidence, 0.05) for patch in patches)
    priors = {
        "gp_ucb": gp_prior_mass,
        "fixed_transfer_ensemble": fixed_prior_mass,
    }
    for patch in patches:
        priors[evolution.patch_mode(patch)] = llm_mass * max(patch.confidence, 0.05) / confidence_total
    return priors


def run_bma_policy(
    source_dataset: str,
    target_dataset: str,
    seed: int,
    patches: tuple[evolution.KernelSkillPatch, ...],
    fixed_scales: tuple[float, ...],
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    initial: int,
    rounds: int,
    normalize: bool,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    evidence_temperature: float,
    gp_prior_mass: float,
    fixed_prior_mass: float,
    evidence_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(adapter, initial, rounds)
    card = transfer.compile_transfer_card(
        source_adapter,
        adapter,
        transfer.source_observations(source_adapter, seed, source_observation_count),
        transfer.role_map_for(source_dataset, target_dataset),
        discount,
        min_source_support,
    )
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
        for candidate in pool
    }
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[:initial]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    priors = expert_priors(patches, gp_prior_mass, fixed_prior_mass)
    mode = f"llm_kernel_online_bma_t{weighted.scale_label(evidence_temperature)}"
    cumulative_predictive_evidence = {name: 0.0 for name in priors}
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []

    for round_index in range(rounds):
        expert_scores: dict[str, dict[str, float]] = {}
        expert_evidence: dict[str, float] = {}
        expert_diagnostics: dict[str, Any] = {}

        scores, evidence, diagnostics = score_scale_ensemble(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            card,
            (0.0,),
            {},
            gp_beta,
            normalize,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            evidence_mode == "prequential",
        )
        expert_scores["gp_ucb"] = scores
        expert_evidence["gp_ucb"] = evidence
        expert_diagnostics["gp_ucb"] = diagnostics

        scores, evidence, diagnostics = score_scale_ensemble(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            card,
            fixed_scales,
            {},
            gp_beta,
            normalize,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            evidence_mode == "prequential",
        )
        expert_scores["fixed_transfer_ensemble"] = scores
        expert_evidence["fixed_transfer_ensemble"] = evidence
        expert_diagnostics["fixed_transfer_ensemble"] = diagnostics

        for patch in patches:
            patch_name = evolution.patch_mode(patch)
            scores, evidence, diagnostics = score_scale_ensemble(
                adapter,
                observed_ids,
                observed,
                features_by_id,
                card,
                patch.scales,
                patch.role_multipliers,
                patch.gp_beta,
                normalize,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
                evidence_mode == "prequential",
            )
            expert_scores[patch_name] = scores
            expert_evidence[patch_name] = evidence
            expert_diagnostics[patch_name] = diagnostics

        if evidence_mode == "prequential":
            evidence_for_selection = cumulative_predictive_evidence
        else:
            evidence_for_selection = expert_evidence
        posterior = normalized_weights({
            name: math.log(max(priors[name], 1e-12))
            + evidence_temperature * evidence_for_selection[name]
            for name in expert_scores
        })
        ranked = {
            name: weighted.rank_normalized(scores)
            for name, scores in expert_scores.items()
        }
        combined_scores = {
            candidate_id: sum(
                posterior[name] * ranked[name][candidate_id]
                for name in ranked
            )
            for candidate_id in next(iter(ranked.values()))
        }
        selected_id = replay.top_candidate(combined_scores)
        selected = by_id[selected_id]
        predictive_log_scores: dict[str, float] = {}
        if evidence_mode == "prequential":
            observed_value = selected.objective_value / 100.0
            for name, diagnostics in expert_diagnostics.items():
                mean_by_id = diagnostics.get("posterior_mean_by_id", {})
                std_by_id = diagnostics.get("posterior_std_by_id", {})
                predictive_log_scores[name] = normal_logpdf(
                    observed_value,
                    mean_by_id.get(selected_id, observed_value),
                    std_by_id.get(selected_id, 0.05),
                )
                cumulative_predictive_evidence[name] += predictive_log_scores[name]
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
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected_id,
                "selected_score": round(combined_scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "hypothesis_snapshot": {
                    "base_acquisition": "mixed_kernel_gp_ucb",
                    "skill_optimization": "llm_kernel_portfolio_online_bayesian_model_average",
                    "expert_priors": priors,
                    "expert_log_marginal_likelihood": expert_evidence,
                    "expert_cumulative_predictive_evidence": cumulative_predictive_evidence,
                    "expert_predictive_log_score": predictive_log_scores,
                    "expert_posterior_weights": posterior,
                    "evidence_mode": evidence_mode,
                    "expert_diagnostics": {
                        name: {
                            key: value
                            for key, value in diagnostics.items()
                            if key not in {"posterior_mean_by_id", "posterior_std_by_id"}
                        }
                        for name, diagnostics in expert_diagnostics.items()
                    },
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
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    evidence_temperature: float,
    gp_prior_mass: float,
    fixed_prior_mass: float,
    evidence_mode: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    baseline_rows, baseline_audits = evolution.run_seed_evaluation(
        seed=seed,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_observation_count=source_observation_count,
        discount=discount,
        min_source_support=min_source_support,
        patches=(),
        fixed_scales=fixed_scales,
        normalize=normalize,
        initial=initial,
        rounds=rounds,
        gp_beta=gp_beta,
        numeric_length_scale=numeric_length_scale,
        categorical_length_scale=categorical_length_scale,
        gp_noise=gp_noise,
    )
    metrics, audit = run_bma_policy(
        source_dataset,
        target_dataset,
        seed,
        patches,
        fixed_scales,
        source_observation_count,
        discount,
        min_source_support,
        initial,
        rounds,
        normalize,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
        evidence_temperature,
        gp_prior_mass,
        fixed_prior_mass,
        evidence_mode,
    )
    baseline_rows.append(metrics)
    baseline_audits[(metrics["mode"], seed)] = audit
    return baseline_rows, baseline_audits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online Bayesian averaging over LLM-generated CARE kernel skills.")
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--patch-ids", default="", help="Optional comma-separated subset of frozen LLM patch ids.")
    parser.add_argument("--source-dataset", default="real_suzuki_miyaura", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seed-start", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=96)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--fixed-ensemble-scales", default="0.5,1,1.5,2,3,4")
    parser.add_argument("--evidence-temperature", type=float, default=1.0)
    parser.add_argument(
        "--evidence-mode",
        choices=("prequential", "marginal_likelihood"),
        default="prequential",
        help="How revealed target outcomes update expert weights.",
    )
    parser.add_argument("--gp-prior-mass", type=float, default=0.30)
    parser.add_argument("--fixed-prior-mass", type=float, default=0.40)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    if args.gp_prior_mass <= 0 or args.fixed_prior_mass <= 0 or args.gp_prior_mass + args.fixed_prior_mass >= 1:
        parser.error("GP and fixed prior masses must be positive and sum to less than 1.")

    target_adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    patches = evolution.normalize_patches(
        {"patches": record.get("normalized_patches", [])},
        target_adapter.decision_columns,
        max_patches=100,
    )
    if args.patch_ids:
        requested_patch_ids = {item.strip() for item in args.patch_ids.split(",") if item.strip()}
        patches = tuple(patch for patch in patches if patch.patch_id in requested_patch_ids)
        found_patch_ids = {patch.patch_id for patch in patches}
        missing_patch_ids = sorted(requested_patch_ids - found_patch_ids)
        if missing_patch_ids:
            raise ValueError(f"Unknown patch ids: {', '.join(missing_patch_ids)}")
    if not patches:
        raise RuntimeError("No normalized LLM patches found in the record.")
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
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
        "evidence_temperature": args.evidence_temperature,
        "gp_prior_mass": args.gp_prior_mass,
        "fixed_prior_mass": args.fixed_prior_mass,
        "evidence_mode": args.evidence_mode,
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

    bma_mode = f"llm_kernel_online_bma_t{weighted.scale_label(args.evidence_temperature)}"
    fixed_mode = f"fixed_scale_ensemble_{weighted.scales_label(fixed_scales)}"
    seed_set = set(seed_values)
    summary = {
        "experiment": "care_llm_kernel_online_bayesian_model_average",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "llm_model": record.get("model"),
        "llm_patch_count": len(patches),
        "evidence_temperature": args.evidence_temperature,
        "evidence_mode": args.evidence_mode,
        "prior_mass": {
            "gp_ucb": args.gp_prior_mass,
            "fixed_transfer_ensemble": args.fixed_prior_mass,
            "llm_portfolio": 1.0 - args.gp_prior_mass - args.fixed_prior_mass,
        },
        "evidence_boundary": (
            "LLM patches are frozen before this run. Online expert weights use only revealed target observations."
        ),
        "aggregate": evolution.summarize(rows, seed_set),
        "paired_vs_gp_ucb": frozen.paired_comparison(rows, bma_mode, "gp_ucb"),
        "paired_vs_fixed_ensemble": frozen.paired_comparison(rows, bma_mode, fixed_mode),
    }
    output_id = f"llm_kernel_online_bma_{args.source_dataset}_to_{args.target_dataset}_seed{args.seed_start}_{args.seeds}seed"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    evolution.write_outputs(output_id, rows, summary, audits, record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
