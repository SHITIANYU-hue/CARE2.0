#!/usr/bin/env python3
"""Calibrate and hold out LLM kernel skills with executable source priors."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

import run_calibrated_frozen_llm_selector as selector
import run_llm_kernel_online_bma as bma
import run_llm_kernel_skill_evolution as evolution
import run_llm_transfer_router as router
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


def prior_role_map_for(source_dataset: str, target_dataset: str) -> dict[str, str]:
    pair = (source_dataset, target_dataset)
    allowed_pairs = transfer.VALUE_PRIOR_FIELDS.get(pair, set())
    available_roles = transfer.descriptor_transfer_role_map_for(
        source_dataset,
        target_dataset,
    )
    return {
        source_field: target_field
        for source_field, target_field in available_roles.items()
        if (source_field, target_field) in allowed_pairs
    }


def run_prior_patch_policy(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    source_observed: list[replay.Candidate],
    card: transfer.TransferCard,
    patch: evolution.KernelSkillPatch,
    normalize: bool,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    prior_card: transfer.TransferCard | None = None,
    enable_identity_cold_start: bool = True,
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
    top10 = {
        candidate.candidate_id
        for candidate in sorted(
            pool,
            key=lambda candidate: candidate.objective_value,
            reverse=True,
        )[:10]
    }
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    source_prior, source_prior_diagnostics = router.aligned_source_prior(
        source_observed,
        target_adapter,
        prior_card or card,
        patch,
    )
    exact_match_candidate_ids = set(
        source_prior_diagnostics.pop("_exact_match_candidate_ids", [])
    )
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    mode = evolution.patch_mode(patch)
    for round_index in range(task.reveal_budget):
        current_beta = weighted.scheduled_gp_beta(
            patch.gp_beta,
            patch.gp_beta_end,
            round_index,
            task.reveal_budget,
        )
        base_scores, kernel_evidence, kernel_diagnostics = bma.score_scale_ensemble(
            target_adapter,
            observed_ids,
            observed,
            features_by_id,
            card,
            patch.scales,
            patch.role_multipliers,
            current_beta,
            normalize,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        prior_adjustments, prior_calibration = router.calibrated_prior_adjustments(
            source_prior,
            observed,
            patch,
        )
        cold_start_diagnostics: dict[str, Any] = {"active": False}
        if enable_identity_cold_start and not prior_calibration.get("active"):
            cold_start_adjustments, cold_start_diagnostics = (
                router.exact_identity_cold_start_adjustments(
                    source_prior,
                    exact_match_candidate_ids,
                    observed,
                    patch,
                )
            )
            if cold_start_diagnostics.get("active"):
                prior_adjustments = cold_start_adjustments
        scores = {
            candidate_id: score + prior_adjustments.get(candidate_id, 0.0)
            for candidate_id, score in base_scores.items()
        }
        selected_id = replay.top_candidate(scores)
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
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected_id,
                "selected_score": round(scores[selected_id], 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "hypothesis_snapshot": {
                    "llm_patch": asdict(patch),
                    "source_prior": source_prior_diagnostics,
                    "prior_calibration": prior_calibration,
                    "identity_cold_start": cold_start_diagnostics,
                    "kernel_log_marginal_likelihood": round(kernel_evidence, 6),
                    "kernel_diagnostics": kernel_diagnostics,
                    "gp_beta": round(current_beta, 6),
                    "prior_applied": bool(
                        prior_calibration.get("active")
                        or cold_start_diagnostics.get("active")
                    ),
                    "evidence_boundary": (
                        "The LLM patch is frozen. The source prior is calibrated only from target "
                        "outcomes revealed before this selection; the current target is hidden."
                    ),
                },
            }
        )
    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": target_adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit


def evaluate_seed_with_priors(
    seed: int,
    split: str,
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
    enable_identity_cold_start: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    rows, audits = selector.evaluate_seed(
        seed=seed,
        split=split,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        patches=(),
        fixed_scales=fixed_scales,
        source_observation_count=source_observation_count,
        discount=discount,
        min_source_support=min_source_support,
        initial=initial,
        rounds=rounds,
        normalize=normalize,
        gp_beta=gp_beta,
        gp_xi=gp_xi,
        numeric_length_scale=numeric_length_scale,
        categorical_length_scale=categorical_length_scale,
        gp_noise=gp_noise,
    )
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    source_observed = transfer.source_observations(
        source_adapter,
        seed,
        source_observation_count,
    )
    card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        source_observed,
        transfer.role_map_for(source_dataset, target_dataset),
        discount,
        min_source_support,
    )
    prior_card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        source_observed,
        prior_role_map_for(source_dataset, target_dataset),
        discount,
        min_source_support,
    )
    for patch in patches:
        metrics, audit = run_prior_patch_policy(
            source_adapter,
            target_adapter,
            task,
            seed,
            source_observed,
            card,
            patch,
            normalize,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            prior_card,
            enable_identity_cold_start,
        )
        metrics["split"] = split
        metrics["selected_source_mode"] = ""
        for event in audit:
            event["split"] = split
        rows.append(metrics)
        audits[(evolution.patch_mode(patch), seed)] = audit
    return rows, audits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate LLM kernel skills with executable, target-calibrated source priors."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--calibration-seed-start", type=int, default=500)
    parser.add_argument("--calibration-seeds", type=int, default=50)
    parser.add_argument("--heldout-seed-start", type=int, default=700)
    parser.add_argument("--heldout-seeds", type=int, default=50)
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
    parser.add_argument("--min-auc-gain", type=float, default=0.0)
    parser.add_argument("--max-final-loss", type=float, default=0.0)
    parser.add_argument("--min-risk-adjusted-composite-gain", type=float, default=0.0)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.6)
    parser.add_argument("--min-final-non-loss-rate", type=float, default=0.55)
    parser.add_argument("--risk-se-multiplier", type=float, default=0.5)
    parser.add_argument(
        "--preselected-patch-id",
        default="",
        help="Freeze one development-selected LLM patch; calibration then selects only the target anchor.",
    )
    parser.add_argument("--enable-identity-cold-start", action="store_true")
    parser.add_argument("--disable-source-prior", action="store_true")
    parser.add_argument("--disable-kernel-transfer", action="store_true")
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    calibration_seeds = set(range(
        args.calibration_seed_start,
        args.calibration_seed_start + args.calibration_seeds,
    ))
    heldout_seeds = set(range(
        args.heldout_seed_start,
        args.heldout_seed_start + args.heldout_seeds,
    ))
    if calibration_seeds & heldout_seeds:
        parser.error("Calibration and held-out seed ranges must not overlap.")
    llm_record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    target_adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    patches = evolution.normalize_patches(
        {"patches": llm_record.get("normalized_patches", [])},
        target_adapter.decision_columns,
        max_patches=100,
    )
    if not patches:
        raise RuntimeError("The LLM record contains no usable frozen patches.")
    if args.preselected_patch_id:
        patches = tuple(
            patch for patch in patches
            if patch.patch_id == args.preselected_patch_id
        )
        if not patches:
            raise ValueError(
                f"Preselected patch {args.preselected_patch_id!r} is not in the LLM record."
            )
    if args.disable_source_prior:
        patches = tuple(
            replace(
                patch,
                source_prior_strength=0.0,
                calibration_mode="off",
            )
            for patch in patches
        )
    if args.disable_kernel_transfer:
        patches = tuple(replace(patch, scales=(0.0,)) for patch in patches)
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
        "enable_identity_cold_start": args.enable_identity_cold_start,
    }
    jobs = [
        *((seed, "calibration") for seed in sorted(calibration_seeds)),
        *((seed, "heldout") for seed in sorted(heldout_seeds)),
    ]
    if args.workers == 1:
        results = [
            evaluate_seed_with_priors(seed=seed, split=split, **worker_args)
            for seed, split in jobs
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    evaluate_seed_with_priors,
                    seed=seed,
                    split=split,
                    **worker_args,
                )
                for seed, split in jobs
            ]
            results = [future.result() for future in futures]
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed_rows, seed_audits in results:
        rows.extend(seed_rows)
        audits.update(seed_audits)
    selected_mode, selection = selector.select_frozen_policy(
        rows,
        calibration_seeds,
        patches,
        args.min_auc_gain,
        args.max_final_loss,
        args.min_risk_adjusted_composite_gain,
        args.calibration_folds,
        args.min_positive_fold_rate,
        args.min_final_non_loss_rate,
        args.risk_se_multiplier,
    )
    if args.preselected_patch_id:
        calibration_selected_mode = selected_mode
        selected_mode = evolution.patch_mode(patches[0])
        selection["calibration_selected_mode"] = calibration_selected_mode
        selection["selected_mode"] = selected_mode
        selection["selected_llm_patch"] = True
        selection["preselected_patch_id"] = args.preselected_patch_id
        selection["rule"] = (
            "The LLM patch was selected from prior development runs and frozen before this "
            "confirmation seed range. Calibration seeds select only the target-only anchor; "
            "they cannot change or reject the preselected patch."
        )
    selector.add_selector_alias(rows, audits, selected_mode, heldout_seeds, selection)
    all_modes = (
        *selector.TARGET_MODES,
        *(evolution.patch_mode(patch) for patch in patches),
        selector.SELECTOR_MODE,
    )
    calibration = selector.mode_means(rows, all_modes, calibration_seeds)
    heldout = selector.mode_means(rows, all_modes, heldout_seeds)
    heldout_pairwise = {
        baseline: {
            field: selector.delta_summary(
                selector.paired_deltas(
                    rows,
                    selector.SELECTOR_MODE,
                    baseline,
                    heldout_seeds,
                    field,
                )
            )
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for baseline in selector.TARGET_MODES
    }
    anchor_heldout = heldout_pairwise[selection["target_anchor_mode"]]
    heldout_validation = {
        "target_anchor_mode": selection["target_anchor_mode"],
        "selected_llm_patch": bool(selection["selected_llm_patch"]),
        "mean_final_and_auc_positive": (
            anchor_heldout["final_best"]["mean"] > 0.0
            and anchor_heldout["best_so_far_auc"]["mean"] > 0.0
        ),
        "at_least_one_metric_95ci_positive": (
            anchor_heldout["final_best"]["normal_95ci_low"] > 0.0
            or anchor_heldout["best_so_far_auc"]["normal_95ci_low"] > 0.0
        ),
        "top10_hit_95ci_positive": (
            anchor_heldout["top10_hit"]["normal_95ci_low"] > 0.0
        ),
    }
    heldout_validation["confirmed_llm_gain"] = bool(
        heldout_validation["selected_llm_patch"]
        and heldout_validation["mean_final_and_auc_positive"]
        and heldout_validation["at_least_one_metric_95ci_positive"]
    )
    heldout_validation["confirmed_top10_gain"] = bool(
        heldout_validation["selected_llm_patch"]
        and heldout_validation["top10_hit_95ci_positive"]
    )
    heldout_validation["confirmed_any_metric_gain"] = bool(
        heldout_validation["confirmed_llm_gain"]
        or heldout_validation["confirmed_top10_gain"]
    )
    summary = {
        "experiment": "care_calibrated_llm_kernel_plus_source_prior_selector",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "llm_model": llm_record.get("model"),
        "kernel_role_map": transfer.role_map_for(
            args.source_dataset,
            args.target_dataset,
        ),
        "source_prior_role_map": prior_role_map_for(
            args.source_dataset,
            args.target_dataset,
        ),
        "llm_generation_call_count": int(llm_record.get("llm_generation_call_count", 1)),
        "new_llm_call_count": 0,
        "patch_execution": (
            "target_only_llm_schedule"
            if args.disable_kernel_transfer
            else (
                "llm_transfer_kernel_only"
                if args.disable_source_prior
                else "kernel_plus_target_calibrated_source_prior"
            )
        ),
        "identity_cold_start_enabled": args.enable_identity_cold_start,
        "source_prior_enabled": not args.disable_source_prior,
        "kernel_transfer_enabled": not args.disable_kernel_transfer,
        "preselected_patch_id": args.preselected_patch_id or None,
        "patches": [asdict(patch) for patch in patches],
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "selection": selection,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_strongest_target_mode_descriptive_only": max(
            selector.TARGET_MODES,
            key=lambda mode: (heldout[mode]["composite"], mode),
        ),
        "heldout_pairwise": heldout_pairwise,
        "heldout_validation": heldout_validation,
        "evidence_boundary": (
            "The LLM patches are frozen before replay. Calibration seeds select one patch or a "
            "target-only fallback. Held-out outcomes never affect selection. Within a replay, "
            "source-prior calibration uses only target outcomes revealed before that round."
        ),
    }
    output_id = (
        f"calibrated_llm_prior_{args.source_dataset}_to_{args.target_dataset}"
    )
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    selector.write_outputs(output_id, rows, audits, summary, llm_record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
