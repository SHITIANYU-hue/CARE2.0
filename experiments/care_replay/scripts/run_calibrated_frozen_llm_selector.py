#!/usr/bin/env python3
"""Select a frozen LLM skill on calibration seeds and confirm it held out."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_llm_kernel_skill_evolution as evolution
import run_llm_transfer_router as router
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
TARGET_MODES = ("gp_ucb", "mixed_kernel_gp_ei", "target_acquisition_portfolio")
SELECTOR_MODE = "llm_calibrated_selector"


def paired_deltas(
    rows: list[dict[str, Any]],
    challenger: str,
    baseline: str,
    seeds: set[int],
    field: str,
) -> list[float]:
    by_key = {(str(row["mode"]), int(row["seed"])): row for row in rows}
    return [
        float(by_key[(challenger, seed)][field])
        - float(by_key[(baseline, seed)][field])
        for seed in sorted(seeds)
        if (challenger, seed) in by_key and (baseline, seed) in by_key
    ]


def delta_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "normal_95ci_low": 0.0,
            "normal_95ci_high": 0.0,
            "win_rate": 0.0,
            "non_loss_rate": 0.0,
        }
    average = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    se = std / math.sqrt(len(values))
    return {
        "n": float(len(values)),
        "mean": round(average, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "normal_95ci_low": round(average - 1.96 * se, 6),
        "normal_95ci_high": round(average + 1.96 * se, 6),
        "win_rate": round(sum(value > 0 for value in values) / len(values), 6),
        "non_loss_rate": round(sum(value >= 0 for value in values) / len(values), 6),
    }


def positive_fold_rate(values: list[float], fold_count: int) -> tuple[float, list[float]]:
    if not values:
        return 0.0, []
    fold_count = max(1, min(fold_count, len(values)))
    folds = [values[index::fold_count] for index in range(fold_count)]
    fold_means = [mean(fold) for fold in folds if fold]
    return (
        sum(value > 0.0 for value in fold_means) / len(fold_means),
        [round(value, 6) for value in fold_means],
    )


def mode_means(
    rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    seeds: set[int],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for mode in modes:
        selected = [
            row for row in rows
            if str(row["mode"]) == mode and int(row["seed"]) in seeds
        ]
        if not selected:
            continue
        output[mode] = {
            field: round(mean(float(row[field]) for row in selected), 6)
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        output[mode]["composite"] = round(
            output[mode]["final_best"] + output[mode]["best_so_far_auc"],
            6,
        )
    return output


def select_frozen_policy(
    rows: list[dict[str, Any]],
    calibration_seeds: set[int],
    patches: tuple[evolution.KernelSkillPatch, ...],
    min_auc_gain: float = 0.0,
    max_final_loss: float = 0.0,
    min_risk_adjusted_composite_gain: float = 0.25,
    calibration_folds: int = 5,
    min_positive_fold_rate: float = 0.8,
    min_final_non_loss_rate: float = 0.55,
    risk_se_multiplier: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    patch_modes = tuple(evolution.patch_mode(patch) for patch in patches)
    target_summary = mode_means(rows, TARGET_MODES, calibration_seeds)
    if set(target_summary) != set(TARGET_MODES):
        raise RuntimeError("Calibration rows do not contain every target-only anchor.")
    anchor_mode = max(
        TARGET_MODES,
        key=lambda mode: (target_summary[mode]["composite"], mode),
    )
    diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for mode in patch_modes:
        final_values = paired_deltas(rows, mode, anchor_mode, calibration_seeds, "final_best")
        auc_values = paired_deltas(rows, mode, anchor_mode, calibration_seeds, "best_so_far_auc")
        composite_values = [
            final_delta + auc_delta
            for final_delta, auc_delta in zip(final_values, auc_values)
        ]
        final_stats = delta_summary(final_values)
        auc_stats = delta_summary(auc_values)
        composite_stats = delta_summary(composite_values)
        fold_positive_rate, fold_composite_means = positive_fold_rate(
            composite_values,
            calibration_folds,
        )
        risk_adjusted_gain = (
            composite_stats["mean"]
            - risk_se_multiplier * composite_stats["se"]
        )
        is_eligible = (
            final_stats["mean"] >= -max_final_loss
            and auc_stats["mean"] >= min_auc_gain
            and risk_adjusted_gain >= min_risk_adjusted_composite_gain
            and fold_positive_rate >= min_positive_fold_rate
            and final_stats["non_loss_rate"] >= min_final_non_loss_rate
        )
        if is_eligible:
            eligible.append(mode)
        diagnostics[mode] = {
            "final_best": final_stats,
            "best_so_far_auc": auc_stats,
            "composite": composite_stats,
            "fold_composite_means": fold_composite_means,
            "positive_fold_rate": round(fold_positive_rate, 6),
            "risk_adjusted_composite_gain": round(risk_adjusted_gain, 6),
            "eligible": is_eligible,
        }

    selected_mode = max(
        eligible,
        key=lambda mode: (
            diagnostics[mode]["risk_adjusted_composite_gain"],
            diagnostics[mode]["best_so_far_auc"]["mean"],
            diagnostics[mode]["final_best"]["mean"],
            mode,
        ),
    ) if eligible else anchor_mode
    return selected_mode, {
        "rule": (
            "Choose the best target-only anchor on calibration composite (final_best + AUC). "
            "Authorize an LLM patch only when paired mean final and AUC are non-negative, final "
            "non-loss rate is sufficiently high, the composite gain is positive in enough "
            "calibration folds, and the risk-adjusted composite gain clears its threshold."
        ),
        "calibration_seed_count": len(calibration_seeds),
        "target_anchor_mode": anchor_mode,
        "target_anchor_summary": target_summary,
        "eligible_patch_modes": sorted(eligible),
        "selected_mode": selected_mode,
        "selected_llm_patch": selected_mode in patch_modes,
        "thresholds": {
            "min_auc_gain": min_auc_gain,
            "max_final_loss": max_final_loss,
            "min_risk_adjusted_composite_gain": min_risk_adjusted_composite_gain,
            "calibration_folds": calibration_folds,
            "min_positive_fold_rate": min_positive_fold_rate,
            "min_final_non_loss_rate": min_final_non_loss_rate,
            "risk_se_multiplier": risk_se_multiplier,
        },
        "patch_diagnostics": diagnostics,
    }


def evaluate_seed(
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
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    rows, audits = evolution.run_seed_evaluation(
        seed,
        source_dataset,
        target_dataset,
        source_observation_count,
        discount,
        min_source_support,
        patches,
        fixed_scales,
        normalize,
        initial,
        rounds,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
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
    audits[("mixed_kernel_gp_ei", seed)] = ei_audit
    portfolio_metrics, portfolio_audit = router.run_target_portfolio(
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
    audits[("target_acquisition_portfolio", seed)] = portfolio_audit
    for row in rows:
        row["split"] = split
        row["selected_source_mode"] = ""
    for audit_rows in audits.values():
        for event in audit_rows:
            event["split"] = split
    return rows, audits


def add_selector_alias(
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    selected_mode: str,
    heldout_seeds: set[int],
    selection: dict[str, Any],
) -> None:
    selected_rows = [
        row for row in rows
        if str(row["mode"]) == selected_mode and int(row["seed"]) in heldout_seeds
    ]
    for source_row in selected_rows:
        row = dict(source_row)
        row["mode"] = SELECTOR_MODE
        row["selected_source_mode"] = selected_mode
        rows.append(row)
        seed = int(row["seed"])
        source_audit = audits[(selected_mode, seed)]
        selector_audit: list[dict[str, Any]] = []
        for source_event in source_audit:
            event = deepcopy(source_event)
            event["mode"] = SELECTOR_MODE
            event["selector_trace"] = {
                "selected_source_mode": selected_mode,
                "selected_llm_patch": bool(selection["selected_llm_patch"]),
                "target_anchor_mode": selection["target_anchor_mode"],
                "selection_rule": selection["rule"],
                "evidence_boundary": (
                    "The selector was frozen using calibration seeds only. This held-out seed "
                    "did not affect target-anchor or LLM-skill selection."
                ),
            }
            selector_audit.append(event)
        audits[(SELECTOR_MODE, seed)] = selector_audit


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    summary: dict[str, Any],
    llm_record: dict[str, Any],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["split"]), int(row["seed"]), str(row["mode"])))
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_RUNS / f"{output_id}_llm_record.json").write_text(
        json.dumps(llm_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for (mode, seed), events in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a frozen LLM skill portfolio and confirm it on independent seeds."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--calibration-seed-start", type=int, default=100)
    parser.add_argument("--calibration-seeds", type=int, default=20)
    parser.add_argument("--heldout-seed-start", type=int, default=200)
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
    parser.add_argument("--min-risk-adjusted-composite-gain", type=float, default=0.25)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.8)
    parser.add_argument("--min-final-non-loss-rate", type=float, default=0.55)
    parser.add_argument("--risk-se-multiplier", type=float, default=1.0)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    calibration_seed_values = set(range(
        args.calibration_seed_start,
        args.calibration_seed_start + args.calibration_seeds,
    ))
    heldout_seed_values = set(range(
        args.heldout_seed_start,
        args.heldout_seed_start + args.heldout_seeds,
    ))
    if calibration_seed_values & heldout_seed_values:
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
    jobs = [
        (seed, "calibration") for seed in sorted(calibration_seed_values)
    ] + [
        (seed, "heldout") for seed in sorted(heldout_seed_values)
    ]
    if args.workers == 1:
        results = [evaluate_seed(seed=seed, split=split, **worker_args) for seed, split in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_seed, seed=seed, split=split, **worker_args)
                for seed, split in jobs
            ]
            results = [future.result() for future in futures]
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed_rows, seed_audits in results:
        rows.extend(seed_rows)
        audits.update(seed_audits)

    selected_mode, selection = select_frozen_policy(
        rows,
        calibration_seed_values,
        patches,
        args.min_auc_gain,
        args.max_final_loss,
        args.min_risk_adjusted_composite_gain,
        args.calibration_folds,
        args.min_positive_fold_rate,
        args.min_final_non_loss_rate,
        args.risk_se_multiplier,
    )
    add_selector_alias(rows, audits, selected_mode, heldout_seed_values, selection)
    all_modes = (*TARGET_MODES, *(evolution.patch_mode(patch) for patch in patches), SELECTOR_MODE)
    calibration = mode_means(rows, all_modes, calibration_seed_values)
    heldout = mode_means(rows, all_modes, heldout_seed_values)
    heldout_pairwise = {
        baseline: {
            field: delta_summary(
                paired_deltas(rows, SELECTOR_MODE, baseline, heldout_seed_values, field)
            )
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for baseline in TARGET_MODES
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
    heldout_strongest_target = max(
        TARGET_MODES,
        key=lambda mode: (heldout[mode]["composite"], mode),
    )
    summary = {
        "experiment": "care_calibrated_frozen_llm_selector",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "llm_model": llm_record.get("model"),
        "llm_generation_call_count": int(
            llm_record.get("llm_generation_call_count", 1)
        ),
        "new_llm_call_count": 0,
        "patches": [asdict(patch) for patch in patches],
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "selection": selection,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_strongest_target_mode_descriptive_only": heldout_strongest_target,
        "heldout_pairwise": heldout_pairwise,
        "heldout_validation": heldout_validation,
        "evidence_boundary": (
            "The LLM record was generated before this experiment from source evidence and public target schema. "
            "Only calibration seeds choose the target anchor or LLM patch. Held-out outcomes never affect selection."
        ),
    }
    output_id = f"calibrated_frozen_llm_{args.source_dataset}_to_{args.target_dataset}"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    write_outputs(output_id, rows, audits, summary, llm_record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
