#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_llm_kernel_skill_evolution as evolution
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


def paired_comparison(
    rows: list[dict[str, Any]],
    challenger: str,
    baseline: str,
) -> dict[str, Any]:
    by_mode_seed = {
        (str(row["mode"]), int(row["seed"])): row
        for row in rows
    }
    seeds = sorted(
        seed for mode, seed in by_mode_seed
        if mode == challenger and (baseline, seed) in by_mode_seed
    )
    out: dict[str, Any] = {"seed_count": len(seeds)}
    for field in ("final_best", "best_so_far_auc", "top10_hit"):
        deltas = [
            float(by_mode_seed[(challenger, seed)][field])
            - float(by_mode_seed[(baseline, seed)][field])
            for seed in seeds
        ]
        delta_mean = mean(deltas)
        delta_std = pstdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * delta_std / math.sqrt(len(deltas)) if deltas else 0.0
        out[field] = {
            "mean_delta": round(delta_mean, 4),
            "std_delta": round(delta_std, 4),
            "normal_95ci_low": round(delta_mean - half_width, 4),
            "normal_95ci_high": round(delta_mean + half_width, 4),
            "win_rate": round(sum(delta > 0 for delta in deltas) / len(deltas), 4),
            "non_loss_rate": round(sum(delta >= 0 for delta in deltas) / len(deltas), 4),
        }
    return out


def load_patch(
    record_path: Path,
    patch_id: str,
    target_fields: tuple[str, ...],
) -> tuple[evolution.KernelSkillPatch, dict[str, Any]]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    parsed = {"patches": record.get("normalized_patches", [])}
    patches = evolution.normalize_patches(parsed, target_fields, max_patches=100)
    for patch in patches:
        if patch.patch_id == patch_id:
            return patch, record
    available = ", ".join(patch.patch_id for patch in patches)
    raise ValueError(f"Patch {patch_id!r} not found. Available patches: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm one frozen LLM-generated CARE kernel patch on fresh seeds.")
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--patch-id", required=True)
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
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    target_adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    patch, llm_record = load_patch(args.llm_record, args.patch_id, target_adapter.decision_columns)
    fixed_scales = weighted.parse_scales(args.fixed_ensemble_scales)
    worker_args = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_observation_count": args.source_observations,
        "discount": args.discount,
        "min_source_support": args.min_source_support,
        "patches": (patch,),
        "fixed_scales": fixed_scales,
        "normalize": not args.no_normalize,
        "initial": args.initial,
        "rounds": args.rounds,
        "gp_beta": args.gp_beta,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
    }
    seed_values = list(range(args.seed_start, args.seed_start + args.seeds))
    if args.workers == 1:
        seed_results = [
            evolution.run_seed_evaluation(seed=seed, **worker_args)
            for seed in seed_values
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evolution.run_seed_evaluation, seed=seed, **worker_args)
                for seed in seed_values
            ]
            seed_results = [future.result() for future in futures]

    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed_rows, seed_audits in seed_results:
        rows.extend(seed_rows)
        audits.update(seed_audits)
    rows.sort(key=lambda row: (int(row["seed"]), str(row["mode"])))

    patch_mode = evolution.patch_mode(patch)
    fixed_mode = f"fixed_scale_ensemble_{weighted.scales_label(fixed_scales)}"
    seed_set = set(seed_values)
    summary = {
        "experiment": "care_frozen_llm_kernel_patch_confirmation",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "confirmation_seed_start": args.seed_start,
        "confirmation_seed_count": args.seeds,
        "frozen_patch": vars(patch),
        "patch_provenance": {
            "model": llm_record.get("model"),
            "usage": llm_record.get("usage", {}),
            "record_path": str(args.llm_record),
            "boundary": (
                "The patch was generated before this confirmation run and is not changed using seeds in this run. "
                "The LLM saw source transfer-card evidence and target public schema, not target outcomes."
            ),
        },
        "aggregate": evolution.summarize(rows, seed_set),
        "paired_vs_gp_ucb": paired_comparison(rows, patch_mode, "gp_ucb"),
        "paired_vs_fixed_ensemble": paired_comparison(rows, patch_mode, fixed_mode),
    }
    output_id = f"frozen_llm_kernel_{args.source_dataset}_to_{args.target_dataset}_{patch.patch_id}_seed{args.seed_start}_{args.seeds}seed"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    evolution.write_outputs(output_id, rows, summary, audits, llm_record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
