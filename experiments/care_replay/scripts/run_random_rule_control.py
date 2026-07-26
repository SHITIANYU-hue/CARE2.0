#!/usr/bin/env python3
"""Run a matched random-rule null against frozen LLM semantic skills.

The null keeps each skill's public field structure, rule count, rule arity, and
execution hyperparameters. It randomizes only condition values and rule signs,
then uses the same target-only semantic executor and the same calibration /
held-out seed protocol. A best random route is selected on calibration only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import llm_semantic_skills as semantic
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


def matched_random_skill(
    skill: semantic.SemanticSkill,
    catalog: dict[str, dict[str, int]],
    rng: random.Random,
    skill_id: str,
) -> semantic.SemanticSkill:
    """Randomize rule values/signs while preserving the LLM rule structure."""
    rules: list[semantic.SemanticRule] = []
    for rule_index, rule in enumerate(skill.rules):
        conditions = tuple(
            (field, rng.choice(tuple(catalog[field])))
            for field, _value in rule.conditions
        )
        magnitude = abs(float(rule.weight))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        rules.append(semantic.SemanticRule(
            rule_id=f"random_{rule_index + 1}",
            conditions=conditions,
            weight=sign * magnitude,
            rationale=(
                "Matched random null: original fields and rule arity preserved; "
                "condition values and sign randomized."
            ),
        ))
    return replace(
        skill,
        skill_id=skill_id,
        rules=tuple(rules),
        hypothesis=(
            "Matched random-rule null preserving the frozen LLM skill's public "
            "feature structure, rule count, and execution schedule."
        ),
    )


def paired_summary(
    rows: list[dict[str, Any]],
    mode: str,
    baseline: str,
    seeds: set[int],
) -> dict[str, float | int]:
    left = {
        int(row["seed"]): row
        for row in rows
        if str(row["mode"]) == mode and int(row["seed"]) in seeds
    }
    right = {
        int(row["seed"]): row
        for row in rows
        if str(row["mode"]) == baseline and int(row["seed"]) in seeds
    }
    common = sorted(set(left) & set(right))
    final = [
        float(left[seed]["final_best"]) - float(right[seed]["final_best"])
        for seed in common
    ]
    auc = [
        float(left[seed]["best_so_far_auc"]) - float(right[seed]["best_so_far_auc"])
        for seed in common
    ]
    composite = [x + y for x, y in zip(final, auc)]
    if not common:
        raise ValueError(f"No paired rows for {mode} versus {baseline}")
    return {
        "seed_count": len(common),
        "final_best_mean_delta": mean(final),
        "final_best_sd_delta": pstdev(final) if len(final) > 1 else 0.0,
        "final_best_win_rate": sum(value > 0.0 for value in final) / len(final),
        "auc_mean_delta": mean(auc),
        "auc_sd_delta": pstdev(auc) if len(auc) > 1 else 0.0,
        "composite_mean_delta": mean(composite),
        "composite_sd_delta": pstdev(composite) if len(composite) > 1 else 0.0,
    }


def select_best_random(
    rows: list[dict[str, Any]],
    modes: tuple[str, ...],
    baseline: str,
    calibration_seeds: set[int],
) -> tuple[str, dict[str, dict[str, float | int]]]:
    diagnostics = {
        mode: paired_summary(rows, mode, baseline, calibration_seeds)
        for mode in modes
    }
    selected = max(
        modes,
        key=lambda mode: (
            float(diagnostics[mode]["composite_mean_delta"]),
            mode,
        ),
    )
    return selected, diagnostics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_seed(
    seed: int,
    target_dataset: str,
    random_skills: tuple[semantic.SemanticSkill, ...],
    initial: int,
    rounds: int,
    calibration_seeds: set[int],
    include_random_warmstart: bool,
) -> list[dict[str, Any]]:
    """Evaluate one seed; safe to run in a process worker."""
    adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(adapter, initial, rounds)
    split = "calibration" if seed in calibration_seeds else "heldout"
    baseline_weights = (1.0,) * (len(adapter.decision_columns) + 1)
    baseline, _baseline_audit = weighted.run_policy(
        adapter,
        task,
        seed,
        "gp_ucb",
        baseline_weights,
        {"scale": 0.0, "weights": list(baseline_weights)},
        1.5,
        0.35,
        3.0,
        0.05,
    )
    baseline["split"] = split
    rows = [baseline]
    for skill in random_skills:
        metrics, _audit = semantic.run_semantic_skill(
            adapter,
            task,
            seed,
            skill,
            0.35,
            3.0,
            0.05,
        )
        metrics["split"] = split
        rows.append(metrics)
        if include_random_warmstart:
            warmstart, _warmstart_audit = semantic.run_llambo_warmstart(
                adapter,
                task,
                seed,
                skill,
                0.35,
                3.0,
                0.05,
            )
            warmstart["split"] = split
            rows.append(warmstart)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--calibration-seed-start", type=int, default=10000)
    parser.add_argument("--calibration-seeds", type=int, default=30)
    parser.add_argument("--heldout-seed-start", type=int, default=11000)
    parser.add_argument("--heldout-seeds", type=int, default=100)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--max-skills", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260724)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--include-random-warmstart",
        action="store_true",
        help="Also run the matched random-rule LLAMBO-style warm-start control.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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

    adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    catalog = semantic.semantic_field_catalog(adapter)
    skills = semantic.normalize_skills(
        {"skills": record.get("normalized_skills", [])},
        catalog,
        max_skills=args.max_skills,
    )
    if not skills:
        raise RuntimeError("The LLM record produced no executable semantic skills.")

    rng = random.Random(args.random_seed)
    random_skills: list[semantic.SemanticSkill] = []
    for replicate in range(args.replicates):
        for skill in skills:
            random_skills.append(matched_random_skill(
                skill,
                catalog,
                rng,
                f"random_rule_r{replicate + 1}_{skill.skill_id}",
            ))

    all_seeds = sorted(calibration_seeds | heldout_seeds)
    worker = partial(
        evaluate_seed,
        target_dataset=args.target_dataset,
        random_skills=tuple(random_skills),
        initial=args.initial,
        rounds=args.rounds,
        calibration_seeds=calibration_seeds,
        include_random_warmstart=args.include_random_warmstart,
    )
    if args.workers == 1:
        batches = [worker(seed) for seed in all_seeds]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(worker, all_seeds))
    rows = [row for batch in batches for row in batch]

    baseline_mode = "gp_ucb"
    random_modes = tuple(sorted({str(skill_mode) for skill_mode in {
        str(row["mode"])
        for row in rows
        if (
            str(row["mode"]).startswith("llm_semantic_random_rule_")
            or str(row["mode"]).startswith("llambo_warmstart_random_rule_")
        )
    }}))
    selected, calibration_diagnostics = select_best_random(
        rows,
        random_modes,
        baseline_mode,
        calibration_seeds,
    )
    selected_heldout = paired_summary(rows, selected, baseline_mode, heldout_seeds)
    all_summaries = {
        mode: {
            "calibration": paired_summary(rows, mode, baseline_mode, calibration_seeds),
            "heldout": paired_summary(rows, mode, baseline_mode, heldout_seeds),
        }
        for mode in random_modes
    }
    summary = {
        "protocol": {
            "target_dataset": args.target_dataset,
            "calibration_seed_start": args.calibration_seed_start,
            "calibration_seed_count": args.calibration_seeds,
            "heldout_seed_start": args.heldout_seed_start,
            "heldout_seed_count": args.heldout_seeds,
            "initial_observations": args.initial,
            "reveal_rounds": args.rounds,
            "baseline": baseline_mode,
            "rule_null": (
                "preserve each LLM skill's rule count, field set, condition arity, "
                "and execution hyperparameters; randomize condition values and signs"
            ),
        },
        "skill_count": len(skills),
        "replicates": args.replicates,
        "selected_on_calibration": selected,
        "selected_calibration": calibration_diagnostics[selected],
        "selected_heldout": selected_heldout,
        "all_summaries": all_summaries,
        "skills": [asdict(skill) for skill in random_skills],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_metrics.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_on_calibration": selected,
        "selected_heldout": selected_heldout,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
