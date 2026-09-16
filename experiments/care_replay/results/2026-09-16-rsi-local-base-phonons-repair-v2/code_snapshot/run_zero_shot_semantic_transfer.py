#!/usr/bin/env python3
"""Evaluate frozen LLM skills on a target without target calibration.

This protocol exists alongside the historical calibration/held-out protocol.
It never chooses a skill from target outcomes: every frozen LLM skill and every
matched random null is reported on the same target seeds. The only target
outcomes used are the normal online observations inside each replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
from functools import partial
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import llm_semantic_skills as semantic
import run_random_rule_control as random_control
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


def paired_delta(
    rows: list[dict[str, Any]],
    mode: str,
    baseline: str,
    seeds: set[int],
) -> dict[str, Any]:
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
    if not common:
        raise ValueError(f"No paired rows for {mode} versus {baseline}")
    final = [float(left[seed]["final_best"]) - float(right[seed]["final_best"]) for seed in common]
    auc = [
        float(left[seed]["best_so_far_auc"]) - float(right[seed]["best_so_far_auc"])
        for seed in common
    ]
    composite = [a + b for a, b in zip(final, auc)]

    def stats(values: list[float]) -> dict[str, float]:
        average = mean(values)
        sd = pstdev(values) if len(values) > 1 else 0.0
        se = sd / (len(values) ** 0.5)
        return {
            "mean": round(average, 6),
            "sd": round(sd, 6),
            "normal_95ci_low": round(average - 1.96 * se, 6),
            "normal_95ci_high": round(average + 1.96 * se, 6),
            "win_rate": round(sum(value > 0.0 for value in values) / len(values), 6),
        }

    return {
        "seed_count": len(common),
        "final_best": stats(final),
        "best_so_far_auc": stats(auc),
        "composite": stats(composite),
    }


def evaluate_seed(
    seed: int,
    target_dataset: str,
    skills: tuple[semantic.SemanticSkill, ...],
    random_skills: tuple[semantic.SemanticSkill, ...],
    initial: int,
    rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    baseline_weights = (1.0,) * (len(adapter.decision_columns) + 1)
    gp_metrics, gp_audit = weighted.run_policy(
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
    rows.append(gp_metrics)
    audits.append({"mode": "gp_ucb", "seed": seed, "events": gp_audit})

    ei_metrics, ei_audit = surrogate.run_policy(
        adapter,
        task,
        seed,
        "mixed_kernel_gp_ei",
        1.5,
        0.01,
        5,
        0.35,
        0.35,
        3.0,
        0.05,
    )
    rows.append(ei_metrics)
    audits.append({"mode": "mixed_kernel_gp_ei", "seed": seed, "events": ei_audit})

    for skill in (*skills, *random_skills):
        metrics, audit = semantic.run_direct_prior_skill(
            adapter,
            task,
            seed,
            skill,
            0.35,
            3.0,
            0.05,
        )
        rows.append(metrics)
        audits.append({"mode": metrics["mode"], "seed": seed, "events": audit})
    return rows, audits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-skills", type=int, default=12)
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260725)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.seeds <= 0 or args.initial <= 0 or args.rounds <= 0:
        parser.error("--seeds, --initial, and --rounds must be positive")
    if args.random_replicates < 0:
        parser.error("--random-replicates cannot be negative")

    adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    catalog = semantic.semantic_field_catalog(adapter)
    skills = semantic.normalize_skills(
        {"skills": record.get("normalized_skills", [])},
        catalog,
        max_skills=args.max_skills,
    )
    if not skills:
        raise RuntimeError("The LLM record produced no executable skills")

    random_skills: list[semantic.SemanticSkill] = []
    rng = random.Random(args.random_seed)
    for replicate in range(args.random_replicates):
        for skill in skills:
            random_skills.append(random_control.matched_random_skill(
                skill,
                catalog,
                rng,
                f"random_rule_r{replicate + 1}_{skill.skill_id}",
            ))

    seeds = set(range(args.seed_start, args.seed_start + args.seeds))
    worker = partial(
        evaluate_seed,
        target_dataset=args.target_dataset,
        skills=skills,
        random_skills=tuple(random_skills),
        initial=args.initial,
        rounds=args.rounds,
    )
    if args.workers == 1:
        batches = [worker(seed) for seed in sorted(seeds)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(worker, sorted(seeds)))
    rows = [row for batch, _audits in batches for row in batch]
    audits = [audit for _batch, batch_audits in batches for audit in batch_audits]

    modes = tuple(sorted({str(row["mode"]) for row in rows}))
    pairwise: dict[str, Any] = {}
    for mode in modes:
        if mode in {"gp_ucb", "mixed_kernel_gp_ei"}:
            continue
        pairwise[mode] = {
            baseline: paired_delta(rows, mode, baseline, seeds)
            for baseline in ("gp_ucb", "mixed_kernel_gp_ei")
        }

    summary = {
        "experiment": "care_zero_shot_semantic_transfer",
        "source_dataset": record.get("source_dataset"),
        "target_dataset": args.target_dataset,
        "evidence_mode": record.get("evidence_mode"),
        "proposal_mode": record.get("proposal_mode", "parametric"),
        "llm_model": record.get("model"),
        "skill_ids": [skill.skill_id for skill in skills],
        "random_null_count": len(random_skills),
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "initial_observations": args.initial,
        "online_rounds": args.rounds,
        "target_calibration_seed_count": 0,
        "target_calibration_rounds": 0,
        "target_predecision_outcome_count": 0,
        "target_observation_budget_per_seed": args.initial + args.rounds,
        "policy_selection": "none; every frozen LLM skill and random null is reported",
        "pairwise_vs_baselines": pairwise,
        "evidence_boundary": (
            "The source record and all skills are frozen before target replay. No target outcome "
            "is used to select a skill, tune a threshold, or construct a random null. Target outcomes "
            "are consumed only by the normal online acquisition loop after each candidate is selected."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (int(row["seed"]), str(row["mode"]))))
    with (args.output_dir / "audits.jsonl").open("w", encoding="utf-8") as handle:
        for audit in sorted(audits, key=lambda row: (int(row["seed"]), str(row["mode"]))):
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
