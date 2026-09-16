#!/usr/bin/env python3
"""Aggregate zero-shot hypothesis-transfer runs without selecting a winner.

The input directories are produced by ``run_zero_shot_semantic_transfer.py``.
Every frozen LLM hypothesis is retained.  A matched random-null comparison is
computed per seed and per hypothesis so the report cannot turn a favorable
random rule into an LLM claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PRIMARY_METRICS = ("final_best", "best_so_far_auc")


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty paired comparison")
    average = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * sd / (len(values) ** 0.5)
    return {
        "mean": round(average, 6),
        "sd": round(sd, 6),
        "normal_95ci_low": round(average - half_width, 6),
        "normal_95ci_high": round(average + half_width, 6),
        "win_rate": round(sum(value > 0.0 for value in values) / len(values), 6),
    }


def paired_metric_delta(
    rows: list[dict[str, str]],
    left_mode: str,
    right_mode: str,
) -> dict[str, Any]:
    left = {
        int(row["seed"]): row
        for row in rows
        if row.get("mode") == left_mode
    }
    right = {
        int(row["seed"]): row
        for row in rows
        if row.get("mode") == right_mode
    }
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError(f"No paired rows for {left_mode} versus {right_mode}")
    final = [
        float(left[seed]["final_best"]) - float(right[seed]["final_best"])
        for seed in seeds
    ]
    auc = [
        float(left[seed]["best_so_far_auc"])
        - float(right[seed]["best_so_far_auc"])
        for seed in seeds
    ]
    return {
        "seed_count": len(seeds),
        "final_best": stats(final),
        "best_so_far_auc": stats(auc),
        "composite": stats([a + b for a, b in zip(final, auc)]),
    }


def paired_mean_random_delta(
    rows: list[dict[str, str]],
    llm_mode: str,
    random_modes: list[str],
) -> dict[str, Any]:
    llm = {
        int(row["seed"]): row
        for row in rows
        if row.get("mode") == llm_mode
    }
    random_by_seed = {
        random_mode: {
            int(row["seed"]): row
            for row in rows
            if row.get("mode") == random_mode
        }
        for random_mode in random_modes
    }
    seeds = sorted(
        set(llm)
        & set.intersection(*(set(by_seed) for by_seed in random_by_seed.values()))
    )
    if not seeds:
        raise ValueError(f"No paired random-null rows for {llm_mode}")
    final: list[float] = []
    auc: list[float] = []
    for seed in seeds:
        random_final = mean(
            float(random_by_seed[mode][seed]["final_best"])
            for mode in random_modes
        )
        random_auc = mean(
            float(random_by_seed[mode][seed]["best_so_far_auc"])
            for mode in random_modes
        )
        final.append(float(llm[seed]["final_best"]) - random_final)
        auc.append(float(llm[seed]["best_so_far_auc"]) - random_auc)
    return {
        "seed_count": len(seeds),
        "random_replicates": len(random_modes),
        "final_best": stats(final),
        "best_so_far_auc": stats(auc),
        "composite": stats([a + b for a, b in zip(final, auc)]),
    }


def stable_gain(comparison: dict[str, Any]) -> bool:
    return any(
        float(comparison[metric]["normal_95ci_low"]) > 0.0
        for metric in PRIMARY_METRICS
    )


def stable_harm(comparison: dict[str, Any]) -> bool:
    return all(
        float(comparison[metric]["normal_95ci_high"]) < 0.0
        for metric in PRIMARY_METRICS
    )


def record_sha256(result_dir: Path, summary: dict[str, Any]) -> str | None:
    record_path = result_dir.parent / "2026-07-25-hypothesis-generation"
    source = summary.get("source_dataset")
    target = summary.get("target_dataset")
    if not source or not target or not record_path.exists():
        return None
    candidates = sorted(record_path.glob("*.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("source_dataset") == source and payload.get("target_dataset") == target:
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def load_run(result_dir: Path) -> dict[str, Any]:
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    with (result_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    llm_modes = [f"llm_direct_prior_{skill_id}" for skill_id in summary["skill_ids"]]
    available_modes = {row.get("mode") for row in rows}
    llm_modes = [mode for mode in llm_modes if mode in available_modes]
    random_modes = sorted(
        mode for mode in available_modes
        if mode and "random_rule_r" in mode
    )
    skills: list[dict[str, Any]] = []
    for mode in llm_modes:
        random_for_skill = [
            random_mode for random_mode in random_modes
            if random_mode.endswith(mode.removeprefix("llm_direct_prior_") )
        ]
        vs_gp = paired_metric_delta(rows, mode, "gp_ucb")
        vs_ei = paired_metric_delta(rows, mode, "mixed_kernel_gp_ei")
        item = {
            "skill_id": mode.removeprefix("llm_direct_prior_"),
            "mode": mode,
            "vs_gp_ucb": vs_gp,
            "vs_mixed_kernel_gp_ei": vs_ei,
            "vs_random_null": paired_mean_random_delta(rows, mode, random_for_skill)
            if random_for_skill else None,
            "stable_gain_vs_mixed_kernel_gp_ei": stable_gain(vs_ei),
            "stable_harm_vs_mixed_kernel_gp_ei": stable_harm(vs_ei),
        }
        skills.append(item)
    best = max(
        skills,
        key=lambda item: item["vs_mixed_kernel_gp_ei"]["composite"]["mean"],
        default=None,
    )
    return {
        "result_dir": result_dir.as_posix(),
        "source_dataset": summary.get("source_dataset"),
        "target_dataset": summary.get("target_dataset"),
        "evidence_mode": summary.get("evidence_mode"),
        "proposal_mode": summary.get("proposal_mode"),
        "llm_model": summary.get("llm_model"),
        "seed_count": int(summary.get("seed_count", 0)),
        "initial_observations": int(summary.get("initial_observations", 0)),
        "online_rounds": int(summary.get("online_rounds", 0)),
        "target_observation_budget_per_seed": int(
            summary.get("target_observation_budget_per_seed", 0)
        ),
        "target_calibration_seed_count": int(summary.get("target_calibration_seed_count", 0)),
        "target_predecision_outcome_count": int(
            summary.get("target_predecision_outcome_count", 0)
        ),
        "random_null_count": len(random_modes),
        "record_sha256": record_sha256(result_dir, summary),
        "skills": skills,
        "best_by_composite_vs_mixed_kernel_gp_ei": best["skill_id"] if best else None,
        "evidence_boundary": summary.get("evidence_boundary"),
    }


def build_report(result_dirs: list[Path]) -> dict[str, Any]:
    runs = [load_run(path) for path in result_dirs]
    stable_gains = sum(
        any(item["stable_gain_vs_mixed_kernel_gp_ei"] for item in run["skills"])
        for run in runs
    )
    stable_harms = sum(
        any(item["stable_harm_vs_mixed_kernel_gp_ei"] for item in run["skills"])
        for run in runs
    )
    skills = [item for run in runs for item in run["skills"]]
    return {
        "study": "care2_zero_shot_hypothesis_transfer_matrix",
        "protocol": {
            "target_calibration": "none",
            "target_predecision_outcomes": 0,
            "policy_selection": "none; every frozen LLM hypothesis is reported",
            "matched_null": (
                "For every LLM hypothesis, random rules preserve the same rule count, "
                "condition arity, executor, and seed schedule; only schema values/signs "
                "are randomized."
            ),
        },
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "skill_count": len(skills),
            "stable_gain_run_count": stable_gains,
            "stable_harm_run_count": stable_harms,
            "stable_gain_fraction": round(stable_gains / len(runs), 6) if runs else 0.0,
            "stable_gain_skill_count": sum(
                item["stable_gain_vs_mixed_kernel_gp_ei"] for item in skills
            ),
            "stable_harm_skill_count": sum(
                item["stable_harm_vs_mixed_kernel_gp_ei"] for item in skills
            ),
            "all_runs_have_zero_target_calibration": all(
                run["target_calibration_seed_count"] == 0
                and run["target_predecision_outcome_count"] == 0
                for run in runs
            ),
        },
        "interpretation": (
            "This matrix is a zero-shot hypothesis audit, not a tuned leaderboard. "
            "A positive row is only called stable when at least one primary metric's "
            "normal 95% CI is above zero versus mixed-kernel GP-EI. The aggregate is "
            "reported at the source-target-pair level, so a few favorable hypotheses "
            "cannot be presented as universal transfer."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# CARE 2.0 Zero-Shot Hypothesis Transfer Matrix",
        "",
        "This report uses frozen LLM hypotheses, no target calibration, and matched "
        "random-rule nulls. Every hypothesis is retained; no target outcome is used "
        "to choose the reported winner.",
        "",
        "## Pair-Level Results",
        "",
        "| Source -> target | Seeds | Hypotheses | Stable positive vs mixed GP-EI | Stable negative | Best composite hypothesis |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        positive = sum(item["stable_gain_vs_mixed_kernel_gp_ei"] for item in run["skills"])
        negative = sum(item["stable_harm_vs_mixed_kernel_gp_ei"] for item in run["skills"])
        lines.append(
            f"| {run['source_dataset']} -> {run['target_dataset']} | {run['seed_count']} | "
            f"{len(run['skills'])} | {positive} | {negative} | "
            f"`{run['best_by_composite_vs_mixed_kernel_gp_ei']}` |"
        )
    lines.extend([
        "",
        "## Protocol Audit",
        "",
        f"Runs: {aggregate['run_count']}; frozen hypotheses: {aggregate['skill_count']}; "
        f"pair-level stable gains: {aggregate['stable_gain_run_count']}/"
        f"{aggregate['run_count']}; pair-level stable harms: {aggregate['stable_harm_run_count']}/"
        f"{aggregate['run_count']}.",
        f"All runs have zero target calibration and zero pre-decision target outcomes: "
        f"{aggregate['all_runs_have_zero_target_calibration']}.",
        "Each run reports GP-UCB, mixed-kernel GP-EI, every LLM hypothesis, and a "
        "same-structure random null. The random null is a control, not a competing "
        "LLM proposal.",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.result_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "zero_shot_transfer_matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "zero_shot_transfer_matrix.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
