#!/usr/bin/env python3
"""Calibrate an LLM semantic skill library and confirm it on held-out seeds."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import llm_semantic_skills as semantic
import run_calibrated_frozen_llm_selector as selector
import run_llm_transfer_router as router
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


LLM_DIRECT_SELECTOR_MODE = "llm_direct_prior_selector"
LLAMBO_WARMSTART_SELECTOR_MODE = "llambo_warmstart_selector"


def expand_skill_variants(
    skills: tuple[semantic.SemanticSkill, ...],
    policy: str,
) -> tuple[semantic.SemanticSkill, ...]:
    if policy == "record":
        return skills
    variants: list[semantic.SemanticSkill] = []
    for skill in skills:
        variants.append(skill)
        variants.append(replace(
            skill,
            skill_id=f"{skill.skill_id}_noprior",
            prior_scale=0.0,
        ))
        if policy == "expanded":
            variants.append(replace(
                skill,
                skill_id=f"{skill.skill_id}_conservative",
                ridge=max(1.5, skill.ridge),
                prior_scale=0.0,
                semantic_mass_start=min(0.30, skill.semantic_mass_start),
                semantic_mass_end=min(0.08, skill.semantic_mass_end),
            ))
    if policy == "expanded" and len(skills) > 1:
        unique_rules: dict[tuple[tuple[str, str], ...], semantic.SemanticRule] = {}
        for skill in skills:
            for rule in skill.rules:
                unique_rules.setdefault(rule.conditions, rule)
        bank_rules = tuple(unique_rules.values())[:36]
        if bank_rules:
            variants.append(semantic.SemanticSkill(
                skill_id="cross_skill_rule_bank",
                rules=bank_rules,
                ridge=4.0,
                prior_scale=0.0,
                semantic_mass_start=0.25,
                semantic_mass_end=0.08,
                ucb_weight=0.60,
                gp_beta_start=2.0,
                gp_beta_end=1.0,
                gp_xi=0.01,
                confidence=max(skill.confidence for skill in skills),
                hypothesis=(
                    "Union of independently proposed LLM rule features. Rule signs are ignored; "
                    "revealed target observations fit a strongly regularized coefficient vector."
                ),
            ))
    return tuple(variants)


def add_mode_alias(
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    source_mode: str,
    alias_mode: str,
    heldout_seeds: set[int],
    selection_rule: str,
) -> None:
    for source_row in list(rows):
        if str(source_row["mode"]) != source_mode or int(source_row["seed"]) not in heldout_seeds:
            continue
        row = dict(source_row)
        row["mode"] = alias_mode
        row["selected_source_mode"] = source_mode
        rows.append(row)
        seed = int(row["seed"])
        alias_audit: list[dict[str, Any]] = []
        for source_event in audits[(source_mode, seed)]:
            event = deepcopy(source_event)
            event["mode"] = alias_mode
            event["external_baseline_selector"] = {
                "selected_source_mode": source_mode,
                "selection_rule": selection_rule,
            }
            alias_audit.append(event)
        audits[(alias_mode, seed)] = alias_audit


def evaluate_seed(
    seed: int,
    split: str,
    target_dataset: str,
    skills: tuple[semantic.SemanticSkill, ...],
    initial: int,
    rounds: int,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    include_llm_baselines: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    baseline_weights = (1.0,) * (len(adapter.decision_columns) + 1)
    baseline_metrics, baseline_audit = weighted.run_policy(
        adapter,
        task,
        seed,
        "gp_ucb",
        baseline_weights,
        {"scale": 0.0, "weights": list(baseline_weights)},
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    ei_metrics, ei_audit = surrogate.run_policy(
        adapter,
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
    portfolio_metrics, portfolio_audit = router.run_target_portfolio(
        adapter,
        task,
        seed,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    for metrics, audit in (
        (baseline_metrics, baseline_audit),
        (ei_metrics, ei_audit),
        (portfolio_metrics, portfolio_audit),
    ):
        metrics["split"] = split
        metrics["selected_source_mode"] = ""
        for event in audit:
            event["split"] = split
        rows.append(metrics)
        audits[(str(metrics["mode"]), seed)] = audit
    for skill in skills:
        metrics, audit = semantic.run_semantic_skill(
            adapter,
            task,
            seed,
            skill,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        metrics["split"] = split
        metrics["selected_source_mode"] = ""
        for event in audit:
            event["split"] = split
        rows.append(metrics)
        audits[(semantic.skill_mode(skill), seed)] = audit
        if include_llm_baselines:
            direct_metrics, direct_audit = semantic.run_direct_prior_skill(
                adapter,
                task,
                seed,
                skill,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            warmstart_metrics, warmstart_audit = semantic.run_llambo_warmstart(
                adapter,
                task,
                seed,
                skill,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            for baseline_metrics, baseline_audit in (
                (direct_metrics, direct_audit),
                (warmstart_metrics, warmstart_audit),
            ):
                baseline_metrics["split"] = split
                baseline_metrics["selected_source_mode"] = ""
                for event in baseline_audit:
                    event["split"] = split
                rows.append(baseline_metrics)
                audits[(str(baseline_metrics["mode"]), seed)] = baseline_audit
    return rows, audits


def select_skill(
    rows: list[dict[str, Any]],
    calibration_seeds: set[int],
    skills: tuple[semantic.SemanticSkill, ...],
    min_risk_adjusted_gain: float,
    min_positive_fold_rate: float,
) -> tuple[str, dict[str, Any]]:
    target_summary = selector.mode_means(rows, selector.TARGET_MODES, calibration_seeds)
    anchor = max(selector.TARGET_MODES, key=lambda mode: (target_summary[mode]["composite"], mode))
    diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for skill in skills:
        mode = semantic.skill_mode(skill)
        final = selector.paired_deltas(rows, mode, anchor, calibration_seeds, "final_best")
        auc = selector.paired_deltas(rows, mode, anchor, calibration_seeds, "best_so_far_auc")
        composite = [left + right for left, right in zip(final, auc)]
        final_stats = selector.delta_summary(final)
        auc_stats = selector.delta_summary(auc)
        composite_stats = selector.delta_summary(composite)
        fold_rate, fold_means = selector.positive_fold_rate(composite, 5)
        risk_adjusted = composite_stats["mean"] - 0.5 * composite_stats["se"]
        accepted = (
            final_stats["mean"] >= 0.0
            and auc_stats["mean"] >= 0.0
            and final_stats["non_loss_rate"] >= 0.55
            and fold_rate >= min_positive_fold_rate
            and risk_adjusted >= min_risk_adjusted_gain
        )
        if accepted:
            eligible.append(mode)
        diagnostics[mode] = {
            "final_best": final_stats,
            "best_so_far_auc": auc_stats,
            "composite": composite_stats,
            "positive_fold_rate": round(fold_rate, 6),
            "fold_composite_means": fold_means,
            "risk_adjusted_composite_gain": round(risk_adjusted, 6),
            "eligible": accepted,
        }
    selected = max(
        eligible,
        key=lambda mode: (
            diagnostics[mode]["risk_adjusted_composite_gain"],
            diagnostics[mode]["best_so_far_auc"]["mean"],
            mode,
        ),
    ) if eligible else anchor
    return selected, {
        "target_anchor_mode": anchor,
        "target_anchor_summary": target_summary,
        "eligible_skill_modes": sorted(eligible),
        "selected_mode": selected,
        "selected_llm_skill": selected.startswith("llm_semantic_"),
        "skill_diagnostics": diagnostics,
        "thresholds": {
            "min_risk_adjusted_composite_gain": min_risk_adjusted_gain,
            "min_positive_fold_rate": min_positive_fold_rate,
            "min_final_non_loss_rate": 0.55,
        },
        "rule": (
            "Choose the strongest target-only anchor on calibration composite. Select an LLM "
            "semantic skill only when final and AUC means are non-negative, fold stability and "
            "non-loss constraints pass, and risk-adjusted composite gain is positive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen calibration and held-out evaluation for LLM semantic skills.")
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--calibration-seed-start", type=int, default=10000)
    parser.add_argument("--calibration-seeds", type=int, default=50)
    parser.add_argument("--heldout-seed-start", type=int, default=11000)
    parser.add_argument("--heldout-seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--gp-xi", type=float, default=0.01)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--min-risk-adjusted-gain", type=float, default=0.25)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.8)
    parser.add_argument("--preselected-skill-id", default="")
    parser.add_argument(
        "--skill-variants",
        choices=("record", "noprior", "expanded"),
        default="expanded",
        help="Predeclared algorithmic variants. expanded adds no-prior, conservative, and rule-bank variants.",
    )
    parser.add_argument(
        "--include-llm-baselines",
        action="store_true",
        help="Evaluate frozen LLM-direct and LLAMBO-style warm-start baselines on the same seeds.",
    )
    parser.add_argument(
        "--disable-rule-prior",
        action="store_true",
        help="Keep the learned semantic rule features but set their LLM coefficient prior to zero.",
    )
    parser.add_argument(
        "--disable-semantic-model",
        action="store_true",
        help="Run only the LLM-chosen GP-UCB/EI schedule and acquisition mix.",
    )
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
    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    catalog = semantic.semantic_field_catalog(adapter)
    skills = semantic.normalize_skills(
        {"skills": record.get("normalized_skills", [])},
        catalog,
        max_skills=100,
    )
    skills = expand_skill_variants(skills, args.skill_variants)
    if args.preselected_skill_id:
        skills = tuple(skill for skill in skills if skill.skill_id == args.preselected_skill_id)
    if args.disable_rule_prior:
        skills = tuple(replace(skill, prior_scale=0.0) for skill in skills)
    if args.disable_semantic_model:
        skills = tuple(replace(
            skill,
            prior_scale=0.0,
            semantic_mass_start=0.0,
            semantic_mass_end=0.0,
        ) for skill in skills)
    if not skills:
        raise RuntimeError("No executable semantic skills remain after normalization.")
    jobs = [
        *((seed, "calibration") for seed in sorted(calibration_seeds)),
        *((seed, "heldout") for seed in sorted(heldout_seeds)),
    ]
    worker_args = {
        "target_dataset": args.target_dataset,
        "skills": skills,
        "initial": args.initial,
        "rounds": args.rounds,
        "gp_beta": args.gp_beta,
        "gp_xi": args.gp_xi,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
        "include_llm_baselines": args.include_llm_baselines,
    }
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
    selected_mode, selection = select_skill(
        rows,
        calibration_seeds,
        skills,
        args.min_risk_adjusted_gain,
        args.min_positive_fold_rate,
    )
    if args.preselected_skill_id:
        selection["calibration_selected_mode"] = selected_mode
        selected_mode = semantic.skill_mode(skills[0])
        selection["selected_mode"] = selected_mode
        selection["selected_llm_skill"] = True
        selection["preselected_skill_id"] = args.preselected_skill_id
        selection["rule"] = (
            "The skill was selected on an earlier development range and frozen before these seeds. "
            "Calibration seeds in this run select only the target-only comparison anchor."
        )
    selector.add_selector_alias(rows, audits, selected_mode, heldout_seeds, {
        "selected_llm_patch": selection["selected_llm_skill"],
        "target_anchor_mode": selection["target_anchor_mode"],
        "rule": selection["rule"],
    })
    external_selection: dict[str, Any] = {}
    external_modes: tuple[str, ...] = ()
    if args.include_llm_baselines:
        direct_modes = tuple(semantic.direct_prior_mode(skill) for skill in skills)
        warmstart_modes = tuple(semantic.llambo_warmstart_mode(skill) for skill in skills)
        external_modes = (*direct_modes, *warmstart_modes)
        for family, modes, alias in (
            ("llm_direct_prior", direct_modes, LLM_DIRECT_SELECTOR_MODE),
            ("llambo_warmstart", warmstart_modes, LLAMBO_WARMSTART_SELECTOR_MODE),
        ):
            calibration_means = selector.mode_means(rows, modes, calibration_seeds)
            chosen = max(
                modes,
                key=lambda mode: (calibration_means[mode]["composite"], mode),
            )
            rule = (
                f"Select the strongest {family} variant on calibration final_best + AUC; "
                "held-out outcomes do not affect this choice."
            )
            add_mode_alias(rows, audits, chosen, alias, heldout_seeds, rule)
            external_selection[family] = {
                "selected_mode": chosen,
                "selector_alias": alias,
                "calibration": calibration_means,
                "rule": rule,
            }
    all_modes = (
        *selector.TARGET_MODES,
        *(semantic.skill_mode(skill) for skill in skills),
        *external_modes,
        *(
            (LLM_DIRECT_SELECTOR_MODE, LLAMBO_WARMSTART_SELECTOR_MODE)
            if args.include_llm_baselines
            else ()
        ),
        selector.SELECTOR_MODE,
    )
    calibration = selector.mode_means(rows, all_modes, calibration_seeds)
    heldout = selector.mode_means(rows, all_modes, heldout_seeds)
    pairwise = {
        baseline: {
            field: selector.delta_summary(selector.paired_deltas(
                rows,
                selector.SELECTOR_MODE,
                baseline,
                heldout_seeds,
                field,
            ))
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for baseline in selector.TARGET_MODES
    }
    strongest_target = max(
        selector.TARGET_MODES,
        key=lambda mode: (heldout[mode]["composite"], mode),
    )
    strongest_stats = pairwise[strongest_target]
    care_vs_external = {
        alias: {
            field: selector.delta_summary(selector.paired_deltas(
                rows,
                selector.SELECTOR_MODE,
                alias,
                heldout_seeds,
                field,
            ))
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for alias in (LLM_DIRECT_SELECTOR_MODE, LLAMBO_WARMSTART_SELECTOR_MODE)
    } if args.include_llm_baselines else {}
    summary = {
        "experiment": "care_calibrated_llm_semantic_skill_selector",
        "source_dataset": record.get("source_dataset"),
        "target_dataset": args.target_dataset,
        "evidence_mode": record.get("evidence_mode", "full"),
        "llm_model": record.get("model"),
        "llm_generation_call_count": int(record.get("llm_generation_call_count", 1)),
        "new_llm_call_count": 0,
        "rule_prior_enabled": not args.disable_rule_prior and not args.disable_semantic_model,
        "semantic_model_enabled": not args.disable_semantic_model,
        "semantic_field_catalog": catalog,
        "skills": [asdict(skill) for skill in skills],
        "skill_variant_policy": args.skill_variants,
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "selection": selection,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_strongest_target_mode_descriptive_only": strongest_target,
        "heldout_pairwise": pairwise,
        "external_llm_baselines": external_selection,
        "care_vs_external_llm": care_vs_external,
        "heldout_validation": {
            "selected_llm_skill": selection["selected_llm_skill"],
            "strongest_target_mode": strongest_target,
            "final_95ci_positive": strongest_stats["final_best"]["normal_95ci_low"] > 0.0,
            "auc_95ci_positive": strongest_stats["best_so_far_auc"]["normal_95ci_low"] > 0.0,
            "top10_95ci_positive": strongest_stats["top10_hit"]["normal_95ci_low"] > 0.0,
        },
        "evidence_boundary": {
            "generation": (
                record.get("prompt_payload", {})
                .get("source_transfer_evidence", {})
                .get("boundary", "Generation evidence boundary was not recorded.")
            ),
            "evaluation": (
                "Calibration selects a skill before held-out evaluation. Every per-round fit uses "
                "only previously revealed target outcomes."
            ),
        },
    }
    output_id = f"calibrated_llm_semantic_{args.target_dataset}"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    selector.write_outputs(output_id, rows, audits, summary, record)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
