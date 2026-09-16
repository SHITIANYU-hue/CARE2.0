#!/usr/bin/env python3
"""Calibrate a frozen source-outcome transfer router and confirm it held out."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import llm_semantic_skills as semantic
import run_calibrated_frozen_llm_selector as selector
import run_calibrated_llm_semantic_selector as semantic_selector
import cross_task_router
import run_llm_kernel_skill_evolution as evolution
import run_llm_transfer_router as outcome_router
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted
import transfer_skill as canonical_skill


SELECTOR_MODE = "care_source_outcome_router"
MATCHED_TARGET_LLM_MODE = "matched_target_only_llm"
WARMSTART_ONLY_MODE = "source_warmstart_only"
DATA_ONLY_MODE = "fixed_data_only_transfer"


def resolve_target_llm_skill(
    record: dict[str, Any],
    target_adapter: replay.DatasetAdapter,
    target_llm_mode: str,
) -> semantic.SemanticSkill | None:
    if target_llm_mode in selector.TARGET_MODES:
        return None
    catalog = semantic.semantic_field_catalog(target_adapter)
    skills = semantic.normalize_skills(
        {"skills": record.get("normalized_skills", [])},
        catalog,
        max_skills=100,
    )
    for skill in semantic_selector.expand_skill_variants(skills, "expanded"):
        if target_llm_mode in {
            semantic.skill_mode(skill),
            semantic.direct_prior_mode(skill),
            semantic.llambo_warmstart_mode(skill),
        }:
            return skill
    raise RuntimeError(
        f"Target-only mode {target_llm_mode!r} is not present in the supplied record."
    )


def target_llm_initial_observations(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    target_llm_mode: str,
    target_llm_skill: semantic.SemanticSkill | None,
) -> list[replay.Candidate]:
    pool = list(adapter.candidates)
    if (
        target_llm_skill is None
        or target_llm_mode != semantic.llambo_warmstart_mode(target_llm_skill)
    ):
        random.Random(seed).shuffle(pool)
        return pool[: task.initial_observations]
    rng = random.Random(seed)
    jitter = {candidate.candidate_id: rng.random() * 1e-8 for candidate in pool}
    ordered = sorted(
        pool,
        key=lambda candidate: (
            semantic.fixed_rule_score(target_llm_skill, candidate)
            + jitter[candidate.candidate_id],
            candidate.candidate_id,
        ),
        reverse=True,
    )
    observed: list[replay.Candidate] = []
    seen_groups: set[str] = set()
    for candidate in ordered:
        if candidate.group in seen_groups and len(seen_groups) < task.initial_observations:
            continue
        observed.append(candidate)
        seen_groups.add(candidate.group)
        if len(observed) == task.initial_observations:
            break
    if len(observed) < task.initial_observations:
        observed_ids = {candidate.candidate_id for candidate in observed}
        observed.extend(
            candidate
            for candidate in ordered
            if candidate.candidate_id not in observed_ids
        )
    return observed[: task.initial_observations]


def target_llm_anchor_scorer(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    target_llm_mode: str,
    target_llm_skill: semantic.SemanticSkill | None,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> Any:
    def score(
        observed_ids: set[str],
        observed: list[replay.Candidate],
        features_by_id: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        beta = gp_beta
        xi = gp_xi
        if target_llm_skill is not None:
            beta = semantic.scheduled_value(
                target_llm_skill.gp_beta_start,
                target_llm_skill.gp_beta_end,
                round_index,
                task.reveal_budget,
            )
            xi = target_llm_skill.gp_xi
        anchors, anchor_diagnostics = outcome_router.target_anchor_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            beta,
            xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        if target_llm_mode == "gp_ucb":
            scores = weighted.rank_normalized(anchors["gp_ucb"])
            return scores, {
                "matched_target_only_mode": target_llm_mode,
                "anchor": anchor_diagnostics,
            }
        if target_llm_mode == "mixed_kernel_gp_ei":
            scores = weighted.rank_normalized(anchors["gp_ei"])
            return scores, {
                "matched_target_only_mode": target_llm_mode,
                "anchor": anchor_diagnostics,
            }
        if target_llm_mode == "target_acquisition_portfolio":
            return anchors["target_acquisition_portfolio"], {
                "matched_target_only_mode": target_llm_mode,
                "anchor": anchor_diagnostics,
            }
        if target_llm_skill is None:
            raise RuntimeError(f"Unsupported target-only anchor {target_llm_mode!r}.")
        ucb_rank = weighted.rank_normalized(anchors["gp_ucb"])
        ei_rank = weighted.rank_normalized(anchors["gp_ei"])
        base_rank = {
            candidate_id: target_llm_skill.ucb_weight * ucb_rank[candidate_id]
            + (1.0 - target_llm_skill.ucb_weight) * ei_rank[candidate_id]
            for candidate_id in ucb_rank
        }
        if target_llm_mode == semantic.llambo_warmstart_mode(target_llm_skill):
            return base_rank, {
                "matched_target_only_mode": target_llm_mode,
                "anchor": anchor_diagnostics,
                "warmstart_only": True,
            }
        semantic_mass = semantic.scheduled_value(
            target_llm_skill.semantic_mass_start,
            target_llm_skill.semantic_mass_end,
            round_index,
            task.reveal_budget,
        )
        if target_llm_mode == semantic.direct_prior_mode(target_llm_skill):
            prior_scores = {
                candidate.candidate_id: semantic.fixed_rule_score(
                    target_llm_skill,
                    candidate,
                )
                for candidate in adapter.candidates
                if candidate.candidate_id not in observed_ids
            }
            semantic_rank = weighted.rank_normalized(prior_scores)
            scores = {
                candidate_id: (1.0 - semantic_mass) * base_rank[candidate_id]
                + semantic_mass * semantic_rank[candidate_id]
                for candidate_id in base_rank
            }
            return scores, {
                "matched_target_only_mode": target_llm_mode,
                "semantic_mass": round(semantic_mass, 6),
                "anchor": anchor_diagnostics,
                "semantic_policy": "frozen_direct_rule_prior",
            }
        if target_llm_mode == semantic.skill_mode(target_llm_skill):
            if semantic_mass <= 0.0:
                return base_rank, {
                    "matched_target_only_mode": target_llm_mode,
                    "semantic_mass": 0.0,
                    "anchor": anchor_diagnostics,
                    "semantic_policy": "target_calibrated_semantic_surrogate",
                }
            semantic_scores, semantic_diagnostics = semantic.semantic_model_scores(
                target_llm_skill,
                adapter,
                observed,
                observed_ids,
                beta,
            )
            semantic_rank = weighted.rank_normalized(semantic_scores)
            scores = {
                candidate_id: (1.0 - semantic_mass) * base_rank[candidate_id]
                + semantic_mass * semantic_rank[candidate_id]
                for candidate_id in base_rank
            }
            semantic_diagnostics.pop("candidate_components", None)
            return scores, {
                "matched_target_only_mode": target_llm_mode,
                "semantic_mass": round(semantic_mass, 6),
                "anchor": anchor_diagnostics,
                "semantic_policy": "target_calibrated_semantic_surrogate",
                "semantic_model": semantic_diagnostics,
            }
        raise RuntimeError(
            f"Mode {target_llm_mode!r} is incompatible with "
            f"skill {target_llm_skill.skill_id!r}."
        )

    return score


def run_matched_target_llm(
    target_adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    target_llm_mode: str,
    target_llm_skill: semantic.SemanticSkill | None,
    existing_rows: list[dict[str, Any]],
    existing_audits: dict[tuple[str, int], list[dict[str, Any]]],
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for row in existing_rows:
        if str(row["mode"]) == target_llm_mode:
            metrics = dict(row)
            audit = deepcopy(existing_audits[(target_llm_mode, seed)])
            break
    else:
        if target_llm_skill is None:
            raise RuntimeError(
                f"Target-only LLM mode {target_llm_mode!r} requires a semantic skill."
            )
        if target_llm_mode == semantic.skill_mode(target_llm_skill):
            metrics, audit = semantic.run_semantic_skill(
                target_adapter,
                task,
                seed,
                target_llm_skill,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
        elif target_llm_mode == semantic.direct_prior_mode(target_llm_skill):
            metrics, audit = semantic.run_direct_prior_skill(
                target_adapter,
                task,
                seed,
                target_llm_skill,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
        elif target_llm_mode == semantic.llambo_warmstart_mode(target_llm_skill):
            metrics, audit = semantic.run_llambo_warmstart(
                target_adapter,
                task,
                seed,
                target_llm_skill,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
        else:
            raise RuntimeError(
                f"Mode {target_llm_mode!r} does not match skill "
                f"{target_llm_skill.skill_id!r}."
            )
    metrics["mode"] = MATCHED_TARGET_LLM_MODE
    metrics["selected_source_mode"] = target_llm_mode
    for event in audit:
        event["mode"] = MATCHED_TARGET_LLM_MODE
        event["matched_target_only_llm"] = {
            "selected_mode": target_llm_mode,
            "selection_boundary": (
                "This target-only strategy identity was frozen by an earlier "
                "calibration run and does not use source evidence."
            ),
        }
    return metrics, audit


def fixed_data_only_patch(
    target_adapter: replay.DatasetAdapter,
    fixed_scales: tuple[float, ...],
    gp_beta: float,
) -> evolution.KernelSkillPatch:
    """One deterministic source-outcome patch with no LLM-generated choices."""

    scales = tuple(dict.fromkeys((0.0, *fixed_scales)))
    return evolution.KernelSkillPatch(
        patch_id="fixed_data_only",
        scales=scales,
        role_multipliers={field: 1.0 for field in target_adapter.decision_columns},
        gp_beta=gp_beta,
        gp_beta_end=gp_beta,
        source_prior_strength=1.0,
        source_similarity_temperature=0.35,
        source_neighbor_count=12,
        calibration_mode="signed",
        min_cv_gain=0.02,
        confidence=1.0,
        reason=(
            "Deterministic non-LLM control: equal role weights and fixed source-prior "
            "settings isolate the value of measured source outcomes."
        ),
        source_interaction_strength=0.75,
        source_interaction_min_support=5,
        canonicalize_source_values=True,
    )


def renamed_control(
    metrics: dict[str, Any],
    audit: list[dict[str, Any]],
    mode: str,
    description: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    renamed_metrics = dict(metrics)
    renamed_metrics["mode"] = mode
    renamed_metrics["split"] = split
    renamed_metrics["selected_source_mode"] = ""
    renamed_audit: list[dict[str, Any]] = []
    for source_event in audit:
        event = deepcopy(source_event)
        event["mode"] = mode
        event["split"] = split
        event["mechanism_control"] = {
            "mode": mode,
            "description": description,
        }
        renamed_audit.append(event)
    return renamed_metrics, renamed_audit


def evaluate_seed(
    seed: int,
    split: str,
    source_dataset: str,
    target_dataset: str,
    patches: tuple[evolution.KernelSkillPatch, ...],
    target_llm_mode: str,
    target_llm_skill: semantic.SemanticSkill | None,
    fixed_scales: tuple[float, ...],
    source_observations: int,
    source_seed: int,
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
    router_min_observations: int,
    router_min_quality: float,
    router_max_transfer_mass: float,
    source_initial_strategy: str,
    include_mechanism_controls: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    initial_observed = target_llm_initial_observations(
        target_adapter,
        task,
        seed,
        target_llm_mode,
        target_llm_skill,
    )
    anchor_scorer = target_llm_anchor_scorer(
        target_adapter,
        task,
        target_llm_mode,
        target_llm_skill,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows, audits = outcome_router.run_seed(
        seed,
        source_dataset,
        target_dataset,
        patches,
        fixed_scales,
        source_observations,
        source_seed,
        discount,
        min_source_support,
        initial,
        rounds,
        normalize,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
        initial_observed,
        anchor_scorer,
        target_llm_mode,
        router_min_observations,
        router_min_quality,
        router_max_transfer_mass,
        source_initial_strategy,
    )
    for row in rows:
        row["split"] = split
        row["selected_source_mode"] = ""
    for events in audits.values():
        for event in events:
            event["split"] = split
    target_metrics, target_audit = run_matched_target_llm(
        target_adapter,
        task,
        seed,
        target_llm_mode,
        target_llm_skill,
        rows,
        audits,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    target_metrics["split"] = split
    for event in target_audit:
        event["split"] = split
    rows.append(target_metrics)
    audits[(MATCHED_TARGET_LLM_MODE, seed)] = target_audit
    if include_mechanism_controls:
        source_adapter = replay.DATASET_BUILDERS[source_dataset]()
        source_observed = transfer.source_observations(
            source_adapter,
            source_seed,
            source_observations,
        )
        card = transfer.compile_transfer_card(
            source_adapter,
            target_adapter,
            source_observed,
            transfer.descriptor_transfer_role_map_for(
                source_dataset,
                target_dataset,
            ),
            discount,
            min_source_support,
        )
        warm_metrics, warm_audit = outcome_router.run_router_policy(
            source_adapter,
            target_adapter,
            task,
            seed,
            card,
            source_observed,
            patches,
            normalize,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            initial_observed,
            anchor_scorer,
            target_llm_mode,
            router_min_observations,
            router_min_quality,
            0.0,
            source_initial_strategy,
        )
        warm_metrics, warm_audit = renamed_control(
            warm_metrics,
            warm_audit,
            WARMSTART_ONLY_MODE,
            (
                "Uses the same source-informed initial design as the full route, "
                "then fixes transfer mass to zero for every sequential round."
            ),
            split,
        )
        rows.append(warm_metrics)
        audits[(WARMSTART_ONLY_MODE, seed)] = warm_audit

        data_patch = fixed_data_only_patch(target_adapter, fixed_scales, gp_beta)
        data_metrics, data_audit = outcome_router.run_router_policy(
            source_adapter,
            target_adapter,
            task,
            seed,
            card,
            source_observed,
            (data_patch,),
            normalize,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            initial_observed,
            anchor_scorer,
            target_llm_mode,
            router_min_observations,
            router_min_quality,
            router_max_transfer_mass,
            source_initial_strategy,
        )
        data_metrics, data_audit = renamed_control(
            data_metrics,
            data_audit,
            DATA_ONLY_MODE,
            (
                "Uses measured source outcomes with a deterministic equal-role patch; "
                "no LLM-generated patch fields are used."
            ),
            split,
        )
        rows.append(data_metrics)
        audits[(DATA_ONLY_MODE, seed)] = data_audit
    return rows, audits


def select_route(
    rows: list[dict[str, Any]],
    calibration_seeds: set[int],
    min_risk_adjusted_gain: float,
    min_positive_fold_rate: float,
    min_final_non_loss_rate: float,
    min_composite_ci_low: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    target_summary = selector.mode_means(
        rows,
        selector.TARGET_MODES,
        calibration_seeds,
    )
    if set(target_summary) != set(selector.TARGET_MODES):
        raise RuntimeError("Calibration rows do not contain all target-only modes.")
    anchor = max(
        selector.TARGET_MODES,
        key=lambda mode: (target_summary[mode]["composite"], mode),
    )
    comparisons: dict[str, dict[str, Any]] = {}
    eligible_by_anchor: dict[str, bool] = {}
    for comparison_anchor in (MATCHED_TARGET_LLM_MODE, anchor):
        final = selector.paired_deltas(
            rows,
            "llm_transfer_router",
            comparison_anchor,
            calibration_seeds,
            "final_best",
        )
        auc = selector.paired_deltas(
            rows,
            "llm_transfer_router",
            comparison_anchor,
            calibration_seeds,
            "best_so_far_auc",
        )
        composite = [left + right for left, right in zip(final, auc)]
        final_stats = selector.delta_summary(final)
        auc_stats = selector.delta_summary(auc)
        composite_stats = selector.delta_summary(composite)
        fold_rate, fold_means = selector.positive_fold_rate(composite, 5)
        risk_adjusted = composite_stats["mean"] - composite_stats["se"]
        eligible_by_anchor[comparison_anchor] = bool(
            final_stats["mean"] >= 0.0
            and auc_stats["mean"] >= 0.0
            and final_stats["non_loss_rate"] >= min_final_non_loss_rate
            and fold_rate >= min_positive_fold_rate
            and risk_adjusted >= min_risk_adjusted_gain
            and composite_stats["normal_95ci_low"] >= min_composite_ci_low
        )
        comparisons[comparison_anchor] = {
            "final_best": final_stats,
            "best_so_far_auc": auc_stats,
            "composite": composite_stats,
            "positive_fold_rate": round(fold_rate, 6),
            "fold_composite_means": fold_means,
            "risk_adjusted_composite_gain": round(risk_adjusted, 6),
            "eligible": eligible_by_anchor[comparison_anchor],
        }
    eligible = all(eligible_by_anchor.values())
    selected = "llm_transfer_router" if eligible else MATCHED_TARGET_LLM_MODE
    return selected, {
        "selected_mode": selected,
        "selected_source_outcome_transfer": eligible,
        "target_anchor_mode": anchor,
        "matched_target_only_llm_mode": MATCHED_TARGET_LLM_MODE,
        "target_anchor_summary": target_summary,
        "source_outcome_diagnostics": comparisons,
        "thresholds": {
            "min_risk_adjusted_gain": min_risk_adjusted_gain,
            "min_positive_fold_rate": min_positive_fold_rate,
            "min_final_non_loss_rate": min_final_non_loss_rate,
            "min_composite_95ci_low": min_composite_ci_low,
        },
        "rule": (
            "Select source-outcome transfer only from calibration seeds. It must pass the "
            "non-negative final/AUC, fold stability, non-loss, one-standard-error gain, "
            "and positive composite 95% confidence-bound requirements against both the "
            "frozen matched target-only LLM and the strongest target-only BO. Otherwise "
            "deploy the frozen matched target-only LLM exactly."
        ),
    }


def add_selector_alias(
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    selected_mode: str,
    heldout_seeds: set[int],
    selection: dict[str, Any],
) -> None:
    for source_row in list(rows):
        seed = int(source_row["seed"])
        if str(source_row["mode"]) != selected_mode or seed not in heldout_seeds:
            continue
        row = dict(source_row)
        row["mode"] = SELECTOR_MODE
        row["selected_source_mode"] = selected_mode
        rows.append(row)
        alias_events: list[dict[str, Any]] = []
        for source_event in audits[(selected_mode, seed)]:
            event = deepcopy(source_event)
            event["mode"] = SELECTOR_MODE
            event["source_outcome_selector"] = {
                "selected_source_mode": selected_mode,
                "selected_source_outcome_transfer": selection[
                    "selected_source_outcome_transfer"
                ],
                "target_anchor_mode": selection["target_anchor_mode"],
                "selection_rule": selection["rule"],
                "evidence_boundary": (
                    "This route was frozen on calibration seeds. The held-out seed did not "
                    "change the source history, patch portfolio, target anchor, or fallback."
                ),
            }
            alias_events.append(event)
        audits[(SELECTOR_MODE, seed)] = alias_events


def component_effect(
    rows: list[dict[str, Any]],
    left_mode: str,
    right_mode: str,
    seeds: set[int],
) -> dict[str, Any]:
    final = selector.paired_deltas(
        rows,
        left_mode,
        right_mode,
        seeds,
        "final_best",
    )
    auc = selector.paired_deltas(
        rows,
        left_mode,
        right_mode,
        seeds,
        "best_so_far_auc",
    )
    composite = [left + right for left, right in zip(final, auc)]
    return {
        "left_mode": left_mode,
        "right_mode": right_mode,
        "final_best": selector.delta_summary(final),
        "best_so_far_auc": selector.delta_summary(auc),
        "composite": selector.delta_summary(composite),
        "exact_seed_match_rate": round(
            sum(
                abs(final_delta) <= 1e-12 and abs(auc_delta) <= 1e-12
                for final_delta, auc_delta in zip(final, auc)
            )
            / max(1, len(composite)),
            6,
        ),
    }


def route_action_diagnostics(
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    mode: str,
    seeds: set[int],
) -> dict[str, Any]:
    round_count = 0
    positive_mass_rounds = 0
    changed_action_rounds = 0
    source_initial_active_seeds = 0
    for seed in seeds:
        events = audits.get((mode, seed), [])
        if events:
            initial = events[0].get("hypothesis_snapshot", {}).get(
                "source_initial_design",
                {},
            )
            if bool(initial.get("source_outcome_active")) and int(
                initial.get("replaced_count", 0)
            ) > 0:
                source_initial_active_seeds += 1
        for event in events:
            snapshot = event.get("hypothesis_snapshot", {})
            router_gate = snapshot.get("router_gate", {})
            round_count += 1
            if float(snapshot.get("transfer_mass", 0.0)) > 0.0:
                positive_mass_rounds += 1
            if (
                router_gate.get("authorized")
                and router_gate.get("anchor_candidate") is not None
                and router_gate.get("selected_candidate")
                != router_gate.get("anchor_candidate")
            ):
                changed_action_rounds += 1
    return {
        "seed_count": len(seeds),
        "round_count": round_count,
        "source_initial_active_seed_rate": round(
            source_initial_active_seeds / max(1, len(seeds)),
            6,
        ),
        "positive_transfer_mass_round_rate": round(
            positive_mass_rounds / max(1, round_count),
            6,
        ),
        "post_initialization_action_change_rate": round(
            changed_action_rounds / max(1, round_count),
            6,
        ),
    }


def mechanism_attribution(
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    heldout_seeds: set[int],
    selected_source_outcome_transfer: bool,
    include_controls: bool,
) -> dict[str, Any]:
    if not include_controls:
        return {
            "status": "not_run",
            "reason": "Mechanism controls were explicitly disabled.",
        }
    warm_start = component_effect(
        rows,
        WARMSTART_ONLY_MODE,
        MATCHED_TARGET_LLM_MODE,
        heldout_seeds,
    )
    continuous = component_effect(
        rows,
        "llm_transfer_router",
        WARMSTART_ONLY_MODE,
        heldout_seeds,
    )
    llm_patch = component_effect(
        rows,
        "llm_transfer_router",
        DATA_ONLY_MODE,
        heldout_seeds,
    )
    actions = route_action_diagnostics(
        audits,
        "llm_transfer_router",
        heldout_seeds,
    )
    if not selected_source_outcome_transfer:
        classification = "offline_selector_fallback"
    elif continuous["exact_seed_match_rate"] == 1.0:
        classification = "source_informed_initial_design_only"
    elif continuous["composite"]["normal_95ci_low"] > 0.0:
        classification = "continuous_source_outcome_gain_supported"
    else:
        classification = "continuous_component_not_established"
    return {
        "status": "complete",
        "classification": classification,
        "source_informed_initial_design_effect": warm_start,
        "post_initialization_source_outcome_effect": continuous,
        "llm_patch_increment_over_fixed_data_only": llm_patch,
        "full_route_action_diagnostics": actions,
        "interpretation_rule": (
            "The full route is called continuous only when it improves over the same "
            "source-informed initial design with transfer mass fixed to zero. LLM patch "
            "credit additionally requires improvement over the fixed data-only route."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen calibration and held-out evaluation for complete source-outcome transfer."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument(
        "--target-llm-record",
        type=Path,
        required=True,
        help="Frozen target-only LLM generation record selected before this experiment.",
    )
    parser.add_argument(
        "--target-llm-mode",
        required=True,
        help="Frozen target-only strategy identity from an earlier calibration run.",
    )
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--calibration-seed-start", type=int, default=32000)
    parser.add_argument("--calibration-seeds", type=int, default=30)
    parser.add_argument("--heldout-seed-start", type=int, default=32100)
    parser.add_argument("--heldout-seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--source-observations", type=int, default=512)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--fixed-ensemble-scales", default="0.5,1,1.5,2,3,4")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--gp-xi", type=float, default=0.01)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--router-min-observations", type=int, default=5)
    parser.add_argument("--router-min-quality", type=float, default=0.20)
    parser.add_argument("--router-max-transfer-mass", type=float, default=0.45)
    parser.add_argument(
        "--source-initial-strategy",
        choices=(
            "matched",
            "source_extremes",
            "source_positive",
            "source_negative",
            "source_positive_quantile",
            "source_negative_quantile",
        ),
        default="matched",
    )
    parser.add_argument("--min-risk-adjusted-gain", type=float, default=0.25)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.8)
    parser.add_argument("--min-final-non-loss-rate", type=float, default=0.6)
    parser.add_argument("--min-composite-ci-low", type=float, default=0.0)
    parser.add_argument(
        "--no-mechanism-controls",
        action="store_true",
        help=(
            "Skip the matched warm-start-only and fixed data-only controls. "
            "The canonical confirmation protocol runs them by default."
        ),
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
    source = replay.DATASET_BUILDERS[args.source_dataset]()
    target = replay.DATASET_BUILDERS[args.target_dataset]()
    route_proposal = cross_task_router.propose_route(
        args.source_dataset,
        args.target_dataset,
        source.decision_columns,
        target.decision_columns,
    ).as_dict()
    target_llm_record = json.loads(args.target_llm_record.read_text(encoding="utf-8"))
    target_llm_skill = resolve_target_llm_skill(
        target_llm_record,
        target,
        args.target_llm_mode,
    )
    patches = evolution.normalize_patches(
        {"patches": record.get("normalized_patches", [])},
        target.decision_columns,
        max_patches=100,
    )
    if not patches:
        raise RuntimeError("The record contains no executable source-outcome patches.")
    fixed_scales = weighted.parse_scales(args.fixed_ensemble_scales)
    source_observed = transfer.source_observations(
        source,
        args.source_seed,
        args.source_observations,
    )
    role_map = transfer.descriptor_transfer_role_map_for(
        args.source_dataset,
        args.target_dataset,
    )
    transfer_card = transfer.compile_transfer_card(
        source,
        target,
        source_observed,
        role_map,
        args.discount,
        args.min_source_support,
    )
    execution = {
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "source_observation_count": args.source_observations,
        "source_seed": args.source_seed,
        "initial_observations": args.initial,
        "reveal_rounds": args.rounds,
        "discount": args.discount,
        "min_source_support": args.min_source_support,
        "fixed_ensemble_scales": list(fixed_scales),
        "normalize_kernel_weights": not args.no_normalize,
        "gp_beta": args.gp_beta,
        "gp_xi": args.gp_xi,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
        "router_min_observations": args.router_min_observations,
        "router_min_quality": args.router_min_quality,
        "router_max_transfer_mass": args.router_max_transfer_mass,
        "source_initial_strategy": args.source_initial_strategy,
        "target_anchor_mode": args.target_llm_mode,
        "include_mechanism_controls": not args.no_mechanism_controls,
    }
    gate_policy = {
        "fallback_mode": MATCHED_TARGET_LLM_MODE,
        "decision_scope": "offline_replay_model_selection",
        "real_experiment_deployment_ready": False,
        "online_router": {
            "min_observations": args.router_min_observations,
            "min_quality": args.router_min_quality,
            "max_transfer_mass": args.router_max_transfer_mass,
        },
        "confirmation": {
            "min_risk_adjusted_gain": args.min_risk_adjusted_gain,
            "min_positive_fold_rate": args.min_positive_fold_rate,
            "min_final_non_loss_rate": args.min_final_non_loss_rate,
            "min_composite_ci_low": args.min_composite_ci_low,
            "must_beat_matched_target_llm": True,
            "must_beat_strongest_target_only_bo": True,
        },
    }
    skill = canonical_skill.compile_transfer_skill(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        route_proposal=route_proposal,
        source_evidence={
            "transfer_card_id": transfer_card.card_id,
            "source_seed": args.source_seed,
            "observation_count": len(source_observed),
            "fixed_across_target_seeds": True,
            "roles": [asdict(role) for role in transfer_card.roles],
            "value_priors": [asdict(prior) for prior in transfer_card.value_priors],
            "evidence_summary": transfer_card.evidence_summary,
        },
        role_map=role_map,
        kernel_patches=[asdict(patch) for patch in patches],
        execution=execution,
        gate=gate_policy,
        provenance={
            "source_patch_model": record.get("model"),
            "source_patch_record": str(args.llm_record),
            "source_patch_record_sha256": canonical_skill.file_sha256(args.llm_record),
            "target_anchor_model": target_llm_record.get("model"),
            "target_anchor_record": str(args.target_llm_record),
            "target_anchor_record_sha256": canonical_skill.file_sha256(
                args.target_llm_record
            ),
        },
    )
    calibration_seeds = set(range(
        skill.execution["calibration_seed_start"],
        skill.execution["calibration_seed_start"]
        + skill.execution["calibration_seed_count"],
    ))
    heldout_seeds = set(range(
        skill.execution["heldout_seed_start"],
        skill.execution["heldout_seed_start"]
        + skill.execution["heldout_seed_count"],
    ))
    jobs = [
        *((seed, "calibration") for seed in sorted(calibration_seeds)),
        *((seed, "heldout") for seed in sorted(heldout_seeds)),
    ]
    worker_args = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "patches": patches,
        "target_llm_mode": args.target_llm_mode,
        "target_llm_skill": target_llm_skill,
        "fixed_scales": fixed_scales,
        "source_observations": skill.execution["source_observation_count"],
        "source_seed": skill.execution["source_seed"],
        "discount": skill.execution["discount"],
        "min_source_support": skill.execution["min_source_support"],
        "initial": skill.execution["initial_observations"],
        "rounds": skill.execution["reveal_rounds"],
        "normalize": skill.execution["normalize_kernel_weights"],
        "gp_beta": skill.execution["gp_beta"],
        "gp_xi": skill.execution["gp_xi"],
        "numeric_length_scale": skill.execution["numeric_length_scale"],
        "categorical_length_scale": skill.execution["categorical_length_scale"],
        "gp_noise": skill.execution["gp_noise"],
        "router_min_observations": skill.execution["router_min_observations"],
        "router_min_quality": skill.execution["router_min_quality"],
        "router_max_transfer_mass": skill.execution["router_max_transfer_mass"],
        "source_initial_strategy": skill.execution["source_initial_strategy"],
        "include_mechanism_controls": skill.execution[
            "include_mechanism_controls"
        ],
    }
    if args.workers == 1:
        results = [
            evaluate_seed(seed=seed, split=split, **worker_args)
            for seed, split in jobs
        ]
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
    selected_mode, selection = select_route(
        rows,
        calibration_seeds,
        skill.gate["confirmation"]["min_risk_adjusted_gain"],
        skill.gate["confirmation"]["min_positive_fold_rate"],
        skill.gate["confirmation"]["min_final_non_loss_rate"],
        skill.gate["confirmation"]["min_composite_ci_low"],
    )
    selection["schema_route_proposal"] = route_proposal
    selection["transfer_skill"] = skill.identity()
    selection["decision_scope"] = "offline_replay_model_selection"
    selection["real_experiment_deployment_ready"] = False
    selection["deployment_note"] = (
        "Calibration seeds reveal archived target outcomes and are valid for offline "
        "benchmark model selection only. They are not a cost-feasible real-lab gate."
    )
    add_selector_alias(rows, audits, selected_mode, heldout_seeds, selection)
    canonical_skill.attach_skill_identity(audits, skill)
    control_modes = (
        (WARMSTART_ONLY_MODE, DATA_ONLY_MODE)
        if not args.no_mechanism_controls
        else ()
    )
    all_modes = (
        *selector.TARGET_MODES,
        MATCHED_TARGET_LLM_MODE,
        "llm_transfer_router",
        *control_modes,
        SELECTOR_MODE,
    )
    calibration = selector.mode_means(rows, all_modes, calibration_seeds)
    heldout = selector.mode_means(rows, all_modes, heldout_seeds)
    pairwise = {
        baseline: {
            field: selector.delta_summary(selector.paired_deltas(
                rows,
                SELECTOR_MODE,
                baseline,
                heldout_seeds,
                field,
            ))
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for baseline in (
            *selector.TARGET_MODES,
            MATCHED_TARGET_LLM_MODE,
            *control_modes,
        )
    }
    attribution = mechanism_attribution(
        rows,
        audits,
        heldout_seeds,
        selection["selected_source_outcome_transfer"],
        not args.no_mechanism_controls,
    )
    selection["mechanism_attribution"] = attribution
    offline_selection_cost = {
        "evaluated_mode_count": len(calibration),
        "calibration_seed_count": len(calibration_seeds),
        "observations_per_policy_replay": args.initial + args.rounds,
        "gross_reveal_equivalents": (
            len(calibration)
            * len(calibration_seeds)
            * (args.initial + args.rounds)
        ),
        "interpretation": (
            "Gross policy-replay count, not a count of unique candidates. It is cheap "
            "for archived lookup but is not an acceptable cost model for wet-lab "
            "deployment."
        ),
    }
    selection["offline_selection_cost"] = offline_selection_cost
    summary = {
        "experiment": "care_calibrated_source_outcome_transfer",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "schema_route_proposal": route_proposal,
        "transfer_skill": skill.as_dict(),
        "source_history": {
            "seed": args.source_seed,
            "observation_count": args.source_observations,
            "fixed_across_target_seeds": True,
        },
        "source_outcome_components": [
            "source-outcome-informed initial target probes",
            "mapped source-neighbor outcome prior",
            "mapped additive source outcome-effect prior",
            "source pairwise interaction residual prior",
            "source-outcome-informed GP categorical geometry",
            "prequential target calibration and exact target-only fallback",
        ],
        "llm_model": record.get("model"),
        "llm_generation_call_count": int(record.get("llm_generation_call_count", 1)),
        "matched_target_only_llm": {
            "model": target_llm_record.get("model"),
            "generation_call_count": int(
                target_llm_record.get("llm_generation_call_count", 1)
            ),
            "frozen_mode": args.target_llm_mode,
            "record_path": str(args.target_llm_record),
        },
        "patches": [asdict(patch) for patch in patches],
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "initial_observations": args.initial,
        "rounds": args.rounds,
        "router": {
            "min_observations": args.router_min_observations,
            "min_quality": args.router_min_quality,
            "max_transfer_mass": args.router_max_transfer_mass,
            "source_initial_strategy": args.source_initial_strategy,
        },
        "selection": selection,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_pairwise": pairwise,
        "mechanism_controls": {
            "enabled": not args.no_mechanism_controls,
            "warmstart_only_mode": WARMSTART_ONLY_MODE,
            "fixed_data_only_mode": DATA_ONLY_MODE,
        },
        "mechanism_attribution": attribution,
        "offline_selection_cost": offline_selection_cost,
        "evidence_boundary": (
            "The source history contains public source features and measured source outcomes. "
            "Source history, mappings, and patches are frozen before target replay. Target "
            "outcomes enter only after reveal and are used prequentially; hidden target rows "
            "never enter source compilation, calibration, routing, or candidate scoring."
        ),
    }
    output_id = (
        f"calibrated_source_outcome_{args.source_dataset}_to_{args.target_dataset}"
    )
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    rows.sort(key=lambda row: (str(row.get("split", "")), int(row["seed"]), str(row["mode"])))
    outcome_router.write_outputs(output_id, rows, audits, summary)
    artifact_root = outcome_router.OUTPUT_RUNS
    skill.write(artifact_root / f"{output_id}_transfer_skill.json")
    trace = canonical_skill.build_canonical_trace(
        skill=skill,
        selection=selection,
        audits=audits,
        selector_mode=SELECTOR_MODE,
        heldout_seeds=heldout_seeds,
    )
    canonical_skill.write_trace(
        artifact_root / f"{output_id}_canonical_trace.jsonl",
        trace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
