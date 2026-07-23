#!/usr/bin/env python3
"""Executable semantic skills compiled from an LLM response."""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any

import run_llm_transfer_router as router
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted


@dataclass(frozen=True)
class SemanticRule:
    rule_id: str
    conditions: tuple[tuple[str, str], ...]
    weight: float
    rationale: str


@dataclass(frozen=True)
class SemanticSkill:
    skill_id: str
    rules: tuple[SemanticRule, ...]
    ridge: float
    prior_scale: float
    semantic_mass_start: float
    semantic_mass_end: float
    ucb_weight: float
    gp_beta_start: float
    gp_beta_end: float
    gp_xi: float
    confidence: float
    hypothesis: str


SEMANTIC_FIELDS: dict[str, tuple[str, ...]] = {
    "real_buchwald_hartwig": (
        "ligand_ligand_family",
        "ligand_has_phosphine",
        "ligand_mw_bin",
        "ligand_logp_bin",
        "ligand_aromatic_ring_bin",
        "base_reagent_base_family",
        "base_functional_class",
        "base_mw_bin",
        "base_tpsa_bin",
        "aryl_halide_halide_type",
        "aryl_halide_has_heteroaromatic",
        "aryl_halide_mw_bin",
        "additive_functional_class",
        "additive_tpsa_bin",
    ),
    "real_suzuki_miyaura": (
        "ligand_ligand_family",
        "ligand_has_phosphine",
        "ligand_mw_bin",
        "ligand_logp_bin",
        "catalyst_functional_class",
        "reagent_reagent_base_family",
        "reagent_functional_class",
        "reactant_1_halide_type",
        "reactant_1_has_heteroaromatic",
        "reactant_2_boron_species",
        "solvent_solvent_family",
        "solvent_solvent_is_protic",
    ),
    "real_chemlex_acidamine": (
        "acid_smiles_length_bin",
        "acid_hetero_atom_bin",
        "acid_aromatic_bin",
        "acid_ring_token_bin",
        "acid_branch_bin",
        "amine_smiles_length_bin",
        "amine_hetero_atom_bin",
        "amine_aromatic_bin",
        "amine_ring_token_bin",
        "amine_branch_bin",
        "reagent_smiles_length_bin",
        "reagent_hetero_atom_bin",
        "reagent_aromatic_bin",
        "reagent_branch_bin",
        "solvent_smiles_length_bin",
        "solvent_hetero_atom_bin",
        "solvent_aromatic_bin",
        "acid_nitrogen_bin",
        "acid_oxygen_bin",
        "acid_carbonyl_bin",
        "acid_amide_flag",
        "acid_nitrile_flag",
        "acid_sulfur_flag",
        "acid_formal_charge_flag",
        "acid_aromatic_hetero_flag",
        "amine_nitrogen_bin",
        "amine_oxygen_bin",
        "amine_carbonyl_bin",
        "amine_amide_flag",
        "amine_nitrile_flag",
        "amine_sulfur_flag",
        "amine_phosphorus_flag",
        "amine_formal_charge_flag",
        "amine_aromatic_hetero_flag",
        "reagent_nitrogen_bin",
        "reagent_oxygen_bin",
        "reagent_sulfur_flag",
        "reagent_phosphorus_flag",
        "reagent_formal_charge_flag",
        "reagent_coupling_family",
        "acid_rdkit_mw_bin",
        "acid_rdkit_logp_bin",
        "acid_rdkit_tpsa_bin",
        "acid_rdkit_hbd_bin",
        "acid_rdkit_hba_bin",
        "acid_rdkit_rotatable_bin",
        "acid_rdkit_aromatic_ring_bin",
        "acid_rdkit_fraction_csp3_bin",
        "acid_rdkit_complexity_bin",
        "acid_rdkit_formal_charge_class",
        "acid_rdkit_ring_system_class",
        "acid_rdkit_acid_functional_class",
        "acid_rdkit_amide_count_bin",
        "amine_rdkit_mw_bin",
        "amine_rdkit_logp_bin",
        "amine_rdkit_tpsa_bin",
        "amine_rdkit_hbd_bin",
        "amine_rdkit_hba_bin",
        "amine_rdkit_rotatable_bin",
        "amine_rdkit_aromatic_ring_bin",
        "amine_rdkit_fraction_csp3_bin",
        "amine_rdkit_complexity_bin",
        "amine_rdkit_formal_charge_class",
        "amine_rdkit_ring_system_class",
        "amine_rdkit_amine_functional_class",
        "amine_rdkit_amide_count_bin",
        "reagent_rdkit_mw_bin",
        "reagent_rdkit_logp_bin",
        "reagent_rdkit_tpsa_bin",
        "reagent_rdkit_formal_charge_class",
        "reagent_rdkit_ring_system_class",
        "reagent_rdkit_functional_class",
    ),
    "real_moleculenet_esol": (
        "smiles_length_bin",
        "hbond_donor_bin",
        "hbond_acceptor_bin",
        "ring_bin",
        "rotatable_bond_bin",
        "polarity_bin",
    ),
    "real_moleculenet_freesolv": (
        "smiles_length_bin",
        "hetero_atom_bin",
        "halogen_bin",
        "aromatic_bin",
        "ring_token_bin",
        "branch_bin",
        "double_bond_bin",
    ),
    "real_moleculenet_lipophilicity": (
        "smiles_length_bin",
        "hetero_atom_bin",
        "halogen_bin",
        "aromatic_bin",
        "ring_token_bin",
        "branch_bin",
        "double_bond_bin",
    ),
}


def semantic_fields_for(adapter: replay.DatasetAdapter) -> tuple[str, ...]:
    if adapter.dataset_id in SEMANTIC_FIELDS:
        return tuple(
            field
            for field in SEMANTIC_FIELDS[adapter.dataset_id]
            if any(field in candidate.metadata for candidate in adapter.candidates)
        )
    if adapter.dataset_id.startswith("real_matbench_"):
        return adapter.decision_columns
    return tuple(
        field
        for field in adapter.decision_columns
        if 1 < len({str(candidate.metadata.get(field, "")) for candidate in adapter.candidates}) <= 20
    )


def semantic_field_catalog(adapter: replay.DatasetAdapter) -> dict[str, dict[str, int]]:
    catalog: dict[str, dict[str, int]] = {}
    for field in semantic_fields_for(adapter):
        counts: dict[str, int] = {}
        for candidate in adapter.candidates:
            value = str(candidate.metadata.get(field, "")).strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
        if 1 < len(counts) <= 24:
            catalog[field] = dict(sorted(counts.items()))
    return catalog


def bounded_float(raw: Any, default: float, lower: float, upper: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def normalize_id(raw: Any, prefix: str, index: int) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_")
    return value[:64] or f"{prefix}_{index + 1}"


def normalize_skills(
    payload: dict[str, Any],
    catalog: dict[str, dict[str, int]],
    max_skills: int = 12,
) -> tuple[SemanticSkill, ...]:
    raw_skills = payload.get("skills", [])
    if not isinstance(raw_skills, list):
        return ()
    skills: list[SemanticSkill] = []
    seen: set[str] = set()
    for skill_index, raw_skill in enumerate(raw_skills[:max_skills]):
        if not isinstance(raw_skill, dict):
            continue
        skill_id = normalize_id(raw_skill.get("skill_id"), "skill", skill_index)
        if skill_id in seen:
            skill_id = f"{skill_id}_{skill_index + 1}"
        raw_rules = raw_skill.get("rules", [])
        rules: list[SemanticRule] = []
        if isinstance(raw_rules, list):
            for rule_index, raw_rule in enumerate(raw_rules[:12]):
                if not isinstance(raw_rule, dict):
                    continue
                raw_conditions = raw_rule.get("conditions", {})
                if isinstance(raw_conditions, list):
                    normalized_conditions: dict[str, Any] = {}
                    for item in raw_conditions:
                        if isinstance(item, dict):
                            normalized_conditions[str(item.get("field", ""))] = item.get("value", "")
                        elif isinstance(item, (list, tuple)) and len(item) == 2:
                            normalized_conditions[str(item[0])] = item[1]
                    raw_conditions = normalized_conditions
                if (
                    isinstance(raw_conditions, dict)
                    and set(raw_conditions) == {"exact_field"}
                    and isinstance(raw_conditions.get("exact_field"), str)
                    and "." in raw_conditions["exact_field"]
                ):
                    field, value = raw_conditions["exact_field"].split(".", 1)
                    raw_conditions = {field: value}
                if not isinstance(raw_conditions, dict):
                    continue
                conditions = tuple(sorted(
                    (str(field), str(value))
                    for field, value in raw_conditions.items()
                    if field in catalog and str(value) in catalog[field]
                ))
                if not conditions:
                    continue
                rules.append(SemanticRule(
                    rule_id=normalize_id(raw_rule.get("rule_id"), "rule", rule_index),
                    conditions=conditions,
                    weight=bounded_float(raw_rule.get("weight"), 0.0, -1.0, 1.0),
                    rationale=str(raw_rule.get("rationale", ""))[:400],
                ))
        if not rules:
            continue
        skills.append(SemanticSkill(
            skill_id=skill_id,
            rules=tuple(rules),
            ridge=bounded_float(raw_skill.get("ridge"), 1.0, 0.05, 20.0),
            prior_scale=bounded_float(raw_skill.get("prior_scale"), 0.25, 0.0, 1.5),
            semantic_mass_start=bounded_float(
                raw_skill.get("semantic_mass_start"), 0.25, 0.0, 0.85
            ),
            semantic_mass_end=bounded_float(
                raw_skill.get("semantic_mass_end"), 0.10, 0.0, 0.85
            ),
            ucb_weight=bounded_float(raw_skill.get("ucb_weight"), 0.5, 0.0, 1.0),
            gp_beta_start=bounded_float(raw_skill.get("gp_beta_start"), 1.5, 0.2, 4.0),
            gp_beta_end=bounded_float(raw_skill.get("gp_beta_end"), 1.0, 0.2, 4.0),
            gp_xi=bounded_float(raw_skill.get("gp_xi"), 0.01, 0.0, 0.20),
            confidence=bounded_float(raw_skill.get("confidence"), 0.5, 0.05, 1.0),
            hypothesis=str(raw_skill.get("hypothesis", ""))[:800],
        ))
        seen.add(skill_id)
    return tuple(skills)


def rule_matches(rule: SemanticRule, candidate: replay.Candidate) -> bool:
    return all(str(candidate.metadata.get(field, "")) == value for field, value in rule.conditions)


def feature_vector(skill: SemanticSkill, candidate: replay.Candidate) -> tuple[float, ...]:
    numeric = candidate.numeric_features or (candidate.x1, candidate.x2, candidate.x3)
    return (
        1.0,
        *(2.0 * float(value) - 1.0 for value in numeric),
        *(1.0 if rule_matches(rule, candidate) else 0.0 for rule in skill.rules),
    )


def cholesky_with_jitter(matrix: list[list[float]]) -> list[list[float]]:
    for attempt in range(6):
        jitter = 10 ** (-9 + attempt)
        candidate = [row[:] for row in matrix]
        for index in range(len(candidate)):
            candidate[index][index] += jitter
        try:
            return surrogate.cholesky_spd(candidate)
        except (ValueError, ZeroDivisionError):
            continue
    raise RuntimeError("Could not factor semantic-skill precision matrix")


def semantic_model_scores(
    skill: SemanticSkill,
    adapter: replay.DatasetAdapter,
    observed: list[replay.Candidate],
    observed_ids: set[str],
    beta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    vectors = [feature_vector(skill, candidate) for candidate in observed]
    y_values = [candidate.objective_value / 100.0 for candidate in observed]
    y_mean = mean(y_values)
    y_scale = max(pstdev(y_values), 0.05)
    y_norm = [(value - y_mean) / y_scale for value in y_values]
    dimension = len(vectors[0])
    precision = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    numeric_dimension = len(vectors[0]) - len(skill.rules)
    prior = [*([0.0] * numeric_dimension), *(
        skill.prior_scale * rule.weight for rule in skill.rules
    )]
    for i in range(dimension):
        precision[i][i] = skill.ridge if i else max(0.02, 0.1 * skill.ridge)
        target[i] = precision[i][i] * prior[i]
    for vector, outcome in zip(vectors, y_norm):
        for i in range(dimension):
            target[i] += vector[i] * outcome
            for j in range(i + 1):
                precision[i][j] += vector[i] * vector[j]
                if i != j:
                    precision[j][i] = precision[i][j]
    lower = cholesky_with_jitter(precision)
    coefficients = surrogate.solve_cholesky(lower, target)
    best_seen = max(y_values)
    scores: dict[str, float] = {}
    selected_components: dict[str, tuple[float, float]] = {}
    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue
        vector = feature_vector(skill, candidate)
        mean_norm = sum(value * coefficient for value, coefficient in zip(vector, coefficients))
        solved = surrogate.solve_cholesky(lower, list(vector))
        variance = max(1e-9, sum(value * item for value, item in zip(vector, solved)))
        prediction = y_mean + y_scale * mean_norm
        uncertainty = y_scale * math.sqrt(variance)
        improvement = prediction - best_seen - skill.gp_xi
        if uncertainty <= 1e-9:
            ei = max(0.0, improvement)
        else:
            z = improvement / uncertainty
            ei = improvement * surrogate.normal_cdf(z) + uncertainty * surrogate.normal_pdf(z)
        ucb = prediction + beta * uncertainty
        scores[candidate.candidate_id] = skill.ucb_weight * ucb + (1.0 - skill.ucb_weight) * ei
        selected_components[candidate.candidate_id] = (prediction, uncertainty)
    return scores, {
        "dimension": dimension,
        "ridge": skill.ridge,
        "prior_scale": skill.prior_scale,
        "coefficients": [round(value, 6) for value in coefficients],
        "y_mean": round(y_mean, 6),
        "y_scale": round(y_scale, 6),
        "candidate_components": selected_components,
    }


def scheduled_value(start: float, end: float, round_index: int, rounds: int) -> float:
    if rounds <= 1:
        return end
    fraction = round_index / (rounds - 1)
    return start + fraction * (end - start)


def skill_mode(skill: SemanticSkill) -> str:
    return f"llm_semantic_{skill.skill_id}"


def direct_prior_mode(skill: SemanticSkill) -> str:
    return f"llm_direct_prior_{skill.skill_id}"


def llambo_warmstart_mode(skill: SemanticSkill) -> str:
    return f"llambo_warmstart_{skill.skill_id}"


def fixed_rule_score(skill: SemanticSkill, candidate: replay.Candidate) -> float:
    denominator = max(1.0, sum(abs(rule.weight) for rule in skill.rules))
    return sum(
        rule.weight for rule in skill.rules if rule_matches(rule, candidate)
    ) / denominator


def run_direct_prior_skill(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    skill: SemanticSkill,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """LLM-only semantic prior blended with BO, without CARE target calibration."""
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
        for candidate in pool
    }
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[:task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {
        candidate.candidate_id
        for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]
    }
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        beta = scheduled_value(skill.gp_beta_start, skill.gp_beta_end, round_index, task.reveal_budget)
        semantic_mass = scheduled_value(
            skill.semantic_mass_start,
            skill.semantic_mass_end,
            round_index,
            task.reveal_budget,
        )
        anchors, anchor_diagnostics = router.target_anchor_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            beta,
            skill.gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        ucb_rank = weighted.rank_normalized(anchors["gp_ucb"])
        ei_rank = weighted.rank_normalized(anchors["gp_ei"])
        anchor_rank = {
            candidate_id: skill.ucb_weight * ucb_rank[candidate_id]
            + (1.0 - skill.ucb_weight) * ei_rank[candidate_id]
            for candidate_id in ucb_rank
        }
        prior_scores = {
            candidate.candidate_id: fixed_rule_score(skill, candidate)
            for candidate in pool
            if candidate.candidate_id not in observed_ids
        }
        prior_rank = weighted.rank_normalized(prior_scores)
        scores = {
            candidate_id: (1.0 - semantic_mass) * anchor_rank[candidate_id]
            + semantic_mass * prior_rank[candidate_id]
            for candidate_id in anchor_rank
        }
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append({
            "dataset_id": adapter.dataset_id,
            "seed": seed,
            "round_index": round_index,
            "mode": direct_prior_mode(skill),
            "public_observed_count": len(observed) - 1,
            "selected_candidate": selected_id,
            "selected_score": round(scores[selected_id], 6),
            "revealed_value": selected.objective_value,
            "best_so_far": best_so_far,
            "hypothesis_snapshot": {
                "skill": asdict(skill),
                "matched_rules": [
                    rule.rule_id for rule in skill.rules if rule_matches(rule, selected)
                ],
                "fixed_llm_prior_score": round(prior_scores[selected_id], 6),
                "semantic_mass": round(semantic_mass, 6),
                "anchor_diagnostics": anchor_diagnostics,
                "evidence_boundary": (
                    "Finite-pool LLM-direct baseline: rule weights remain frozen and are not "
                    "calibrated from target observations. Target observations only update the "
                    "shared GP-UCB/EI anchor."
                ),
            },
        })
    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": adapter.dataset_id,
        "mode": direct_prior_mode(skill),
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit


def run_llambo_warmstart(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    skill: SemanticSkill,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Finite-pool adaptation of LLAMBO zero-shot LLM warmstarting."""
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    rng = random.Random(seed)
    jitter = {candidate.candidate_id: rng.random() * 1e-8 for candidate in pool}
    ordered = sorted(
        pool,
        key=lambda candidate: (
            fixed_rule_score(skill, candidate) + jitter[candidate.candidate_id],
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
        selected_ids = {candidate.candidate_id for candidate in observed}
        observed.extend(
            candidate
            for candidate in ordered
            if candidate.candidate_id not in selected_ids
        )
        observed = observed[:task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    warmstart_ids = sorted(observed_ids)
    top10 = {
        candidate.candidate_id
        for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]
    }
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
        for candidate in pool
    }
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        beta = scheduled_value(skill.gp_beta_start, skill.gp_beta_end, round_index, task.reveal_budget)
        anchors, anchor_diagnostics = router.target_anchor_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            beta,
            skill.gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        ucb_rank = weighted.rank_normalized(anchors["gp_ucb"])
        ei_rank = weighted.rank_normalized(anchors["gp_ei"])
        scores = {
            candidate_id: skill.ucb_weight * ucb_rank[candidate_id]
            + (1.0 - skill.ucb_weight) * ei_rank[candidate_id]
            for candidate_id in ucb_rank
        }
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append({
            "dataset_id": adapter.dataset_id,
            "seed": seed,
            "round_index": round_index,
            "mode": llambo_warmstart_mode(skill),
            "public_observed_count": len(observed) - 1,
            "selected_candidate": selected_id,
            "selected_score": round(scores[selected_id], 6),
            "revealed_value": selected.objective_value,
            "best_so_far": best_so_far,
            "hypothesis_snapshot": {
                "skill": asdict(skill),
                "warmstart_candidates": warmstart_ids,
                "anchor_diagnostics": anchor_diagnostics,
                "evidence_boundary": (
                    "Finite-pool adaptation of LLAMBO zero-shot warmstarting. The LLM skill "
                    "selects only the initial batch; subsequent rounds use the shared target-only "
                    "GP-UCB/EI anchor."
                ),
            },
        })
    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": adapter.dataset_id,
        "mode": llambo_warmstart_mode(skill),
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit


def run_semantic_skill(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    skill: SemanticSkill,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
        for candidate in pool
    }
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    observed = shuffled[:task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {
        candidate.candidate_id
        for candidate in sorted(pool, key=lambda item: item.objective_value, reverse=True)[:10]
    }
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        beta = scheduled_value(
            skill.gp_beta_start,
            skill.gp_beta_end,
            round_index,
            task.reveal_budget,
        )
        semantic_mass = scheduled_value(
            skill.semantic_mass_start,
            skill.semantic_mass_end,
            round_index,
            task.reveal_budget,
        )
        anchors, anchor_diagnostics = router.target_anchor_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            beta,
            skill.gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        ucb_rank = weighted.rank_normalized(anchors["gp_ucb"])
        ei_rank = weighted.rank_normalized(anchors["gp_ei"])
        anchor_rank = {
            candidate_id: skill.ucb_weight * ucb_rank[candidate_id]
            + (1.0 - skill.ucb_weight) * ei_rank[candidate_id]
            for candidate_id in ucb_rank
        }
        if semantic_mass > 0.0:
            semantic_scores, semantic_diagnostics = semantic_model_scores(
                skill,
                adapter,
                observed,
                observed_ids,
                beta,
            )
            semantic_rank = weighted.rank_normalized(semantic_scores)
            scores = {
                candidate_id: (1.0 - semantic_mass) * anchor_rank[candidate_id]
                + semantic_mass * semantic_rank[candidate_id]
                for candidate_id in anchor_rank
            }
        else:
            semantic_diagnostics = {
                "candidate_components": {},
                "skipped": True,
                "reason": "semantic_mass_is_zero",
            }
            scores = anchor_rank
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        candidate_components = semantic_diagnostics.pop("candidate_components")
        prediction, uncertainty = candidate_components.get(selected_id, (0.0, 0.0))
        matched_rules = [
            rule.rule_id for rule in skill.rules if rule_matches(rule, selected)
        ]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append({
            "dataset_id": adapter.dataset_id,
            "seed": seed,
            "round_index": round_index,
            "mode": skill_mode(skill),
            "public_observed_count": len(observed) - 1,
            "selected_candidate": selected_id,
            "selected_score": round(scores[selected_id], 6),
            "revealed_value": selected.objective_value,
            "best_so_far": best_so_far,
            "hypothesis_snapshot": {
                "skill": asdict(skill),
                "matched_rules": matched_rules,
                "semantic_mass": round(semantic_mass, 6),
                "gp_beta": round(beta, 6),
                "semantic_prediction": round(prediction, 6),
                "semantic_uncertainty": round(uncertainty, 6),
                "semantic_model": semantic_diagnostics,
                "anchor_diagnostics": anchor_diagnostics,
                "evidence_boundary": (
                    "The LLM skill and rule vocabulary were frozen before this replay. "
                    "Only target outcomes revealed in earlier rounds fit the semantic surrogate."
                ),
            },
        })
    final_best = max(candidate.objective_value for candidate in observed)
    return {
        "dataset": adapter.dataset_id,
        "mode": skill_mode(skill),
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }, audit
