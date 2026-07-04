#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

TransferMode = str
DEFAULT_MODES: tuple[TransferMode, ...] = (
    "no_care_random",
    "incumbent",
    "target_local_no_gate",
    "target_local_gate_v1",
    "transfer_no_gate",
    "transfer_gate_v1",
    "transfer_plus_local_gate_v1",
    "transfer_strict_no_gate",
    "transfer_strict_gate_v1",
    "transfer_strict_plus_local_gate_v1",
    "transfer_value_prior_no_gate",
    "transfer_value_prior_gate_v1",
    "transfer_value_prior_strict_gate_v1",
    "transfer_descriptor_value_prior_no_gate",
    "transfer_descriptor_value_prior_gate_v1",
    "transfer_descriptor_value_prior_strict_gate_v1",
    "llm_transfer_gate_v1",
    "llm_transfer_strict_gate_v1",
    "llm_audit_transfer_gate_v1",
    "llm_audit_transfer_strict_gate_v1",
)


@dataclass(frozen=True)
class TransferRole:
    source_field: str
    target_field: str
    source_support_count: int
    source_mean_abs_effect: float
    source_max_abs_effect: float
    confidence: float
    transfer_weight: float


@dataclass(frozen=True)
class TransferValuePrior:
    source_field: str
    target_field: str
    value: str
    source_support_count: int
    source_effect: float
    confidence: float
    transfer_weight: float


@dataclass(frozen=True)
class TransferCard:
    card_id: str
    source_dataset: str
    target_dataset: str
    source_observation_count: int
    discount: float
    min_source_support: int
    role_map: dict[str, str]
    roles: tuple[TransferRole, ...]
    value_priors: tuple[TransferValuePrior, ...]
    evidence_summary: str


def parse_modes(raw: str) -> tuple[TransferMode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(DEFAULT_MODES)}")
    return modes


def is_llm_transfer_mode(mode: TransferMode) -> bool:
    return mode in {"llm_transfer_gate_v1", "llm_transfer_strict_gate_v1"}


def is_llm_audit_transfer_mode(mode: TransferMode) -> bool:
    return mode in {"llm_audit_transfer_gate_v1", "llm_audit_transfer_strict_gate_v1"}


def is_any_llm_mode(mode: TransferMode) -> bool:
    return is_llm_transfer_mode(mode) or is_llm_audit_transfer_mode(mode)


def is_value_prior_mode(mode: TransferMode) -> bool:
    return mode in {
        "transfer_value_prior_no_gate",
        "transfer_value_prior_gate_v1",
        "transfer_value_prior_strict_gate_v1",
        "transfer_descriptor_value_prior_no_gate",
        "transfer_descriptor_value_prior_gate_v1",
        "transfer_descriptor_value_prior_strict_gate_v1",
    }


def is_descriptor_value_prior_mode(mode: TransferMode) -> bool:
    return mode in {
        "transfer_descriptor_value_prior_no_gate",
        "transfer_descriptor_value_prior_gate_v1",
        "transfer_descriptor_value_prior_strict_gate_v1",
    }


ROLE_MAPS: dict[tuple[str, str], dict[str, str]] = {
    (
        "real_buchwald_hartwig",
        "real_suzuki_miyaura",
    ): {
        "ligand": "ligand",
        "base": "reagent",
        "aryl_halide": "reactant_1",
        "additive": "solvent",
    },
    (
        "real_suzuki_miyaura",
        "real_buchwald_hartwig",
    ): {
        "ligand": "ligand",
        "reagent": "base",
        "reactant_1": "aryl_halide",
        "solvent": "additive",
    },
    (
        "real_suzuki_miyaura",
        "real_chemlex_acidamine",
    ): {
        "reactant_1": "acid",
        "reactant_2": "amine",
        "reagent": "reagent",
        "solvent": "solvent",
    },
    (
        "real_chemlex_acidamine",
        "real_suzuki_miyaura",
    ): {
        "acid": "reactant_1",
        "amine": "reactant_2",
        "reagent": "reagent",
        "solvent": "solvent",
    },
    (
        "real_buchwald_hartwig",
        "real_chemlex_acidamine",
    ): {
        "aryl_halide": "acid",
        "base": "reagent",
        "additive": "solvent",
    },
    (
        "real_chemlex_acidamine",
        "real_buchwald_hartwig",
    ): {
        "acid": "aryl_halide",
        "reagent": "base",
        "solvent": "additive",
    },
    (
        "synthetic_chemlex_i",
        "real_chemlex_acidamine",
    ): {
        "acid": "acid",
        "amine": "amine",
        "base_equivalents": "reagent",
        "solvent_polarity": "solvent",
    },
    (
        "synthetic_suzuki_i",
        "real_suzuki_miyaura",
    ): {
        "ligand_identity": "ligand",
        "catalyst_loading": "catalyst",
        "temperature": "reagent",
        "residence_time": "solvent",
    },
    (
        "synthetic_materials_i",
        "real_matbench_expt_gap",
    ): {
        "dopant": "dominant_family",
        "dopant_ratio": "max_element_fraction_bin",
        "anneal_temperature": "mean_atomic_number_bin",
        "dwell_time": "element_count_bin",
    },
    (
        "real_moleculenet_freesolv",
        "real_moleculenet_lipophilicity",
    ): {
        "smiles_length_bin": "smiles_length_bin",
        "hetero_atom_bin": "hetero_atom_bin",
        "halogen_bin": "halogen_bin",
        "aromatic_bin": "aromatic_bin",
        "ring_token_bin": "ring_token_bin",
        "branch_bin": "branch_bin",
        "double_bond_bin": "double_bond_bin",
    },
    (
        "real_moleculenet_lipophilicity",
        "real_moleculenet_freesolv",
    ): {
        "smiles_length_bin": "smiles_length_bin",
        "hetero_atom_bin": "hetero_atom_bin",
        "halogen_bin": "halogen_bin",
        "aromatic_bin": "aromatic_bin",
        "ring_token_bin": "ring_token_bin",
        "branch_bin": "branch_bin",
        "double_bond_bin": "double_bond_bin",
    },
    (
        "real_moleculenet_esol",
        "real_moleculenet_freesolv",
    ): {
        "smiles_length_bin": "smiles_length_bin",
        "hbond_donor_bin": "hetero_atom_bin",
        "ring_bin": "ring_token_bin",
        "rotatable_bond_bin": "branch_bin",
    },
    (
        "real_moleculenet_esol",
        "real_moleculenet_lipophilicity",
    ): {
        "smiles_length_bin": "smiles_length_bin",
        "hbond_donor_bin": "hetero_atom_bin",
        "ring_bin": "ring_token_bin",
        "rotatable_bond_bin": "branch_bin",
    },
}


REACTION_DESCRIPTOR_ROLE_MAPS: dict[tuple[str, str], dict[str, str]] = {
    (
        "real_buchwald_hartwig",
        "real_suzuki_miyaura",
    ): {
        "ligand_ligand_family": "ligand_ligand_family",
        "ligand_has_phosphine": "ligand_has_phosphine",
        "ligand_has_phosphorus": "ligand_has_phosphorus",
        "ligand_mw_bin": "ligand_mw_bin",
        "ligand_logp_bin": "ligand_logp_bin",
        "ligand_aromatic_ring_bin": "ligand_aromatic_ring_bin",
        "ligand_functional_class": "ligand_functional_class",
        "base_reagent_base_family": "reagent_reagent_base_family",
        "base_functional_class": "reagent_functional_class",
        "base_mw_bin": "reagent_mw_bin",
        "base_tpsa_bin": "reagent_tpsa_bin",
        "base_has_phosphorus": "reagent_has_phosphorus",
        "aryl_halide_halide_type": "reactant_1_halide_type",
        "aryl_halide_has_aryl_halide": "reactant_1_has_aryl_halide",
        "aryl_halide_has_boron": "reactant_1_has_boron",
        "aryl_halide_boron_species": "reactant_1_boron_species",
        "aryl_halide_has_heteroaromatic": "reactant_1_has_heteroaromatic",
        "aryl_halide_functional_class": "reactant_1_functional_class",
        "aryl_halide_mw_bin": "reactant_1_mw_bin",
        "additive_functional_class": "solvent_functional_class",
        "additive_mw_bin": "solvent_mw_bin",
        "additive_tpsa_bin": "solvent_tpsa_bin",
    },
    (
        "real_suzuki_miyaura",
        "real_buchwald_hartwig",
    ): {
        "ligand_ligand_family": "ligand_ligand_family",
        "ligand_has_phosphine": "ligand_has_phosphine",
        "ligand_has_phosphorus": "ligand_has_phosphorus",
        "ligand_mw_bin": "ligand_mw_bin",
        "ligand_logp_bin": "ligand_logp_bin",
        "ligand_aromatic_ring_bin": "ligand_aromatic_ring_bin",
        "ligand_functional_class": "ligand_functional_class",
        "reagent_reagent_base_family": "base_reagent_base_family",
        "reagent_functional_class": "base_functional_class",
        "reagent_mw_bin": "base_mw_bin",
        "reagent_tpsa_bin": "base_tpsa_bin",
        "reagent_has_phosphorus": "base_has_phosphorus",
        "reactant_1_halide_type": "aryl_halide_halide_type",
        "reactant_1_has_aryl_halide": "aryl_halide_has_aryl_halide",
        "reactant_1_has_boron": "aryl_halide_has_boron",
        "reactant_1_boron_species": "aryl_halide_boron_species",
        "reactant_1_has_heteroaromatic": "aryl_halide_has_heteroaromatic",
        "reactant_1_functional_class": "aryl_halide_functional_class",
        "reactant_1_mw_bin": "aryl_halide_mw_bin",
        "solvent_functional_class": "additive_functional_class",
        "solvent_mw_bin": "additive_mw_bin",
        "solvent_tpsa_bin": "additive_tpsa_bin",
    },
}


def descriptor_role_map_for(source_dataset: str, target_dataset: str) -> dict[str, str]:
    return dict(REACTION_DESCRIPTOR_ROLE_MAPS.get((source_dataset, target_dataset), {}))


def descriptor_transfer_role_map_for(source_dataset: str, target_dataset: str) -> dict[str, str]:
    role_map = role_map_for(source_dataset, target_dataset)
    role_map.update(descriptor_role_map_for(source_dataset, target_dataset))
    return role_map


VALUE_PRIOR_FIELDS: dict[tuple[str, str], set[tuple[str, str]]] = {
    (
        "real_moleculenet_freesolv",
        "real_moleculenet_lipophilicity",
    ): {
        ("smiles_length_bin", "smiles_length_bin"),
        ("hetero_atom_bin", "hetero_atom_bin"),
        ("halogen_bin", "halogen_bin"),
        ("aromatic_bin", "aromatic_bin"),
        ("ring_token_bin", "ring_token_bin"),
        ("branch_bin", "branch_bin"),
        ("double_bond_bin", "double_bond_bin"),
    },
    (
        "real_moleculenet_lipophilicity",
        "real_moleculenet_freesolv",
    ): {
        ("smiles_length_bin", "smiles_length_bin"),
        ("hetero_atom_bin", "hetero_atom_bin"),
        ("halogen_bin", "halogen_bin"),
        ("aromatic_bin", "aromatic_bin"),
        ("ring_token_bin", "ring_token_bin"),
        ("branch_bin", "branch_bin"),
        ("double_bond_bin", "double_bond_bin"),
    },
    (
        "real_moleculenet_esol",
        "real_moleculenet_freesolv",
    ): {("smiles_length_bin", "smiles_length_bin")},
    (
        "real_moleculenet_esol",
        "real_moleculenet_lipophilicity",
    ): {("smiles_length_bin", "smiles_length_bin")},
}


for _reaction_pair, _descriptor_role_map in REACTION_DESCRIPTOR_ROLE_MAPS.items():
    VALUE_PRIOR_FIELDS.setdefault(_reaction_pair, set()).update(_descriptor_role_map.items())


def role_map_for(source_dataset: str, target_dataset: str) -> dict[str, str]:
    if (source_dataset, target_dataset) in ROLE_MAPS:
        return dict(ROLE_MAPS[(source_dataset, target_dataset)])
    raise ValueError(f"No role map defined for {source_dataset} -> {target_dataset}")


def source_observations(adapter: replay.DatasetAdapter, seed: int, count: int) -> list[replay.Candidate]:
    shuffled = list(adapter.candidates)
    random.Random(10_000 + seed).shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def compile_transfer_card(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    observed: list[replay.Candidate],
    role_map: dict[str, str],
    discount: float,
    min_source_support: int,
    source_value_effect_threshold: float = 3.0,
) -> TransferCard:
    source_factor_columns = tuple(dict.fromkeys((*source_adapter.decision_columns, *role_map.keys())))
    factor_summary = replay.factor_stats(observed, source_factor_columns)
    global_mean = replay.observed_mean(observed)
    effects_by_field: dict[str, list[float]] = {field: [] for field in role_map}
    for (field_name, _value), (count, value_mean) in factor_summary.items():
        if field_name not in role_map or count < min_source_support:
            continue
        effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
        effects_by_field[field_name].append(effect)

    roles: list[TransferRole] = []
    for source_field, target_field in role_map.items():
        effects = effects_by_field[source_field]
        source_support_count = len(effects)
        mean_abs = mean(abs(effect) for effect in effects) if effects else 0.0
        max_abs = max((abs(effect) for effect in effects), default=0.0)
        support_confidence = min(1.0, math.log1p(source_support_count) / math.log1p(12.0))
        effect_confidence = min(1.0, max_abs / 18.0)
        confidence = round(discount * support_confidence * effect_confidence, 4)
        transfer_weight = round(min(1.20, 0.35 + confidence), 4) if source_support_count else 0.0
        roles.append(
            TransferRole(
                source_field=source_field,
                target_field=target_field,
                source_support_count=source_support_count,
                source_mean_abs_effect=round(mean_abs, 4),
                source_max_abs_effect=round(max_abs, 4),
                confidence=confidence,
                transfer_weight=transfer_weight,
            )
        )

    target_value_fields = tuple(dict.fromkeys((*target_adapter.decision_columns, *role_map.values())))
    target_value_counts_by_field = {
        field: Counter(str(candidate.metadata.get(field, "")) for candidate in target_adapter.candidates)
        for field in target_value_fields
    }
    target_values_by_field = {field: set(counts) for field, counts in target_value_counts_by_field.items()}
    source_values_by_field: dict[str, set[str]] = {}
    for (field_name, value), (_count, _value_mean) in factor_summary.items():
        source_values_by_field.setdefault(field_name, set()).add(str(value))
    active_value_fields = VALUE_PRIOR_FIELDS.get((source_adapter.dataset_id, target_adapter.dataset_id), set())
    value_priors: list[TransferValuePrior] = []
    role_by_pair = {(role.source_field, role.target_field): role for role in roles}
    for source_field, target_field in sorted(active_value_fields):
        if role_map.get(source_field) != target_field:
            continue
        role = role_by_pair.get((source_field, target_field))
        if role is None or role.transfer_weight <= 0.0:
            continue
        target_values = target_values_by_field.get(target_field, set())
        if len(source_values_by_field.get(source_field, set())) < 2 or len(target_values) < 2:
            continue
        for (field_name, value), (count, value_mean) in factor_summary.items():
            if field_name != source_field or count < min_source_support or str(value) not in target_values:
                continue
            source_coverage = count / max(1, len(observed))
            target_coverage = target_value_counts_by_field.get(target_field, Counter()).get(str(value), 0) / max(
                1, len(target_adapter.candidates)
            )
            if source_coverage > 0.85 or target_coverage > 0.85:
                continue
            effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
            if abs(effect) < source_value_effect_threshold:
                continue
            support_confidence = min(1.0, math.log1p(count) / math.log1p(10.0))
            effect_confidence = min(1.0, abs(effect) / 16.0)
            confidence = round(discount * support_confidence * effect_confidence, 4)
            transfer_weight = round(min(0.08, abs(effect / 100.0) * role.transfer_weight), 6)
            value_priors.append(
                TransferValuePrior(
                    source_field=source_field,
                    target_field=target_field,
                    value=str(value),
                    source_support_count=count,
                    source_effect=round(effect, 4),
                    confidence=confidence,
                    transfer_weight=transfer_weight,
                )
            )

    active = [role for role in roles if role.transfer_weight > 0.0]
    summary = (
        f"Compiled {len(active)} active role transfers from {source_adapter.dataset_id} "
        f"using {len(observed)} observed source rows and {len(value_priors)} conservative "
        "shared-vocabulary value priors. Direction is not transferred as a raw chemical "
        "rule; the card transfers role-level evidence strength and only transfers "
        "source value priors when source and target fields share the same public "
        "descriptor vocabulary."
    )
    return TransferCard(
        card_id=f"transfer_{source_adapter.dataset_id}_to_{target_adapter.dataset_id}",
        source_dataset=source_adapter.dataset_id,
        target_dataset=target_adapter.dataset_id,
        source_observation_count=len(observed),
        discount=discount,
        min_source_support=min_source_support,
        role_map=dict(role_map),
        roles=tuple(roles),
        value_priors=tuple(value_priors),
        evidence_summary=summary,
    )


def transfer_adjustments(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    strict: bool = False,
    min_role_confidence: float = 0.18,
    min_positive_roles: int = 2,
    max_negative_roles: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    if len(observed) < 8:
        return adjustments, {
            "skills": {"cross_domain_transfer_card": {"active": False, "reason": "observed_count_below_8"}},
            "max_abs_adjustment": 0.0,
        }

    target_factor_columns = tuple(
        dict.fromkeys((*adapter.decision_columns, *(role.target_field for role in card.roles)))
    )
    factor_summary = replay.factor_stats(observed, target_factor_columns)
    global_mean = replay.observed_mean(observed)
    role_by_target = {
        role.target_field: role
        for role in card.roles
        if role.transfer_weight > 0.0 and (not strict or role.confidence >= min_role_confidence)
    }
    applied_specs: list[dict[str, Any]] = []
    positive = 0
    negative = 0
    scored = 0
    strict_rejected = 0

    for c in pool:
        if c.candidate_id not in adjustments:
            continue
        signals: list[float] = []
        for target_field, role in role_by_target.items():
            value = str(c.metadata.get(target_field, ""))
            if not value:
                continue
            count, value_mean = factor_summary.get((target_field, value), (0, global_mean))
            if count < min_target_support:
                continue
            effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
            if abs(effect) < effect_threshold:
                continue
            signals.append((effect / 100.0) * role.transfer_weight)
        if not signals:
            continue
        if strict:
            positive_signals = [signal for signal in signals if signal > 0]
            negative_signals = [signal for signal in signals if signal < 0]
            if len(positive_signals) < min_positive_roles or len(negative_signals) > max_negative_roles:
                strict_rejected += 1
                continue
            bounded = max(0.0, min(0.055, mean(positive_signals)))
        else:
            bounded = max(-0.10, min(0.10, mean(signals)))
        adjustments[c.candidate_id] += bounded
        scored += 1
        positive += int(bounded > 0)
        negative += int(bounded < 0)

    for target_field, role in role_by_target.items():
        active_values = []
        for (field_name, value), (count, value_mean) in factor_summary.items():
            if field_name != target_field or count < min_target_support:
                continue
            effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
            if abs(effect) >= effect_threshold:
                active_values.append(
                    {
                        "value": value,
                        "count": count,
                        "effect": round(effect, 4),
                    }
                )
        applied_specs.append(
            {
                "source_field": role.source_field,
                "target_field": target_field,
                "transfer_weight": role.transfer_weight,
                "confidence": role.confidence,
                "strict_enabled": strict,
                "active_target_values": sorted(active_values, key=lambda item: abs(item["effect"]), reverse=True)[:8],
            }
        )

    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert = {
        "skills": {
            "cross_domain_transfer_card": {
                "active": scored > 0,
                "card_id": card.card_id,
                "scored_candidates": scored,
                "positive_adjustments": positive,
                "negative_adjustments": negative,
                "strict_rejected_candidates": strict_rejected,
                "strict_min_role_confidence": min_role_confidence if strict else 0.0,
                "strict_min_positive_roles": min_positive_roles if strict else 0,
                "strict_max_negative_roles": max_negative_roles if strict else 0,
                "applied_specs": applied_specs,
            }
        },
        "max_abs_adjustment": round(max_abs, 6),
    }
    return adjustments, cert


def source_value_prior_adjustments(
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    card: TransferCard,
    strict: bool = False,
    min_prior_confidence: float = 0.10,
    min_positive_priors: int = 1,
    max_negative_priors: int = 0,
    skill_id: str = "source_value_prior",
    signed_adjustment_cap: float = 0.08,
    positive_adjustment_cap: float = 0.06,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    active_priors = [
        prior
        for prior in card.value_priors
        if prior.transfer_weight > 0.0 and (not strict or prior.confidence >= min_prior_confidence)
    ]
    if not active_priors:
        return adjustments, {
            "skills": {
                skill_id: {
                    "active": False,
                    "reason": "no_shared_vocabulary_value_priors",
                }
            },
            "max_abs_adjustment": 0.0,
        }

    priors_by_target: dict[tuple[str, str], list[TransferValuePrior]] = {}
    for prior in active_priors:
        priors_by_target.setdefault((prior.target_field, prior.value), []).append(prior)

    scored = 0
    positive = 0
    negative = 0
    strict_rejected = 0
    for candidate in pool:
        if candidate.candidate_id not in adjustments:
            continue
        signals: list[float] = []
        for (target_field, value), priors in priors_by_target.items():
            if str(candidate.metadata.get(target_field, "")) != value:
                continue
            for prior in priors:
                direction = 1.0 if prior.source_effect > 0 else -1.0
                signals.append(direction * prior.transfer_weight)
        if not signals:
            continue
        if strict:
            positive_signals = [signal for signal in signals if signal > 0]
            negative_signals = [signal for signal in signals if signal < 0]
            if len(positive_signals) < min_positive_priors or len(negative_signals) > max_negative_priors:
                strict_rejected += 1
                continue
            bounded = max(0.0, min(positive_adjustment_cap, mean(positive_signals)))
        else:
            bounded = max(-signed_adjustment_cap, min(signed_adjustment_cap, mean(signals)))
        adjustments[candidate.candidate_id] += bounded
        scored += 1
        positive += int(bounded > 0)
        negative += int(bounded < 0)

    applied_specs = [
        {
            "source_field": prior.source_field,
            "target_field": prior.target_field,
            "value": prior.value,
            "source_support_count": prior.source_support_count,
            "source_effect": prior.source_effect,
            "confidence": prior.confidence,
            "transfer_weight": prior.transfer_weight,
        }
        for prior in sorted(active_priors, key=lambda item: abs(item.source_effect), reverse=True)[:20]
    ]
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert = {
        "skills": {
            skill_id: {
                "active": scored > 0,
                "card_id": card.card_id,
                "scored_candidates": scored,
                "positive_adjustments": positive,
                "negative_adjustments": negative,
                "strict_rejected_candidates": strict_rejected,
                "strict_enabled": strict,
                "min_prior_confidence": min_prior_confidence if strict else 0.0,
                "min_positive_priors": min_positive_priors if strict else 0,
                "max_negative_priors": max_negative_priors if strict else 0,
                "signed_adjustment_cap": signed_adjustment_cap,
                "positive_adjustment_cap": positive_adjustment_cap if strict else 0.0,
                "applied_specs": applied_specs,
            }
        },
        "max_abs_adjustment": round(max_abs, 6),
    }
    return adjustments, cert


def llm_transfer_prompt_payload(
    adapter: replay.DatasetAdapter,
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    strict: bool,
    min_role_confidence: float,
    min_positive_roles: int,
    max_negative_roles: int,
) -> dict[str, Any]:
    active_roles = [asdict(role) for role in card.roles if role.transfer_weight > 0.0]
    return {
        "target_evidence": replay.observed_evidence_payload(adapter, observed),
        "transfer_card": {
            "card_id": card.card_id,
            "source_dataset": card.source_dataset,
            "target_dataset": card.target_dataset,
            "source_observation_count": card.source_observation_count,
            "discount": card.discount,
            "role_map": card.role_map,
            "roles": active_roles,
            "evidence_summary": card.evidence_summary,
        },
        "transfer_boundary": (
            "Use the transfer card only as source-to-target role-level evidence strength. "
            "Do not assume hidden target outcomes, and do not transfer source factor values directly."
        ),
        "selection_constraints": {
            "allowed_fields": sorted({role["target_field"] for role in active_roles}),
            "min_target_support": min_target_support,
            "effect_threshold": effect_threshold,
            "strict": strict,
            "strict_min_role_confidence": min_role_confidence if strict else 0.0,
            "strict_min_positive_roles": min_positive_roles if strict else 0,
            "strict_max_negative_roles": max_negative_roles if strict else 0,
        },
        "output_contract": {
            "adjustments": [
                {
                    "field": "one allowed target decision column",
                    "value": "one target factor value supported by factor_evidence/top_revealed/bottom_revealed",
                    "direction": "prefer or penalize",
                    "weight": "number between 0.0 and 0.08",
                    "reason": "short evidence-based reason",
                }
            ],
            "confidence": "number between 0 and 1",
        },
    }


def llm_transfer_adjustments(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    seed: int,
    round_index: int,
    mode: TransferMode,
    config: replay.LLMConfig,
    strict: bool = False,
    min_role_confidence: float = 0.18,
    min_positive_roles: int = 2,
    max_negative_roles: int = 0,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    if len(observed) < 8:
        cert = {
            "skills": {"llm_cross_domain_transfer_card": {"active": False, "reason": "observed_count_below_8"}},
            "max_abs_adjustment": 0.0,
        }
        return adjustments, cert, {"called": False, "reason": "observed_count_below_8"}

    role_by_target = {
        role.target_field: role
        for role in card.roles
        if role.transfer_weight > 0.0 and (not strict or role.confidence >= min_role_confidence)
    }
    if not role_by_target:
        cert = {
            "skills": {"llm_cross_domain_transfer_card": {"active": False, "reason": "no_active_transfer_roles"}},
            "max_abs_adjustment": 0.0,
        }
        return adjustments, cert, {"called": False, "reason": "no_active_transfer_roles"}

    prompt_payload = llm_transfer_prompt_payload(
        adapter,
        observed,
        card,
        min_target_support,
        effect_threshold,
        strict,
        min_role_confidence,
        min_positive_roles,
        max_negative_roles,
    )
    system = (
        "You are a CARE 2.0 cross-domain transfer policy proposer for scientific finite-pool replay. "
        "Use only the revealed target observations and the source-to-target role-level transfer card in the user JSON. "
        "Do not assume hidden outcomes for unrevealed candidates. Do not transfer source factor values directly. "
        "Return only one JSON object with an adjustments array and confidence. No markdown. No prose. No chain-of-thought."
    )
    user = (
        "Propose bounded target factor-level score adjustments for the next candidate selection. "
        "Use fields only from selection_constraints.allowed_fields. Use values supported by target factor_evidence, "
        "top_revealed, or bottom_revealed. The transfer card tells you which target roles are reliable enough to reuse; "
        "target observations determine the direction. Max 4 adjustments. Return exactly this shape: "
        "{\"adjustments\":[{\"field\":\"ligand\",\"value\":\"L2\",\"direction\":\"prefer\",\"weight\":0.05,\"reason\":\"short evidence reason\"}],\"confidence\":0.7}. "
        "JSON input:\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
    )
    content, response_meta = replay.chat_completion_text(
        config,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )

    parse_error = ""
    try:
        parsed = replay.extract_json_object(content)
    except ValueError as exc:
        parsed = {"adjustments": [], "confidence": 0.0}
        parse_error = str(exc)
    if "adjustments" not in parsed and {"field", "value", "direction"} <= set(parsed):
        parsed = {"adjustments": [parsed], "confidence": parsed.get("confidence", 0.5)}

    factor_summary = replay.factor_stats(observed, adapter.decision_columns)
    global_mean = replay.observed_mean(observed)
    applied_specs: list[dict[str, Any]] = []
    rejected_specs: list[dict[str, Any]] = []
    positive = 0
    negative = 0
    scored = 0
    strict_rejected_candidates = 0

    for item in list(parsed.get("adjustments", []))[:4]:
        if not isinstance(item, dict):
            continue
        field_name = str(item.get("field", ""))
        value = str(item.get("value", ""))
        direction = str(item.get("direction", "")).lower()
        role = role_by_target.get(field_name)
        if role is None or direction not in {"prefer", "penalize"}:
            rejected_specs.append({"item": item, "reason": "field_or_direction_not_allowed"})
            continue
        count, value_mean = factor_summary.get((field_name, value), (0, global_mean))
        if count < min_target_support:
            rejected_specs.append({"item": item, "reason": "insufficient_target_support", "target_support_count": count})
            continue
        target_effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
        if abs(target_effect) < effect_threshold:
            rejected_specs.append({"item": item, "reason": "target_effect_below_threshold", "target_effect": round(target_effect, 4)})
            continue
        if strict:
            if role.confidence < min_role_confidence:
                rejected_specs.append({"item": item, "reason": "role_confidence_below_strict_threshold", "role_confidence": role.confidence})
                continue
            if direction != "prefer" or target_effect <= 0:
                rejected_specs.append({"item": item, "reason": "strict_mode_requires_positive_target_signal", "target_effect": round(target_effect, 4)})
                continue
        try:
            magnitude = min(0.08, max(0.0, abs(float(item.get("weight", 0.0)))))
        except (TypeError, ValueError):
            rejected_specs.append({"item": item, "reason": "invalid_weight"})
            continue
        scaled = magnitude * min(1.0, role.transfer_weight)
        cap = 0.055 if strict else 0.10
        delta = min(cap, scaled) if direction == "prefer" else -min(cap, scaled)
        matched = 0
        for c in pool:
            if c.candidate_id not in adjustments:
                continue
            if str(c.metadata.get(field_name, "")) != value:
                continue
            adjustments[c.candidate_id] += delta
            matched += 1
        if matched == 0:
            rejected_specs.append({"item": item, "reason": "no_unobserved_candidates_matched"})
            continue
        scored += matched
        positive += int(delta > 0) * matched
        negative += int(delta < 0) * matched
        applied_specs.append(
            {
                "field": field_name,
                "value": value,
                "direction": direction,
                "weight": round(delta, 6),
                "matched_candidates": matched,
                "target_support_count": count,
                "target_effect": round(target_effect, 4),
                "source_field": role.source_field,
                "role_confidence": role.confidence,
                "role_transfer_weight": role.transfer_weight,
                "reason": str(item.get("reason", ""))[:240],
            }
        )

    if strict and applied_specs:
        by_id = {c.candidate_id: c for c in pool}
        for cid, value in list(adjustments.items()):
            if value == 0.0:
                continue
            candidate = by_id[cid]
            positive_matches = 0
            negative_matches = 0
            for spec in applied_specs:
                if str(candidate.metadata.get(spec["field"], "")) != spec["value"]:
                    continue
                positive_matches += int(float(spec["weight"]) > 0)
                negative_matches += int(float(spec["weight"]) < 0)
            if positive_matches < min_positive_roles or negative_matches > max_negative_roles:
                adjustments[cid] = 0.0
                strict_rejected_candidates += 1

    scored = sum(1 for value in adjustments.values() if value != 0.0)
    positive = sum(1 for value in adjustments.values() if value > 0.0)
    negative = sum(1 for value in adjustments.values() if value < 0.0)
    for cid, value in list(adjustments.items()):
        adjustments[cid] = max(-0.12, min(0.12, value))
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert = {
        "skills": {
            "llm_cross_domain_transfer_card": {
                "active": bool(applied_specs),
                "model": response_meta["model"],
                "scored_candidates": scored,
                "positive_adjustments": positive,
                "negative_adjustments": negative,
                "strict_enabled": strict,
                "strict_min_positive_roles": min_positive_roles if strict else 0,
                "strict_max_negative_roles": max_negative_roles if strict else 0,
                "strict_rejected_candidates": strict_rejected_candidates,
                "applied_specs": applied_specs,
                "rejected_specs": rejected_specs,
                "parse_error": parse_error,
            }
        },
        "max_abs_adjustment": round(max_abs, 6),
    }
    record = {
        "called": True,
        "mode": mode,
        "seed": seed,
        "round_index": round_index,
        "model": response_meta["model"],
        "usage": response_meta["usage"],
        "prompt_payload": prompt_payload,
        "raw_response": content,
        "parsed_response": parsed,
        "applied_specs": applied_specs,
        "rejected_specs": rejected_specs,
        "parse_error": parse_error,
    }
    return adjustments, cert, record


def compact_public_candidate(
    adapter: replay.DatasetAdapter,
    candidate: replay.Candidate,
    base_score: float,
    adjusted_score: float,
    adjustment: float,
    transfer_cert: dict[str, Any],
) -> dict[str, Any]:
    hidden_keys = {
        adapter.hidden_target,
        "yield_value",
        "conversion_value",
        "stability_score",
        "normalized_solubility_score",
        "hydration_affinity_score",
        "normalized_lipophilicity_score",
        "normalized_band_gap_score",
    }
    public_metadata = {key: value for key, value in candidate.metadata.items() if key not in hidden_keys}
    applied_specs = []
    transfer_skill = next(iter(transfer_cert.get("skills", {}).values()), {})
    for spec in transfer_skill.get("applied_specs", []):
        field = spec.get("target_field") or spec.get("field")
        if not field:
            continue
        value = candidate.metadata.get(str(field))
        if value in ("", None):
            continue
        matched_values = []
        for active in spec.get("active_target_values", []):
            if str(active.get("value", "")) == str(value):
                matched_values.append(active)
        if matched_values:
            applied_specs.append(
                {
                    "target_field": field,
                    "target_value": value,
                    "source_field": spec.get("source_field"),
                    "transfer_weight": spec.get("transfer_weight"),
                    "role_confidence": spec.get("confidence"),
                    "matched_target_evidence": matched_values[:3],
                }
            )
    return {
        "candidate_id": candidate.candidate_id,
        "group": candidate.group,
        "public_features": {"x1": round(candidate.x1, 4), "x2": round(candidate.x2, 4), "x3": round(candidate.x3, 4)},
        "metadata": public_metadata,
        "base_score": round(base_score, 6),
        "adjusted_score": round(adjusted_score, 6),
        "transfer_adjustment": round(adjustment, 6),
        "matched_transfer_specs": applied_specs[:8],
    }


def llm_audit_gate(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed: list[replay.Candidate],
    card: TransferCard,
    transfer_cert: dict[str, Any],
    base_scores: dict[str, float],
    adjusted_scores: dict[str, float],
    adjustments: dict[str, float],
    gate: replay.GateCertificate,
    seed: int,
    round_index: int,
    mode: TransferMode,
    config: replay.LLMConfig,
    candidate_count: int = 8,
) -> tuple[replay.GateCertificate, dict[str, Any]]:
    if not gate.authorized:
        return gate, {"called": False, "reason": "gate_not_authorized"}
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    candidate_ids = [cid for cid, _score in sorted(adjusted_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:candidate_count]]
    candidate_ids = list(dict.fromkeys([gate.incumbent_candidate, gate.challenger_candidate, *candidate_ids]))
    candidates = [
        compact_public_candidate(
            adapter,
            by_id[cid],
            base_scores[cid],
            adjusted_scores[cid],
            adjustments.get(cid, 0.0),
            transfer_cert,
        )
        for cid in candidate_ids
        if cid in by_id
    ]
    prompt_payload = {
        "target_evidence": replay.observed_evidence_payload(adapter, observed),
        "transfer_card": {
            "card_id": card.card_id,
            "source_dataset": card.source_dataset,
            "target_dataset": card.target_dataset,
            "source_observation_count": card.source_observation_count,
            "discount": card.discount,
            "role_map": card.role_map,
            "roles": [asdict(role) for role in card.roles if role.transfer_weight > 0.0],
            "evidence_summary": card.evidence_summary,
        },
        "proposed_gate": asdict(gate),
        "candidate_shortlist": candidates,
        "transfer_certificate": transfer_cert,
        "audit_instruction": (
            "Audit only whether the challenger has enough revealed target evidence and transfer-card support "
            "to override the incumbent. Do not use hidden outcomes for shortlisted candidates. "
            "When the evidence is close, sparse, or mostly inherited from broad transfer priors, reject."
        ),
        "output_contract": {
            "decision": "one of: approve, reject",
            "confidence": "number between 0 and 1",
            "reason": "short evidence-based reason",
        },
    }
    system = (
        "You are a CARE 2.0 cross-domain transfer auditor. "
        "Use only revealed target evidence, public candidate features, and the source-to-target role transfer card. "
        "Do not infer hidden outcomes for candidate IDs. Return only one JSON object. No markdown. No chain-of-thought."
    )
    user = (
        "Decide whether to approve the proposed challenger over the incumbent. "
        "Approve only when the challenger has clear target-observation support on transferred roles and no obvious negative evidence. "
        "Reject if evidence is weak, broad, contradictory, close to the incumbent, or mostly due to a single noisy factor. "
        "Return exactly one JSON object with keys decision, confidence, and reason. "
        "The decision value must be either approve or reject. "
        "JSON input:\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
    )
    content, response_meta = replay.chat_completion_text(
        config,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    parse_error = ""
    try:
        parsed = replay.extract_json_object(content)
    except ValueError as exc:
        parsed = {"decision": "reject", "confidence": 0.0, "reason": "parse_error_defaults_to_reject"}
        parse_error = str(exc)
    decision = str(parsed.get("decision", "")).strip().lower()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0
    approved = decision == "approve" and confidence >= 0.6 and not parse_error
    reason = str(parsed.get("reason", ""))[:240]
    applied_skill_ids = tuple(dict.fromkeys([*gate.applied_skill_ids, "llm_transfer_auditor"]))
    if approved:
        audited_gate = replay.GateCertificate(
            gate_version=gate.gate_version,
            incumbent_candidate=gate.incumbent_candidate,
            challenger_candidate=gate.challenger_candidate,
            selected_candidate=gate.selected_candidate,
            authorized=True,
            gate_margin=gate.gate_margin,
            acquisition_loss=gate.acquisition_loss,
            row_order_stable=gate.row_order_stable,
            applied_skill_ids=applied_skill_ids,
            reason="llm_audit_approved_challenger",
        )
    else:
        audited_gate = replay.GateCertificate(
            gate_version=gate.gate_version,
            incumbent_candidate=gate.incumbent_candidate,
            challenger_candidate=gate.challenger_candidate,
            selected_candidate=gate.incumbent_candidate,
            authorized=False,
            gate_margin=gate.gate_margin,
            acquisition_loss=gate.acquisition_loss,
            row_order_stable=gate.row_order_stable,
            applied_skill_ids=applied_skill_ids,
            reason="llm_audit_rejected_challenger",
        )
    record = {
        "called": True,
        "mode": mode,
        "seed": seed,
        "round_index": round_index,
        "model": response_meta["model"],
        "usage": response_meta["usage"],
        "decision": decision,
        "approved": approved,
        "confidence": confidence,
        "reason": reason,
        "prompt_payload": prompt_payload,
        "raw_response": content,
        "parsed_response": parsed,
        "parse_error": parse_error,
    }
    return audited_gate, record


def transfer_row_order_stability_check(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    reference_adjustments: dict[str, float],
    strict: bool = False,
    min_role_confidence: float = 0.18,
    min_positive_roles: int = 2,
    max_negative_roles: int = 0,
) -> bool:
    shuffled = list(pool)
    random.Random(20_000 + len(observed)).shuffle(shuffled)
    shuffled_adjustments, _ = transfer_adjustments(
        adapter,
        tuple(shuffled),
        observed_ids,
        observed,
        card,
        min_target_support,
        effect_threshold,
        strict,
        min_role_confidence,
        min_positive_roles,
        max_negative_roles,
    )
    return all(abs(reference_adjustments[k] - shuffled_adjustments[k]) < 1e-12 for k in reference_adjustments)


def combine_adjustments(*items: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for adjustments in items:
        for key, value in adjustments.items():
            out[key] = out.get(key, 0.0) + value
    for key, value in list(out.items()):
        out[key] = max(-0.14, min(0.14, value))
    return out


def run_target_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: TransferMode,
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    strict_min_role_confidence: float,
    strict_min_positive_roles: int,
    strict_max_negative_roles: int,
    llm_config: replay.LLMConfig | None = None,
) -> tuple[dict[str, Any], list[replay.AuditEntry], replay.HypothesisEntry]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {c.candidate_id: c for c in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {c.candidate_id for c in observed}
    skills = replay.make_skills(adapter)
    hypothesis = replay.make_hypothesis(adapter)
    if mode == "no_care_random":
        hypothesis.status = "inactive"
        hypothesis.evidence_summary = "No CARE hypothesis or transfer card is used in this random-search baseline."

    audit: list[replay.AuditEntry] = []
    top10 = {c.candidate_id for c in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    best_trace: list[float] = []
    intervention_count = 0
    bad_interventions = 0
    rejected_good_challengers = 0
    selected_top10 = any(c.candidate_id in top10 for c in observed)
    transfer_active_rounds = 0
    transfer_scored_candidates_total = 0
    llm_call_count = 0
    llm_parse_error_count = 0

    for round_index in range(task.reveal_budget):
        transfer_cert: dict[str, Any] | None = None
        value_prior_cert: dict[str, Any] | None = None
        llm_record: dict[str, Any] | None = None
        round_transfer_active = False
        if mode == "no_care_random":
            selected_candidate = rng.choice([c for c in pool if c.candidate_id not in observed_ids])
            gate = replay.GateCertificate(
                gate_version="no_care",
                incumbent_candidate=selected_candidate.candidate_id,
                challenger_candidate=selected_candidate.candidate_id,
                selected_candidate=selected_candidate.candidate_id,
                authorized=False,
                gate_margin=0.0,
                acquisition_loss=0.0,
                row_order_stable=True,
                applied_skill_ids=(),
                reason="baseline_random_search_no_care",
            )
        else:
            base_scores = replay.public_incumbent_scores(adapter, observed_ids, observed)
            if mode == "incumbent":
                incumbent = replay.top_candidate(base_scores)
                gate = replay.GateCertificate(
                    gate_version="none",
                    incumbent_candidate=incumbent,
                    challenger_candidate=incumbent,
                    selected_candidate=incumbent,
                    authorized=False,
                    gate_margin=0.0,
                    acquisition_loss=0.0,
                    row_order_stable=True,
                    applied_skill_ids=(),
                    reason="baseline_incumbent_only",
                )
            else:
                local_adjustments: dict[str, float] | None = None
                local_cert: dict[str, Any] | None = None
                transfer_adjustment_values: dict[str, float] | None = None
                value_prior_adjustment_values: dict[str, float] | None = None
                row_order_stable = True
                active_skill_ids: list[str] = []

                if mode in {"target_local_no_gate", "target_local_gate_v1", "transfer_plus_local_gate_v1", "transfer_strict_plus_local_gate_v1"}:
                    local_adjustments, local_cert = replay.skill_adjustments(
                        adapter, pool, observed_ids, observed, skills, round_index
                    )
                    row_order_stable = row_order_stable and replay.row_order_stability_check(
                        adapter, pool, observed_ids, observed, skills, round_index, local_adjustments
                    )
                    active_skill_ids.extend(k for k, v in local_cert["skills"].items() if v.get("active"))

                if mode in {
                    "transfer_no_gate",
                    "transfer_gate_v1",
                    "transfer_plus_local_gate_v1",
                    "transfer_strict_no_gate",
                    "transfer_strict_gate_v1",
                    "transfer_strict_plus_local_gate_v1",
                    "transfer_value_prior_no_gate",
                    "transfer_value_prior_gate_v1",
                    "transfer_value_prior_strict_gate_v1",
                    "llm_transfer_gate_v1",
                    "llm_transfer_strict_gate_v1",
                    "llm_audit_transfer_gate_v1",
                    "llm_audit_transfer_strict_gate_v1",
                }:
                    strict_transfer = mode in {
                        "transfer_strict_no_gate",
                        "transfer_strict_gate_v1",
                        "transfer_strict_plus_local_gate_v1",
                        "transfer_value_prior_strict_gate_v1",
                        "transfer_descriptor_value_prior_strict_gate_v1",
                        "llm_transfer_strict_gate_v1",
                        "llm_audit_transfer_strict_gate_v1",
                    }
                    if is_llm_transfer_mode(mode):
                        if llm_config is None:
                            raise RuntimeError("LLM transfer mode requested but no LLM config was provided.")
                        transfer_adjustment_values, transfer_cert, llm_record = llm_transfer_adjustments(
                            adapter,
                            pool,
                            observed_ids,
                            observed,
                            card,
                            min_target_support,
                            effect_threshold,
                            seed,
                            round_index,
                            mode,
                            llm_config,
                            strict_transfer,
                            strict_min_role_confidence,
                            strict_min_positive_roles,
                            strict_max_negative_roles,
                        )
                        llm_call_count += int(bool(llm_record.get("called")))
                        llm_parse_error_count += int(bool(llm_record.get("parse_error")))
                        transfer_skill = transfer_cert["skills"]["llm_cross_domain_transfer_card"]
                    else:
                        transfer_adjustment_values, transfer_cert = transfer_adjustments(
                            adapter,
                            pool,
                            observed_ids,
                            observed,
                            card,
                            min_target_support,
                            effect_threshold,
                            strict_transfer,
                            strict_min_role_confidence,
                            strict_min_positive_roles,
                            strict_max_negative_roles,
                        )
                        row_order_stable = row_order_stable and transfer_row_order_stability_check(
                            adapter,
                            pool,
                            observed_ids,
                            observed,
                            card,
                            min_target_support,
                            effect_threshold,
                            transfer_adjustment_values,
                            strict_transfer,
                            strict_min_role_confidence,
                            strict_min_positive_roles,
                            strict_max_negative_roles,
                        )
                        transfer_skill = transfer_cert["skills"]["cross_domain_transfer_card"]
                    if transfer_skill.get("active"):
                        round_transfer_active = True
                        transfer_scored_candidates_total += int(transfer_skill.get("scored_candidates", 0))
                        active_skill_ids.append("llm_cross_domain_transfer_card" if is_llm_transfer_mode(mode) else "cross_domain_transfer_card")

                if is_value_prior_mode(mode):
                    strict_value_prior = mode in {
                        "transfer_value_prior_strict_gate_v1",
                        "transfer_descriptor_value_prior_strict_gate_v1",
                    }
                    value_prior_skill_id = (
                        "reaction_descriptor_value_prior"
                        if is_descriptor_value_prior_mode(mode)
                        else "source_value_prior"
                    )
                    min_positive_priors = max(1, strict_min_positive_roles - 1)
                    if is_descriptor_value_prior_mode(mode) and strict_value_prior:
                        min_positive_priors = max(1, strict_min_positive_roles)
                    signed_adjustment_cap = 0.08
                    positive_adjustment_cap = 0.06
                    value_prior_adjustment_values, value_prior_cert = source_value_prior_adjustments(
                        pool,
                        observed_ids,
                        card,
                        strict_value_prior,
                        strict_min_role_confidence,
                        min_positive_priors,
                        strict_max_negative_roles,
                        value_prior_skill_id,
                        signed_adjustment_cap,
                        positive_adjustment_cap,
                    )
                    value_prior_skill = value_prior_cert["skills"][value_prior_skill_id]
                    if value_prior_skill.get("active"):
                        round_transfer_active = True
                        transfer_scored_candidates_total += int(value_prior_skill.get("scored_candidates", 0))
                        active_skill_ids.append(value_prior_skill_id)

                if mode in {"transfer_plus_local_gate_v1", "transfer_strict_plus_local_gate_v1"}:
                    adjustments = combine_adjustments(local_adjustments or {}, transfer_adjustment_values or {})
                elif is_value_prior_mode(mode):
                    adjustments = combine_adjustments(transfer_adjustment_values or {}, value_prior_adjustment_values or {})
                else:
                    adjustments = local_adjustments or transfer_adjustment_values or {
                        c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids
                    }
                adjusted_scores = {cid: base_scores[cid] + adjustments.get(cid, 0.0) for cid in base_scores}
                if mode in {
                    "target_local_no_gate",
                    "transfer_no_gate",
                    "transfer_strict_no_gate",
                    "transfer_value_prior_no_gate",
                    "transfer_descriptor_value_prior_no_gate",
                }:
                    gate = replay.no_gate_decision(base_scores, adjusted_scores, row_order_stable, tuple(active_skill_ids))
                else:
                    gate = replay.gate_decision("gate_v1", base_scores, adjusted_scores, adjustments, row_order_stable, tuple(active_skill_ids))

                if is_llm_audit_transfer_mode(mode):
                    if llm_config is None:
                        raise RuntimeError("LLM audit transfer mode requested but no LLM config was provided.")
                    gate, llm_record = llm_audit_gate(
                        adapter,
                        pool,
                        observed,
                        card,
                        transfer_cert or {},
                        base_scores,
                        adjusted_scores,
                        adjustments,
                        gate,
                        seed,
                        round_index,
                        mode,
                        llm_config,
                    )
                    llm_call_count += int(bool(llm_record.get("called")))
                    llm_parse_error_count += int(bool(llm_record.get("parse_error")))

                if gate.authorized:
                    intervention_count += 1
                    if by_id[gate.challenger_candidate].objective_value < by_id[gate.incumbent_candidate].objective_value:
                        bad_interventions += 1
                elif by_id[gate.challenger_candidate].objective_value > by_id[gate.incumbent_candidate].objective_value:
                    rejected_good_challengers += 1

        if round_transfer_active:
            transfer_active_rounds += 1
        selected = by_id[gate.selected_candidate]
        if mode != "no_care_random":
            replay.update_hypothesis_from_reveal(hypothesis, selected, observed, round_index, adapter.preferred_groups)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(c.objective_value for c in observed)
        best_trace.append(best_so_far)
        hypothesis_snapshot = asdict(hypothesis)
        hypothesis_snapshot["transfer_card"] = asdict(card)
        if transfer_cert is not None:
            hypothesis_snapshot["transfer_policy"] = transfer_cert
        if value_prior_cert is not None:
            hypothesis_snapshot["source_value_prior_policy"] = value_prior_cert
        if llm_record is not None:
            hypothesis_snapshot["llm_transfer_policy"] = llm_record
        audit.append(
            replay.AuditEntry(
                dataset_id=adapter.dataset_id,
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by=mode if gate.authorized or mode in {"no_care_random", "incumbent"} else "incumbent",
                gate=gate,
                revealed_value=selected.objective_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=hypothesis_snapshot,
            )
        )

    final_best = max(c.objective_value for c in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
        "intervention_count": intervention_count,
        "bad_intervention_count": bad_interventions,
        "rejected_good_challenger_count": rejected_good_challengers,
        "transfer_active_rounds": transfer_active_rounds,
        "transfer_scored_candidates_total": transfer_scored_candidates_total,
        "llm_call_count": llm_call_count,
        "llm_parse_error_count": llm_parse_error_count,
        "source_observation_count": card.source_observation_count,
        "mean_transfer_confidence": round(mean(role.confidence for role in card.roles), 4),
        "hypothesis_confidence": round(hypothesis.confidence, 4),
        "hypothesis_support_count": hypothesis.support_count,
    }
    return metrics, audit, hypothesis


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    numeric_fields = [
        "final_best",
        "best_so_far_auc",
        "simple_regret",
        "top10_hit",
        "intervention_count",
        "bad_intervention_count",
        "rejected_good_challenger_count",
        "transfer_active_rounds",
        "transfer_scored_candidates_total",
        "llm_call_count",
        "llm_parse_error_count",
        "source_observation_count",
        "mean_transfer_confidence",
        "hypothesis_confidence",
        "hypothesis_support_count",
    ]
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in numeric_fields:
            vals = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(vals), 4),
                "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def llm_config_from_args(args: argparse.Namespace, modes: tuple[TransferMode, ...]) -> replay.LLMConfig | None:
    if not any(is_any_llm_mode(mode) for mode in modes):
        return None
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(f"Set {args.llm_api_key_env}, COMMONSTACK_API_KEY, or CARE_LLM_API_KEY before running LLM transfer modes.")
    return replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
    )


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[replay.AuditEntry]],
    hypotheses: dict[tuple[str, int], replay.HypothesisEntry],
    cards: dict[int, TransferCard],
    descriptor_cards: dict[int, TransferCard] | None = None,
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for seed, card in sorted(cards.items()):
        (OUTPUT_RUNS / f"{output_id}_transfer_card_seed{seed}.json").write_text(
            json.dumps(asdict(card), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for seed, card in sorted((descriptor_cards or {}).items()):
        (OUTPUT_RUNS / f"{output_id}_descriptor_transfer_card_seed{seed}.json").write_text(
            json.dumps(asdict(card), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for (mode, seed), audit in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for entry in audit:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    for (mode, seed), hypothesis in sorted(hypotheses.items()):
        (OUTPUT_RUNS / f"{output_id}_knowledge_{mode}_seed{seed}.json").write_text(
            json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def run_transfer_ablation(
    source_dataset: str,
    target_dataset: str,
    seeds: int,
    rounds: int,
    initial: int,
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    min_target_support: int,
    effect_threshold: float,
    strict_min_role_confidence: float,
    strict_min_positive_roles: int,
    strict_max_negative_roles: int,
    modes: tuple[TransferMode, ...],
    llm_config: replay.LLMConfig | None,
    output_tag: str,
) -> dict[str, Any]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    role_map = role_map_for(source_dataset, target_dataset)
    descriptor_role_map = descriptor_role_map_for(source_dataset, target_dataset)
    descriptor_modes_requested = any(is_descriptor_value_prior_mode(mode) for mode in modes)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[replay.AuditEntry]] = {}
    hypotheses: dict[tuple[str, int], replay.HypothesisEntry] = {}
    cards: dict[int, TransferCard] = {}
    descriptor_cards: dict[int, TransferCard] = {}
    for seed in range(seeds):
        source_observed = source_observations(source_adapter, seed, source_observation_count)
        card = compile_transfer_card(
            source_adapter,
            target_adapter,
            source_observed,
            role_map,
            discount,
            min_source_support,
        )
        cards[seed] = card
        descriptor_card = card
        if descriptor_modes_requested and descriptor_role_map:
            descriptor_card = compile_transfer_card(
                source_adapter,
                target_adapter,
                source_observed,
                descriptor_transfer_role_map_for(source_dataset, target_dataset),
                discount,
                min_source_support,
            )
            descriptor_cards[seed] = descriptor_card
        for mode in modes:
            active_card = descriptor_card if is_descriptor_value_prior_mode(mode) else card
            metrics, audit, hypothesis = run_target_policy(
                target_adapter,
                task,
                seed,
                mode,
                active_card,
                min_target_support,
                effect_threshold,
                strict_min_role_confidence,
                strict_min_positive_roles,
                strict_max_negative_roles,
                llm_config,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit
            hypotheses[(mode, seed)] = hypothesis

    output_id = f"transfer_{source_dataset}_to_{target_dataset}" if not output_tag else f"transfer_{source_dataset}_to_{target_dataset}_{output_tag}"
    summary = {
        "experiment": "care_cross_domain_transfer_ablation",
        "output_id": output_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "role_map": role_map,
        "descriptor_role_map": descriptor_role_map,
        "transfer_boundary": (
            "Source outcomes are used to estimate role-level evidence strength. "
            "Direct source value priors are transferred only for whitelisted source-target "
            "fields that share the same public descriptor vocabulary; no target hidden "
            "outcomes are used."
        ),
        "source_observation_count": source_observation_count,
        "discount": discount,
        "min_source_support": min_source_support,
        "min_target_support": min_target_support,
        "effect_threshold": effect_threshold,
        "strict_min_role_confidence": strict_min_role_confidence,
        "strict_min_positive_roles": strict_min_positive_roles,
        "strict_max_negative_roles": strict_max_negative_roles,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "llm": None
        if llm_config is None
        else {
            "base_url": llm_config.base_url,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
        },
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, summary, audits, hypotheses, cards, descriptor_cards)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE 2.0 cross-domain transfer ablations.")
    parser.add_argument("--source-dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_suzuki_miyaura", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=48)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--min-target-support", type=int, default=2)
    parser.add_argument("--effect-threshold", type=float, default=4.0)
    parser.add_argument("--strict-min-role-confidence", type=float, default=0.18)
    parser.add_argument("--strict-min-positive-roles", type=int, default=2)
    parser.add_argument("--strict-max-negative-roles", type=int, default=0)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(DEFAULT_MODES)}")
    parser.add_argument("--llm-base-url", default=os.environ.get("CARE_LLM_BASE_URL", "https://api.commonstack.ai/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("CARE_LLM_MODEL", "moonshotai/kimi-k2.7-code"))
    parser.add_argument("--llm-api-key-env", default="CARE_LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=500)
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()
    modes = parse_modes(args.modes)
    llm_config = llm_config_from_args(args, modes)

    summary = run_transfer_ablation(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        seeds=args.seeds,
        rounds=args.rounds,
        initial=args.initial,
        source_observation_count=args.source_observations,
        discount=args.discount,
        min_source_support=args.min_source_support,
        min_target_support=args.min_target_support,
        effect_threshold=args.effect_threshold,
        strict_min_role_confidence=args.strict_min_role_confidence,
        strict_min_positive_roles=args.strict_min_positive_roles,
        strict_max_negative_roles=args.strict_max_negative_roles,
        modes=modes,
        llm_config=llm_config,
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
