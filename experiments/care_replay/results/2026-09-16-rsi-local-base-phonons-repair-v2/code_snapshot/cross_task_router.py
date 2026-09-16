#!/usr/bin/env python3
"""Schema-only route proposals for cross-task CARE 2.0 transfer.

This module deliberately does not inspect target outcomes.  It proposes which
transfer family is worth offering to the calibration gate from public task
metadata alone.  The proposal is therefore a candidate ordering, not a claim
that transfer is safe; the held-out experiment still decides whether the
source-outcome route is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


KNOWN_DATASET_COLUMNS: dict[str, tuple[str, ...]] = {
    "real_buchwald_hartwig": ("ligand", "additive", "base", "aryl_halide"),
    "real_chemlex_acidamine": ("acid", "amine", "reagent", "solvent"),
    "real_suzuki_miyaura": (
        "reactant_1",
        "reactant_2",
        "catalyst",
        "ligand",
        "reagent",
        "solvent",
    ),
    "real_matbench_dielectric": (
        "anion_family",
        "element_count_bin",
        "dominant_family",
        "transition_metal_flag",
        "lanthanide_flag",
        "mean_atomic_number_bin",
        "max_element_fraction_bin",
    ),
    "real_matbench_expt_gap": (
        "anion_family",
        "element_count_bin",
        "dominant_family",
        "transition_metal_flag",
        "lanthanide_flag",
        "mean_atomic_number_bin",
        "max_element_fraction_bin",
    ),
    "real_matbench_log_kvrh": (
        "anion_family",
        "element_count_bin",
        "dominant_family",
        "transition_metal_flag",
        "lanthanide_flag",
        "mean_atomic_number_bin",
        "max_element_fraction_bin",
    ),
    "real_matbench_phonons": (
        "anion_family",
        "element_count_bin",
        "dominant_family",
        "transition_metal_flag",
        "lanthanide_flag",
        "mean_atomic_number_bin",
        "max_element_fraction_bin",
    ),
    "real_moleculenet_esol": (
        "molecular_weight_bin",
        "hbond_donor_bin",
        "ring_bin",
        "rotatable_bond_bin",
        "polar_surface_area_bin",
        "smiles_length_bin",
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
    "real_moleculenet_freesolv_continuous": (
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


REACTION_DATASETS = frozenset(
    {
        "real_buchwald_hartwig",
        "real_chemlex_acidamine",
        "real_suzuki_miyaura",
    }
)


@dataclass(frozen=True)
class RouteProposal:
    source_dataset: str
    target_dataset: str
    source_family: str
    target_family: str
    shared_field_count: int
    shared_field_ratio: float
    shared_role_count: int
    candidate_route_families: tuple[str, ...]
    recommended_candidate: str
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "source_family": self.source_family,
            "target_family": self.target_family,
            "shared_field_count": self.shared_field_count,
            "shared_field_ratio": self.shared_field_ratio,
            "shared_role_count": self.shared_role_count,
            "candidate_route_families": list(self.candidate_route_families),
            "recommended_candidate": self.recommended_candidate,
            "rationale": self.rationale,
            "outcome_gate_required": True,
        }


def task_family(dataset_id: str) -> str:
    if dataset_id.startswith("real_matbench_"):
        return "materials"
    if dataset_id.startswith("real_moleculenet_"):
        return "molecules"
    if dataset_id in REACTION_DATASETS:
        return "reactions"
    if dataset_id.startswith("synthetic_materials"):
        return "materials"
    if dataset_id.startswith("synthetic_chemlex") or dataset_id.startswith("synthetic_suzuki"):
        return "reactions"
    return "unknown"


def _field_role(field: str) -> str | None:
    name = field.lower()
    if "solvent" in name:
        return "solvent"
    if any(token in name for token in ("ligand", "catalyst")):
        return "catalyst"
    if any(token in name for token in ("acid", "amine", "aryl", "reactant")):
        return "substrate"
    if any(token in name for token in ("base", "additive", "reagent")):
        return "condition"
    if any(token in name for token in ("temperature", "dwell", "residence")):
        return "schedule"
    return None


def _roles(columns: Iterable[str]) -> set[str]:
    return {role for column in columns if (role := _field_role(column)) is not None}


def _columns_for(dataset_id: str, columns: Iterable[str] | None) -> tuple[str, ...]:
    if columns is not None:
        return tuple(columns)
    return KNOWN_DATASET_COLUMNS.get(dataset_id, ())


def propose_route(
    source_dataset: str,
    target_dataset: str,
    source_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] | None = None,
) -> RouteProposal:
    source = _columns_for(source_dataset, source_columns)
    target = _columns_for(target_dataset, target_columns)
    source_set = set(source)
    target_set = set(target)
    shared_fields = source_set & target_set
    shared_roles = _roles(source) & _roles(target)
    ratio = len(shared_fields) / max(1, min(len(source_set), len(target_set)))
    source_family = task_family(source_dataset)
    target_family = task_family(target_dataset)

    if source_dataset == target_dataset:
        candidates = ("target_only",)
        rationale = "Source and target are the same task; no transfer is needed."
    elif source_family == "reactions" and target_family == "reactions" and shared_roles:
        candidates = ("reaction_role_source_outcome", "target_only")
        rationale = (
            "Reaction tasks share public component roles; offer role-level source-outcome "
            "transfer, then let target calibration gate it."
        )
    elif source_family == target_family and ratio >= 0.5:
        candidates = ("shared_descriptor_source_outcome", "target_only")
        rationale = (
            "The tasks share at least half of their public descriptor vocabulary; offer "
            "descriptor-aligned source-outcome transfer, then gate it on target evidence."
        )
    elif shared_fields or shared_roles:
        candidates = ("schema_aligned_source_outcome", "target_only")
        rationale = (
            "Some public fields or component roles align, but the overlap is below the "
            "strong-transfer threshold; offer a conservative candidate only."
        )
    else:
        candidates = ("target_only",)
        rationale = "No public source-target alignment is visible; use target-only search."

    return RouteProposal(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_family=source_family,
        target_family=target_family,
        shared_field_count=len(shared_fields),
        shared_field_ratio=round(ratio, 6),
        shared_role_count=len(shared_roles),
        candidate_route_families=candidates,
        recommended_candidate=candidates[0],
        rationale=rationale,
    )
