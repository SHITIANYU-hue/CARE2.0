#!/usr/bin/env python3
"""Canonical, executable CARE 2.0 transfer-skill artifact.

The older replay code has several useful internal objects (transfer cards,
kernel patches, semantic skills).  This module does not replace their math.  It
freezes the pieces that are actually executed for one source-target pair into
one versioned artifact and emits a compact end-to-end trace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "care.transfer_skill/v1"
TRACE_SCHEMA_VERSION = "care.transfer_trace/v1"

KNOWN_OPERATORS = frozenset(
    {
        "source_informed_initial_design",
        "mapped_neighbor_outcome_prior",
        "mapped_additive_outcome_prior",
        "mapped_interaction_residual_prior",
        "source_weighted_kernel_geometry",
        "prequential_expert_routing",
        "calibrated_exact_fallback",
    }
)

FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "heldout_outcomes",
        "hidden_target_outcomes",
        "target_outcomes",
        "oracle_values",
    }
)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_EVIDENCE_KEYS:
                return child_path
            violation = _find_forbidden_key(child, child_path)
            if violation:
                return violation
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violation = _find_forbidden_key(child, f"{path}[{index}]")
            if violation:
                return violation
    return None


@dataclass(frozen=True)
class TransferOperator:
    name: str
    source_outcome_required: bool
    parameters: dict[str, Any]

    def validate(self) -> None:
        if self.name not in KNOWN_OPERATORS:
            raise ValueError(f"Unknown transfer operator: {self.name}")


@dataclass(frozen=True)
class TransferSkill:
    schema_version: str
    skill_id: str
    source_dataset: str
    target_dataset: str
    hypothesis: dict[str, Any]
    execution_contract: dict[str, Any]
    source_evidence: dict[str, Any]
    role_map: dict[str, str]
    operators: tuple[TransferOperator, ...]
    kernel_patches: tuple[dict[str, Any], ...]
    execution: dict[str, Any]
    gate: dict[str, Any]
    provenance: dict[str, Any]
    evidence_boundary: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported transfer-skill schema: {self.schema_version}")
        if not self.source_dataset or not self.target_dataset:
            raise ValueError("Both source_dataset and target_dataset are required.")
        if not self.operators:
            raise ValueError("At least one executable transfer operator is required.")
        for operator in self.operators:
            operator.validate()
        if not any(operator.source_outcome_required for operator in self.operators):
            raise ValueError("A transfer skill must execute at least one source-outcome operator.")
        if not self.kernel_patches:
            raise ValueError("A transfer skill must contain at least one frozen kernel patch.")
        required_execution = {
            "calibration_seed_start",
            "calibration_seed_count",
            "heldout_seed_start",
            "heldout_seed_count",
            "router_min_observations",
            "router_min_quality",
            "router_max_transfer_mass",
        }
        missing_execution = required_execution - set(self.execution)
        if missing_execution:
            raise ValueError(
                f"TransferSkill execution is missing: {sorted(missing_execution)}"
            )
        for name in ("router_min_quality", "router_max_transfer_mass"):
            value = float(self.execution[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        confirmation = self.gate.get("confirmation", {})
        for name in ("min_positive_fold_rate", "min_final_non_loss_rate"):
            if name in confirmation and not 0.0 <= float(confirmation[name]) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        violation = _find_forbidden_key(self.as_dict(include_fingerprint=False))
        if violation:
            raise ValueError(f"Hidden target evidence is forbidden in a TransferSkill: {violation}")

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.as_dict(include_fingerprint=False))

    def identity(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "fingerprint": self.fingerprint,
        }

    def as_dict(self, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "hypothesis": _json_clone(self.hypothesis),
            "execution_contract": _json_clone(self.execution_contract),
            "source_evidence": _json_clone(self.source_evidence),
            "role_map": dict(self.role_map),
            "operators": [asdict(operator) for operator in self.operators],
            "kernel_patches": _json_clone(self.kernel_patches),
            "execution": _json_clone(self.execution),
            "gate": _json_clone(self.gate),
            "provenance": _json_clone(self.provenance),
            "evidence_boundary": self.evidence_boundary,
        }
        if include_fingerprint:
            payload["fingerprint"] = _sha256_json(payload)
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def compile_transfer_skill(
    *,
    source_dataset: str,
    target_dataset: str,
    route_proposal: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    role_map: Mapping[str, str],
    kernel_patches: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    gate: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> TransferSkill:
    patches = tuple(_json_clone(patch) for patch in kernel_patches)
    patch_digest = _sha256_json(patches)[:12]
    route_name = str(route_proposal.get("recommended_candidate", "source_outcome"))
    reasons = [
        {
            "patch_id": str(patch.get("patch_id", f"patch_{index + 1}")),
            "reason": str(patch.get("reason", "")),
            "confidence": float(patch.get("confidence", 0.0)),
        }
        for index, patch in enumerate(patches)
    ]
    operators = (
        TransferOperator(
            "source_informed_initial_design",
            True,
            {"strategy": execution["source_initial_strategy"]},
        ),
        TransferOperator(
            "mapped_neighbor_outcome_prior",
            True,
            {"enabled_by_patch": "source_prior_strength > 0"},
        ),
        TransferOperator(
            "mapped_additive_outcome_prior",
            True,
            {"minimum_source_support": execution["min_source_support"]},
        ),
        TransferOperator(
            "mapped_interaction_residual_prior",
            True,
            {"enabled_by_patch": "source_interaction_strength > 0"},
        ),
        TransferOperator(
            "source_weighted_kernel_geometry",
            True,
            {"fixed_ensemble_scales": list(execution["fixed_ensemble_scales"])},
        ),
        TransferOperator(
            "prequential_expert_routing",
            True,
            {
                "minimum_target_observations": execution["router_min_observations"],
                "minimum_quality": execution["router_min_quality"],
                "maximum_transfer_mass": execution["router_max_transfer_mass"],
            },
        ),
        TransferOperator(
            "calibrated_exact_fallback",
            False,
            {"fallback_mode": gate["fallback_mode"]},
        ),
    )
    return TransferSkill(
        schema_version=SCHEMA_VERSION,
        skill_id=f"care2::{source_dataset}::{target_dataset}::{patch_digest}",
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        hypothesis={
            "claim": (
                f"Frozen measured outcomes from {source_dataset} can improve finite-pool "
                f"search on {target_dataset} through the {route_name} route without "
                "reading hidden target outcomes."
            ),
            "route_family": route_name,
            "route_rationale": str(route_proposal.get("rationale", "")),
            "candidate_patch_rationales": reasons,
            "failure_condition": (
                "Reject the transfer route when calibration does not clear every frozen "
                "gain, stability, and confidence requirement against both target anchors."
            ),
        },
        execution_contract={
            "artifact_type": "source_outcome_transfer_policy",
            "llm_role": (
                "Offline proposal of a bounded kernel-patch portfolio. The LLM is not "
                "called during target replay."
            ),
            "llm_generated_executable_patch_fields": [
                "scales",
                "role_multipliers",
                "gp_beta",
                "gp_beta_end",
                "source_prior_strength",
                "source_similarity_temperature",
                "source_neighbor_count",
                "calibration_mode",
                "min_cv_gain",
                "confidence",
                "source_interaction_strength",
                "source_interaction_min_support",
                "canonicalize_source_values",
            ],
            "advisory_only_fields": [
                "hypothesis.claim",
                "hypothesis.route_rationale",
                "hypothesis.candidate_patch_rationales[].reason",
                "hypothesis.failure_condition",
            ],
            "causal_credit_rule": (
                "LLM patch credit requires a held-out gain over the fixed data-only "
                "transfer control; a gain over target-only alone is insufficient."
            ),
        },
        source_evidence=_json_clone(source_evidence),
        role_map=dict(role_map),
        operators=operators,
        kernel_patches=patches,
        execution=_json_clone(execution),
        gate=_json_clone(gate),
        provenance=_json_clone(provenance),
        evidence_boundary=(
            "The skill is compiled from public schemas, a fixed measured source history, "
            "and frozen LLM records. Hidden target outcomes are absent. Target outcomes "
            "enter only after each selected target candidate is revealed."
        ),
    )


def attach_skill_identity(
    audits: Mapping[tuple[str, int], list[dict[str, Any]]],
    skill: TransferSkill,
) -> None:
    identity = skill.identity()
    for events in audits.values():
        for event in events:
            event["transfer_skill"] = identity


def build_canonical_trace(
    *,
    skill: TransferSkill,
    selection: Mapping[str, Any],
    audits: Mapping[tuple[str, int], list[dict[str, Any]]],
    selector_mode: str,
    heldout_seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Flatten the method into a compact source-to-deployment event stream."""

    events: list[dict[str, Any]] = []

    def append(stage: str, event_type: str, payload: Mapping[str, Any]) -> None:
        events.append(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "sequence": len(events),
                "stage": stage,
                "event_type": event_type,
                "transfer_skill": skill.identity(),
                "payload": _json_clone(payload),
            }
        )

    append("compile", "source_evidence_frozen", skill.source_evidence)
    append("compile", "hypothesis_generated", skill.hypothesis)
    append(
        "compile",
        "transfer_skill_compiled",
        {
            "role_map": skill.role_map,
            "operators": [asdict(operator) for operator in skill.operators],
            "kernel_patch_ids": [
                patch.get("patch_id", "") for patch in skill.kernel_patches
            ],
            "execution": skill.execution,
        },
    )
    append(
        "calibration",
        "calibration_gate_decision",
        {
            "selected_mode": selection["selected_mode"],
            "selected_source_outcome_transfer": selection[
                "selected_source_outcome_transfer"
            ],
            "target_anchor_mode": selection["target_anchor_mode"],
            "thresholds": selection["thresholds"],
            "diagnostics": selection["source_outcome_diagnostics"],
            "decision_scope": selection.get("decision_scope"),
            "real_experiment_deployment_ready": selection.get(
                "real_experiment_deployment_ready"
            ),
            "offline_selection_cost": selection.get("offline_selection_cost"),
        },
    )
    if "mechanism_attribution" in selection:
        append(
            "evaluation",
            "mechanism_attribution",
            selection["mechanism_attribution"],
        )

    for seed in sorted(heldout_seeds):
        seed_events = audits.get((selector_mode, seed), [])
        append(
            "heldout",
            "heldout_replay_started",
            {"seed": seed, "round_count": len(seed_events)},
        )
        for source_event in seed_events:
            snapshot = source_event.get("hypothesis_snapshot", {})
            router_gate = snapshot.get("router_gate", {})
            append(
                "heldout",
                "acquisition_decision",
                {
                    "seed": seed,
                    "round_index": source_event.get("round_index"),
                    "public_observed_count": source_event.get("public_observed_count"),
                    "anchor_candidate": router_gate.get("anchor_candidate"),
                    "router_candidate": router_gate.get("router_candidate"),
                    "selected_candidate": source_event.get("selected_candidate"),
                    "router_authorized": router_gate.get("authorized"),
                    "router_reason": router_gate.get("reason"),
                    "transfer_mass": snapshot.get("transfer_mass", 0.0),
                    "active_experts": sorted(snapshot.get("expert_weights", {})),
                    "revealed_value": source_event.get("revealed_value"),
                    "best_so_far": source_event.get("best_so_far"),
                },
            )
        if seed_events:
            append(
                "heldout",
                "heldout_replay_completed",
                {
                    "seed": seed,
                    "final_best": seed_events[-1].get("best_so_far"),
                    "selected_mode": selection["selected_mode"],
                },
            )
    return events


def write_trace(path: Path, events: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
