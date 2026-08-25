#!/usr/bin/env python3
"""Materialize the frozen Amylase protocol from an outcome-blind LLM record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import run_multisource_transfer_baselines as multisource
import run_source_only_family_gate as gate
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_DIR = (
    ROOT / "preregistrations/2026-08-25-flip2-amylase-coverage-gate-v1"
)
LOCK_PATH = (
    "preregistrations/2026-08-25-flip2-amylase-coverage-gate-v1/"
    "protocol_preregistration.json"
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(
    spec: Mapping[str, Any], record: Mapping[str, Any]
) -> str:
    if spec.get("status") != "prepared_before_dataset_download":
        raise ValueError("The public task specification is not outcome-blind.")
    if record.get("status") != "generated_before_dataset_download":
        raise ValueError("The LLM record was not generated before data download.")
    if record.get("target_outcomes_available") is not False:
        raise ValueError("The LLM record does not exclude target outcomes.")
    if record.get("public_task_spec") != spec:
        raise ValueError("The LLM record embeds a different public task specification.")
    recommended = str(
        record.get("parsed_hypothesis", {}).get("recommended_skill", "")
    )
    if recommended not in gate.OUTCOME_BLIND_LLM_SKILLS:
        raise ValueError(f"Unknown outcome-blind LLM skill: {recommended}")
    return recommended


def build_config(
    spec: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    record_relative_path: str,
    record_sha256: str,
    public_task_spec_relative_path: str,
    public_task_spec_sha256: str,
) -> dict[str, Any]:
    recommended = validate_inputs(spec, record)
    thresholds = dict(spec["coverage_aware_gate"])
    thresholds.pop("eligibility_rule", None)
    thresholds.pop("coverage_rescue_rule", None)
    config = {
        "protocol": {
            "version": "care2_flip2_amylase_coverage_gate_v1",
            "status": "frozen_before_execution",
            "frozen_on": "2026-08-25",
            "analysis_status": "prospective_preregistered_before_test_outcomes",
            "evidence_class": "prospective_external_coverage_aware_router_confirmation",
            "dataset": {
                "name": str(spec["dataset"]),
                "official_split": str(spec["official_split"]),
                "url": str(spec["source_url"]),
                "local_filename": "flip2_amylase_one_to_many.csv.gz",
                "license": str(spec["license"]),
                "citation": str(spec["source"]),
                "public_metadata": dict(spec["public_variant_counts"]),
            },
            "preregistration": {"lock_path": LOCK_PATH},
            "semantic_hypothesis": {
                "status": "generated_before_dataset_download",
                "model": str(record["model"]),
                "record_path": record_relative_path,
                "record_sha256": record_sha256,
                "public_task_spec_path": public_task_spec_relative_path,
                "public_task_spec_sha256": public_task_spec_sha256,
                "recommended_skill": recommended,
                "decision_authority": (
                    "may_abstain_after_coverage_gate_but_cannot_force_transfer"
                ),
                "target_outcomes_available_to_model": False,
            },
            "calibration_routes": [
                {
                    "route_id": "train_to_validation",
                    "source_task_ids": ["real_flip2_amylase_train"],
                    "pseudo_target_task_id": "real_flip2_amylase_validation",
                    "heldout_seed_start": 99400,
                    "heldout_seed_count": 100,
                }
            ],
            "deployment": {
                "source_task_ids": ["real_flip2_amylase_train_validation"],
                "public_target_task_id": "public_flip2_amylase_test",
                "target_task_id": "real_flip2_amylase_test",
                "heldout_seed_start": 99600,
                "heldout_seed_count": 100,
            },
            "gate": {
                "candidate_methods": [multisource.ADDITIVE_MUTATION_POLICY_MODE],
                "eligibility_rule": gate.COVERAGE_CONDITIONED_ELIGIBILITY_RULE,
                "coverage_thresholds": thresholds,
                "no_gate_selection_rule": (
                    "coverage_gate_then_frozen_llm_abstention"
                ),
                "fallback_policy": "target_gp_ucb",
            },
            "success_rules": {
                "candidate_efficacy": (
                    "official_test_paired_auc_ci95_lower_bound_above_zero"
                ),
                "router_correctness": (
                    "transfer_if_candidate_effect_is_positive_or_hold_if_candidate_effect_is_nonpositive"
                ),
                "llm_abstention": (
                    "if_recommended_skill_is_abstain_then_deploy_target_gp_ucb"
                ),
                "no_optional_stopping": True,
            },
            "source_seed": 0,
            "source_observation_count_per_task": 5000,
            "source_inducing_limit": 128,
            "initial_observations": 3,
            "reveal_rounds": 12,
            "rgpe_draws": 256,
            "multitask_rho_grid": [
                -0.9,
                -0.75,
                -0.5,
                -0.25,
                0.0,
                0.25,
                0.5,
                0.75,
                0.9,
            ],
            "bma_temperature": 1.0,
            "skill_prior_mass_start": 0.5,
            "skill_prior_mass_end": 0.1,
            "kernel": {
                "gp_beta": 1.5,
                "numeric_length_scale": 0.45,
                "categorical_length_scale": 4.0,
                "gp_noise": 0.05,
            },
            "constraints": {
                "test_partition_outcomes_excluded_from_gate_selection": True,
                "public_target_adapter_contains_no_measured_outcomes": True,
                "structural_coverage_is_target_outcome_invariant": True,
                "calibration_uses_official_train_and_validation_partitions_only": True,
                "official_one_to_many_split_preserved": True,
                "llm_hypothesis_generated_before_dataset_download": True,
                "same_initial_observations_within_seed": True,
                "same_target_reveal_budget": True,
                "all_declared_seeds_retained": True,
                "no_optional_stopping": True,
            },
            "claim_boundary": str(spec["claim_boundary"]),
        }
    }
    gate.validate_protocol(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = replay.RAW_DATA / "flip2_amylase_one_to_many.csv.gz"
    if raw_path.exists():
        raise ValueError("Refusing protocol preparation after Amylase download.")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    record_relative = str(args.record.resolve().relative_to(ROOT.resolve()))
    spec_relative = str(args.spec.resolve().relative_to(ROOT.resolve()))
    config = build_config(
        spec,
        record,
        record_relative_path=record_relative,
        record_sha256=sha256_path(args.record),
        public_task_spec_relative_path=spec_relative,
        public_task_spec_sha256=sha256_path(args.spec),
    )
    encoded = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != encoded:
        raise ValueError("Existing protocol config differs from the LLM record.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "canonical_config_sha256": gate.canonical_sha256(config),
                "recommended_skill": config["protocol"]["semantic_hypothesis"][
                    "recommended_skill"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
