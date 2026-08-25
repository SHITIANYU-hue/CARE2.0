from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import prepare_amylase_coverage_gate_protocol as prepare
import freeze_prospective_family_gate as freeze_gate
import run_source_only_family_gate as gate


def public_spec() -> dict:
    return json.loads(
        (
            ROOT
            / "preregistrations/2026-08-25-flip2-amylase-coverage-gate-v1/"
            "public_task_spec.json"
        ).read_text(encoding="utf-8")
    )


def llm_record(spec: dict, recommended_skill: str) -> dict:
    return {
        "status": "generated_before_dataset_download",
        "target_outcomes_available": False,
        "model": "anthropic/claude-opus-4-8",
        "public_task_spec": spec,
        "parsed_hypothesis": {"recommended_skill": recommended_skill},
    }


@pytest.mark.parametrize(
    "recommended_skill",
    ["source_additive_mutation_prior", "abstain"],
)
def test_build_config_supports_both_frozen_llm_decisions(
    recommended_skill: str,
) -> None:
    spec = public_spec()
    config = prepare.build_config(
        spec,
        llm_record(spec, recommended_skill),
        record_relative_path="preregistrations/example/llm_hypothesis_record.json",
        record_sha256="a" * 64,
        public_task_spec_relative_path=(
            "preregistrations/2026-08-25-flip2-amylase-coverage-gate-v1/"
            "public_task_spec.json"
        ),
        public_task_spec_sha256="b" * 64,
    )
    gate.validate_protocol(config)
    protocol = config["protocol"]
    assert protocol["semantic_hypothesis"]["recommended_skill"] == recommended_skill
    assert protocol["deployment"]["public_target_task_id"] == (
        "public_flip2_amylase_test"
    )
    assert protocol["gate"]["coverage_thresholds"][
        "minimum_deployment_exact_token_coverage"
    ] == 0.9
    assert protocol["semantic_hypothesis"]["decision_authority"] == (
        "may_abstain_after_coverage_gate_but_cannot_force_transfer"
    )
    assert protocol["gate"]["no_gate_selection_rule"] == (
        "coverage_gate_then_frozen_llm_abstention"
    )


def test_build_config_rejects_mismatched_embedded_spec() -> None:
    spec = public_spec()
    record = llm_record({**spec, "dataset": "different"}, "abstain")
    with pytest.raises(ValueError, match="different public task specification"):
        prepare.build_config(
            spec,
            record,
            record_relative_path="preregistrations/example/record.json",
            record_sha256="a" * 64,
            public_task_spec_relative_path=(
                "preregistrations/2026-08-25-flip2-amylase-coverage-gate-v1/"
                "public_task_spec.json"
            ),
            public_task_spec_sha256="b" * 64,
        )


def test_freeze_rejects_changed_public_task_spec_hash(tmp_path: Path) -> None:
    spec = public_spec()
    spec_path = tmp_path / "public_task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    hypothesis = {
        "public_task_spec_path": str(spec_path),
        "public_task_spec_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="hash does not match"):
        freeze_gate.verify_public_task_spec(hypothesis, llm_record(spec, "abstain"))
