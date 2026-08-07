from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import transfer_skill  # noqa: E402


def make_skill() -> transfer_skill.TransferSkill:
    return transfer_skill.compile_transfer_skill(
        source_dataset="source_task",
        target_dataset="target_task",
        route_proposal={
            "recommended_candidate": "shared_descriptor_source_outcome",
            "rationale": "The public descriptor roles align.",
        },
        source_evidence={
            "source_seed": 0,
            "observation_count": 32,
            "roles": [{"source_field": "a", "target_field": "b"}],
        },
        role_map={"a": "b"},
        kernel_patches=[
            {
                "patch_id": "frozen_patch",
                "scales": [0.0, 1.0],
                "source_prior_strength": 0.8,
                "confidence": 0.7,
                "reason": "Test the mapped descriptor effect.",
            }
        ],
        execution={
            "calibration_seed_start": 100,
            "calibration_seed_count": 5,
            "heldout_seed_start": 200,
            "heldout_seed_count": 10,
            "source_initial_strategy": "matched",
            "min_source_support": 3,
            "fixed_ensemble_scales": [0.5, 1.0],
            "router_min_observations": 5,
            "router_min_quality": 0.2,
            "router_max_transfer_mass": 0.45,
        },
        gate={
            "fallback_mode": "matched_target_only_llm",
            "confirmation": {
                "min_risk_adjusted_gain": 0.25,
                "min_positive_fold_rate": 0.8,
            },
        },
        provenance={"model": "test-model", "record_sha256": "abc"},
    )


class TransferSkillTests(unittest.TestCase):
    def test_fingerprint_is_stable_and_serialized(self) -> None:
        first = make_skill()
        second = make_skill()
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.as_dict()["fingerprint"], first.fingerprint)
        self.assertTrue(first.skill_id.endswith(first.skill_id.rsplit("::", 1)[-1]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skill.json"
            first.write(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["fingerprint"], first.fingerprint)
        self.assertEqual(payload["operators"][0]["name"], "source_informed_initial_design")
        self.assertEqual(
            payload["execution_contract"]["artifact_type"],
            "source_outcome_transfer_policy",
        )
        self.assertIn(
            "hypothesis.failure_condition",
            payload["execution_contract"]["advisory_only_fields"],
        )

    def test_hidden_target_outcomes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hidden target evidence"):
            transfer_skill.compile_transfer_skill(
                source_dataset="source_task",
                target_dataset="target_task",
                route_proposal={"recommended_candidate": "source_outcome"},
                source_evidence={"hidden_target_outcomes": [1.0]},
                role_map={"a": "b"},
                kernel_patches=[{"patch_id": "bad"}],
                execution={
                    "calibration_seed_start": 100,
                    "calibration_seed_count": 5,
                    "heldout_seed_start": 200,
                    "heldout_seed_count": 10,
                    "source_initial_strategy": "matched",
                    "min_source_support": 3,
                    "fixed_ensemble_scales": [1.0],
                    "router_min_observations": 5,
                    "router_min_quality": 0.2,
                    "router_max_transfer_mass": 0.45,
                },
                gate={"fallback_mode": "matched_target_only_llm"},
                provenance={},
            )

    def test_trace_links_hypothesis_gate_and_round_decisions(self) -> None:
        skill = make_skill()
        selection = {
            "selected_mode": "llm_transfer_router",
            "selected_source_outcome_transfer": True,
            "target_anchor_mode": "gp_ucb",
            "thresholds": {"min_positive_fold_rate": 0.8},
            "source_outcome_diagnostics": {"gp_ucb": {"eligible": True}},
            "mechanism_attribution": {
                "classification": "continuous_component_not_established"
            },
        }
        audits = {
            ("care_source_outcome_router", 101): [
                {
                    "round_index": 0,
                    "public_observed_count": 5,
                    "selected_candidate": "candidate-7",
                    "revealed_value": 82.0,
                    "best_so_far": 82.0,
                    "hypothesis_snapshot": {
                        "transfer_mass": 0.2,
                        "expert_weights": {"frozen_patch": 1.0},
                        "router_gate": {
                            "anchor_candidate": "candidate-2",
                            "router_candidate": "candidate-7",
                            "authorized": True,
                            "reason": "quality_gate_passed",
                        },
                    },
                }
            ]
        }
        events = transfer_skill.build_canonical_trace(
            skill=skill,
            selection=selection,
            audits=audits,
            selector_mode="care_source_outcome_router",
            heldout_seeds={101},
        )
        event_types = [event["event_type"] for event in events]
        self.assertEqual(
            event_types[:4],
            [
                "source_evidence_frozen",
                "hypothesis_generated",
                "transfer_skill_compiled",
                "calibration_gate_decision",
            ],
        )
        acquisition = next(
            event for event in events if event["event_type"] == "acquisition_decision"
        )
        self.assertEqual(acquisition["payload"]["selected_candidate"], "candidate-7")
        self.assertEqual(acquisition["payload"]["transfer_mass"], 0.2)
        mechanism = next(
            event for event in events if event["event_type"] == "mechanism_attribution"
        )
        self.assertEqual(
            mechanism["payload"]["classification"],
            "continuous_component_not_established",
        )


if __name__ == "__main__":
    unittest.main()
