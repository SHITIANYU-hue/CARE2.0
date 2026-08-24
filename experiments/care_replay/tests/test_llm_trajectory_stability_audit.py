from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_llm_trajectory_stability_audit as audit  # noqa: E402


class LlmTrajectoryStabilityAuditTest(unittest.TestCase):
    def test_strict_sign_flips_are_not_hidden_by_aggregate_counts(self) -> None:
        early = [
            {
                "case_id": "a",
                "domain": "chemistry",
                "evidence_tier": "development",
                audit.EARLY_FIELD: "1.0",
            },
            {
                "case_id": "b",
                "domain": "materials",
                "evidence_tier": "holdout",
                audit.EARLY_FIELD: "-2.0",
            },
            {
                "case_id": "c",
                "domain": "molecular",
                "evidence_tier": "holdout",
                audit.EARLY_FIELD: "0.0",
            },
        ]
        repeat = [
            {
                "case_id": "a",
                "domain": "chemistry",
                "evidence_tier": "development",
                audit.REPEAT_FIELD: "-1.0",
            },
            {
                "case_id": "b",
                "domain": "materials",
                "evidence_tier": "holdout",
                audit.REPEAT_FIELD: "2.0",
            },
            {
                "case_id": "c",
                "domain": "molecular",
                "evidence_tier": "holdout",
                audit.REPEAT_FIELD: "0.0",
            },
        ]
        result = audit.build_audit(early, repeat)
        self.assertEqual(result["strict_sign_flip_count"], 2)
        self.assertEqual(result["exact_sign_agreement_count"], 1)
        self.assertEqual(result["claim_decision"], "descriptive_only_incomplete_frozen_repetition")

    def test_second_repeat_is_attached_without_changing_pairing(self) -> None:
        early = [{
            "case_id": "a",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.EARLY_FIELD: "0.2",
        }]
        repeat = [{
            "case_id": "a",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.REPEAT_FIELD: "1.4",
        }]
        second = [{"case_id": "a", audit.REPEAT_FIELD: "0.3"}]
        result = audit.build_audit(early, repeat, second)
        self.assertEqual(result["second_frozen_repeat_route_count"], 1)
        self.assertEqual(result["routes"][0]["second_frozen_repeat_auc_delta"], 0.3)

    def test_route_mismatch_is_rejected(self) -> None:
        early = [{
            "case_id": "a",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.EARLY_FIELD: "1.0",
        }]
        repeat = [{
            "case_id": "b",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.REPEAT_FIELD: "1.0",
        }]
        with self.assertRaisesRegex(ValueError, "Route sets do not match"):
            audit.build_audit(early, repeat)

    def test_svg_marks_descriptive_boundary(self) -> None:
        early = [{
            "case_id": "a",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.EARLY_FIELD: "1.0",
        }]
        repeat = [{
            "case_id": "a",
            "domain": "chemistry",
            "evidence_tier": "holdout",
            audit.REPEAT_FIELD: "-1.0",
        }]
        result = audit.build_audit(early, repeat)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stability.svg"
            audit.write_stability_svg(result, output)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("SIGN FLIP", svg)
        self.assertIn("Descriptive audit only", svg)


if __name__ == "__main__":
    unittest.main()
