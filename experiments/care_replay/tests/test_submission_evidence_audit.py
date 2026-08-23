from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_submission_evidence_audit as audit  # noqa: E402


class SubmissionEvidenceAuditTest(unittest.TestCase):
    def test_exact_two_sided_sign_test_excludes_ties(self) -> None:
        self.assertAlmostEqual(audit.exact_two_sided_sign_p(6, 3), 0.5078125)
        self.assertAlmostEqual(audit.exact_two_sided_sign_p(3, 1), 0.625)
        self.assertIsNone(audit.exact_two_sided_sign_p(0, 0))

    def test_leave_one_route_out_range(self) -> None:
        self.assertEqual(audit.leave_one_out_mean_range([1.0, 2.0, 3.0]), (1.5, 2.5))

    def test_single_route_is_never_called_a_bootstrap_confirmation(self) -> None:
        summary = audit.summarize_values(
            [0.28],
            bootstrap_samples=100,
            bootstrap_seed=3,
        )
        self.assertFalse(summary["bootstrap_ci_excludes_zero"])

    def test_frozen_extension_is_not_promoted_without_positive_uncertainty(self) -> None:
        rows = [
            {
                "phase": "development",
                "case_id": "dev_positive",
                "domain": "materials_property",
                "target_task": "dev",
                audit.DELTA_FIELD: "3.0",
            },
            {
                "phase": "frozen_extension",
                "case_id": "holdout_positive",
                "domain": "reaction_optimization_cn",
                "target_task": "holdout_a",
                audit.DELTA_FIELD: "1.0",
            },
            {
                "phase": "frozen_extension",
                "case_id": "holdout_negative",
                "domain": "reaction_optimization_cn",
                "target_task": "holdout_b",
                audit.DELTA_FIELD: "-1.0",
            },
        ]
        result = audit.build_audit(
            rows,
            bootstrap_samples=500,
            bootstrap_seed=7,
        )
        self.assertFalse(result["confirmatory_positive_transfer_supported"])
        self.assertEqual(result["phases"]["frozen_extension"]["wins"], 1)
        self.assertEqual(result["phases"]["frozen_extension"]["losses"], 1)
        self.assertEqual(result["routes"][1]["evidence_tier"], "post_freeze_extension")


if __name__ == "__main__":
    unittest.main()
