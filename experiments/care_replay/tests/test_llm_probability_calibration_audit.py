from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_llm_probability_calibration as audit  # noqa: E402


class LlmProbabilityCalibrationAuditTests(unittest.TestCase):
    def test_candidate_menu_record_uses_selected_candidate(self) -> None:
        prompt = {
            "candidate_menu": {
                "columns": ["candidate_id", "gp_probability_improvement"],
                "rows": [["a", 0.2], ["b", 0.7]],
            },
        }
        self.assertEqual(
            audit.candidate_menu_record(prompt, "b")["gp_probability_improvement"],
            0.7,
        )

    def test_calibration_bins_include_probability_one(self) -> None:
        bins = audit.calibration_bins(
            [0.0, 0.2, 0.4, 0.8, 1.0],
            [0, 0, 1, 1, 1],
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(sum(row["count"] for row in bins), 5)
        self.assertEqual(bins[-1]["count"], 2)

    def test_gp_probability_can_be_reconstructed_from_posterior(self) -> None:
        probability, source = audit.gp_probability_for_candidate(
            {"gp_posterior_mean": 5.0, "gp_posterior_std": 2.0},
            5.0,
        )
        self.assertAlmostEqual(probability, 0.5)
        self.assertEqual(source, "reconstructed")

    def test_recorded_gp_probability_takes_precedence(self) -> None:
        probability, source = audit.gp_probability_for_candidate(
            {
                "gp_probability_improvement": 0.7,
                "gp_posterior_mean": 5.0,
                "gp_posterior_std": 2.0,
            },
            5.0,
        )
        self.assertAlmostEqual(probability, 0.7)
        self.assertEqual(source, "recorded")

    def test_binary_metrics_match_known_brier_and_bias(self) -> None:
        metrics = audit.binary_metrics(
            [0.2, 0.8],
            [0, 1],
            [0.0, 0.5, 1.0],
        )
        self.assertAlmostEqual(metrics["brier_score"], 0.04)
        self.assertAlmostEqual(metrics["mean_probability_bias"], 0.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)

    def test_roc_auc_handles_ties(self) -> None:
        self.assertAlmostEqual(audit.roc_auc([0.5, 0.5], [1, 0]), 0.5)
        self.assertIsNone(audit.roc_auc([0.2, 0.3], [0, 0]))

    def test_route_bootstrap_is_reproducible(self) -> None:
        first = audit.route_bootstrap_interval(
            [-0.2, 0.0, 0.1],
            samples=1000,
            seed=7,
        )
        second = audit.route_bootstrap_interval(
            [-0.2, 0.0, 0.1],
            samples=1000,
            seed=7,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
