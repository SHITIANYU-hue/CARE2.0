from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_goal_report as report  # noqa: E402


class BuildGoalReportTest(unittest.TestCase):
    def test_pairwise_gain_requires_primary_ci_above_zero(self) -> None:
        result = report.classify_pairwise({
            "gp_ucb": {
                "final_best": {"mean": 2.0, "normal_95ci_low": 0.1, "normal_95ci_high": 3.9},
                "best_so_far_auc": {"mean": 0.1, "normal_95ci_low": -0.5, "normal_95ci_high": 0.7},
            },
            "mixed_kernel_gp_ei": {
                "final_best": {"mean": 1.0, "normal_95ci_low": -0.2, "normal_95ci_high": 2.2},
                "best_so_far_auc": {"mean": -1.0, "normal_95ci_low": -2.0, "normal_95ci_high": -0.1},
            },
        })
        self.assertTrue(result["gp_ucb"]["primary_gain"])
        self.assertFalse(result["mixed_kernel_gp_ei"]["primary_gain"])

    def test_random_null_ci_is_computed_from_heldout_sd(self) -> None:
        item = report.load_random_null_from_summary({
            "protocol": {"target_dataset": "target"},
            "selected_on_calibration": "random",
            "selected_heldout": {
                "seed_count": 4,
                "final_best_mean_delta": 1.0,
                "final_best_sd_delta": 0.0,
                "auc_mean_delta": 0.5,
                "auc_sd_delta": 0.0,
                "final_best_win_rate": 0.75,
            },
        })
        self.assertEqual(item["final_best_ci"], [1.0, 1.0])
        self.assertTrue(item["stable_primary_gain"])

    def test_schema_route_proposal_is_public_metadata_only(self) -> None:
        proposal = report.cross_task_router.propose_route(
            "real_matbench_dielectric",
            "real_matbench_expt_gap",
        )
        self.assertEqual(
            proposal.recommended_candidate,
            "shared_descriptor_source_outcome",
        )


if __name__ == "__main__":
    unittest.main()
