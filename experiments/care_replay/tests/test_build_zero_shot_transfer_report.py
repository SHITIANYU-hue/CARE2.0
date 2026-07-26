from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_zero_shot_transfer_report as report  # noqa: E402


class BuildZeroShotTransferReportTest(unittest.TestCase):
    def test_stable_gain_and_harm_require_primary_ci_separation(self) -> None:
        positive = {
            "final_best": {"normal_95ci_low": 0.1, "normal_95ci_high": 1.0},
            "best_so_far_auc": {"normal_95ci_low": -1.0, "normal_95ci_high": 1.0},
        }
        negative = {
            "final_best": {"normal_95ci_low": -1.0, "normal_95ci_high": -0.1},
            "best_so_far_auc": {"normal_95ci_low": -1.0, "normal_95ci_high": -0.1},
        }
        self.assertTrue(report.stable_gain(positive))
        self.assertFalse(report.stable_harm(positive))
        self.assertTrue(report.stable_harm(negative))

    def test_paired_random_null_uses_the_same_seed_schedule(self) -> None:
        rows = []
        for seed in (1, 2, 3):
            rows.append({
                "seed": str(seed), "mode": "llm_direct_prior_skill",
                "final_best": "10", "best_so_far_auc": "5",
            })
            rows.append({
                "seed": str(seed), "mode": "llm_direct_prior_random_rule_r1_skill",
                "final_best": "8", "best_so_far_auc": "4",
            })
            rows.append({
                "seed": str(seed), "mode": "llm_direct_prior_random_rule_r2_skill",
                "final_best": "9", "best_so_far_auc": "5",
            })
        result = report.paired_mean_random_delta(
            rows,
            "llm_direct_prior_skill",
            [
                "llm_direct_prior_random_rule_r1_skill",
                "llm_direct_prior_random_rule_r2_skill",
            ],
        )
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["final_best"]["mean"], 1.5)
        self.assertEqual(result["best_so_far_auc"]["mean"], 0.5)

    def test_load_run_reports_every_llm_skill_and_null(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "run"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text(json.dumps({
                "source_dataset": "source",
                "target_dataset": "target",
                "evidence_mode": "source_schema_only",
                "proposal_mode": "hypothesis_only",
                "llm_model": "test-model",
                "skill_ids": ["skill"],
                "seed_count": 2,
                "initial_observations": 5,
                "online_rounds": 2,
                "target_observation_budget_per_seed": 7,
                "target_calibration_seed_count": 0,
                "target_predecision_outcome_count": 0,
                "evidence_boundary": "frozen",
            }), encoding="utf-8")
            rows = []
            for seed in (1, 2):
                for mode, final, auc in (
                    ("gp_ucb", 10, 5),
                    ("mixed_kernel_gp_ei", 10, 5),
                    ("llm_direct_prior_skill", 12, 7),
                    ("llm_direct_prior_random_rule_r1_skill", 11, 6),
                ):
                    rows.append({
                        "seed": str(seed), "mode": mode,
                        "final_best": str(final), "best_so_far_auc": str(auc),
                    })
            with (result_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            loaded = report.load_run(result_dir)
            self.assertEqual(loaded["random_null_count"], 1)
            self.assertEqual(loaded["skills"][0]["skill_id"], "skill")
            self.assertTrue(loaded["skills"][0]["stable_gain_vs_mixed_kernel_gp_ei"])


if __name__ == "__main__":
    unittest.main()
