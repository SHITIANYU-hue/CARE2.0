from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_zero_shot_semantic_transfer as zero_shot  # noqa: E402


class ZeroShotSemanticTransferTest(unittest.TestCase):
    def test_paired_delta_reports_without_selection(self) -> None:
        rows = []
        for seed in range(3):
            rows.extend([
                {
                    "mode": "gp_ucb",
                    "seed": seed,
                    "final_best": 10.0,
                    "best_so_far_auc": 8.0,
                },
                {
                    "mode": "llm_direct_prior_skill",
                    "seed": seed,
                    "final_best": 11.0 + seed,
                    "best_so_far_auc": 8.5,
                },
            ])
        summary = zero_shot.paired_delta(
            rows,
            "llm_direct_prior_skill",
            "gp_ucb",
            {0, 1, 2},
        )
        self.assertEqual(summary["seed_count"], 3)
        self.assertEqual(summary["final_best"]["mean"], 2.0)
        self.assertEqual(summary["final_best"]["win_rate"], 1.0)
        self.assertGreater(summary["final_best"]["normal_95ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
