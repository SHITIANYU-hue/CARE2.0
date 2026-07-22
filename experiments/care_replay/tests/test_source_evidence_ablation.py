from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_source_evidence_ablation as ablation  # noqa: E402


class SourceEvidenceAblationTest(unittest.TestCase):
    def test_paired_summary_uses_only_matched_seeds(self) -> None:
        left = {
            1: {"final_best": "5", "best_so_far_auc": "4", "top10_hit": "1"},
            2: {"final_best": "7", "best_so_far_auc": "6", "top10_hit": "1"},
            3: {"final_best": "99", "best_so_far_auc": "99", "top10_hit": "1"},
        }
        right = {
            1: {"final_best": "3", "best_so_far_auc": "3", "top10_hit": "0"},
            2: {"final_best": "5", "best_so_far_auc": "5", "top10_hit": "1"},
        }
        summary = ablation.paired_summary(left, right)
        self.assertEqual(summary["paired_seed_count"], 2)
        self.assertEqual(summary["metrics"]["final_best"]["mean"], 2.0)
        self.assertEqual(summary["metrics"]["best_so_far_auc"]["mean"], 1.0)
        self.assertEqual(summary["metrics"]["top10_hit"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
