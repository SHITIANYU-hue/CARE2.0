from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_portfolio_report as report  # noqa: E402


def summary(selected: bool) -> dict:
    return {
        "source_dataset": "source",
        "target_dataset": "target",
        "calibration_seed_count": 2,
        "heldout_seed_count": 3,
        "selection": {
            "selected_source_outcome_transfer": selected,
            "selected_candidate_id": "positive" if selected else None,
            "target_anchor_mode": "gp_ucb",
        },
        "heldout": {
            "matched_target_only_llm": {"final_best": 10.0, "best_so_far_auc": 8.0},
            "gp_ucb": {"final_best": 9.0, "best_so_far_auc": 7.0, "composite": 16.0},
            "care_source_outcome_portfolio": {"final_best": 12.0, "best_so_far_auc": 11.0},
        },
    }


class BuildPortfolioReportTest(unittest.TestCase):
    def test_selected_route_is_compared_to_both_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "raw_summaries"
            summary_path.mkdir()
            path = summary_path / "summary.json"
            path.write_text(json.dumps(summary(True)), encoding="utf-8")
            item = report.load_portfolio_summary(path)
        self.assertEqual(item["final_delta_vs_matched"], 2.0)
        self.assertEqual(item["auc_delta_vs_matched"], 3.0)
        self.assertEqual(item["final_delta_vs_strongest_bo"], 3.0)

    def test_rejected_route_reports_exact_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "raw_summaries"
            summary_path.mkdir()
            path = summary_path / "summary.json"
            path.write_text(json.dumps(summary(False)), encoding="utf-8")
            item = report.load_portfolio_summary(path)
        self.assertFalse(item["transfer_deployed"])
        self.assertEqual(item["final_delta_vs_matched"], 0.0)
        self.assertEqual(item["final_delta_vs_strongest_bo"], 1.0)


if __name__ == "__main__":
    unittest.main()
