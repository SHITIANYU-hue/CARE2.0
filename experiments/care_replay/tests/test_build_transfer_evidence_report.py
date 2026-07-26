from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_transfer_evidence_report as report  # noqa: E402


class BuildTransferEvidenceReportTest(unittest.TestCase):
    def test_execution_kind_separates_warm_start_and_continuous_routes(self) -> None:
        self.assertEqual(
            report.execution_kind("llambo_warmstart_source_skill"),
            "warm-start",
        )
        self.assertEqual(
            report.execution_kind("llm_direct_prior_source_skill"),
            "continuous semantic route",
        )

    def test_primary_gain_requires_a_positive_primary_ci(self) -> None:
        self.assertTrue(report.primary_gain({
            "metrics": {
                "final_best": {"normal_95ci_low": -1.0},
                "best_so_far_auc": {"normal_95ci_low": 0.1},
            }
        }))
        self.assertFalse(report.primary_gain({
            "metrics": {
                "final_best": {"normal_95ci_low": -1.0},
                "best_so_far_auc": {"normal_95ci_low": 0.0},
            }
        }))

    def test_build_report_counts_evidence_layers(self) -> None:
        comparison = {
            "metrics": {
                "final_best": {"normal_95ci_low": 0.1},
                "best_so_far_auc": {"normal_95ci_low": -0.1},
            }
        }
        source = {
            "pairs": [
                {
                    "source_dataset": "source_a",
                    "target_dataset": "target_a",
                    "pair_id": "a",
                    "route": "llm_direct_prior_a",
                    "execution_kind": "continuous semantic route",
                    "source_schema_gain_vs_matched_llm": True,
                    "source_schema_gain_vs_strongest_bo": True,
                    "matched_llm_comparison": comparison,
                    "strongest_bo_comparison": comparison,
                },
                {
                    "source_dataset": "source_b",
                    "target_dataset": "target_b",
                    "pair_id": "b",
                    "route": "llambo_warmstart_b",
                    "execution_kind": "warm-start",
                    "source_schema_gain_vs_matched_llm": True,
                    "source_schema_gain_vs_strongest_bo": False,
                    "matched_llm_comparison": comparison,
                    "strongest_bo_comparison": comparison,
                },
            ]
        }
        portfolio = {
            "portfolio_count": 3,
            "source_informed_deployment_count": 1,
            "warm_start_count": 1,
            "continuous_transfer_candidate_count": 0,
            "exact_fallback_count": 2,
            "portfolios": [],
        }
        result = report.build_report(source, portfolio, {
            "random_null_count": 2,
            "random_null_stable_gain_count": 0,
            "traditional_transfer_suite_count": 1,
        })
        self.assertEqual(result["aggregate"]["continuous_semantic_gain_count"], 1)
        self.assertEqual(result["aggregate"]["warm_start_gain_count"], 1)
        self.assertEqual(result["aggregate"]["source_outcome_exact_fallback_count"], 2)


if __name__ == "__main__":
    unittest.main()
