from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import ingest_hypothesis_transfer as ingest


class IngestHypothesisTransferTest(unittest.TestCase):
    def test_cards_keep_failure_conditions_and_result_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record_path = root / "record.json"
            record_path.write_text(json.dumps({
                "model": "test-model",
                "source_dataset": "source",
                "target_dataset": "target",
                "evidence_mode": "source_schema_only",
                "proposal_mode": "hypothesis_only",
                "hypothesis_compilation": {"fixed_execution_parameters": True},
                "parsed_response": {"hypotheses": [{
                    "hypothesis_id": "mechanism_one",
                    "claim": "A increases the outcome.",
                    "mechanism": "A stabilizes the transition state.",
                    "conditions": {"field": "value"},
                    "expected_direction": "positive",
                    "failure_conditions": ["when the substrate class changes"],
                    "confidence": 0.7,
                }]},
            }), encoding="utf-8")
            report = {
                "runs": [{
                    "source_dataset": "source",
                    "target_dataset": "target",
                    "result_dir": "results/run",
                    "seed_count": 10,
                    "random_null_count": 1,
                    "target_calibration_seed_count": 0,
                    "target_predecision_outcome_count": 0,
                    "target_observation_budget_per_seed": 13,
                    "record_sha256": "abc",
                    "skills": [{
                        "skill_id": "mechanism_one",
                        "vs_mixed_kernel_gp_ei": {
                            "seed_count": 10,
                            "final_best": {"mean": 1.0, "normal_95ci_low": 0.2, "normal_95ci_high": 1.8},
                            "best_so_far_auc": {"mean": 0.5, "normal_95ci_low": -0.2, "normal_95ci_high": 1.2},
                        },
                        "vs_random_null": {
                            "seed_count": 10,
                            "final_best": {"mean": 0.5, "normal_95ci_low": -0.1, "normal_95ci_high": 1.1},
                            "best_so_far_auc": {"mean": 0.2, "normal_95ci_low": -0.2, "normal_95ci_high": 0.6},
                        },
                        "stable_gain_vs_mixed_kernel_gp_ei": True,
                        "stable_harm_vs_mixed_kernel_gp_ei": False,
                    }],
                }],
            }
            cards = ingest.build_cards([record_path], report, "2026-07-25")
            hypothesis = next(card for card in cards if card["type"] == "hypothesis")
            result = next(card for card in cards if card["type"] == "experiment_result")
            run_log = next(card for card in cards if card["type"] == "run_log")
            self.assertIn("Failure conditions", hypothesis["content"])
            self.assertEqual(hypothesis["status"], "candidate")
            self.assertIn("no calibration outcomes", result["content"])
            self.assertEqual(run_log["status"], "done")


if __name__ == "__main__":
    unittest.main()
