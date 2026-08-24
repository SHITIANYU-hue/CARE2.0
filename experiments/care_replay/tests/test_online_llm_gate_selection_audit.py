import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "gate_selection",
    SCRIPTS / "build_online_llm_gate_selection_audit.py",
)
selection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(selection)


class OnlineLlmGateSelectionAuditTests(unittest.TestCase):
    def test_route_split_is_disjoint(self):
        config = selection.load_config(
            ROOT / "configs" / "online_llm_gate_selection_audit_v1.json"
        )
        protocol = config["protocol"]
        training = selection.portfolio.load_portfolio_config(
            ROOT / protocol["training_config"]
        )
        evaluation = selection.portfolio.load_portfolio_config(
            ROOT / protocol["evaluation_config"]
        )
        result = selection.validate_split(training, evaluation)
        self.assertEqual(result["training_route_count"], 11)
        self.assertEqual(result["evaluation_route_count"], 6)
        self.assertEqual(result["overlap_count"], 0)

    def test_selection_ignores_evaluation_metrics(self):
        candidates = [
            {
                "policy": {
                    "policy_id": "a",
                    "parameter_count": 1,
                    "gate_round": 1,
                    "threshold": None,
                    "hard_abstention": False,
                },
                "training": {
                    "losses": 0,
                    "mean_auc_delta": 1.0,
                    "worst_route_auc_delta": 0.0,
                    "switch_count": 3,
                },
                "evaluation": {"mean_auc_delta": -999.0},
            },
            {
                "policy": {
                    "policy_id": "b",
                    "parameter_count": 1,
                    "gate_round": 1,
                    "threshold": None,
                    "hard_abstention": False,
                },
                "training": {
                    "losses": 1,
                    "mean_auc_delta": 100.0,
                    "worst_route_auc_delta": -0.1,
                    "switch_count": 1,
                },
                "evaluation": {"mean_auc_delta": 999.0},
            },
        ]
        self.assertEqual(selection.select_candidate(candidates)["policy"]["policy_id"], "a")
        candidates[0]["evaluation"]["mean_auc_delta"] = -1e9
        candidates[1]["evaluation"]["mean_auc_delta"] = 1e9
        self.assertEqual(selection.select_candidate(candidates)["policy"]["policy_id"], "a")

    def test_bounded_authority_candidates_are_explicit(self):
        config = selection.load_config(
            ROOT / "configs" / "online_llm_gate_selection_audit_v1.json"
        )
        candidates = selection.policy_candidates(config["protocol"])
        bounded = [row for row in candidates if row["kind"] == "bounded_authority"]
        self.assertEqual(len(bounded), 5)
        self.assertTrue(all(row["force_fallback"] for row in bounded))
        self.assertTrue(all(row["threshold"] is None for row in bounded))

    def test_route_disjoint_confirmation_is_frozen_and_complete(self):
        config = selection.matched.load_json(
            ROOT
            / "configs"
            / "online_llm_bounded_authority_route_disjoint_confirmation_v1.json"
        )
        self.assertEqual(config["protocol"]["status"], "frozen_before_execution")
        self.assertEqual(config["protocol"]["replicates_per_route"], 30)
        self.assertEqual(len(config["cases"]), 6)
        self.assertEqual(config["calibration_gate"]["evaluation_after_reveals"], 1)
        self.assertTrue(
            config["calibration_gate"]["force_fallback_after_evaluation"]
        )
        self.assertNotIn("prediction_mae_threshold", config["calibration_gate"])
        for case in config["cases"]:
            self.assertTrue((ROOT / case["initial_record"]).is_file())


if __name__ == "__main__":
    unittest.main()
