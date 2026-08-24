import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "calibration_gate",
    SCRIPTS / "build_online_llm_calibration_gate_audit.py",
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


class OnlineLlmCalibrationGateAuditTests(unittest.TestCase):
    def test_calibration_features_exclude_gp_deferral(self):
        rounds = [
            {"expected_outcome": 20.0, "revealed_value": 10.0, "hard_abstention": False},
            {"expected_outcome": None, "revealed_value": 80.0, "hard_abstention": False},
            {"expected_outcome": 40.0, "revealed_value": 20.0, "hard_abstention": False},
        ]
        result = gate.calibration_features(rounds, 3)
        self.assertEqual(result["scored_prediction_count"], 2)
        self.assertEqual(result["mean_absolute_prediction_error"], 15.0)

    def test_gate_uses_error_or_explicit_abstention(self):
        self.assertTrue(gate.gate_switches({"hard_abstention": False, "mean_absolute_prediction_error": 20.0}, 10.0))
        self.assertFalse(gate.gate_switches({"hard_abstention": False, "mean_absolute_prediction_error": 5.0}, 10.0))
        self.assertTrue(gate.gate_switches({"hard_abstention": True, "mean_absolute_prediction_error": 0.0}, 10.0))


if __name__ == "__main__":
    unittest.main()
