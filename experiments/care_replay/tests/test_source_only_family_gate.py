from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_source_only_family_gate as gate


def summary(effects: dict[str, tuple[float, float, float]]) -> dict:
    return {
        "comparisons_vs_target_gp_ucb": {
            method: {
                "best_so_far_auc": {
                    "mean_delta": values[0],
                    "normal_95ci_low": values[1],
                    "normal_95ci_high": values[2],
                }
            }
            for method, values in effects.items()
        }
    }


class SourceOnlyFamilyGateTests(unittest.TestCase):
    def test_rejects_when_no_method_is_positive_on_every_source_route(self):
        routes = {
            "a_to_b": summary({"rgpe": (2.0, 1.0, 3.0), "icm": (-1.0, -2.0, 0.0)}),
            "b_to_a": summary({"rgpe": (-0.5, -1.0, 0.0), "icm": (1.0, 0.2, 1.8)}),
        }
        selected = gate.select_source_only_policy(routes, ["rgpe", "icm"])
        self.assertEqual(selected["eligible_methods"], [])
        self.assertEqual(selected["deployed_policy"], "target_gp_ucb")
        self.assertTrue(selected["fallback_triggered"])

    def test_selects_best_method_that_is_positive_on_every_source_route(self):
        routes = {
            "a_to_b": summary({"rgpe": (2.0, 1.0, 3.0), "icm": (3.0, 2.0, 4.0)}),
            "b_to_a": summary({"rgpe": (2.5, 1.5, 3.5), "icm": (1.0, 0.2, 1.8)}),
        }
        selected = gate.select_source_only_policy(routes, ["rgpe", "icm"])
        self.assertEqual(selected["eligible_methods"], ["rgpe", "icm"])
        self.assertEqual(selected["deployed_policy"], "rgpe")
        self.assertFalse(selected["fallback_triggered"])

    def test_protocol_rejects_deployment_target_in_source_only_calibration(self):
        config = {
            "protocol": {
                "status": "frozen_before_execution",
                "calibration_routes": [
                    {
                        "route_id": "leaky",
                        "source_task_ids": ["real_flip2_hydro_p01053"],
                        "pseudo_target_task_id": "real_flip2_hydro_p06241",
                    }
                ],
                "deployment": {
                    "source_task_ids": ["real_flip2_hydro_p01053"],
                    "target_task_id": "real_flip2_hydro_p06241",
                },
                "gate": {
                    "candidate_methods": ["multisource_rgpe"],
                    "fallback_policy": "target_gp_ucb",
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "absent from source-only"):
            gate.validate_protocol(config)


if __name__ == "__main__":
    unittest.main()
