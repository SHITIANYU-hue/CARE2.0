from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_multidomain_llm_report as report  # noqa: E402


class MultidomainLlmReportTest(unittest.TestCase):
    def test_majority_and_domain_claims_use_predeclared_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = []
            for index, (target, selected, low, seeds) in enumerate((
                ("materials", True, 1.0, 500),
                ("molecular", True, 0.2, 500),
                ("reaction", False, -1.0, 100),
                ("wetlab", True, 0.3, 20),
            )):
                path = root / f"{index}.json"
                path.write_text(json.dumps({
                    "target_dataset": target,
                    "domain": target,
                    "heldout_seed_count": seeds,
                    "selection": {
                        "selected_llm_skill": selected,
                        "selected_mode": "llm" if selected else "gp_ucb",
                    },
                    "heldout_strongest_target_mode_descriptive_only": "gp_ucb",
                    "heldout": {"llm_calibrated_selector": {"final_best": 1.0, "best_so_far_auc": 1.0}},
                    "heldout_pairwise": {
                        "gp_ucb": {
                            "final_best": {"mean": low, "normal_95ci_low": low, "normal_95ci_high": low + 1},
                            "best_so_far_auc": {"mean": low, "normal_95ci_low": low, "normal_95ci_high": low + 1},
                            "top10_hit": {"mean": 0.0, "normal_95ci_low": -0.1, "normal_95ci_high": 0.1},
                        }
                    },
                }), encoding="utf-8")
                paths.append(path)
            result = report.aggregate(paths)
            self.assertTrue(result["majority_gain"])
            self.assertTrue(result["multidomain_gain"])
            self.assertEqual(result["fallback_count"], 1)
            self.assertEqual(result["evaluated_run_count"], 3)
            self.assertEqual(result["development_run_count"], 1)
            self.assertEqual(result["development_signal_count"], 1)


if __name__ == "__main__":
    unittest.main()
