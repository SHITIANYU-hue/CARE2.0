from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_refined_llm_kernel_skills as refinement  # noqa: E402


class LlmKernelSkillRefinementTest(unittest.TestCase):
    def test_payload_uses_calibration_diagnostics_but_not_heldout(self) -> None:
        record = {
            "prompt_payload": {"target_schema": {"dataset_id": "target"}},
            "normalized_patches": [{"patch_id": "p1", "scales": [0.0, 1.0]}],
        }
        diagnostics = {
            "final_best": {"mean": 1.0},
            "best_so_far_auc": {"mean": 2.0},
            "composite": {"mean": 3.0},
            "positive_fold_rate": 0.8,
            "fold_composite_means": [1.0, 2.0],
            "risk_adjusted_composite_gain": 2.5,
        }
        summary = {
            "source_dataset": "source",
            "target_dataset": "target",
            "selection": {
                "target_anchor_mode": "gp_ucb",
                "calibration_seed_count": 20,
                "patch_diagnostics": {"llm_evolved_kernel_p1": diagnostics},
            },
            "heldout": {"secret_marker": 999.0},
        }
        payload = refinement.build_refinement_payload(record, summary, 6)
        serialized = json.dumps(payload)
        self.assertIn("calibration_only_diagnostics_vs_target_anchor", serialized)
        self.assertNotIn("secret_marker", serialized)
        self.assertNotIn("999.0", serialized)


if __name__ == "__main__":
    unittest.main()
