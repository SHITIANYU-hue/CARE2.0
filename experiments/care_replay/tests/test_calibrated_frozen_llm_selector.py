from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_calibrated_frozen_llm_selector as selector  # noqa: E402
import run_llm_kernel_skill_evolution as evolution  # noqa: E402


def patch(patch_id: str) -> evolution.KernelSkillPatch:
    return evolution.KernelSkillPatch(
        patch_id=patch_id,
        scales=(0.0, 1.0),
        role_multipliers={},
        gp_beta=1.5,
        gp_beta_end=1.0,
        source_prior_strength=0.0,
        source_similarity_temperature=0.5,
        source_neighbor_count=3,
        calibration_mode="off",
        min_cv_gain=0.0,
        confidence=0.7,
        reason="test",
    )


def row(mode: str, seed: int, final_best: float, auc: float) -> dict[str, object]:
    return {
        "mode": mode,
        "seed": seed,
        "final_best": final_best,
        "best_so_far_auc": auc,
        "top10_hit": 0,
    }


class CalibratedFrozenSelectorTest(unittest.TestCase):
    def target_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in range(4):
            rows.extend(
                [
                    row("gp_ucb", seed, 80.0, 70.0),
                    row("mixed_kernel_gp_ei", seed, 79.0, 69.0),
                    row("target_acquisition_portfolio", seed, 78.0, 68.0),
                ]
            )
        return rows

    def test_selects_positive_llm_patch(self) -> None:
        rows = self.target_rows()
        mode = evolution.patch_mode(patch("positive"))
        rows.extend(row(mode, seed, 80.2, 70.5) for seed in range(4))
        selected, diagnostics = selector.select_frozen_policy(
            rows,
            set(range(4)),
            (patch("positive"),),
        )
        self.assertEqual(selected, mode)
        self.assertTrue(diagnostics["selected_llm_patch"])

    def test_falls_back_when_llm_patch_is_negative(self) -> None:
        rows = self.target_rows()
        mode = evolution.patch_mode(patch("negative"))
        rows.extend(row(mode, seed, 78.0, 66.0) for seed in range(4))
        selected, diagnostics = selector.select_frozen_policy(
            rows,
            set(range(4)),
            (patch("negative"),),
        )
        self.assertEqual(selected, "gp_ucb")
        self.assertFalse(diagnostics["selected_llm_patch"])

    def test_heldout_rows_do_not_change_selection(self) -> None:
        rows = self.target_rows()
        positive_mode = evolution.patch_mode(patch("positive"))
        rows.extend(row(positive_mode, seed, 80.2, 70.5) for seed in range(4))
        for seed in range(100, 104):
            rows.extend(
                [
                    row("gp_ucb", seed, 90.0, 90.0),
                    row("mixed_kernel_gp_ei", seed, 90.0, 90.0),
                    row("target_acquisition_portfolio", seed, 90.0, 90.0),
                    row(positive_mode, seed, 0.0, 0.0),
                ]
            )
        selected, _diagnostics = selector.select_frozen_policy(
            rows,
            set(range(4)),
            (patch("positive"),),
        )
        self.assertEqual(selected, positive_mode)

    def test_rejects_gain_that_is_not_stable_across_calibration_folds(self) -> None:
        rows: list[dict[str, object]] = []
        unstable_mode = evolution.patch_mode(patch("unstable"))
        gains = (3.0, 3.0, 3.0, -1.0, -1.0)
        for seed, gain in enumerate(gains):
            rows.extend(
                [
                    row("gp_ucb", seed, 80.0, 70.0),
                    row("mixed_kernel_gp_ei", seed, 79.0, 69.0),
                    row("target_acquisition_portfolio", seed, 78.0, 68.0),
                    row(unstable_mode, seed, 80.0 + gain, 70.0 + gain),
                ]
            )
        selected, diagnostics = selector.select_frozen_policy(
            rows,
            set(range(5)),
            (patch("unstable"),),
        )
        self.assertEqual(selected, "gp_ucb")
        self.assertEqual(
            diagnostics["patch_diagnostics"][unstable_mode]["positive_fold_rate"],
            0.6,
        )


if __name__ == "__main__":
    unittest.main()
