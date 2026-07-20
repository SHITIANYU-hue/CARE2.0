from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_synthetic_suzuki as replay  # noqa: E402
import run_llm_kernel_skill_evolution as evolution  # noqa: E402
import run_llm_transfer_router as router  # noqa: E402
import run_transfer_weighted_kernel as weighted  # noqa: E402


def make_adapter() -> replay.DatasetAdapter:
    candidates = (
        replay.Candidate("a1", "g", 0.0, 0.0, 0.0, 10.0, {"catalyst": "A"}),
        replay.Candidate("a2", "g", 0.0, 0.0, 0.0, 12.0, {"catalyst": "A"}),
        replay.Candidate("a3", "g", 0.0, 0.0, 0.0, 11.0, {"catalyst": "A"}),
        replay.Candidate("b1", "g", 0.0, 0.0, 0.0, 80.0, {"catalyst": "B"}),
        replay.Candidate("b2", "g", 0.0, 0.0, 0.0, 90.0, {"catalyst": "B"}),
        replay.Candidate("b3", "g", 0.0, 0.0, 0.0, 85.0, {"catalyst": "B"}),
        replay.Candidate("c1", "g", 0.0, 0.0, 0.0, 50.0, {"catalyst": "C"}),
        replay.Candidate("c2", "g", 0.0, 0.0, 0.0, 55.0, {"catalyst": "C"}),
    )
    return replay.DatasetAdapter(
        dataset_id="unit_test",
        title="Unit test adapter",
        objective="maximize score",
        decision_columns=("catalyst",),
        hidden_target="score",
        group_column="group",
        preferred_groups=(),
        failure_note="test only",
        candidates=candidates,
    )


class ExplorationPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = make_adapter()
        self.observed = [
            self.adapter.candidates[0],
            self.adapter.candidates[1],
            self.adapter.candidates[3],
            self.adapter.candidates[4],
        ]
        self.observed_ids = {candidate.candidate_id for candidate in self.observed}

    def test_coverage_exposes_public_unseen_values_without_targets(self) -> None:
        payload = replay.exploration_evidence_payload(self.adapter, self.observed, 0)
        unseen = [row for row in payload["factor_coverage"] if row["coverage_state"] == "unseen"]
        self.assertEqual([row["value"] for row in unseen], ["C"])
        self.assertNotIn("objective_value", str(payload["factor_coverage"]))
        self.assertNotIn("50.0", str(payload["factor_coverage"]))
        self.assertEqual(set(payload["top_revealed"][0]["metadata"]), {"catalyst"})

    def test_normalizer_accepts_explore_exploit_and_avoid_with_evidence(self) -> None:
        parsed = {
            "confidence": 0.83,
            "adjustments": [
                {
                    "field": "catalyst",
                    "value": "C",
                    "direction": "prefer",
                    "intent": "explore",
                    "weight": 0.035,
                },
                {
                    "field": "catalyst",
                    "value": "B",
                    "direction": "prefer",
                    "intent": "exploit",
                    "weight": 0.075,
                },
                {
                    "field": "catalyst",
                    "value": "A",
                    "direction": "penalize",
                    "intent": "avoid",
                    "weight": 0.065,
                },
            ],
        }
        adjustments, specs, confidence = replay.normalize_llm_adjustments(
            self.adapter,
            self.adapter.candidates,
            self.observed_ids,
            self.observed,
            parsed,
            True,
        )
        self.assertEqual({spec["intent"] for spec in specs}, {"explore", "exploit", "avoid"})
        self.assertGreater(adjustments["c1"], 0.0)
        self.assertGreater(adjustments["b3"], 0.0)
        self.assertLess(adjustments["a3"], 0.0)
        self.assertNotEqual(confidence["raw"], confidence["calibrated"])

    def test_normalizer_rejects_exploration_of_covered_value(self) -> None:
        parsed = {
            "confidence": 0.9,
            "adjustments": [
                {
                    "field": "catalyst",
                    "value": "B",
                    "direction": "prefer",
                    "intent": "explore",
                    "weight": 0.04,
                }
            ],
        }
        _, specs, _ = replay.normalize_llm_adjustments(
            self.adapter,
            self.adapter.candidates,
            self.observed_ids,
            self.observed,
            parsed,
            True,
        )
        self.assertEqual(specs, [])

    def test_exploration_gate_is_deterministic_and_bounded(self) -> None:
        kwargs = {
            "gate_version": "llm_explore_gate_v1",
            "base_scores": {"inc": 1.0, "new": 0.97},
            "adjusted_scores": {"inc": 1.0, "new": 1.01},
            "adjustments": {"inc": 0.0, "new": 0.04},
            "novelty_scores": {"new": 1.0},
            "row_order_stable": True,
            "active_skill_ids": ("llm_exploration_policy",),
            "seed": 3,
            "round_index": 0,
        }
        first = replay.exploration_gate_decision(**kwargs)
        second = replay.exploration_gate_decision(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first.challenger_candidate, "new")
        self.assertLessEqual(first.acquisition_loss, 0.10)

    def test_exploration_gate_grants_one_early_bounded_quota(self) -> None:
        gate = replay.exploration_gate_decision(
            gate_version="llm_explore_gate_v1",
            base_scores={"inc": 1.0, "new": 0.98},
            adjusted_scores={"inc": 1.0, "new": 1.01},
            adjustments={"inc": 0.0, "new": 0.03},
            novelty_scores={"new": 0.9},
            row_order_stable=True,
            active_skill_ids=("llm_exploration_policy",),
            seed=11,
            round_index=1,
            proposal_confidence=0.65,
            prior_exploration_interventions=0,
            public_best_value=75.0,
        )
        self.assertTrue(gate.authorized)
        self.assertEqual(gate.reason, "authorized_exploration_quota")

    def test_beta_schedule_decays_from_exploration_to_exploitation(self) -> None:
        values = [weighted.scheduled_gp_beta(2.8, 0.9, index, 5) for index in range(5)]
        self.assertEqual(values[0], 2.8)
        self.assertAlmostEqual(values[-1], 0.9)
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))

    def test_relative_loo_gain_rewards_lower_revealed_target_error(self) -> None:
        self.assertAlmostEqual(router.relative_loo_gain(0.2, 0.1), 0.5)
        self.assertLess(router.relative_loo_gain(0.1, 0.2), 0.0)

    def test_router_gate_requires_target_warmup_and_online_quality(self) -> None:
        common = {
            "anchor_candidate": "anchor",
            "router_candidate": "transfer",
            "anchor_loss": 0.02,
            "risk_budget": 0.05,
            "transfer_mass": 0.2,
        }
        warmup = router.router_gate_decision(
            observed_count=9,
            max_quality=0.4,
            **common,
        )
        weak = router.router_gate_decision(
            observed_count=10,
            max_quality=0.14,
            **common,
        )
        accepted = router.router_gate_decision(
            observed_count=10,
            max_quality=0.2,
            **common,
        )
        self.assertEqual(warmup, (False, "target_warmup_incomplete"))
        self.assertEqual(weak, (False, "online_quality_below_threshold"))
        self.assertEqual(accepted, (True, "online_evidence_authorized_transfer"))

    def test_nested_adjustments_are_repaired_and_audited(self) -> None:
        parsed, repair = replay.repair_llm_response_shape(
            {
                "decision_summary": {
                    "hypothesis": "test C",
                    "adjustments": [{"field": "catalyst", "value": "C"}],
                    "confidence": 0.61,
                }
            }
        )
        self.assertEqual(repair, "hoisted_adjustments_from_decision_summary")
        self.assertEqual(parsed["adjustments"][0]["value"], "C")
        self.assertEqual(parsed["confidence"], 0.61)
        self.assertEqual(parsed["decision_summary"], {"hypothesis": "test C"})

    def test_robust_patch_selector_prefers_confident_anchor_within_equivalence_set(self) -> None:
        patches = (
            evolution.KernelSkillPatch(
                "noisy_best", (1.0,), {}, 1.5, 1.0, 0.0, 0.5, 8, "off", 0.05, 0.35, ""
            ),
            evolution.KernelSkillPatch(
                "safe_anchor", (0.0, 1.0), {}, 1.5, 1.0, 0.0, 0.5, 8, "off", 0.05, 0.82, ""
            ),
            evolution.KernelSkillPatch(
                "clearly_worse", (0.0, 2.0), {}, 1.5, 1.0, 0.0, 0.5, 8, "off", 0.05, 0.95, ""
            ),
        )
        rows = []
        for seed, best_score, safe_score, worse_score in (
            (0, 180.0, 179.9, 170.0),
            (1, 170.0, 170.1, 160.0),
            (2, 176.0, 175.8, 165.0),
        ):
            for patch, score in zip(patches, (best_score, safe_score, worse_score)):
                rows.append(
                    {
                        "mode": evolution.patch_mode(patch),
                        "seed": seed,
                        "final_best": score / 2.0,
                        "best_so_far_auc": score / 2.0,
                    }
                )
        selected, diagnostics = evolution.select_calibration_robust_patch(
            rows,
            {0, 1, 2},
            patches,
        )
        self.assertEqual(selected, "llm_evolved_kernel_safe_anchor")
        self.assertFalse(
            diagnostics["modes"]["llm_evolved_kernel_clearly_worse"]["eligible"]
        )


if __name__ == "__main__":
    unittest.main()
