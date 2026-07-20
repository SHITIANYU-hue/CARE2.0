from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_synthetic_suzuki as replay  # noqa: E402
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

    def test_beta_schedule_decays_from_exploration_to_exploitation(self) -> None:
        values = [weighted.scheduled_gp_beta(2.8, 0.9, index, 5) for index in range(5)]
        self.assertEqual(values[0], 2.8)
        self.assertAlmostEqual(values[-1], 0.9)
        self.assertTrue(all(left > right for left, right in zip(values, values[1:])))

    def test_relative_loo_gain_rewards_lower_revealed_target_error(self) -> None:
        self.assertAlmostEqual(router.relative_loo_gain(0.2, 0.1), 0.5)
        self.assertLess(router.relative_loo_gain(0.1, 0.2), 0.0)


if __name__ == "__main__":
    unittest.main()
