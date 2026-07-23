from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_round_efficiency_summary as rounds  # noqa: E402


class RoundEfficiencySummaryTest(unittest.TestCase):
    def test_first_round_to_threshold_handles_initial_hit_and_censoring(self) -> None:
        events = [
            {"round_index": 0, "best_so_far": 4.0},
            {"round_index": 1, "best_so_far": 7.0},
        ]
        self.assertEqual(rounds.first_round_to_threshold(events, 5.0, 5.0, 3), (0, True))
        self.assertEqual(rounds.first_round_to_threshold(events, 2.0, 6.0, 3), (2, True))
        self.assertEqual(rounds.first_round_to_threshold(events, 2.0, 8.0, 3), (3, False))

    def test_first_round_to_top10_is_one_based(self) -> None:
        events = [
            {"round_index": 0, "selected_candidate": "ordinary"},
            {"round_index": 1, "selected_candidate": "top"},
        ]
        self.assertEqual(rounds.first_round_to_top10(events, False, {"top"}, 3), (2, True))
        self.assertEqual(rounds.first_round_to_top10(events, True, {"top"}, 3), (0, True))
        self.assertEqual(rounds.first_round_to_top10(events, False, {"other"}, 3), (3, False))

    def test_best_at_round_uses_initial_value_for_zero_budget(self) -> None:
        events = [
            {"round_index": 0, "best_so_far": 4.0},
            {"round_index": 1, "best_so_far": 7.0},
        ]
        self.assertEqual(rounds.best_at_round(events, 3.0, 0), 3.0)
        self.assertEqual(rounds.best_at_round(events, 3.0, 1), 4.0)
        self.assertEqual(rounds.best_at_round(events, 3.0, 2), 7.0)

    def test_llm_initial_context_uses_warmstart_candidates(self) -> None:
        candidates = (
            rounds.replay.Candidate("low", "a", 0, 0, 0, 1.0, {}),
            rounds.replay.Candidate("high", "b", 0, 0, 0, 9.0, {}),
        )
        adapter = rounds.replay.DatasetAdapter(
            "test",
            "test",
            "maximize",
            (),
            "value",
            "group",
            (),
            "",
            candidates,
        )
        events = [{
            "hypothesis_snapshot": {
                "warmstart_candidates": ["high"],
            }
        }]
        self.assertEqual(
            rounds.llm_initial_context(adapter, 0, 1, {"high"}, events),
            (9.0, True),
        )


if __name__ == "__main__":
    unittest.main()
