from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import plot_round_efficiency as plot  # noqa: E402


class PlotRoundEfficiencyTest(unittest.TestCase):
    def test_trace_has_top10_respects_round_budget(self) -> None:
        events = [
            {"round_index": 0, "selected_candidate": "ordinary"},
            {"round_index": 1, "selected_candidate": "top"},
        ]
        self.assertFalse(plot.trace_has_top10(events, False, {"top"}, 1))
        self.assertTrue(plot.trace_has_top10(events, False, {"top"}, 2))
        self.assertTrue(plot.trace_has_top10(events, True, {"top"}, 0))

    def test_mean_ci_is_exact_for_constant_values(self) -> None:
        self.assertEqual(plot.mean_ci([2.0, 2.0, 2.0]), (2.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
