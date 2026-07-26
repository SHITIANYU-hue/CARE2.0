from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_calibrated_source_outcome_portfolio as portfolio  # noqa: E402


class SourceOutcomePortfolioTests(unittest.TestCase):
    def test_parse_portfolio_is_deterministic_and_validates_mass(self) -> None:
        candidates = portfolio.parse_portfolio(
            "standard:0.45:matched,conservative:0.25:source_extremes"
        )
        self.assertEqual([candidate.candidate_id for candidate in candidates], [
            "standard",
            "conservative",
        ])
        self.assertEqual(candidates[1].mode, "source_outcome_candidate__conservative")
        with self.assertRaises(ValueError):
            portfolio.parse_portfolio("too_strong:0.5:matched")

    def test_parse_portfolio_rejects_duplicate_or_unknown_entries(self) -> None:
        with self.assertRaises(ValueError):
            portfolio.parse_portfolio("same:0.2:matched,same:0.3:matched")
        with self.assertRaises(ValueError):
            portfolio.parse_portfolio("bad:0.2:target_only")

    def test_route_metadata_distinguishes_initial_design_from_continuous_transfer(self) -> None:
        candidates = portfolio.parse_portfolio("warmstart:0.0:source_extremes")
        self.assertEqual(candidates[0].source_initial_strategy, "source_extremes")
        self.assertEqual(candidates[0].router_max_transfer_mass, 0.0)


if __name__ == "__main__":
    unittest.main()
