from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_source_outcome_goal as summary  # noqa: E402
import run_source_outcome_suite as suite  # noqa: E402
import summarize_source_outcome_round_efficiency as efficiency  # noqa: E402


class SourceOutcomeGoalSummaryTests(unittest.TestCase):
    def test_paired_summary_uses_pairwise_uncertainty(self) -> None:
        result = summary.paired_summary([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["normal_95ci_low"], 1.0)
        self.assertEqual(result["non_loss_rate"], 1.0)

    def test_markdown_report_names_goal_status(self) -> None:
        report = {
            "pairs": [{
                "source_dataset": "source",
                "target_dataset": "target",
                "selected_source_outcome_transfer": True,
                "statistically_positive": True,
                "non_negative": True,
                "final_best": {"mean": 1.0},
                "best_so_far_auc": {"mean": 2.0},
                "composite": {
                    "normal_95ci_low": 1.0,
                    "normal_95ci_high": 5.0,
                },
                "raw_source_route": {
                    "composite": {"mean": 3.0},
                },
            }],
            "goal": {
                "all_pairs_non_negative": True,
                "statistically_positive_pair_count": 1,
                "pair_count": 1,
                "majority_statistically_positive": True,
                "achieved": True,
            },
        }
        rendered = summary.markdown_report(report)
        self.assertIn("Goal achieved: **True**", rendered)
        self.assertIn("source -> target", rendered)

    def test_suite_command_uses_frozen_strategy_and_seed_ranges(self) -> None:
        pair = {
            "pair_id": "source_to_target",
            "source_dataset": "source",
            "target_dataset": "target",
            "source_observations": 32,
            "llm_record": "configs/source.json",
            "target_llm_record": "configs/target.json",
            "target_llm_mode": "gp_ucb",
            "frozen_policy": {
                "initial_observations": 2,
                "reveal_rounds": 13,
                "source_initial_strategy": "source_negative",
                "router_min_observations": 5,
            },
        }
        protocol = {
            "source_seed": 0,
            "initial_observations": 5,
            "reveal_rounds": 10,
            "calibration_seed_count": 50,
            "heldout_seed_count": 100,
            "source_initial_strategy": "source_extremes",
            "router_min_quality": 0.25,
            "router_max_transfer_mass": 0.15,
        }
        command = suite.build_command(
            pair,
            protocol,
            {"router_min_observations": 8},
            35000,
            36000,
            20,
            "frozen",
        )
        self.assertIn("--source-initial-strategy", command)
        self.assertIn("source_negative", command)
        self.assertIn("13", command)
        self.assertIn("35000", command)
        self.assertIn("36000", command)
        self.assertIn("frozen_source_to_target", command)

    def test_round_efficiency_reads_source_initial_trace(self) -> None:
        events = [{
            "round_index": 0,
            "hypothesis_snapshot": {
                "source_initial_design": {
                    "revealed_initial_observations": [
                        {
                            "candidate_id": "a",
                            "revealed_value": 5.0,
                        },
                        {
                            "candidate_id": "b",
                            "revealed_value": 8.0,
                        },
                    ]
                }
            },
        }]
        initial = efficiency.initial_from_source_trace(events)
        self.assertEqual(initial, (8.0, {"a", "b"}))

    def test_round_efficiency_censors_missed_threshold(self) -> None:
        events = [
            {"round_index": 0, "best_so_far": 2.0},
            {"round_index": 1, "best_so_far": 3.0},
        ]
        self.assertEqual(
            efficiency.first_round_to_threshold(
                1.0,
                events,
                threshold=4.0,
                censor_round=3,
            ),
            (3, False),
        )

    def test_round_efficiency_resolves_copied_record_by_basename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "copied" / "record.json"
            record.parent.mkdir()
            record.write_bytes(b'{"frozen": true}\n')
            with patch.object(efficiency, "ROOT", root):
                resolved = efficiency.resolve_record_path(str(root / "missing" / record.name))
            self.assertEqual(resolved, record)

    def test_round_efficiency_missing_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(efficiency, "ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    efficiency.resolve_record_path(str(root / "missing" / "record.json"))

    def test_round_efficiency_identical_restored_record_uses_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archived = root / "results" / "record.json"
            frozen = root / "skill_banks" / "frozen_records" / "record.json"
            for record in (archived, frozen):
                record.parent.mkdir(parents=True)
                record.write_bytes(b'{"frozen": true}\n')
            with patch.object(efficiency, "ROOT", root):
                resolved = efficiency.resolve_record_path(str(root / "missing" / frozen.name))
            self.assertEqual(resolved, frozen)

    def test_round_efficiency_identical_copies_use_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("z_last", "a_first"):
                record = root / name / "record.json"
                record.parent.mkdir()
                record.write_bytes(b'{"frozen": true}\n')
            with patch.object(efficiency, "ROOT", root):
                resolved = efficiency.resolve_record_path(str(root / "missing" / "record.json"))
            self.assertEqual(resolved, root / "a_first" / "record.json")

    def test_round_efficiency_conflicting_frozen_and_restored_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archived = root / "results" / "record.json"
            frozen = root / "skill_banks" / "frozen_records" / "record.json"
            for record, contents in ((archived, b'{"value": 1}'), (frozen, b'{"value": 2}')):
                record.parent.mkdir(parents=True)
                record.write_bytes(contents)
            with patch.object(efficiency, "ROOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "different contents"):
                    efficiency.resolve_record_path(str(root / "missing" / frozen.name))

    def test_round_efficiency_existing_explicit_path_has_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "results" / "record.json"
            frozen = root / "skill_banks" / "frozen_records" / "record.json"
            for record, contents in ((explicit, b'{"value": 1}'), (frozen, b'{"value": 2}')):
                record.parent.mkdir(parents=True)
                record.write_bytes(contents)
            with patch.object(efficiency, "ROOT", root):
                resolved = efficiency.resolve_record_path(str(explicit))
            self.assertEqual(resolved, explicit)


if __name__ == "__main__":
    unittest.main()
