from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_repeated_online_llm_confirmation as repeated  # noqa: E402
import run_repeated_online_llm_confirmation_v2 as repeated_v2  # noqa: E402


class RepeatedOnlineConfirmationTests(unittest.TestCase):
    def frozen_suite(self) -> dict:
        return {
            "suite": "test",
            "protocol": {
                "status": "frozen_before_execution",
                "primary_estimand": "mean route delta",
                "primary_success_rule": "95% CI lower > 0",
                "replicate_start": 10,
                "replicates_per_route": 3,
                "bootstrap_samples": 200,
                "bootstrap_seed": 7,
            },
            "runner": {
                "model": "test-model",
                "base_url": "https://example.test/v1",
                "api_key_env": "TEST_KEY",
                "api_mode": "chat",
                "temperature": 0.2,
                "rounds": 2,
                "decision_policy": "high_authority",
                "deliberation_mode": "proposal_critic",
                "initial_design_mode": "auto",
                "max_tokens": 100,
                "repair_attempts": 0,
                "fail_on_llm_error": True,
            },
            "menu": {
                "gp_count": 5,
                "source_count": 3,
                "consensus_count": 3,
                "diversity_count": 1,
                "max_transfer_gp_rank": 5,
                "safety_fallback_gp_count": 5,
                "force_first_consensus": False,
            },
            "cases": [
                {
                    "case_id": "route_a",
                    "domain": "a",
                    "evidence_tier": "development",
                    "config": "missing.json",
                    "initial_record": "missing_record.json",
                },
                {
                    "case_id": "route_b",
                    "domain": "b",
                    "evidence_tier": "holdout",
                    "config": "missing.json",
                    "initial_record": "missing_record.json",
                },
            ],
        }

    def test_protocol_must_be_frozen(self) -> None:
        suite = self.frozen_suite()
        suite["protocol"]["status"] = "development"
        with self.assertRaisesRegex(ValueError, "frozen"):
            repeated.validate_suite(suite, check_paths=False)

    def test_replicate_ids_are_declared_by_protocol(self) -> None:
        self.assertEqual(repeated.replicate_ids(self.frozen_suite()), [10, 11, 12])

    def test_bh_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = repeated.benjamini_hochberg([0.01, 0.04, 0.03, None])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertIsNone(adjusted[3])

    def test_hierarchical_bootstrap_is_reproducible(self) -> None:
        values = {"a": [1.0, 2.0, 3.0], "b": [0.5, 1.0, 1.5]}
        first = repeated.hierarchical_bootstrap(values, samples=500, seed=9)
        second = repeated.hierarchical_bootstrap(values, samples=500, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_of_route_means"], 1.5)
        self.assertIsNotNone(first["bootstrap_95ci_low"])

    def test_partial_execution_disables_confirmatory_decision(self) -> None:
        suite = self.frozen_suite()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = repeated.trajectory_dir(root, "route_a", 10)
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "online_llm_scientist": {
                                "llm_participation_rate": 1.0,
                                "llm_decision_authority_rate": 1.0,
                                "llm_gp_override_rate": 0.5,
                                "llm_critic_revision_rate": 0.1,
                                "source_transfer_active_rate": 0.8,
                            }
                        },
                        "deltas": {
                            "online_llm_increment_over_same_initial_gp": {
                                "best_so_far_auc": 2.0,
                                "final_best": 1.0,
                            }
                        },
                        "usage": {"total_tokens": 100},
                    }
                ),
                encoding="utf-8",
            )
            (output / "llm_trace.jsonl").write_text("{}\n", encoding="utf-8")
            report = repeated.aggregate(root, suite)
            self.assertFalse(report["protocol_complete"])
            self.assertEqual(
                report["claim_decision"],
                "not_evaluated_incomplete_protocol",
            )


class RetryResilientConfirmationTests(unittest.TestCase):
    def suite(self) -> dict:
        suite = RepeatedOnlineConfirmationTests().frozen_suite()
        suite["suite"] = "retry-test"
        suite["retry_policy"] = {
            "max_infrastructure_attempts": 3,
            "model_validation_failure": "terminal",
            "scientific_outcome_failure": "not_retryable",
        }
        return suite

    def test_classifies_cost_cap_as_retryable_infrastructure(self) -> None:
        error = RuntimeError("HTTP 429: Access key max cost limit exceeded")
        self.assertTrue(repeated_v2.retryable_infrastructure_error(error))

    def test_does_not_retry_schema_validation_failure(self) -> None:
        error = ValueError("LLM response omitted selected_candidate_id")
        self.assertFalse(repeated_v2.retryable_infrastructure_error(error))

    def test_infrastructure_failure_is_preserved_before_success(self) -> None:
        suite = self.suite()
        case = suite["cases"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repeated.write_json(root / "protocol_lock.json", {"test": True})
            trajectory = repeated.trajectory_dir(root, "route_a", 10)
            calls = 0

            def fake_online(args: argparse.Namespace) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("HTTP 429 rate limit")
                repeated.write_json(
                    args.output_dir / "summary.json",
                    {"metrics": {}, "deltas": {}, "usage": {}},
                )
                (args.output_dir / "llm_trace.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )

            with mock.patch.object(repeated_v2.online, "run_online", fake_online):
                repeated_v2.run_trajectory(
                    case,
                    suite,
                    trajectory,
                    10,
                    {"git_commit": "test"},
                )

            self.assertEqual(calls, 2)
            self.assertTrue(
                (trajectory / "attempts" / "attempt_01" / "error.json").is_file()
            )
            self.assertTrue(
                (trajectory / "attempts" / "attempt_02" / "summary.json").is_file()
            )
            self.assertTrue((trajectory / "summary.json").is_file())
            metadata = repeated.load_json(
                trajectory / "trajectory_metadata.json"
            )
            self.assertEqual(metadata["successful_attempt"], 2)

    def test_model_validation_failure_is_terminal(self) -> None:
        suite = self.suite()
        case = suite["cases"][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repeated.write_json(root / "protocol_lock.json", {"test": True})
            trajectory = repeated.trajectory_dir(root, "route_a", 10)
            with mock.patch.object(
                repeated_v2.online,
                "run_online",
                side_effect=ValueError("invalid LLM response schema"),
            ) as patched:
                with self.assertRaisesRegex(ValueError, "invalid LLM"):
                    repeated_v2.run_trajectory(
                        case,
                        suite,
                        trajectory,
                        10,
                        {"git_commit": "test"},
                    )
            self.assertEqual(patched.call_count, 1)
            self.assertTrue((trajectory / "error.json").is_file())


if __name__ == "__main__":
    unittest.main()
