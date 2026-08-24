from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import urllib.error


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_repeated_online_llm_confirmation as repeated  # noqa: E402
import run_repeated_online_llm_confirmation_v2 as repeated_v2  # noqa: E402
import run_repeated_online_llm_confirmation_v3 as repeated_v3  # noqa: E402
import run_repeated_online_llm_confirmation_v4 as repeated_v4  # noqa: E402
import run_repeated_online_llm_confirmation_v5 as repeated_v5  # noqa: E402
import run_repeated_online_llm_confirmation_v6 as repeated_v6  # noqa: E402


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


class ProviderAwareConfirmationTests(unittest.TestCase):
    def rate_control(self) -> dict:
        return {
            "min_request_interval_seconds": 0.0,
            "max_call_attempts": 2,
            "rate_limit_wait_seconds": 0.0,
            "transient_wait_seconds": 0.0,
            "request_timeout_seconds": 1.0,
            "reasoning_effort": "low",
        }

    def test_rate_control_is_required(self) -> None:
        suite = RetryResilientConfirmationTests().suite()
        with self.assertRaisesRegex(ValueError, "rate_control"):
            repeated_v3.validate_suite(suite, check_paths=False)

    def test_invalid_reasoning_effort_is_rejected(self) -> None:
        suite = RetryResilientConfirmationTests().suite()
        suite["rate_control"] = self.rate_control()
        suite["rate_control"]["reasoning_effort"] = "disabled"
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            repeated_v3.validate_suite(suite, check_paths=False)

    def test_http_429_retries_same_call_and_preserves_payload(self) -> None:
        client = repeated_v3.ProviderAwareChatClient(self.rate_control())
        config = SimpleNamespace(
            api_mode="chat",
            structured_mode="json",
            model="glm-5.3",
            base_url="https://example.test/v1",
            api_key="test-only",
            temperature=0.2,
            max_tokens=100,
        )
        rate_error = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "rate limited",
            {"Retry-After": "0"},
            io.BytesIO(b'{"error":{"message":"rate limited"}}'),
        )

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "model": "glm-5.3",
                        "choices": [
                            {
                                "message": {"content": '{"selected":"a"}'},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"total_tokens": 42},
                    }
                ).encode("utf-8")

        captured_payloads = []

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            if len(captured_payloads) == 1:
                raise rate_error
            return FakeResponse()

        with mock.patch.object(
            repeated_v3.urllib.request,
            "urlopen",
            side_effect=fake_urlopen,
        ), mock.patch.object(repeated_v3.online.replay, "trace_llm_event"):
            content, metadata = client(
                config,
                [{"role": "user", "content": "choose"}],
            )

        self.assertEqual(content, '{"selected":"a"}')
        self.assertEqual(metadata["usage"]["total_tokens"], 42)
        self.assertEqual(len(captured_payloads), 2)
        self.assertEqual(captured_payloads[0], captured_payloads[1])
        self.assertEqual(captured_payloads[0]["reasoning_effort"], "low")

    def test_single_answer_envelope_is_removed_without_semantic_edits(self) -> None:
        payload = {
            "answer": {
                "selected_candidate_id": "candidate_a",
                "confidence": 0.6,
            }
        }
        with mock.patch.object(
            repeated_v4,
            "_ORIGINAL_EXTRACT_JSON_OBJECT",
            return_value=payload,
        ):
            parsed = repeated_v4.extract_provider_json_object("ignored")
        self.assertEqual(parsed, payload["answer"])

    def test_non_envelope_response_is_unchanged(self) -> None:
        payload = {
            "answer": {"selected_candidate_id": "candidate_a"},
            "metadata": {"provider": "test"},
        }
        with mock.patch.object(
            repeated_v4,
            "_ORIGINAL_EXTRACT_JSON_OBJECT",
            return_value=payload,
        ):
            parsed = repeated_v4.extract_provider_json_object("ignored")
        self.assertIs(parsed, payload)

    def test_file_content_transport_envelope_is_unwrapped(self) -> None:
        payload = {
            "file_path": "ignored.json",
            "content": '{"selected_candidate_id":"candidate_a","confidence":0.6}',
        }
        nested = {
            "selected_candidate_id": "candidate_a",
            "confidence": 0.6,
        }
        with mock.patch.object(
            repeated_v5,
            "_ORIGINAL_EXTRACT_JSON_OBJECT",
            side_effect=[payload, nested],
        ):
            parsed = repeated_v5.extract_provider_transport_object("ignored")
        self.assertEqual(parsed, nested)

    def test_transport_envelope_with_extra_key_is_not_unwrapped(self) -> None:
        payload = {
            "file_path": "ignored.json",
            "content": '{"selected_candidate_id":"candidate_a"}',
            "comment": "extra",
        }
        with mock.patch.object(
            repeated_v5,
            "_ORIGINAL_EXTRACT_JSON_OBJECT",
            return_value=payload,
        ):
            parsed = repeated_v5.extract_provider_transport_object("ignored")
        self.assertIs(parsed, payload)

    def test_schema_repair_retains_original_scientific_context(self) -> None:
        responses = [
            (None, {"usage": {"total_tokens": 100}}),
            (
                '{"selected_candidate_id":"candidate_a"}',
                {"usage": {"total_tokens": 50}},
            ),
        ]
        messages = [
            {"role": "system", "content": "science system"},
            {"role": "user", "content": "ORIGINAL SCIENCE AND CANDIDATE MENU"},
        ]
        with mock.patch.object(
            repeated_v6.online.replay,
            "chat_completion_text",
            side_effect=responses,
        ) as chat:
            normalized, _content, _metadata, _parsed, attempts = (
                repeated_v6.context_preserving_validated_llm_json(
                    SimpleNamespace(),
                    messages,
                    lambda parsed: dict(parsed),
                    {"candidate_ids": ["candidate_a"]},
                    1,
                )
            )
        repaired_messages = chat.call_args_list[1].args[1]
        self.assertEqual(normalized["selected_candidate_id"], "candidate_a")
        self.assertFalse(attempts[0]["valid"])
        self.assertTrue(attempts[1]["valid"])
        self.assertNotIn(
            {"role": "assistant", "content": ""},
            repaired_messages,
        )
        self.assertIn(
            "ORIGINAL SCIENCE AND CANDIDATE MENU",
            [message["content"] for message in repaired_messages],
        )
        self.assertIn("validation_error", repaired_messages[-1]["content"])

    def test_schema_repair_keeps_nonempty_previous_response(self) -> None:
        responses = [
            ("not valid json", {"usage": {"total_tokens": 100}}),
            (
                '{"selected_candidate_id":"candidate_a"}',
                {"usage": {"total_tokens": 50}},
            ),
        ]
        messages = [
            {"role": "system", "content": "science system"},
            {"role": "user", "content": "ORIGINAL SCIENCE AND CANDIDATE MENU"},
        ]
        with mock.patch.object(
            repeated_v6.online.replay,
            "chat_completion_text",
            side_effect=responses,
        ) as chat:
            repeated_v6.context_preserving_validated_llm_json(
                SimpleNamespace(),
                messages,
                lambda parsed: dict(parsed),
                {"candidate_ids": ["candidate_a"]},
                1,
            )
        repaired_messages = chat.call_args_list[1].args[1]
        self.assertEqual(
            repaired_messages[-2],
            {"role": "assistant", "content": "not valid json"},
        )
        self.assertEqual(repaired_messages[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
