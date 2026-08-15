from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_llm_initial_design_hypothesis as initial_design  # noqa: E402
import run_multisource_warmstart as warmstart  # noqa: E402
import run_online_llm_scientist as online  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "baumgartner_multisource_warmstart_v1.json"
SUZUKI_CONFIG = ROOT / "configs" / "baumgartner_multisource_warmstart_v2.json"
TARGET_ID = "real_baumgartner_cn_morpholine_alphos"
SOURCE_ID = "real_baumgartner_cn_aniline_alphos"


class OnlineLlmScientistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.target = replay.DATASET_BUILDERS[TARGET_ID]()
        self.shortlist, self.views, _ = initial_design.candidate_shortlist(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
            8,
        )

    def test_initial_prompt_is_outcome_blind(self) -> None:
        prompt = online.build_initial_prompt(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
            self.shortlist,
            self.views,
        )
        online.assert_no_unrevealed_outcomes(prompt)
        encoded = json.dumps(prompt, ensure_ascii=False)
        self.assertNotIn("yield_value", encoded)
        self.assertNotIn("objective_value", encoded)

    def test_candidate_menu_contains_predictions_but_no_outcomes(self) -> None:
        observed = [0, 1, 2]
        source_prior, _ = warmstart.build_source_consensus(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
        )
        menu, diagnostics = online.build_candidate_menu(
            self.target,
            observed,
            source_prior,
            self.config["protocol"]["kernel"],
            4,
            2,
            2,
            2,
            1,
        )
        observed_ids = {self.target.candidates[index].candidate_id for index in observed}
        self.assertTrue(menu)
        self.assertFalse(observed_ids & {row["candidate_id"] for row in menu})
        self.assertIn("gp_incumbent_candidate", diagnostics)
        for row in menu:
            self.assertNotIn("objective_value", row)
            self.assertNotIn("yield_value", row.get("conditions", {}))
            self.assertIn("gp_posterior_mean", row["model_evidence"])
            self.assertIn("consensus_rank", row["model_evidence"])
        self.assertEqual(
            sum(row["model_evidence"]["decision_eligible"] for row in menu),
            2,
        )

    def test_candidate_menu_routes_to_target_gp_after_transfer_stop(self) -> None:
        observed = [0, 1, 2]
        source_prior, _ = warmstart.build_source_consensus(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
        )
        menu, diagnostics = online.build_candidate_menu(
            self.target,
            observed,
            source_prior,
            self.config["protocol"]["kernel"],
            4,
            2,
            2,
            2,
            1,
            "target_gp",
        )
        eligible = [
            row for row in menu if row["model_evidence"]["decision_eligible"]
        ]
        self.assertEqual(
            sorted(row["model_evidence"]["gp_rank"] for row in eligible),
            [1, 2],
        )
        self.assertEqual(diagnostics["eligibility_mode"], "target_gp")

    def test_transfer_rank_gate_falls_back_to_gp_rank_one(self) -> None:
        observed = [0, 1, 2]
        source_prior, _ = warmstart.build_source_consensus(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
        )
        menu, diagnostics = online.build_candidate_menu(
            self.target,
            observed,
            source_prior,
            self.config["protocol"]["kernel"],
            4,
            2,
            2,
            2,
            1,
            "transfer_consensus",
            0,
        )
        eligible = [
            row for row in menu if row["model_evidence"]["decision_eligible"]
        ]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0]["model_evidence"]["gp_rank"], 1)
        self.assertEqual(
            diagnostics["eligibility_mode"],
            "target_gp_safety_fallback",
        )

    def test_transfer_rank_gate_can_preserve_bounded_llm_choice(self) -> None:
        observed = [0, 1, 2]
        source_prior, _ = warmstart.build_source_consensus(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
        )
        menu, diagnostics = online.build_candidate_menu(
            self.target,
            observed,
            source_prior,
            self.config["protocol"]["kernel"],
            4,
            2,
            2,
            2,
            1,
            "transfer_consensus",
            0,
            3,
        )
        eligible = [
            row for row in menu if row["model_evidence"]["decision_eligible"]
        ]
        self.assertEqual(
            sorted(row["model_evidence"]["gp_rank"] for row in eligible),
            [1, 2, 3],
        )
        self.assertEqual(diagnostics["safety_fallback_gp_count"], 3)

    def test_round_response_must_pass_calibrated_decision_set(self) -> None:
        menu = [
            {
                "candidate_id": "candidate-a",
                "model_evidence": {"decision_eligible": True},
            },
            {
                "candidate_id": "candidate-b",
                "model_evidence": {"decision_eligible": False},
            },
        ]
        with self.assertRaisesRegex(ValueError, "calibrated decision set"):
            online.normalize_round_response(
                {
                    "selected_candidate_id": "candidate-b",
                    "hypothesis_status": "mixed",
                },
                menu,
            )

    def test_round_response_must_select_from_menu(self) -> None:
        menu = [{"candidate_id": "candidate-a"}]
        with self.assertRaisesRegex(ValueError, "outside the menu"):
            online.normalize_round_response(
                {
                    "selected_candidate_id": "candidate-b",
                    "hypothesis_status": "mixed",
                },
                menu,
            )

    def test_round_response_requires_json_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            online.normalize_round_response(
                {
                    "selected_candidate_id": "candidate-a",
                    "hypothesis_status": "mixed",
                    "continue_source_transfer": "false",
                },
                [{"candidate_id": "candidate-a"}],
            )

    def test_invalid_json_shape_is_repaired_once(self) -> None:
        responses = iter(
            [
                (
                    '{"selected_candidate_id":"missing"}',
                    {"model": "test-model", "usage": {"total_tokens": 4}},
                ),
                (
                    '{"selected_candidate_id":"candidate-a",'
                    '"hypothesis_status":"mixed",'
                    '"continue_source_transfer":true}',
                    {"model": "test-model", "usage": {"total_tokens": 5}},
                ),
            ]
        )
        config = replay.LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="test-model",
            temperature=0.0,
            max_tokens=100,
        )
        with mock.patch.object(
            replay, "chat_completion_text", side_effect=lambda *_: next(responses)
        ) as completion:
            normalized, _content, _metadata, _parsed, attempts = (
                online.validated_llm_json(
                    config,
                    [{"role": "user", "content": "choose"}],
                    lambda response: online.normalize_round_response(
                        response, [{"candidate_id": "candidate-a"}]
                    ),
                    {"allowed_candidate_ids": ["candidate-a"]},
                    1,
                )
            )
        self.assertEqual(normalized["selected_candidate_id"], "candidate-a")
        self.assertEqual([item["valid"] for item in attempts], [False, True])
        repair_request = completion.call_args_list[1].args[1]
        self.assertIn("validation_error", repair_request[-1]["content"])

    def test_imports_existing_outcome_blind_initial_record(self) -> None:
        policy = online.initial_policy_from_record(
            {
                "schema_version": "care.llm_initial_design_hypothesis/v1",
                "frozen_hypothesis": {
                    "hypothesis": "Transferred hypothesis",
                    "mechanism": "Transferred mechanism",
                    "selected_source_view": "all_sources",
                    "selected_candidate_ids": ["a", "b", "c"],
                    "selection_roles": [
                        {"candidate_id": "a", "role": "quality_anchor"}
                    ],
                    "failure_conditions": ["failure"],
                    "confidence": 0.7,
                },
            }
        )
        self.assertEqual(policy["selected_candidate_ids"], ["a", "b", "c"])
        self.assertEqual(policy["candidate_assessments"][0]["role"], "quality_anchor")
        self.assertIsNone(policy["candidate_assessments"][0]["expected_outcome"])

    def test_auto_initial_design_keeps_direct_for_single_group_task(self) -> None:
        selected_ids = [str(row["candidate_id"]) for row in self.shortlist[:3]]
        indices, mode, compiler = online.resolve_initial_design(
            self.target,
            [SOURCE_ID],
            self.config["protocol"],
            {
                "selected_source_view": "all_sources",
                "selected_candidate_ids": selected_ids,
                "candidate_assessments": [],
            },
            "auto",
        )
        self.assertEqual(mode, "direct")
        self.assertEqual(
            [self.target.candidates[index].candidate_id for index in indices],
            selected_ids,
        )
        self.assertEqual(compiler["compiler"], "direct_llm_selection/v1")

    def test_auto_initial_design_compiles_multigroup_task(self) -> None:
        config = json.loads(SUZUKI_CONFIG.read_text(encoding="utf-8"))
        target = replay.DATASET_BUILDERS["real_baumgartner_suzuki_minlp2"]()
        source_id = "real_baumgartner_suzuki_minlp1"
        shortlist, _views, _ = initial_design.candidate_shortlist(
            target,
            [source_id],
            config["protocol"],
            6,
        )
        selected_ids = [str(row["candidate_id"]) for row in shortlist[:3]]
        indices, mode, compiler = online.resolve_initial_design(
            target,
            [source_id],
            config["protocol"],
            {
                "hypothesis": "test",
                "selected_source_view": "all_sources",
                "selected_candidate_ids": selected_ids,
                "candidate_assessments": [
                    {
                        "candidate_id": selected_ids[0],
                        "role": "quality_anchor",
                    }
                ],
                "failure_conditions": [],
            },
            "auto",
        )
        self.assertEqual(mode, "compiled")
        self.assertEqual(len(indices), 3)
        self.assertEqual(
            compiler["compiler"],
            "llm_semantic_anchor_plus_quality_bounded_maximin/v1",
        )

    def test_online_loop_preserves_budget_and_uses_llm_each_round(self) -> None:
        selected_ids = [str(row["candidate_id"]) for row in self.shortlist[:3]]
        record = {
            "schema_version": online.INITIAL_SCHEMA_VERSION,
            "model": "test-initial-model",
            "config_fingerprint": warmstart.config_fingerprint(self.config),
            "target_task": TARGET_ID,
            "source_tasks": [SOURCE_ID],
            "frozen_initial_policy": {
                "hypothesis": "A falsifiable test hypothesis.",
                "mechanism": "Source conditions may transfer.",
                "selected_source_view": "all_sources",
                "selected_candidate_ids": selected_ids,
                "candidate_assessments": [
                    {
                        "candidate_id": candidate_id,
                        "role": "probe",
                        "expected_outcome": 50.0,
                        "confidence": 0.5,
                        "reason": "test",
                    }
                    for candidate_id in selected_ids
                ],
                "failure_conditions": ["low observed outcome"],
                "confidence": 0.5,
            },
        }

        def fake_completion(_config, messages):
            prompt = json.loads(messages[-1]["content"])
            candidate_id = prompt["candidate_menu"]["rows"][0][0]
            response = {
                "hypothesis_status": "mixed",
                "updated_hypothesis": "Updated after observed evidence.",
                "selected_candidate_id": candidate_id,
                "decision_type": "exploit",
                "expected_outcome": 50.0,
                "probability_of_improving_current_best": 0.5,
                "confidence": 0.6,
                "evidence_for": ["GP-supported candidate"],
                "evidence_against": ["Limited target evidence"],
                "reasoning_summary": "Best available tradeoff.",
                "continue_source_transfer": True,
            }
            return json.dumps(response), {
                "model": "test-online-model",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            initial_path = tmp_path / "initial.json"
            online.write_json(initial_path, record)
            online.write_fingerprint(initial_path)
            args = argparse.Namespace(
                config=CONFIG,
                initial_record=initial_path,
                output_dir=tmp_path / "output",
                rounds=2,
                initial_design_mode="direct",
                menu_gp_count=4,
                menu_source_count=2,
                menu_consensus_count=2,
                force_first_consensus=True,
                menu_diversity_count=1,
                fail_on_llm_error=True,
                llm_base_url="https://example.invalid/v1",
                llm_model="test-online-model",
                llm_api_key_env="TEST_LLM_API_KEY",
                llm_api_mode="chat",
                llm_temperature=0.0,
                llm_max_tokens=200,
                llm_repair_attempts=1,
            )
            with mock.patch.dict(os.environ, {"TEST_LLM_API_KEY": "test-key"}), mock.patch.object(
                replay, "chat_completion_text", side_effect=fake_completion
            ):
                with mock.patch("builtins.print"):
                    online.run_online(args)
            summary = json.loads(
                (args.output_dir / "summary.json").read_text(encoding="utf-8")
            )
            metrics = summary["metrics"][online.MODE]
            self.assertEqual(metrics["reveal_rounds"], 2)
            self.assertEqual(metrics["llm_selected_rounds"], 2)
            self.assertEqual(metrics["llm_participation_rate"], 1.0)
            events = [
                json.loads(line)
                for line in (args.output_dir / "llm_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                sum(event["event"] == "target_reveal" for event in events), 2
            )

    def test_online_loop_caps_rounds_to_remaining_pool(self) -> None:
        selected_ids = [
            candidate.candidate_id for candidate in self.target.candidates[:-1]
        ]
        record = {
            "schema_version": online.INITIAL_SCHEMA_VERSION,
            "model": "test-initial-model",
            "config_fingerprint": warmstart.config_fingerprint(self.config),
            "target_task": TARGET_ID,
            "source_tasks": [SOURCE_ID],
            "frozen_initial_policy": {
                "hypothesis": "test",
                "mechanism": "test",
                "selected_source_view": "all_sources",
                "selected_candidate_ids": selected_ids,
                "candidate_assessments": [],
                "failure_conditions": [],
                "confidence": 0.5,
            },
        }

        def fake_completion(_config, messages):
            prompt = json.loads(messages[-1]["content"])
            candidate_id = prompt["candidate_menu"]["rows"][0][0]
            return json.dumps(
                {
                    "hypothesis_status": "mixed",
                    "selected_candidate_id": candidate_id,
                    "continue_source_transfer": False,
                }
            ), {"model": "test-online-model", "usage": {}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            initial_path = tmp_path / "initial.json"
            online.write_json(initial_path, record)
            args = argparse.Namespace(
                config=CONFIG,
                initial_record=initial_path,
                output_dir=tmp_path / "output",
                rounds=10,
                initial_design_mode="direct",
                menu_gp_count=4,
                menu_source_count=2,
                menu_consensus_count=2,
                force_first_consensus=True,
                menu_diversity_count=1,
                fail_on_llm_error=True,
                llm_base_url="https://example.invalid/v1",
                llm_model="test-online-model",
                llm_api_key_env="TEST_LLM_API_KEY",
                llm_api_mode="chat",
                llm_temperature=0.0,
                llm_max_tokens=200,
                llm_repair_attempts=1,
            )
            with mock.patch.dict(
                os.environ, {"TEST_LLM_API_KEY": "test-key"}
            ), mock.patch.object(
                replay, "chat_completion_text", side_effect=fake_completion
            ), mock.patch("builtins.print"):
                online.run_online(args)
            summary = json.loads(
                (args.output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["requested_reveal_rounds"], 10)
            self.assertEqual(summary["actual_reveal_rounds"], 1)


if __name__ == "__main__":
    unittest.main()
