from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import audit_hidden_target_noninterference as audit  # noqa: E402
import run_llm_initial_design_hypothesis as initial_design  # noqa: E402
import run_multisource_warmstart as warmstart  # noqa: E402
import run_online_llm_scientist as online  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


class HiddenTargetNoninterferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "configs" / "online_llm_generalization_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.target = replay.DATASET_BUILDERS["real_moleculenet_freesolv"]()
        self.source_ids = ["real_moleculenet_lipophilicity"]
        self.views = initial_design.source_views(self.target, self.source_ids)
        self.selected_view = next(iter(self.views))
        self.source_prior, _ = warmstart.build_source_consensus(
            self.target,
            self.views[self.selected_view],
            self.config["protocol"],
        )
        self.observed = [0, 1, 2]
        self.menu, self.diagnostics = online.build_candidate_menu(
            self.target,
            self.observed,
            self.source_prior,
            self.config["protocol"]["kernel"],
            5,
            3,
            3,
            3,
            1,
            "full_menu",
            5,
            5,
        )
        self.initial_policy = {
            "hypothesis": "public test hypothesis",
            "mechanism": "public test mechanism",
            "selected_source_view": self.selected_view,
            "selected_candidate_ids": [
                self.target.candidates[index].candidate_id
                for index in self.observed
            ],
            "failure_conditions": [],
            "confidence": 0.5,
            "selected_source_evidence": {},
        }
        self.prompt = online.build_round_prompt(
            self.target,
            self.initial_policy,
            self.observed,
            self.menu,
            0,
            10,
            None,
            self.diagnostics,
        )

    def test_shadow_target_preserves_reveals_and_public_rows(self) -> None:
        shadow, changed = audit.shadow_target(self.target, self.observed, 17)
        self.assertGreater(changed, 0)
        self.assertEqual(
            [self.target.candidates[index].objective_value for index in self.observed],
            [shadow.candidates[index].objective_value for index in self.observed],
        )
        self.assertEqual(
            audit.public_candidate_fingerprint(self.target),
            audit.public_candidate_fingerprint(shadow),
        )

    def test_trace_path_can_be_relative_to_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary_path = Path(temporary_directory) / "summary.json"
            trace_path = summary_path.parent / "llm_trace.jsonl"
            summary_path.write_text("{}\n", encoding="utf-8")
            trace_path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                audit.resolve_trace_path(summary_path, "llm_trace.jsonl"),
                trace_path.resolve(),
            )

    def test_production_state_is_exactly_invariant_to_hidden_labels(self) -> None:
        event = {
            "prompt": self.prompt,
            "prompt_sha256": audit.payload_sha256(self.prompt),
            "menu_diagnostics": self.diagnostics,
        }
        result = audit.audit_request_state(
            case_id="test_route",
            target=self.target,
            source_ids=self.source_ids,
            protocol=self.config["protocol"],
            event=event,
        )
        self.assertTrue(result["pass"])
        self.assertTrue(result["source_prior_equal"])
        self.assertTrue(result["candidate_menu_invariant"])
        self.assertTrue(result["llm_prompt_invariant"])

    def test_archived_prompt_hash_drift_is_detected(self) -> None:
        leaky_prompt = dict(self.prompt)
        leaky_prompt["unrevealed_oracle_hint"] = 123.0
        event = {
            "prompt": leaky_prompt,
            "prompt_sha256": audit.payload_sha256(self.prompt),
            "menu_diagnostics": self.diagnostics,
        }
        result = audit.audit_request_state(
            case_id="test_route",
            target=self.target,
            source_ids=self.source_ids,
            protocol=self.config["protocol"],
            event=event,
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["saved_prompt_declared_hash_valid"])


if __name__ == "__main__":
    unittest.main()
