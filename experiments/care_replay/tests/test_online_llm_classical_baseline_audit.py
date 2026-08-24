from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_online_llm_classical_baseline_audit as audit  # noqa: E402


class OnlineLLMClassicalBaselineAuditTests(unittest.TestCase):
    def test_protocol_is_explicitly_posthoc_and_same_start(self) -> None:
        config = audit.load_json(
            ROOT / "configs" / "online_llm_matched_classical_audit_v1.json"
        )
        protocol = config["protocol"]
        self.assertEqual(protocol["status"], "posthoc_descriptive_stress_test")
        self.assertTrue(
            protocol["constraints"][
                "fixed_initial_candidate_ids_from_online_llm_summary"
            ]
        )
        self.assertTrue(
            protocol["constraints"]["target_gp_must_reproduce_online_summary"]
        )
        self.assertEqual(len(config["cases"]), 3)

    def test_aggregate_retains_suzuki_failure_and_source_hashes(self) -> None:
        output = (
            ROOT
            / "results"
            / "2026-08-24-online-llm-matched-classical-audit-v1"
        )
        aggregate = audit.load_json(output / "aggregate.json")
        self.assertEqual(aggregate["completed_routes"], 3)
        self.assertEqual(
            aggregate["routes_where_online_llm_beats_every_realized_baseline_auc"],
            2,
        )
        suzuki = next(
            report
            for report in aggregate["route_reports"]
            if report["case_id"] == "reizman_cases_123_to_case4"
        )
        self.assertFalse(suzuki["online_llm_beats_every_realized_baseline_auc"])
        self.assertAlmostEqual(
            suzuki["strongest_realized_baseline_posthoc"][
                "online_llm_minus_baseline_auc"
            ],
            -1.96,
        )
        for figure in aggregate["figure_files"]:
            path = Path(figure["path"])
            if not path.is_absolute():
                path = ROOT / path.relative_to("experiments/care_replay")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                figure["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
