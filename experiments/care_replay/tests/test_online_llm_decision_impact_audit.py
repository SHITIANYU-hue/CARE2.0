import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "decision_impact",
    SCRIPTS / "build_online_llm_decision_impact_audit.py",
)
impact = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(impact)


class OnlineLlmDecisionImpactAuditTests(unittest.TestCase):
    def test_audit_distinguishes_participation_from_override(self):
        records = [
            {
                "case_id": "route_a",
                "changed_gp_rank_one": False,
                "continue_source_transfer": True,
                "critic_revised_choice": False,
                "online_auc_delta_vs_same_initial_gp": 0.0,
                "complete_system_auc_delta_vs_fixed_v2": 1.0,
                "total_tokens": 100,
                "fallback_rounds": 9,
                "nominal_llm_calls_avoided": 18,
            },
            {
                "case_id": "route_b",
                "changed_gp_rank_one": True,
                "continue_source_transfer": False,
                "critic_revised_choice": True,
                "online_auc_delta_vs_same_initial_gp": -0.5,
                "complete_system_auc_delta_vs_fixed_v2": 0.0,
                "total_tokens": 200,
                "fallback_rounds": 9,
                "nominal_llm_calls_avoided": 18,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            audit = impact.build_audit(records, Path(directory))
        self.assertEqual(audit["decision_impact"]["gp_override_count"], 1)
        self.assertEqual(audit["decision_impact"]["gp_rank_one_choice_count"], 1)
        self.assertEqual(
            audit["performance"]["online_auc_sign_counts"],
            {"wins": 0, "ties": 1, "losses": 1},
        )
        self.assertEqual(audit["cost"]["total_tokens"], 300)

    def test_trace_extracts_executable_decision_and_gp_default(self):
        events = [
            {
                "event": "llm_round_request",
                "prompt": {
                    "decision_context": {"gp_default_candidate": "gp_1"}
                },
            },
            {
                "event": "llm_round_response",
                "round_index": 0,
                "normalized_decision": {
                    "selected_candidate_id": "source_2",
                    "continue_source_transfer": True,
                },
            },
        ]
        decision, round_index = impact.final_llm_decision(events)
        self.assertEqual(impact.gp_default_candidate(events), "gp_1")
        self.assertEqual(decision["selected_candidate_id"], "source_2")
        self.assertEqual(round_index, "0")


if __name__ == "__main__":
    unittest.main()
