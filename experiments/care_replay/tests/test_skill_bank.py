from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_reizman_skill_bank  # noqa: E402
import skill_bank  # noqa: E402


class SkillBankTests(unittest.TestCase):
    def test_bank_is_task_disjoint_and_writes_agent_skill(self) -> None:
        bank = build_reizman_skill_bank.build_bank()
        self.assertFalse(
            set(bank.development_task_ids) & set(bank.evaluation_task_ids)
        )
        with tempfile.TemporaryDirectory() as directory:
            bank.write(Path(directory))
            skill_text = (Path(directory) / "SKILL.md").read_text(encoding="utf-8")
            self.assertIn("name: care2-wetlab-transfer", skill_text)
            self.assertIn("ask", skill_text)
            self.assertTrue((Path(directory) / "references" / "evidence.json").exists())
            self.assertTrue((Path(directory) / "manifest.json").exists())

    def test_agent_prompt_rejects_hidden_target_outcomes(self) -> None:
        bank = build_reizman_skill_bank.build_bank()
        with self.assertRaisesRegex(ValueError, "hidden target evidence"):
            bank.build_agent_prompt({"target_outcomes": [1.0, 2.0]})
        prompt = bank.build_agent_prompt(
            {
                "target_schema": {
                    "continuous": ["temperature"],
                    "categorical": ["catalyst"],
                },
                "revealed_observations": [],
            }
        )
        self.assertIn("Current target context", prompt)

    def test_development_selection_is_consolidated_as_abstention_evidence(self) -> None:
        selection = {
            "config_fingerprint": "frozen-config",
            "selection_scope": "leave_one_development_task_out_only",
            "selection_metric": "best_so_far_auc",
            "selection_required_ci_low": 0.0,
            "evaluation_task_ids_not_loaded": ["real_reizman_suzuki_case_4"],
            "selected_route": {"mode": "target_gp_ucb"},
            "candidate_comparisons": {
                "target_gp_ucb": {
                    "fold_seed_count": 90,
                    "best_so_far_auc": {"normal_95ci_low": 0.0},
                }
            },
        }
        bank = build_reizman_skill_bank.build_bank(selection)
        self.assertIn(
            "reizman_development_transfer_gate_v1",
            {item.evidence_id for item in bank.evidence},
        )
        self.assertIn("confidence gate", bank.render_skill_md())
        json.dumps(bank.as_dict())

    def test_overlap_is_rejected(self) -> None:
        evidence = skill_bank.SkillEvidence(
            evidence_id="e",
            source_tasks=("task",),
            task_family="test",
            status="candidate",
            lesson="Keep a fallback.",
            applicability=("test",),
            failure_modes=("none",),
            trace_references=("trace",),
            observation_count=1,
            provenance={},
        )
        reusable = skill_bank.ReusableSkill(
            skill_id="s",
            title="S",
            task_families=("test",),
            instructions=("Do the thing.",),
            abstain_when=("unsupported",),
            evidence_ids=("e",),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            skill_bank.compile_skill_bank(
                bank_id="bank",
                development_task_ids=("task",),
                evaluation_task_ids=("task",),
                evidence=(evidence,),
                skills=(reusable,),
                provenance={},
            )


if __name__ == "__main__":
    unittest.main()
