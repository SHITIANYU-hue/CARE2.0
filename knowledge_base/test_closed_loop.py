from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import close_loop


class ClosedLoopTest(unittest.TestCase):
    def make_result(self, root: Path, promotable: bool) -> Path:
        result = root / "results" / "confirmed-transfer"
        result.mkdir(parents=True)
        (result / "run_manifest.json").write_text(
            json.dumps({
                "study": "closed-loop-test",
                "date": "2026-08-26",
                "knowledge_update": {
                    "allow_skill_promotion": promotable,
                    "task_disjoint_confirmation": promotable,
                    "protocol_frozen_before_evaluation": promotable,
                    "external_outcomes_not_used_during_selection": promotable,
                },
            }),
            encoding="utf-8",
        )
        with (result / "headline_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source", "target", "evidence_mode", "selected_skill",
                "rule_prior", "baseline", "seeds", "delta_final_best",
                "final_ci_low", "final_ci_high", "delta_auc", "auc_ci_low",
                "auc_ci_high", "rounds_saved_top10",
            ])
            writer.writeheader()
            writer.writerow({
                "source": "source_a",
                "target": "target_b",
                "evidence_mode": "full_source_outcome",
                "selected_skill": "source_guided_warmstart",
                "rule_prior": "source_outcome",
                "baseline": "target_gp_ucb",
                "seeds": "50",
                "delta_final_best": "2.0",
                "final_ci_low": "0.5",
                "final_ci_high": "3.5",
                "delta_auc": "1.5",
                "auc_ci_low": "0.2",
                "auc_ci_high": "2.8",
                "rounds_saved_top10": "1.0",
            })
        (result / "llm_skill_record.json").write_text(
            json.dumps({
                "normalized_skills": [{
                    "skill_id": "source_guided_warmstart",
                    "hypothesis": "A source-supported region improves initial design.",
                    "confidence": 0.8,
                    "rules": [{
                        "rule_id": "prefer_shared_region",
                        "conditions": [["region", "shared"]],
                        "weight": 0.5,
                        "rationale": "The region is supported by completed source experiments.",
                    }],
                    "semantic_mass_start": 0.2,
                    "semantic_mass_end": 0.0,
                }]
            }),
            encoding="utf-8",
        )
        return result

    def paths(self, root: Path) -> dict[str, Path]:
        kb = root / "kb"
        kb.mkdir()
        (kb / "seed.json").write_text("[]\n", encoding="utf-8")
        return {
            "seed": kb / "seed.json",
            "generated_dir": kb / "generated",
            "state_path": kb / "state.json",
            "audit_path": kb / "audit.jsonl",
            "db": kb / "care.sqlite",
            "export": kb / "index.md",
            "embeddings": kb / "embeddings.jsonl",
        }

    def test_confirmed_experience_returns_to_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_result(root, promotable=True)
            report = close_loop.close_loop([result], **self.paths(root))

            receipt = report["receipts"][0]
            self.assertTrue(receipt["closed_loop"]["qualified_skill_to_runtime"])
            self.assertEqual(len(receipt["skill_states"]["active"]), 1)
            retrieved = {
                card_id
                for item in receipt["return_channel"]["feedback"]
                for card_id in item["retrieved_card_ids"]
            }
            self.assertIn(receipt["skill_states"]["active"][0], retrieved)
            self.assertTrue((result / "knowledge_feedback.json").exists())
            cards = json.loads(
                Path(receipt["generated_card_file"]).read_text(encoding="utf-8")
            )
            skill = next(card for card in cards if card["type"] == "skill")
            self.assertIn("prefer_shared_region", skill["content"])
            self.assertIn("executable-skill", skill["tags"])

    def test_unconfirmed_experience_is_archived_but_cannot_steer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_result(root, promotable=False)
            report = close_loop.close_loop([result], **self.paths(root))

            receipt = report["receipts"][0]
            self.assertFalse(receipt["closed_loop"]["qualified_skill_to_runtime"])
            self.assertEqual(len(receipt["skill_states"]["candidate"]), 1)
            retrieved = {
                card_id
                for item in receipt["return_channel"]["feedback"]
                for card_id in item["retrieved_card_ids"]
            }
            self.assertNotIn(receipt["skill_states"]["candidate"][0], retrieved)


if __name__ == "__main__":
    unittest.main()
