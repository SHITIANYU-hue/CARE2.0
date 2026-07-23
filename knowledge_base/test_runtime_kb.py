from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import build_kb
import ingest_experiment_results as ingest
import retrieval


class RuntimeKnowledgeBaseTest(unittest.TestCase):
    def test_ingest_and_retrieve_public_skill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result_dir = root / "2026-07-21-example"
            result_dir.mkdir()
            (result_dir / "run_manifest.json").write_text(json.dumps({
                "study": "example",
                "date": "2026-07-21",
                "negative_results": [
                    {
                        "source": "source_a",
                        "target": "target_b",
                        "case": "gate-reject",
                        "result": "Calibration rejected transfer.",
                    },
                    {
                        "source": "source_c",
                        "target": "target_d",
                        "case": "negative-heldout",
                        "result": "Held-out replay rejected transfer.",
                    },
                ],
            }), encoding="utf-8")
            (result_dir / "headline_results.csv").write_text(
                "target,evidence_mode,selected_skill,rule_prior,baseline,seeds,delta_final_best,final_ci_low,final_ci_high,delta_auc,auc_ci_low,auc_ci_high,rounds_saved_top10\n"
                "target_a,target_only,ring_rule,disabled,gp_ucb,100,2.0,1.0,3.0,1.5,0.5,2.5,0.4\n",
                encoding="utf-8",
            )
            cards = ingest.cards_from_result_dir(result_dir)
            db = root / "kb.sqlite"
            build_kb.build_sqlite(cards, db)
            found = retrieval.retrieve_runtime_cards(db, "ring rule target calibration", 5)
            self.assertTrue(any(card["type"] == "skill" for card in found))
            con = sqlite3.connect(db)
            self.assertEqual(con.execute("select count(*) from cards").fetchone()[0], 5)
            con.close()

    def test_source_outcome_skill_explains_outcome_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            result_dir = Path(raw) / "2026-07-24-source-outcome"
            result_dir.mkdir()
            (result_dir / "run_manifest.json").write_text(
                json.dumps({
                    "study": "source_outcome",
                    "date": "2026-07-24",
                }),
                encoding="utf-8",
            )
            (result_dir / "headline_results.csv").write_text(
                "source,target,evidence_mode,selected_skill,rule_prior,baseline,seeds,"
                "delta_final_best,final_ci_low,final_ci_high,delta_auc,auc_ci_low,"
                "auc_ci_high,rounds_saved_top10\n"
                "source_a,target_b,full_source_outcome,source_outcome_router,"
                "source_outcome,matched_target_only_llm,100,2.0,1.0,3.0,1.5,"
                "0.5,2.5,1.0\n",
                encoding="utf-8",
            )
            cards = ingest.cards_from_result_dir(result_dir)
            skill = next(card for card in cards if card["type"] == "skill")
            self.assertIn("measured source outcomes", skill["summary"])
            self.assertIn("exact target-only fallback", skill["content"])


if __name__ == "__main__":
    unittest.main()
