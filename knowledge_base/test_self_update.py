from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import retrieval
import self_update as updater


class SelfUpdateTest(unittest.TestCase):
    def make_result(self, root: Path, promotable: bool = False) -> Path:
        result = root / "results" / "2026-08-13-example"
        result.mkdir(parents=True)
        manifest = {
            "study": "self-update-example",
            "date": "2026-08-13",
            "knowledge_update": {
                "allow_skill_promotion": promotable,
                "task_disjoint_confirmation": promotable,
                "protocol_frozen_before_evaluation": promotable,
                "external_outcomes_not_used_during_selection": promotable,
            },
        }
        (result / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (result / "headline_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source",
                    "target",
                    "evidence_mode",
                    "selected_skill",
                    "rule_prior",
                    "baseline",
                    "seeds",
                    "delta_final_best",
                    "final_ci_low",
                    "final_ci_high",
                    "delta_auc",
                    "auc_ci_low",
                    "auc_ci_high",
                    "rounds_saved_top10",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source": "source_a",
                    "target": "target_b",
                    "evidence_mode": "full_source_outcome",
                    "selected_skill": "quality_bounded_warmstart",
                    "rule_prior": "source_outcome",
                    "baseline": "target_gp_ucb",
                    "seeds": "100",
                    "delta_final_best": "2.0",
                    "final_ci_low": "1.0",
                    "final_ci_high": "3.0",
                    "delta_auc": "1.5",
                    "auc_ci_low": "0.5",
                    "auc_ci_high": "2.5",
                    "rounds_saved_top10": "1.0",
                }
            )
        return result

    def paths(self, root: Path) -> dict[str, Path]:
        kb = root / "kb"
        return {
            "seed": kb / "seed.json",
            "generated_dir": kb / "generated",
            "state_path": kb / "state.json",
            "audit_path": kb / "audit.jsonl",
            "db": kb / "care.sqlite",
            "export": kb / "index.md",
            "embeddings": kb / "embeddings.jsonl",
        }

    def make_llm_suite(self, root: Path) -> Path:
        result = root / "results" / "2026-08-13-llm-suite"
        aggregate = result / "aggregate"
        aggregate.mkdir(parents=True)
        (aggregate / "suite_summary.json").write_text(
            json.dumps({
                "schema_version": "care.llm_initial_design_suite/v1",
                "evidence_class": "retrospective_multi_target_component_ablation",
                "task_count": 1,
                "route": {
                    "name": "task_structure_route/v1",
                    "uses_target_outcomes": False,
                    "status": "exploratory_candidate_not_externally_confirmed",
                },
                "auc_delta_vs_fixed_v2": {
                    "task_mean": 2.0,
                    "task_95ci_low": -1.0,
                    "task_95ci_high": 5.0,
                    "task_win_rate": 1.0,
                    "task_nonloss_rate": 1.0,
                },
                "final_best_delta_vs_fixed_v2": {"task_mean": 0.0},
                "per_task": [{
                    "target_task": "target_a",
                    "route_mode": "llm_hypothesis_initial_design",
                    "route_reason": "multiple completed sources",
                    "routed_auc_delta_vs_fixed": 2.0,
                    "routed_final_delta_vs_fixed": 0.0,
                    "llm_confidence": 0.7,
                    "hypothesis": "Prefer the source-supported high-temperature region.",
                    "hypothesis_record": "results/target_a/llm_hypothesis_record.json",
                }],
            }),
            encoding="utf-8",
        )
        return result

    def make_reflective_suite(self, root: Path) -> Path:
        result = root / "results" / "2026-08-14-reflective-suite"
        aggregate = result / "aggregate"
        aggregate.mkdir(parents=True)
        (aggregate / "suite_summary.json").write_text(
            json.dumps({
                "schema_version": "care.llm_reflective_scientist_suite/v1",
                "task_count": 1,
                "reflection_hypothesis_status_counts": {"falsified": 1},
                "reflection_stop_rate": 1.0,
                "gate": {"proposal_count": 0, "acceptance_count": 0},
                "auc_delta_vs_fixed_v2": {
                    "task_mean": 1.0,
                    "task_95ci_low": -1.0,
                    "task_95ci_high": 3.0,
                },
                "per_task": [{
                    "target_task": "target_a",
                    "hypothesis_status": "falsified",
                    "stop_transfer": True,
                    "auc_delta_vs_fixed": 1.0,
                    "gate_proposal_count": 0,
                    "gate_acceptance_count": 0,
                    "revised_hypothesis": "The source mechanism does not transfer.",
                    "evidence_interpretation": "Initial evidence contradicted the claim.",
                    "reflection_record": "results/target_a/reflection.json",
                }],
            }),
            encoding="utf-8",
        )
        return result

    def test_update_stages_candidate_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_result(root)
            paths = self.paths(root)
            paths["seed"].parent.mkdir(parents=True)
            paths["seed"].write_text("[]\n", encoding="utf-8")

            report = updater.self_update(
                [result],
                **paths,
                timestamp="2026-08-13T12:00:00+00:00",
            )
            self.assertEqual(report["status"], "updated")
            self.assertEqual(report["promoted_skill_ids"], [])
            generated = next(paths["generated_dir"].glob("auto-*.json"))
            cards = json.loads(generated.read_text(encoding="utf-8"))
            skill = next(card for card in cards if card["type"] == "skill")
            run_log = next(card for card in cards if card["type"] == "run_log")
            self.assertEqual(skill["status"], "candidate")
            self.assertIn("self-update", skill["tags"])
            self.assertNotIn(str(Path.home()), run_log["content"])

            found = retrieval.retrieve_runtime_cards(
                paths["db"], "quality bounded warmstart", 5
            )
            self.assertFalse(any(card["type"] == "skill" for card in found))
            embedding_lines = paths["embeddings"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(embedding_lines), report["total_card_count"])

            second = updater.self_update([result], **paths)
            self.assertEqual(second["status"], "noop")

    def test_promotion_requires_manifest_and_explicit_enable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_result(root, promotable=True)
            paths = self.paths(root)
            paths["seed"].parent.mkdir(parents=True)
            paths["seed"].write_text("[]\n", encoding="utf-8")

            report = updater.self_update(
                [result],
                **paths,
                allow_promotion=True,
                timestamp="2026-08-13T12:00:00+00:00",
            )
            self.assertEqual(len(report["promoted_skill_ids"]), 1)
            con = sqlite3.connect(paths["db"])
            status = con.execute(
                "select status from cards where type = 'skill'"
            ).fetchone()[0]
            con.close()
            self.assertEqual(status, "active")
            found = retrieval.retrieve_runtime_cards(
                paths["db"], "quality bounded warmstart", 5
            )
            self.assertTrue(any(card["type"] == "skill" for card in found))

    def test_discovers_and_ingests_llm_hypothesis_suite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_llm_suite(root)
            paths = self.paths(root)
            paths["seed"].parent.mkdir(parents=True)
            paths["seed"].write_text("[]\n", encoding="utf-8")

            self.assertEqual(updater.discover_result_dirs(root / "results"), [result])
            report = updater.self_update([result], **paths)
            self.assertEqual(report["new_card_count"], 3)
            generated = next(paths["generated_dir"].glob("auto-*.json"))
            cards = json.loads(generated.read_text(encoding="utf-8"))
            hypothesis = next(card for card in cards if card["type"] == "hypothesis")
            result_card = next(card for card in cards if card["type"] == "experiment_result")
            self.assertEqual(hypothesis["status"], "candidate")
            self.assertIn("candidate claim", hypothesis["content"])
            self.assertIn("CI crosses zero", result_card["summary"])

    def test_discovers_and_ingests_reflective_scientist_suite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            result = self.make_reflective_suite(root)
            paths = self.paths(root)
            paths["seed"].parent.mkdir(parents=True)
            paths["seed"].write_text("[]\n", encoding="utf-8")

            self.assertEqual(updater.discover_result_dirs(root / "results"), [result])
            report = updater.self_update([result], **paths)
            self.assertEqual(report["new_card_count"], 3)
            generated = next(paths["generated_dir"].glob("auto-*.json"))
            cards = json.loads(generated.read_text(encoding="utf-8"))
            hypothesis = next(card for card in cards if card["type"] == "hypothesis")
            self.assertEqual(hypothesis["status"], "candidate")
            self.assertIn("hypothesis-falsified", hypothesis["tags"])
            self.assertIn("stop_transfer=True", hypothesis["content"])


if __name__ == "__main__":
    unittest.main()
