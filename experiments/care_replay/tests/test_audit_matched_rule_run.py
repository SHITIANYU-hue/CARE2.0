from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import random
from statistics import mean
import sys
import tempfile
from types import SimpleNamespace
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_matched_rule_run as audit  # noqa: E402
import run_matched_random_rule_controls as study  # noqa: E402


class MatchedRuleRunAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.run = Path(self.temp.name) / "run"
        self.repo.mkdir()
        self.run.mkdir()
        self.source = self.repo / "executor.py"
        self.source.write_text("# frozen test source\n", encoding="utf-8")
        (self.repo / "data.csv").write_text("frozen test data\n", encoding="utf-8")
        self.catalog = {"a": {"one": 18}}
        self.candidates = [SimpleNamespace(candidate_id=f"c{index:02d}", objective_value=index + 0.125,
                                            metadata={"a": "one"}) for index in range(18)]
        self.skill = study.semantic.SemanticSkill(
            skill_id="source", rules=(study.semantic.SemanticRule("r", (("a", "one"),), 0.8, "test"),),
            ridge=2.0, prior_scale=0.3, semantic_mass_start=0.3, semantic_mass_end=0.1,
            ucb_weight=0.5, gp_beta_start=1.5, gp_beta_end=1.0, gp_xi=0.01,
            confidence=0.7, hypothesis="test",
        )
        self.config = {
            "calibration_seed_start": 100, "calibration_seed_count": 5,
            "heldout_seed_start": 200, "heldout_seed_count": 2,
            "null_replicates": 2, "random_seed": 71, "initial_observations": 2,
            "reveal_rounds": 3, "expected_candidate_count": 18,
            "target_dataset": "real_buchwald_hartwig", "field_types": {"a": "category"},
            "min_risk_adjusted_gain": 0.25, "min_positive_fold_rate": 0.8,
        }
        self.runner = SimpleNamespace(
            BASELINES=study.BASELINES, make_libraries=study.make_libraries,
            validate_config=study.validate_config, validate_rows=study.validate_rows,
            freeze_selection=study.freeze_selection, summarize=study.summarize,
            offline_guard=study.offline_guard,
            semantic=SimpleNamespace(SemanticRule=study.semantic.SemanticRule,
                                     SemanticSkill=study.semantic.SemanticSkill,
                                     skill_mode=study.semantic.skill_mode,
                                     semantic_field_catalog=lambda adapter: self.catalog),
            replay=SimpleNamespace(DATASET_BUILDERS={
                "real_buchwald_hartwig": lambda: SimpleNamespace(candidates=self.candidates)}),
        )

    def save(self, name, value):
        study.save_json(self.run / name, value)

    def seal_integrity(self):
        self.save("integrity.json", {path.name: audit.file_digest(path)
                                     for path in self.run.iterdir()
                                     if path.is_file() and path.name != "integrity.json"})

    def fixture(self, compressed):
        self.config["require_protocol_lock"] = compressed
        self.save("config.json", self.config)
        banks = study.make_libraries((self.skill,), self.catalog, self.config)
        all_skills = tuple(skill for bank in banks.values() for skill in bank)
        _, aliases = study.execution_plan(all_skills, enabled=compressed)
        calibration, heldout = study.validate_config(self.config)
        initial = {}
        for seed in calibration | heldout:
            pool = list(self.candidates)
            random.Random(seed).shuffle(pool)
            initial[str(seed)] = [candidate.candidate_id for candidate in pool[:2]]
        manifest = {
            "config": self.config, "config_sha256": audit.file_digest(self.run / "config.json"),
            "source_sha256": {"executor.py": audit.file_digest(self.source)},
            "data_sha256": {"data.csv": audit.file_digest(self.repo / "data.csv")},
            "public_catalog": self.catalog, "libraries": {
                name: [asdict(skill) for skill in bank] for name, bank in banks.items()},
            "initial_candidates": initial, "model_api_calls": 0,
            "planned_trajectories": 7 * (3 + len(all_skills)),
        }
        if compressed:
            manifest.update(execution_aliases=aliases,
                            planned_physical_trajectories=7 * (3 + len(set(aliases.values()))))
            protocol = {key: manifest[key] for key in ("config_sha256", "source_sha256", "data_sha256")}
            protocol.update(libraries_sha256=audit.digest(manifest["libraries"]),
                            execution_aliases_sha256=audit.digest(aliases))
            self.save("frozen_protocol.json", protocol)
            manifest["frozen_protocol_sha256"] = audit.file_digest(self.run / "frozen_protocol.json")
        self.save("manifest.json", manifest)
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        top10 = {candidate.candidate_id for candidate in self.candidates[-10:]}
        all_rows = {}
        for split, seeds in (("calibration", calibration), ("heldout", heldout)):
            metrics, events = [], []
            for seed in sorted(seeds):
                physical_metrics = {}
                for index, mode in enumerate(sorted(set(study.BASELINES) | set(aliases.values()))):
                    seen = set(initial[str(seed)])
                    best = max(by_id[c].objective_value for c in seen)
                    available = [candidate for candidate in self.candidates if candidate.candidate_id not in seen]
                    reveal = available[index:index + 3]
                    best_trace = []
                    for step, candidate in enumerate(reveal):
                        before = len(seen)
                        seen.add(candidate.candidate_id)
                        best = max(best, candidate.objective_value)
                        best_trace.append(best)
                        events.append({"mode": mode, "seed": seed, "round_index": step,
                                       "public_observed_count": before,
                                       "selected_candidate": candidate.candidate_id,
                                       "revealed_value": candidate.objective_value, "best_so_far": best})
                    physical_metrics[mode] = {"mode": mode, "seed": seed, "split": split,
                                              "final_best": round(best, 4),
                                              "best_so_far_auc": round(mean(best_trace), 4),
                                              "simple_regret": round(17.125 - best, 4),
                                              "top10_hit": int(bool(seen & top10))}
                metrics.extend(physical_metrics[mode] for mode in study.BASELINES)
                metrics.extend(dict(physical_metrics[representative], mode=mode)
                               for mode, representative in aliases.items())
            study.nulls.write_csv(self.run / f"{split}_metrics.csv", metrics)
            suffix = ".gz" if compressed else ""
            opener = gzip.open if compressed else open
            with opener(self.run / f"{split}_traces.jsonl{suffix}", "wt", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            all_rows[split] = metrics
        selections = study.freeze_selection(all_rows["calibration"], banks, calibration, self.config)
        lock = {"selections": selections, "heldout_started": False,
                "manifest_sha256": audit.file_digest(self.run / "manifest.json"),
                "calibration_metrics_sha256": audit.file_digest(self.run / "calibration_metrics.csv")}
        if compressed:
            lock["frozen_protocol_sha256"] = manifest["frozen_protocol_sha256"]
        self.save("selection_lock.json", lock)
        summary = study.summarize(all_rows["heldout"], banks, selections, heldout)
        summary.update(status="complete", completed_trajectories=manifest["planned_trajectories"],
                       model_api_calls=0, selection_lock_sha256=audit.file_digest(self.run / "selection_lock.json"))
        self.save("summary.json", summary)
        self.seal_integrity()
        return manifest, banks

    def run_audit(self):
        return audit.audit_run(self.repo, self.run, runner=self.runner)

    def test_old_uncompressed_run_reconstructs_every_trajectory(self):
        manifest, _ = self.fixture(compressed=False)
        result = self.run_audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["trajectories_reconstructed"], manifest["planned_trajectories"])
        self.assertEqual(result["physical_trajectories_reconstructed"], manifest["planned_trajectories"])
        self.assertFalse(result["frozen_protocol_verified"])

    def test_compressed_alias_run_reconstructs_logical_metrics_without_expanded_traces(self):
        manifest, _ = self.fixture(compressed=True)
        result = self.run_audit()
        self.assertEqual(result["trajectories_reconstructed"], 56)
        self.assertEqual(result["physical_trajectories_reconstructed"], manifest["planned_physical_trajectories"])
        self.assertLess(result["physical_trajectories_reconstructed"], 56)
        self.assertTrue(result["frozen_protocol_verified"])
        self.assertTrue(result["source_and_data_hashes_unchanged"])

    def test_source_change_requires_matching_historical_snapshot(self):
        self.fixture(compressed=True)
        original = self.source.read_bytes()
        self.source.write_text("# changed source\n", encoding="utf-8")
        with self.assertRaisesRegex(audit.AuditError, "Source hash mismatch"):
            self.run_audit()
        snapshot = self.run / "source_snapshot"
        snapshot.mkdir()
        (snapshot / self.source.name).write_bytes(original)
        result = self.run_audit()
        self.assertFalse(result["source_and_data_hashes_unchanged"])
        self.assertEqual(result["historical_source_snapshots_verified"], ["executor.py"])

    def test_data_change_is_rejected(self):
        self.fixture(compressed=True)
        (self.repo / "data.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(audit.AuditError, "Data hash mismatch"):
            self.run_audit()

    def test_alias_cannot_hide_changed_execution_parameters(self):
        manifest, banks = self.fixture(compressed=True)
        aliases = manifest["execution_aliases"]
        alias = next(mode for mode, representative in aliases.items() if mode != representative)
        changed = {}
        for name, bank in banks.items():
            changed[name] = tuple(replace(skill, ridge=999.0) if study.semantic.skill_mode(skill) == alias else skill
                                  for skill in bank)
        with self.assertRaisesRegex(audit.AuditError, "Alias changes execution inputs"):
            audit.validate_aliases(manifest, changed, self.runner)

    def test_missing_or_chained_alias_is_rejected(self):
        manifest, banks = self.fixture(compressed=True)
        invalid = deepcopy(manifest)
        del invalid["execution_aliases"][next(iter(invalid["execution_aliases"]))]
        with self.assertRaisesRegex(audit.AuditError, "cover exactly"):
            audit.validate_aliases(invalid, banks, self.runner)
        invalid = deepcopy(manifest)
        aliases = invalid["execution_aliases"]
        alias = next(mode for mode, representative in aliases.items() if mode != representative)
        aliases[aliases[alias]] = alias
        with self.assertRaisesRegex(audit.AuditError, "Noncanonical alias chain"):
            audit.validate_aliases(invalid, banks, self.runner)

    def test_frozen_protocol_cannot_substitute_libraries(self):
        self.fixture(compressed=True)
        protocol = audit.load_json(self.run / "frozen_protocol.json")
        protocol["libraries_sha256"] = "wrong"
        self.save("frozen_protocol.json", protocol)
        self.seal_integrity()
        with self.assertRaisesRegex(audit.AuditError, "Frozen protocol differs: libraries_sha256"):
            self.run_audit()

    def test_metric_tampering_is_detected_even_if_integrity_is_resealed(self):
        self.fixture(compressed=True)
        path = self.run / "heldout_metrics.csv"
        rows = audit.read_metrics(path)
        rows[-1]["best_so_far_auc"] = float(rows[-1]["best_so_far_auc"]) + 1
        study.nulls.write_csv(path, rows)
        self.seal_integrity()
        with self.assertRaisesRegex(audit.AuditError, "Metric mismatch"):
            self.run_audit()

    def test_reveals_and_duplicate_queries_are_checked(self):
        self.fixture(compressed=True)
        path = self.run / "heldout_traces.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = [json.loads(line) for line in handle]
        events[0]["revealed_value"] += 1
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.writelines(json.dumps(event) + "\n" for event in events)
        self.seal_integrity()
        with self.assertRaisesRegex(audit.AuditError, "Reveal value mismatch"):
            self.run_audit()
        events[0]["revealed_value"] -= 1
        events[1]["selected_candidate"] = events[0]["selected_candidate"]
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.writelines(json.dumps(event) + "\n" for event in events)
        self.seal_integrity()
        with self.assertRaisesRegex(audit.AuditError, "Duplicate candidate query"):
            self.run_audit()

    def test_initial_candidates_are_recreated_from_seed_not_only_trusted(self):
        manifest, _ = self.fixture(compressed=False)
        manifest["initial_candidates"]["100"] = list(reversed(manifest["initial_candidates"]["100"]))
        self.save("manifest.json", manifest)
        lock = audit.load_json(self.run / "selection_lock.json")
        lock["manifest_sha256"] = audit.file_digest(self.run / "manifest.json")
        self.save("selection_lock.json", lock)
        summary = audit.load_json(self.run / "summary.json")
        summary["selection_lock_sha256"] = audit.file_digest(self.run / "selection_lock.json")
        self.save("summary.json", summary)
        self.seal_integrity()
        with self.assertRaisesRegex(audit.AuditError, "seeded shared start"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
