from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_matched_rule_stability as analysis  # noqa: E402


class MatchedRuleStabilityTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {**analysis.PROTOCOL, "banks_per_family": 2,
                         "skills_per_bank": 2, "calibration_seed_count": 5,
                         "heldout_seed_count": 10, "bootstrap_replicates": 200,
                         "calibration_seed_start": 100, "heldout_seed_start": 200}
        libraries = {key: [{"skill_id": f"{key}_a"}, {"skill_id": f"{key}_b"}]
                     for key in ("llm", "r1_001", "r1_002", "r2_001", "r2_002")}
        self.manifest = {"config": {"null_replicates": 2, "target_dataset": "real_buchwald_hartwig",
                                    "calibration_seed_start": 100, "calibration_seed_count": 5,
                                    "heldout_seed_start": 200, "heldout_seed_count": 10},
                         "libraries": libraries}
        self.manifest["config"].update(protocol_id=self.protocol["run_protocol_id"],
                                        random_seed=self.protocol["random_seed"],
                                        stability_analysis=analysis.expected_analysis_plan(self.protocol))
        self.lock = {"selections": {
            key: {"ungated_calibration_winner": analysis.skill_mode(bank[0]),
                  "selected_mode": "gp_ucb",
                  "selection": {"target_anchor_mode": "gp_ucb"}}
            for key, bank in libraries.items()}}
        self.rows = []
        for seed in range(200, 210):
            for mode in analysis.BASELINES:
                self.rows.append(self.row(mode, seed, 40))
            for key, bank in libraries.items():
                # Unselected b is intentionally better in heldout. Analysis
                # must use the frozen a choice even in the face of this fact.
                self.rows.append(self.row(analysis.skill_mode(bank[0]), seed,
                                          60 if key == "llm" else 55))
                self.rows.append(self.row(analysis.skill_mode(bank[1]), seed, 90))

    @staticmethod
    def row(mode, seed, auc):
        return {"mode": mode, "seed": seed, "split": "heldout",
                "best_so_far_auc": auc, "final_best": auc + 5,
                "top10_hit": int(auc >= 60)}

    def run_analysis(self, rows=None):
        return analysis.analyze(self.manifest, self.lock,
                                self.rows if rows is None else rows, protocol=self.protocol)

    def test_bootstrap_product_weights_match_literal_crossed_resampling(self):
        matrix = np.array([[1, 2, 9], [10, 30, 50]])
        bank_indices = [[0, 0], [1, 0], [1, 1]]
        seed_indices = [[0, 2, 2], [1, 0, 1], [2, 2, 0]]
        bank_counts = np.array([np.bincount(x, minlength=2) for x in bank_indices])
        seed_counts = np.array([np.bincount(x, minlength=3) for x in seed_indices])
        actual = analysis.crossed_bootstrap_means(matrix, bank_counts, seed_counts)
        expected = [matrix[np.ix_(banks, seeds)].mean()
                    for banks, seeds in zip(bank_indices, seed_indices)]
        np.testing.assert_allclose(actual, expected)

    def test_shared_llm_seed_variation_does_not_shrink_when_banks_replicate(self):
        # Every control is constant and only the shared LLM varies by start.
        seeds = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]])
        single = analysis.crossed_bootstrap_means([[0, 10, 20]], np.ones((4, 1)), seeds)
        repeated = analysis.crossed_bootstrap_means(
            np.tile([0, 10, 20], (100, 1)), np.ones((4, 100)), seeds)
        np.testing.assert_array_equal(single, repeated)
        self.assertGreater(np.std(repeated), 0)

    def test_bootstrap_rejects_invalid_axis_counts_and_nonfinite_cells(self):
        for matrix, banks, seeds in (
            ([[1, 2]], [[0]], [[1, 1]]),
            ([[1, 2]], [[1]], [[1, 0]]),
            ([[1, float("nan")]], [[1]], [[1, 1]]),
            ([[1, 2]], [[1]], [[-1, 3]]),
        ):
            with self.subTest(matrix=matrix, banks=banks, seeds=seeds):
                with self.assertRaises(ValueError):
                    analysis.crossed_bootstrap_means(matrix, banks, seeds)

    def test_primary_uses_frozen_raw_choices_and_keeps_gate_secondary_separate(self):
        result = self.run_analysis()
        for family in analysis.FAMILIES:
            item = result["families"][family]
            self.assertEqual(item["primary_ungated_auc"]["mean_delta"], 5)
            self.assertEqual(item["secondary_deployed_auc"]["mean_delta"], 0)
            self.assertEqual(item["primary_ungated_auc"]["positive_bank_fraction"], 1)
            self.assertEqual(item["secondary_deployed_auc"]["positive_bank_fraction"], 0)
        self.assertTrue(result["stable_advantage"])

    def test_fixed_seed_analysis_is_repeatable_and_row_order_independent(self):
        rows = deepcopy(self.rows)
        for row in rows:
            if "llm_semantic_" in row["mode"]:
                row["best_so_far_auc"] += (row["seed"] % 3) * len(row["mode"])
        first = self.run_analysis(rows)
        self.assertEqual(first, self.run_analysis(list(reversed(rows))))
        self.assertEqual(first, self.run_analysis(rows))

    def test_adjusted_primary_interval_uses_correct_bonferroni_tails(self):
        actual = analysis.percentile_interval(np.arange(10001), 0.975)
        self.assertAlmostEqual(actual["lower"], 125)
        self.assertAlmostEqual(actual["upper"], 9875)
        result = self.run_analysis()
        self.assertEqual(result["families"]["r1"]["primary_ungated_auc"]["interval"]["confidence_level"], 0.975)
        self.assertEqual(result["families"]["r1"]["secondary_deployed_auc"]["interval"]["confidence_level"], 0.95)

    def test_negative_seed_block_prevents_stable_advantage_despite_positive_mean(self):
        rows = deepcopy(self.rows)
        for row in rows:
            if row["mode"] == "llm_semantic_llm_a":
                row["best_so_far_auc"] = 50 if row["seed"] < 202 else 80
        result = self.run_analysis(rows)
        primary = result["families"]["r1"]["primary_ungated_auc"]
        self.assertGreater(primary["mean_delta"], 0)
        self.assertGreater(primary["interval"]["lower"], 0)
        self.assertFalse(primary["all_block_means_positive"])
        self.assertFalse(result["stable_advantage"])
        self.assertEqual([block["seed_count"] for block in primary["ordered_seed_blocks"]], [2] * 5)

    def test_both_families_must_pass_and_zero_is_not_positive(self):
        rows = deepcopy(self.rows)
        for row in rows:
            if row["mode"].startswith("llm_semantic_r2_") and row["mode"].endswith("_a"):
                row["best_so_far_auc"] = 60
        result = self.run_analysis(rows)
        self.assertTrue(result["families"]["r1"]["primary_family_stability_pass"])
        self.assertFalse(result["families"]["r2"]["primary_family_stability_pass"])
        self.assertFalse(result["stable_advantage"])

    def test_descriptive_metrics_have_no_inferential_interval(self):
        result = self.run_analysis()
        for family in analysis.FAMILIES:
            for endpoint in result["families"][family]["descriptive_other_metrics"].values():
                for stats in endpoint.values():
                    self.assertNotIn("interval", stats)

    def test_missing_duplicate_wrong_split_and_nonfinite_rows_fail_closed(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]]]
        for field, value in (("split", "calibration"), ("best_so_far_auc", float("nan")),
                             ("seed", 200.5), ("top10_hit", 2)):
            rows = deepcopy(self.rows)
            rows[0][field] = value
            variants.append(rows)
        for rows in variants:
            with self.subTest(last=rows[-1]):
                with self.assertRaises(ValueError):
                    self.run_analysis(rows)

    def test_foreign_bank_winner_or_anchor_is_rejected(self):
        for key, value in (("ungated_calibration_winner", "llm_semantic_r2_001_a"),
                           ("selected_mode", "mixed_kernel_gp_ei")):
            lock = deepcopy(self.lock)
            lock["selections"]["r1_001"][key] = value
            with self.assertRaises(ValueError):
                analysis.analyze(self.manifest, lock, self.rows, protocol=self.protocol)

    def test_protocol_counts_and_calibration_separation_cannot_silently_change(self):
        with self.assertRaises(ValueError):
            analysis.analyze(self.manifest, self.lock, self.rows)
        changed = deepcopy(self.manifest)
        changed["config"]["calibration_seed_start"] = 200
        with self.assertRaises(ValueError):
            analysis.analyze(changed, self.lock, self.rows, protocol=self.protocol)

    def test_same_size_run_with_changed_analysis_plan_or_seed_is_rejected(self):
        for key, value in (("bootstrap_seed", 999), ("bootstrap_replicates", 999),
                           ("primary_policy", "selected_mode"), ("heldout_blocks", 2)):
            changed = deepcopy(self.manifest)
            changed["config"]["stability_analysis"][key] = value
            with self.assertRaisesRegex(ValueError, "analysis plan"):
                analysis.analyze(changed, self.lock, self.rows, protocol=self.protocol)
        changed = deepcopy(self.manifest)
        changed["config"]["random_seed"] += 1
        with self.assertRaisesRegex(ValueError, "random_seed"):
            analysis.analyze(changed, self.lock, self.rows, protocol=self.protocol)

    def coverage_fixture(self):
        protocol = {**self.protocol,
                    "run_protocol_id": analysis.COVERAGE_PROTOCOL["run_protocol_id"],
                    "analysis_id": analysis.COVERAGE_PROTOCOL["analysis_id"],
                    "random_seed": analysis.COVERAGE_PROTOCOL["random_seed"],
                    "bootstrap_seed": analysis.COVERAGE_PROTOCOL["bootstrap_seed"]}
        manifest = deepcopy(self.manifest)
        manifest["config"].update(protocol_id=protocol["run_protocol_id"],
                                  random_seed=protocol["random_seed"],
                                  stability_analysis=analysis.expected_analysis_plan(protocol),
                                  coverage_matching=dict(analysis.COVERAGE_MATCHING))
        manifest["coverage_matching_diagnostics"] = {"synthetic_public_coverage": {"pool_count": 20}}
        return protocol, manifest

    def test_registry_preserves_original_and_selects_new_coverage_protocol(self):
        for protocol in (analysis.PROTOCOL, analysis.COVERAGE_PROTOCOL):
            selected = analysis.protocol_for({"protocol_id": protocol["run_protocol_id"]})
            self.assertEqual(selected, protocol)
            self.assertIsNot(selected, protocol)
        self.assertEqual(analysis.PROTOCOL["calibration_seed_start"], 202610183000)
        self.assertEqual(analysis.PROTOCOL["bootstrap_seed"], 2026101803)
        self.assertEqual(analysis.COVERAGE_PROTOCOL["calibration_seed_start"], 202610185000)
        self.assertEqual(analysis.COVERAGE_PROTOCOL["heldout_seed_start"], 202610186000)
        self.assertEqual(analysis.COVERAGE_PROTOCOL["random_seed"], 2026101804)
        self.assertEqual(analysis.COVERAGE_PROTOCOL["bootstrap_seed"], 2026101805)
        for key in ("banks_per_family", "skills_per_bank", "calibration_seed_count", "heldout_seed_count",
                    "bootstrap_replicates", "block_count", "primary_confidence_level",
                    "secondary_confidence_level", "primary_selection_key", "secondary_selection_key"):
            self.assertEqual(analysis.PROTOCOL[key], analysis.COVERAGE_PROTOCOL[key])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            analysis.protocol_for({"protocol_id": "unregistered"})

    def test_coverage_protocol_is_dispatched_by_manifest_without_override(self):
        protocol, manifest = self.coverage_fixture()
        with patch.dict(analysis.COVERAGE_PROTOCOL, protocol):
            result = analysis.analyze(manifest, self.lock, self.rows)
        self.assertEqual(result["analysis_protocol"]["analysis_id"], "bh_coverage_matched_rule_stability_v1")
        for family in analysis.FAMILIES:
            self.assertEqual(result["families"][family]["primary_ungated_auc"]["mean_delta"], 5)
            self.assertEqual(result["families"][family]["secondary_deployed_auc"]["mean_delta"], 0)
        self.assertIn("完整联合激活直方图", analysis.render_text(result))

    def test_coverage_configuration_is_exact_including_boolean_types(self):
        protocol, manifest = self.coverage_fixture()
        for key, value in (("absolute_fraction_tolerance", 0.01),
                           ("absolute_fraction_tolerance", "0.005"),
                           ("match_ordered_rule_support", False),
                           ("match_ordered_rule_support", 1),
                           ("match_joint_activation_histogram", False),
                           ("sampling", "unconditioned"),
                           ("identity_and_duplicates", "forbidden"),
                           ("extra", True)):
            changed = deepcopy(manifest)
            changed["config"]["coverage_matching"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "Coverage-matching configuration"):
                    analysis.analyze(changed, self.lock, self.rows, protocol=protocol)
        for value in (None, {}, []):
            changed = deepcopy(manifest)
            changed["config"]["coverage_matching"] = value
            with self.assertRaisesRegex(ValueError, "Coverage-matching configuration"):
                analysis.analyze(changed, self.lock, self.rows, protocol=protocol)
        changed = deepcopy(self.manifest)
        changed["config"]["coverage_matching"] = dict(analysis.COVERAGE_MATCHING)
        with self.assertRaisesRegex(ValueError, "original random-rule protocol"):
            analysis.analyze(changed, self.lock, self.rows, protocol=self.protocol)

    def test_frozen_coverage_hash_binds_public_diagnostics(self):
        _protocol, manifest = self.coverage_fixture()
        frozen = {"coverage_matching_sha256": analysis.digest(manifest["coverage_matching_diagnostics"])}
        analysis.validate_coverage_provenance(manifest, frozen)
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            analysis.validate_coverage_provenance(manifest, {})
        changed = deepcopy(manifest)
        changed["coverage_matching_diagnostics"]["synthetic_public_coverage"]["pool_count"] = 21
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            analysis.validate_coverage_provenance(changed, frozen)
        for invalid in (None, {}, []):
            changed = deepcopy(manifest)
            changed["coverage_matching_diagnostics"] = invalid
            with self.assertRaisesRegex(ValueError, "nonempty public coverage diagnostics"):
                analysis.validate_coverage_provenance(changed, frozen)
        # Old protocols neither invent nor require a new coverage hash.
        analysis.validate_coverage_provenance(self.manifest, {})

    def test_render_labels_secondary_and_interpretation_limit(self):
        text = analysis.render_text(self.run_analysis())
        self.assertIn("部署次终点", text)
        self.assertIn("探索性未调整", text)
        self.assertIn("不代表新模型生成或新任务", text)

    def test_predeclared_bootstrap_dimensions_preserve_known_constant_effect(self):
        # Full production resampling dimensions, using only synthetic effects.
        rng = np.random.default_rng(analysis.PROTOCOL["bootstrap_seed"])
        seeds = rng.multinomial(100, np.full(100, 0.01), size=10000)
        banks = rng.multinomial(100, np.full(100, 0.01), size=10000)
        actual = analysis.crossed_bootstrap_means(np.full((100, 100), 3.5), banks, seeds)
        np.testing.assert_allclose(actual, 3.5)
        self.assertEqual(actual.shape, (10000,))

    def write_synthetic_run(self, root, manifest=None, coverage_hash=None, source_hash=None):
        def save(name, value):
            (root / name).write_text(json.dumps(value), encoding="utf-8")

        manifest = deepcopy(self.manifest if manifest is None else manifest)
        manifest.update(model_api_calls=0, config_sha256=analysis.digest(manifest["config"]),
                        source_sha256={analysis.ANALYSIS_SOURCE: source_hash or analysis.file_hash(Path(analysis.__file__))},
                        data_sha256={}, execution_aliases={})
        frozen = {key: manifest[key] for key in ("config_sha256", "source_sha256", "data_sha256")}
        frozen.update(libraries_sha256=analysis.digest(manifest["libraries"]),
                      execution_aliases_sha256=analysis.digest(manifest["execution_aliases"]))
        if "coverage_matching_diagnostics" in manifest:
            frozen["coverage_matching_sha256"] = coverage_hash or analysis.digest(manifest["coverage_matching_diagnostics"])
        save("frozen_protocol.json", frozen)
        manifest["frozen_protocol_sha256"] = analysis.file_hash(root / "frozen_protocol.json")
        save("manifest.json", manifest)
        # Intentionally not even a usable calibration table: only its hash
        # belongs to analysis. Frozen choices are the sole selection input.
        (root / "calibration_metrics.csv").write_text("opaque calibration provenance", encoding="utf-8")
        lock = {**self.lock, "manifest_sha256": analysis.file_hash(root / "manifest.json"),
                "calibration_metrics_sha256": analysis.file_hash(root / "calibration_metrics.csv"),
                "frozen_protocol_sha256": manifest["frozen_protocol_sha256"]}
        save("selection_lock.json", lock)
        save("summary.json", {"status": "complete", "model_api_calls": 0,
                              "selection_lock_sha256": analysis.file_hash(root / "selection_lock.json")})
        with (root / "heldout_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        save("integrity.json", {path.name: analysis.file_hash(path) for path in root.iterdir()})

    def test_file_workflow_verifies_provenance_and_refuses_overwrite_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_synthetic_run(root)
            with patch.dict(analysis.PROTOCOL, self.protocol):
                result = analysis.analyze_run(root, root / "analysis")
                self.assertTrue(result["stable_advantage"])
                self.assertFalse(result["provenance"]["calibration_values_read"])
                self.assertTrue((root / "analysis.json").is_file())
                self.assertTrue((root / "analysis.txt").is_file())
                with self.assertRaisesRegex(ValueError, "already exist"):
                    analysis.analyze_run(root, root / "analysis")
                (root / "heldout_metrics.csv").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                    analysis.analyze_run(root, root / "another_analysis")

    def test_coverage_file_workflow_enforces_diagnostics_hash_and_source_freeze(self):
        protocol, manifest = self.coverage_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_synthetic_run(root, manifest)
            with patch.dict(analysis.COVERAGE_PROTOCOL, protocol):
                result = analysis.analyze_run(root, root / "analysis")
            self.assertEqual(result["analysis_protocol"]["analysis_id"], "bh_coverage_matched_rule_stability_v1")
            self.assertEqual(result["coverage_matching"], analysis.COVERAGE_MATCHING)
        # These fixtures have internally consistent file-level hashes; the
        # extra semantic provenance check must still reject incorrect claims.
        for kwargs, error in (({"coverage_hash": "0" * 64}, "diagnostics hash"),
                              ({"source_hash": "0" * 64}, "source differs")):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.write_synthetic_run(root, manifest, **kwargs)
                with patch.dict(analysis.COVERAGE_PROTOCOL, protocol):
                    with self.assertRaisesRegex(ValueError, error):
                        analysis.analyze_run(root, root / "analysis")


if __name__ == "__main__":
    unittest.main()
