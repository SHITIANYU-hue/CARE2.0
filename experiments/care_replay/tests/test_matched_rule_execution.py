"""Exact execution reuse must preserve every original library selection cell."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_matched_random_rule_controls as study  # noqa: E402


class MatchedRuleExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = semantic.SemanticSkill(
            skill_id="original",
            rules=(
                semantic.SemanticRule("first", (("a", "one"), ("b", "two")),
                                      0.8, "first explanation"),
                semantic.SemanticRule("second", (("c", "three"),),
                                      -0.6, "second explanation"),
            ),
            ridge=1.5, prior_scale=0.5, semantic_mass_start=0.4,
            semantic_mass_end=0.2, ucb_weight=0.7, gp_beta_start=2.5,
            gp_beta_end=1.5, gp_xi=0.1, confidence=0.8,
            hypothesis="Original hypothesis",
        )
        self.alias = replace(
            self.original, skill_id="alias", hypothesis="Different prose",
            rules=tuple(replace(rule, rule_id=f"alias_{index}", rationale="Other rationale")
                        for index, rule in enumerate(self.original.rules)),
        )

    def test_signature_ignores_only_ids_and_explanatory_text(self):
        self.assertEqual(study.skill_execution_signature(self.original),
                         study.skill_execution_signature(self.alias))
        self.assertEqual(study.skill_execution_signature(self.original),
                         study.skill_execution_signature(deepcopy(self.original)))

    def test_signature_retains_every_numeric_skill_field(self):
        signature = study.skill_execution_signature(self.original)
        for field in fields(self.original):
            value = getattr(self.original, field.name)
            if isinstance(value, (int, float)):
                with self.subTest(field=field.name):
                    changed = replace(self.original, **{field.name: value + 0.000123})
                    self.assertNotEqual(signature, study.skill_execution_signature(changed))

    def test_signature_retains_rule_weights_fields_values_and_order(self):
        first, second = self.original.rules
        variants = {
            "weight": (replace(first, weight=-first.weight), second),
            "field": (replace(first, conditions=(("different", "one"), ("b", "two"))), second),
            "value": (replace(first, conditions=(("a", "different"), ("b", "two"))), second),
            "condition_order": (replace(first, conditions=tuple(reversed(first.conditions))), second),
            "rule_order": (second, first),
            "rule_multiplicity": (first, second, second),
            "condition_multiplicity": (replace(first, conditions=first.conditions + first.conditions[:1]), second),
        }
        signature = study.skill_execution_signature(self.original)
        for name, rules in variants.items():
            with self.subTest(change=name):
                self.assertNotEqual(signature, study.skill_execution_signature(
                    replace(self.original, rules=rules)))

    def test_execution_plan_preserves_first_representative_and_all_original_aliases(self):
        different = replace(self.original, skill_id="different", prior_scale=0.6)
        third_alias = replace(self.alias, skill_id="third_alias")
        skills = (self.original, self.alias, different, third_alias)
        representatives, aliases = study.execution_plan(skills)
        self.assertEqual(representatives, (self.original, different))
        self.assertEqual(list(aliases), [semantic.skill_mode(skill) for skill in skills])
        self.assertEqual(list(aliases.values()), [semantic.skill_mode(self.original),
                         semantic.skill_mode(self.original), semantic.skill_mode(different),
                         semantic.skill_mode(self.original)])
        full_representatives, identities = study.execution_plan(skills, enabled=False)
        self.assertEqual(full_representatives, skills)
        self.assertEqual(identities, {semantic.skill_mode(skill): semantic.skill_mode(skill)
                                      for skill in skills})

    def test_metric_expansion_preserves_every_nonmode_key_and_original_cells(self):
        representatives, aliases = study.execution_plan((self.original, self.alias))
        self.assertEqual(representatives, (self.original,))
        mode = semantic.skill_mode(self.original)
        baseline = {"mode": study.BASELINES[0], "seed": 3, "split": "calibration",
                    "final_best": 10.0, "custom": "baseline metadata"}
        metric = {"mode": mode, "seed": 3, "split": "calibration", "final_best": 12.0,
                  "best_so_far_auc": 11.2345, "simple_regret": 0.3, "top10_hit": 1,
                  "selected_source_mode": "", "custom": {"nested": [1, 2]}}
        rows = [baseline, metric]
        before = deepcopy(rows)
        expanded = study.expand_metric_aliases(rows, aliases)
        self.assertEqual(rows, before)
        self.assertEqual(expanded, [baseline, metric,
                         {**metric, "mode": semantic.skill_mode(self.alias)}])
        second_seed = [{**row, "seed": 4} for row in rows]
        with self.assertRaises(ValueError):
            study.expand_metric_aliases(rows + second_seed, aliases)
        expanded_two = expanded + study.expand_metric_aliases(second_seed, aliases)
        self.assertEqual(len(expanded_two), 6)
        self.assertEqual({(row["mode"], row["seed"]) for row in expanded_two},
                         {(mode, seed) for mode in (study.BASELINES[0], *aliases)
                          for seed in (3, 4)})

    def test_aliases_preserve_canonical_selection_and_id_tiebreak(self):
        libraries = {"llm": (self.original, self.alias)}
        seeds = set(range(5))
        physical_rows = []
        original_mode = semantic.skill_mode(self.original)
        for seed in sorted(seeds):
            for mode in study.BASELINES:
                physical_rows.append({"mode": mode, "seed": seed, "split": "calibration",
                                      "final_best": 10.0, "best_so_far_auc": 10.0,
                                      "top10_hit": 0})
            physical_rows.append({"mode": original_mode, "seed": seed, "split": "calibration",
                                  "final_best": 15.0, "best_so_far_auc": 15.0,
                                  "top10_hit": 1})
        _, aliases = study.execution_plan(libraries["llm"])
        expanded = [row for seed in sorted(seeds)
                    for row in study.expand_metric_aliases(
                        [physical for physical in physical_rows if physical["seed"] == seed], aliases)]
        config = {"min_risk_adjusted_gain": 0.25, "min_positive_fold_rate": 0.8}
        selections = study.freeze_selection(expanded, libraries, seeds, config)
        mode, detail = study.calibrated.select_skill(expanded, seeds, libraries["llm"], 0.25, 0.8)
        self.assertEqual(selections["llm"]["selected_mode"], mode)
        self.assertEqual(mode, max(aliases))
        self.assertEqual(set(detail["skill_diagnostics"]), set(aliases))

    def test_cache_is_cleared_even_when_canonical_execution_fails(self):
        config = {"target_dataset": "real_buchwald_hartwig", "initial_observations": 5,
                  "reveal_rounds": 10, "gp_beta": 1.5, "gp_xi": 0.01,
                  "numeric_length_scale": 0.35, "categorical_length_scale": 3.0,
                  "gp_noise": 0.05}
        cache = study.calibrated.surrogate._FEATURE_ARRAY_CACHE
        original_connect = socket.socket.connect
        cache["sentinel"] = object()
        with patch.object(study.calibrated, "evaluate_seed", side_effect=ValueError("test failure")):
            with self.assertRaisesRegex(ValueError, "test failure"):
                study.evaluate_seed(3, "calibration", config, (self.original,))
        self.assertEqual(cache, {})
        self.assertIs(socket.socket.connect, original_connect)

    def test_feature_cache_is_bounded_and_preserves_values_and_hits(self):
        surrogate = study.calibrated.surrogate
        if surrogate.np is None:
            self.skipTest("NumPy feature-array backend is unavailable")
        adapter = SimpleNamespace(candidates=[SimpleNamespace(candidate_id="a"),
                                              SimpleNamespace(candidate_id="b")])
        cache = surrogate._FEATURE_ARRAY_CACHE
        cache.clear()
        # Retain old objects so identity reuse cannot turn this into a spurious
        # hit; each dictionary has equal feature values but a distinct identity.
        inputs = []
        try:
            for index in range(8):
                features = {"a": ((0.1, 0.2), ("x", "y")),
                            "b": ((0.3, 0.4), ("z", "w"))}
                inputs.append(features)
                arrays = surrogate._cached_feature_arrays(adapter, features)
                again = surrogate._cached_feature_arrays(adapter, features)
                with self.subTest(dictionary=index):
                    self.assertLessEqual(len(cache), 4)
                    for first, second in zip(arrays, again):
                        self.assertIs(first, second)
                    self.assertEqual(arrays[0], ("a", "b"))
                    self.assertEqual(arrays[1], {"a": 0, "b": 1})
                    self.assertEqual(arrays[2].tolist(), [[0.1, 0.2], [0.3, 0.4]])
                    self.assertEqual(arrays[3].tolist(), [["x", "y"], ["z", "w"]])
            # The earliest entry was evicted, but reconstructs identically.
            arrays = surrogate._cached_feature_arrays(adapter, inputs[0])
            self.assertEqual(arrays[2].tolist(), [[0.1, 0.2], [0.3, 0.4]])
            self.assertEqual(arrays[3].tolist(), [["x", "y"], ["z", "w"]])
            self.assertLessEqual(len(cache), 4)
        finally:
            cache.clear()


class ExistingPilotExecutionEquivalenceTest(unittest.TestCase):
    def test_real_bh_replay_ignores_only_alias_prose_and_ids(self):
        # An already-used pilot calibration seed verifies optimization parity;
        # this test never inspects the expanded study's calibration/held-out data.
        config_path = SCRIPTS.parent / "configs" / "bh_random_rule_controls_pilot_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with study.offline_guard():
            record = json.loads((study.ROOT / config["llm_record"]).read_text(encoding="utf-8"))
            catalog = record["prompt_payload"]["target"]["public_semantic_fields"]
            original = semantic.normalize_skills({"skills": record["normalized_skills"]},
                                                  catalog, 100)[0]
            alias = replace(original, skill_id="execution_equivalence_alias", hypothesis="Alias prose",
                            rules=tuple(replace(rule, rule_id=f"alias_{index}", rationale="Alias text")
                                        for index, rule in enumerate(original.rules)))
            self.assertEqual(study.skill_execution_signature(original),
                             study.skill_execution_signature(alias))
            rows, traces, _ = study.evaluate_seed(config["calibration_seed_start"], "calibration",
                                                  config, (original, alias))
        original_mode, alias_mode = semantic.skill_mode(original), semantic.skill_mode(alias)
        original_rows = [row for row in rows if row["mode"] == original_mode]
        alias_rows = [dict(row, mode=original_mode) for row in rows if row["mode"] == alias_mode]
        self.assertEqual(len(original_rows), 1)
        self.assertEqual(original_rows, alias_rows)
        original_trace = [event for event in traces if event["mode"] == original_mode]
        alias_trace = [dict(event, mode=original_mode) for event in traces if event["mode"] == alias_mode]
        self.assertEqual(len(original_trace), config["reveal_rounds"])
        self.assertEqual(original_trace, alias_trace)
        self.assertEqual(study.calibrated.surrogate._FEATURE_ARRAY_CACHE, {})


if __name__ == "__main__":
    unittest.main()
