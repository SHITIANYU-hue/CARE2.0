from __future__ import annotations

import random
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_random_rule_control as null_control  # noqa: E402


class RandomRuleControlTest(unittest.TestCase):
    """All fixtures are public metadata; no dataset or model is needed."""

    def setUp(self) -> None:
        # Unsorted names/values exercise legacy R1 and canonical R2 ordering.
        self.catalog = {
            "choice_z": {"z2": 10, "z1": 1},
            "choice_a": {"a2": 7, "a1": 3},
            "choice_m": {"m2": 4, "m1": 6},
            "flag_z": {"on": 9, "off": 1},
            "flag_a": {"present": 2, "absent": 8},
        }
        self.field_types = {
            "choice_z": "categorical", "choice_a": "categorical",
            "choice_m": "categorical", "flag_z": "binary", "flag_a": "binary",
        }
        self.skill = semantic.SemanticSkill(
            skill_id="source",
            rules=(
                semantic.SemanticRule("r1", (("choice_z", "z1"), ("choice_a", "a1")), 0.8, "test"),
                semantic.SemanticRule("r2", (("choice_z", "z2"), ("flag_z", "on")), -0.4, "test"),
                semantic.SemanticRule("r3", (("flag_z", "off"),), 0.0, "test"),
            ),
            ridge=2.0, prior_scale=0.3, semantic_mass_start=0.3,
            semantic_mass_end=0.1, ucb_weight=0.5, gp_beta_start=1.5,
            gp_beta_end=1.0, gp_xi=0.01, confidence=0.7, hypothesis="test",
        )

    def draw(self, null_type: str, seed: int = 9) -> semantic.SemanticSkill:
        return null_control.matched_random_skill(
            self.skill, self.catalog, random.Random(seed), "random",
            null_type=null_type, field_types=self.field_types,
        )

    def assert_matched_structure(self, randomized: semantic.SemanticSkill) -> None:
        self.assertEqual(randomized.skill_id, "random")
        self.assertEqual(len(randomized.rules), len(self.skill.rules))
        for before, after in zip(self.skill.rules, randomized.rules):
            self.assertEqual(len(after.conditions), len(before.conditions))
            self.assertEqual(abs(after.weight), abs(before.weight))
            self.assertEqual(len({field for field, _ in after.conditions}), len(after.conditions))
            for field, value in after.conditions:
                self.assertIn(value, self.catalog[field])
        for field in asdict(self.skill):
            if field not in {"skill_id", "rules", "hypothesis"}:
                self.assertEqual(getattr(randomized, field), getattr(self.skill, field), field)

    def test_both_nulls_preserve_shape_magnitudes_and_all_hyperparameters(self) -> None:
        for null_type in ("r1", "r2"):
            with self.subTest(null_type=null_type):
                self.assert_matched_structure(self.draw(null_type))

    def test_r1_preserves_exact_field_positions(self) -> None:
        self.assertEqual(
            [tuple(field for field, _ in rule.conditions) for rule in self.draw("r1").rules],
            [tuple(field for field, _ in rule.conditions) for rule in self.skill.rules],
        )

    def test_r1_matches_legacy_draws_and_rng_state(self) -> None:
        for seed in (0, 1, 9, 100):
            actual_rng = random.Random(seed)
            legacy_rng = random.Random(seed)
            randomized = null_control.matched_random_skill(self.skill, self.catalog, actual_rng, "random")
            expected_rules = []
            for index, rule in enumerate(self.skill.rules):
                conditions = tuple((field, legacy_rng.choice(tuple(self.catalog[field]))) for field, _ in rule.conditions)
                sign = -1.0 if legacy_rng.random() < 0.5 else 1.0
                expected_rules.append(semantic.SemanticRule(
                    rule_id=f"random_{index + 1}", conditions=conditions,
                    weight=sign * abs(float(rule.weight)),
                    rationale="Matched random null: original fields and rule arity preserved; condition values and sign randomized.",
                ))
            self.assertEqual(randomized.rules, tuple(expected_rules))
            self.assertEqual(randomized.hypothesis,
                             "Matched random-rule null preserving the frozen LLM skill's public feature structure, rule count, and execution schedule.")
            self.assertEqual(actual_rng.getstate(), legacy_rng.getstate())
            self.assertEqual(randomized, self.draw("r1", seed))

    def test_r2_type_matching_injection_and_reuse(self) -> None:
        for seed in range(30):
            randomized = self.draw("r2", seed)
            mapping = {}
            for before, after in zip(self.skill.rules, randomized.rules):
                for (source, _), (target, _) in zip(before.conditions, after.conditions):
                    self.assertEqual(self.field_types[source], self.field_types[target])
                    if source in mapping:
                        self.assertEqual(mapping[source], target)
                    mapping[source] = target
            self.assertEqual(len(set(mapping.values())), len(mapping))
            self.assert_matched_structure(randomized)

    def test_r2_reproducible_and_changes_fields_across_draws(self) -> None:
        seen = set()
        for seed in range(50):
            left = self.draw("r2", seed)
            self.assertEqual(left, self.draw("r2", seed))
            seen.add(left.rules[0].conditions[0][0])
        self.assertEqual(seen, {"choice_a", "choice_m", "choice_z"})

    def test_r2_allows_identity_mapping(self) -> None:
        original = [tuple(field for field, _ in rule.conditions) for rule in self.skill.rules]
        observed = [
            [tuple(field for field, _ in rule.conditions) for rule in self.draw("r2", seed).rules]
            for seed in range(100)
        ]
        self.assertIn(original, observed)

    def test_r2_is_independent_of_dictionary_insertion_order(self) -> None:
        reversed_catalog = {
            field: dict(reversed(tuple(values.items())))
            for field, values in reversed(tuple(self.catalog.items()))
        }
        reversed_types = dict(reversed(tuple(self.field_types.items())))
        for seed in range(10):
            reordered = null_control.matched_random_skill(
                self.skill, reversed_catalog, random.Random(seed), "random",
                null_type="r2", field_types=reversed_types,
            )
            self.assertEqual(reordered, self.draw("r2", seed))

    def test_value_counts_do_not_bias_draws(self) -> None:
        changed_counts = {
            field: {value: 10000 - count for value, count in values.items()}
            for field, values in self.catalog.items()
        }
        for null_type in ("r1", "r2"):
            randomized = null_control.matched_random_skill(
                self.skill, changed_counts, random.Random(9), "random",
                null_type=null_type, field_types=self.field_types,
            )
            self.assertEqual(randomized, self.draw(null_type))

    def test_generation_does_not_mutate_source_or_metadata(self) -> None:
        skill_before = asdict(self.skill)
        catalog_before = {field: dict(values) for field, values in self.catalog.items()}
        types_before = dict(self.field_types)
        self.draw("r1")
        self.draw("r2")
        self.assertEqual(asdict(self.skill), skill_before)
        self.assertEqual(self.catalog, catalog_before)
        self.assertEqual(self.field_types, types_before)

    def assert_rejected_without_rng_draw(self, **kwargs: object) -> None:
        rng = random.Random(10)
        state = rng.getstate()
        parameters = dict(skill=self.skill, catalog=self.catalog, rng=rng,
                          skill_id="random", null_type="r2", field_types=self.field_types)
        parameters.update(kwargs)
        with self.assertRaises(ValueError):
            null_control.matched_random_skill(**parameters)
        self.assertEqual(rng.getstate(), state)

    def test_unknown_null_type_is_rejected(self) -> None:
        self.assert_rejected_without_rng_draw(null_type="r3")

    def test_r2_requires_complete_explicit_field_types(self) -> None:
        for field_types in (
            None, {}, {"choice_z": "categorical"},
            {**self.field_types, "missing": "categorical"},
            {**self.field_types, "choice_z": ""},
            {**self.field_types, "choice_z": None},
        ):
            with self.subTest(field_types=field_types):
                self.assert_rejected_without_rng_draw(field_types=field_types)

    def test_r1_validates_field_types_if_provided(self) -> None:
        self.assert_rejected_without_rng_draw(null_type="r1", field_types={})

    def test_invalid_catalog_is_rejected(self) -> None:
        invalid_catalogs = [
            {}, {"": {"a": 1}}, {"choice_z": {}}, {"choice_z": {"": 1}},
            {"choice_z": {1: 2}}, {"choice_z": {"z1": 0}},
            {"choice_z": {"z1": -1}}, {"choice_z": {"z1": 1.5}},
            {"choice_z": {"z1": True}},
        ]
        for catalog in invalid_catalogs:
            with self.subTest(catalog=catalog):
                self.assert_rejected_without_rng_draw(catalog=catalog)

    def test_unknown_rule_field_or_value_is_rejected(self) -> None:
        for conditions in ((("missing", "x"),), (("choice_z", "missing"),)):
            invalid = replace(self.skill, rules=(replace(self.skill.rules[0], conditions=conditions),))
            self.assert_rejected_without_rng_draw(skill=invalid)

    def test_duplicate_condition_fields_are_rejected(self) -> None:
        for null_type in ("r1", "r2"):
            invalid = replace(self.skill, rules=(replace(
                self.skill.rules[0], conditions=(("choice_z", "z1"), ("choice_z", "z2")),
            ),))
            self.assert_rejected_without_rng_draw(skill=invalid, null_type=null_type)

    def test_empty_rules_or_conditions_are_rejected(self) -> None:
        self.assert_rejected_without_rng_draw(skill=replace(self.skill, rules=()))
        self.assert_rejected_without_rng_draw(skill=replace(
            self.skill, rules=(replace(self.skill.rules[0], conditions=()),),
        ))

    def test_nonfinite_or_invalid_weights_are_rejected(self) -> None:
        for weight in (float("inf"), float("nan"), "invalid", None):
            invalid = replace(self.skill, rules=(replace(self.skill.rules[0], weight=weight),))
            self.assert_rejected_without_rng_draw(skill=invalid)

    def test_single_field_type_has_legal_identity_mapping(self) -> None:
        skill = replace(self.skill, rules=(self.skill.rules[2],))
        catalog = {"flag_z": self.catalog["flag_z"]}
        randomized = null_control.matched_random_skill(
            skill, catalog, random.Random(3), "random", null_type="r2",
            field_types={"flag_z": "binary"},
        )
        self.assertEqual(randomized.rules[0].conditions[0][0], "flag_z")


if __name__ == "__main__":
    unittest.main()
