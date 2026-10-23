from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import random
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import coverage_matched_rule_controls as coverage  # noqa: E402
import llm_semantic_skills as semantic  # noqa: E402


class WhitelistedMetadata(Mapping):
    """A metadata mapping that forbids even iterating its hidden target key."""

    def __init__(self, public, hidden):
        self.public, self.hidden = public, hidden
        self.accesses = []

    def __getitem__(self, key):
        self.accesses.append(key)
        if key not in self.public:
            raise AssertionError(f"Nonpublic metadata accessed: {key}")
        return self.public[key]

    def __iter__(self):
        raise AssertionError("Complete metadata must not be iterated or hashed")

    def __len__(self):
        raise AssertionError("Complete metadata must not be inspected")


class MetadataOnlyCandidate:
    def __init__(self, metadata):
        self.metadata = metadata

    def __getattr__(self, name):
        raise AssertionError(f"Candidate property outside metadata accessed: {name}")


class ControlledRandom:
    def __init__(self, configuration_index, signs):
        self.configuration_index = configuration_index
        self.signs = iter(signs)

    def choice(self, choices):
        return choices[self.configuration_index]

    def random(self):
        return next(self.signs)


class CoverageMatchedRuleControlTest(unittest.TestCase):
    def setUp(self):
        # Identical public columns deliberately create many different legal
        # assignments with the same activation. They must retain multiplicity.
        self.metadata = [{"a": value, "b": value, "c": value}
                         for value in ("0", "0", "1", "1")]
        self.catalog = {field: {"0": 2, "1": 2} for field in ("a", "b", "c")}
        self.field_types = {field: "binary" for field in self.catalog}
        self.skill = semantic.SemanticSkill(
            skill_id="source",
            rules=(semantic.SemanticRule("first", (("a", "0"),), 0.8, "first rule"),
                   semantic.SemanticRule("second", (("b", "0"),), -0.4, "second rule")),
            ridge=2.0, prior_scale=0.3, semantic_mass_start=0.3, semantic_mass_end=0.1,
            ucb_weight=0.5, gp_beta_start=1.5, gp_beta_end=1.0, gp_xi=0.01,
            confidence=0.7, hypothesis="template hypothesis",
        )

    def pools(self, **kwargs):
        return coverage.build_pools((self.skill,), self.catalog, self.field_types,
                                    self.metadata, **kwargs)

    @staticmethod
    def observed_coverage(configuration, metadata):
        activations = [tuple(all(row.get(field, "") == value for field, value in conditions)
                             for conditions in configuration) for row in metadata]
        supports = tuple(sum(row[index] for row in activations) for index in range(len(configuration)))
        histogram = Counter(sum((1 << index) for index, active in enumerate(row) if active)
                            for row in activations)
        return supports, histogram

    def test_r1_joint_constraint_rejects_same_marginals_with_wrong_coactivation(self):
        pool = self.pools(tolerance=0.0)["r1"]["source"]
        self.assertEqual(pool.diagnostics["total_condition_configurations"], 4)
        self.assertEqual(pool.diagnostics["accepted_condition_configurations"], 2)
        self.assertEqual(set(pool.configurations), {
            ((("a", "0"),), (("b", "0"),)),
            ((("a", "1"),), (("b", "1"),)),
        })
        original = tuple(rule.conditions for rule in self.skill.rules)
        crossed = ((("a", "0"),), (("b", "1"),))
        self.assertEqual(self.observed_coverage(original, self.metadata)[0],
                         self.observed_coverage(crossed, self.metadata)[0])
        self.assertNotEqual(self.observed_coverage(original, self.metadata)[1],
                            self.observed_coverage(crossed, self.metadata)[1])

    def test_r2_exhausts_uniform_injective_maps_without_activation_deduplication(self):
        pool = self.pools(tolerance=0.0)["r2"]["source"]
        self.assertEqual(pool.diagnostics["field_mapping_count"], 6)
        self.assertEqual(pool.diagnostics["condition_configurations_per_mapping"], 4)
        self.assertEqual(pool.diagnostics["total_condition_configurations"], 24)
        self.assertEqual(len(pool.configurations), 12)
        assignments_per_map = Counter(tuple(rule[0][0] for rule in configuration)
                                      for configuration in pool.configurations)
        self.assertEqual(set(assignments_per_map.values()), {2})
        self.assertEqual(len(assignments_per_map), 6)
        self.assertTrue(all(left != right for left, right in assignments_per_map))
        activation_patterns = {tuple(tuple(all(row[field] == value for field, value in rule)
                                           for rule in configuration) for row in self.metadata)
                               for configuration in pool.configurations}
        self.assertEqual(len(activation_patterns), 2)
        self.assertGreater(len(pool.configurations), len(activation_patterns))
        self.assertIn(tuple(rule.conditions for rule in self.skill.rules), pool.configurations)

    def test_every_configuration_matches_reference_support_and_all_joint_cells(self):
        original = tuple(rule.conditions for rule in self.skill.rules)
        original_support, original_joint = self.observed_coverage(original, self.metadata)
        for family, bank in self.pools(tolerance=0.0).items():
            pool = bank["source"]
            for configuration in pool.configurations:
                with self.subTest(family=family, configuration=configuration):
                    self.assertEqual(self.observed_coverage(configuration, self.metadata),
                                     (original_support, original_joint))
            self.assertEqual(pool.diagnostics["original_rule_support_counts"], list(original_support))
            self.assertEqual(pool.diagnostics["original_joint_activation_counts"],
                             {str(key): value for key, value in original_joint.items()})
            self.assertEqual(pool.diagnostics["max_accepted_joint_activation_fraction_deviation"], 0.0)
            json.dumps(pool.diagnostics, allow_nan=False)

    def test_inclusive_fraction_boundary_uses_exact_candidate_counts(self):
        metadata = [{"a": "0" if index < 101 else "1", "b": "0" if index < 100 else "1"}
                    for index in range(200)]
        catalog = {"a": {"0": 101, "1": 99}, "b": {"0": 100, "1": 100}}
        field_types = {"a": "binary", "b": "binary"}
        skill = replace(self.skill, rules=(self.skill.rules[0],))
        at_boundary = coverage.build_pools((skill,), catalog, field_types, metadata,
                                           tolerance=0.005)["r2"]["source"]
        below_boundary = coverage.build_pools((skill,), catalog, field_types, metadata,
                                              tolerance=0.00499)["r2"]["source"]
        self.assertEqual(at_boundary.diagnostics["allowed_count_deviation"], 1)
        self.assertEqual(set(at_boundary.configurations), {((("a", "0"),),),
                                                          ((("b", "0"),),), ((("b", "1"),),)})
        self.assertEqual(at_boundary.diagnostics["max_accepted_rule_support_fraction_deviation"], 0.005)
        self.assertEqual(at_boundary.diagnostics["max_accepted_joint_activation_fraction_deviation"], 0.005)
        self.assertEqual(below_boundary.configurations, (((("a", "0"),),),))

    def test_signed_draws_cover_independent_sign_combinations_and_keep_parameters(self):
        pool = self.pools()["r2"]["source"]
        signs_seen = set()
        for signs in product((0.1, 0.9), repeat=2):
            sampled = coverage.draw_skill(pool, ControlledRandom(0, signs), "draw")
            self.assertEqual(tuple(rule.conditions for rule in sampled.rules), pool.configurations[0])
            signs_seen.add(tuple(rule.weight for rule in sampled.rules))
            self.assertEqual([abs(rule.weight) for rule in sampled.rules], [0.8, 0.4])
            self.assertEqual(sampled.skill_id, "draw")
            for field, value in asdict(self.skill).items():
                if field not in {"skill_id", "rules", "hypothesis"}:
                    self.assertEqual(getattr(sampled, field), value)
        self.assertEqual(signs_seen, set(product((-0.8, 0.8), (-0.4, 0.4))))
        self.assertEqual(pool.diagnostics["accepted_signed_configurations"], len(pool.configurations) * 4)
        self.assertEqual(pool.diagnostics["total_signed_configurations"], 24 * 4)

    def test_uniform_draw_delegates_to_uniform_choice_of_full_configuration_pool(self):
        pool = self.pools()["r2"]["source"]
        # Visiting every equiprobable choice index recovers exactly the finite
        # conditional distribution, including activation-equivalent entries.
        recovered = [tuple(rule.conditions for rule in coverage.draw_skill(
            pool, ControlledRandom(index, (0.1, 0.9)), "draw").rules)
            for index in range(len(pool.configurations))]
        self.assertEqual(tuple(recovered), pool.configurations)

    def test_draw_is_reproducible_with_replacement_and_preserves_template(self):
        pool = self.pools()["r2"]["source"]
        before = asdict(self.skill)
        left, right = random.Random(4), random.Random(4)
        left_draws = [coverage.draw_skill(pool, left, f"draw_{index}") for index in range(40)]
        right_draws = [coverage.draw_skill(pool, right, f"draw_{index}") for index in range(40)]
        self.assertEqual(left_draws, right_draws)
        conditions = [tuple(rule.conditions for rule in skill.rules) for skill in left_draws]
        self.assertLess(len(set(conditions)), len(conditions))
        self.assertEqual(asdict(self.skill), before)

    def test_reused_field_mapping_is_global_and_types_are_preserved(self):
        catalog = {**self.catalog, "flag": {"off": 2, "on": 2}, "flag2": {"no": 2, "yes": 2}}
        types = {**self.field_types, "flag": "boolean", "flag2": "boolean"}
        metadata = [dict(row, flag="on" if index < 2 else "off", flag2="yes" if index < 2 else "no")
                    for index, row in enumerate(self.metadata)]
        skill = replace(self.skill, rules=(
            semantic.SemanticRule("a", (("a", "0"), ("flag", "on")), 0.8, "test"),
            semantic.SemanticRule("b", (("a", "1"), ("b", "1")), -0.4, "test"),
        ))
        pool = coverage.build_pools((skill,), catalog, types, metadata, tolerance=1.0)["r2"]["source"]
        for configuration in pool.configurations:
            mapping = {}
            for before, after in zip(skill.rules, configuration):
                self.assertEqual(len(before.conditions), len(after))
                for (source, _), (target, _) in zip(before.conditions, after):
                    self.assertEqual(types[source], types[target])
                    self.assertEqual(mapping.setdefault(source, target), target)
            self.assertEqual(len(set(mapping.values())), len(mapping))

    def test_hidden_yield_metadata_and_objective_properties_are_never_accessed(self):
        guarded = [MetadataOnlyCandidate(WhitelistedMetadata(dict(row), hidden=index * 1e9))
                   for index, row in enumerate(self.metadata)]
        actual = coverage.build_pools((self.skill,), self.catalog, self.field_types, guarded)
        expected = self.pools()
        self.assertEqual(actual, expected)
        for candidate in guarded:
            self.assertEqual(set(candidate.metadata.accesses), set(self.catalog))

    def test_hidden_label_permutation_does_not_change_pool_hashes_or_draws(self):
        left_metadata = [dict(row, yield_value=index) for index, row in enumerate(self.metadata)]
        right_metadata = [dict(row, yield_value=1000 - index) for index, row in enumerate(self.metadata)]
        left = coverage.build_pools((self.skill,), self.catalog, self.field_types, left_metadata)
        right = coverage.build_pools((self.skill,), self.catalog, self.field_types, right_metadata)
        self.assertEqual(left, right)
        for family in left:
            self.assertEqual(coverage.draw_skill(left[family]["source"], random.Random(3), "draw"),
                             coverage.draw_skill(right[family]["source"], random.Random(3), "draw"))

    def test_metadata_and_dictionary_order_do_not_change_pool_or_diagnostics(self):
        reordered_catalog = {field: dict(reversed(tuple(values.items())))
                             for field, values in reversed(tuple(self.catalog.items()))}
        reordered = coverage.build_pools((self.skill,), reordered_catalog,
                                         dict(reversed(tuple(self.field_types.items()))),
                                         list(reversed(self.metadata)))
        self.assertEqual(reordered, self.pools())

    def test_cache_reuses_only_same_public_problem_and_tolerance(self):
        cache = {}
        first = self.pools(cache=cache)
        second = coverage.build_pools((self.skill,), self.catalog, self.field_types,
                                       list(reversed(self.metadata)), cache=cache)
        for family in first:
            self.assertIs(first[family]["source"], second[family]["source"])
        changed = self.pools(cache=cache, tolerance=1.0)
        self.assertIsNot(first["r1"]["source"], changed["r1"]["source"])
        public_change = [*self.metadata, {"a": "0", "b": "1", "c": "0"}]
        third = coverage.build_pools((self.skill,), self.catalog, self.field_types, public_change, cache=cache)
        self.assertNotEqual(first["r1"]["source"].diagnostics["public_metadata_sha256"],
                            third["r1"]["source"].diagnostics["public_metadata_sha256"])

    def test_r2_unequal_cardinality_fails_instead_of_biasing_original_null(self):
        catalog = {**self.catalog, "b": {"0": 2, "1": 2, "2": 1}}
        with self.assertRaisesRegex(ValueError, "equal value cardinality"):
            coverage.build_pools((self.skill,), catalog, self.field_types, self.metadata)
        r1 = coverage.build_pools((self.skill,), catalog, self.field_types, self.metadata, families=("r1",))
        self.assertEqual(r1["r1"]["source"].diagnostics["total_condition_configurations"], 6)

    def test_enumeration_limit_fails_without_approximate_or_unmatched_fallback(self):
        cache = {}
        with self.assertRaisesRegex(ValueError, "No approximate or unmatched fallback"):
            self.pools(families=("r2",), max_configurations=23, cache=cache)
        self.assertEqual(cache, {})

    def test_invalid_tolerance_limit_families_or_metadata_are_rejected(self):
        for tolerance in (-0.1, 1.1, True, float("nan"), float("inf"), "0.005"):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.pools(tolerance=tolerance)
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.pools(max_configurations=limit)
        for families in ((), ("r1", "r1"), ("r3",)):
            with self.subTest(families=families), self.assertRaises(ValueError):
                self.pools(families=families)
        with self.assertRaisesRegex(ValueError, "at least one public candidate"):
            coverage.build_pools((self.skill,), self.catalog, self.field_types, [])

    def test_invalid_skill_or_incomplete_types_are_rejected_before_metadata_access(self):
        class ForbiddenCandidate:
            @property
            def metadata(self):
                raise AssertionError("Invalid input should fail before metadata access")

        with self.assertRaisesRegex(ValueError, "cover exactly"):
            coverage.build_pools((self.skill,), self.catalog, {}, [ForbiddenCandidate()])
        duplicate = replace(self.skill, rules=(replace(self.skill.rules[0],
                            conditions=(("a", "0"), ("a", "1"))),))
        with self.assertRaisesRegex(ValueError, "repeats condition field"):
            coverage.build_pools((duplicate,), self.catalog, self.field_types, [ForbiddenCandidate()])
        with self.assertRaisesRegex(ValueError, "unique skill IDs"):
            coverage.build_pools((self.skill, self.skill), self.catalog, self.field_types, [ForbiddenCandidate()])


if __name__ == "__main__":
    unittest.main()
