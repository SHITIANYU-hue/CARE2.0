"""Offline equivalence and rejection tests for paired BH semantic labels."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import llm_semantic_skills as semantic
import paired_semantic_anonymization as paired


class PairedSemanticAnonymizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.materials = [paired.build_pair(i, 2026102001 + i) for i in range(10)]
        cls.record = json.loads(paired.RECORD.read_text(encoding="utf-8"))
        cls.library = {"skills": copy.deepcopy(cls.record["normalized_skills"])}
        for skill in cls.library["skills"]:
            for rule in skill["rules"]:
                rule["conditions"] = dict(rule["conditions"])

    def library_for(self, material, arm):
        library = copy.deepcopy(self.library)
        if arm == "anonymous":
            mapping = material["private_mapping"]["forward"]
            for skill in library["skills"]:
                for rule in skill["rules"]:
                    rule["conditions"] = {mapping["fields"][f]: mapping["values"][v]
                                          for f, v in rule["conditions"].items()}
        return library

    def test_all_input_hashes_are_exported_and_record_is_frozen(self):
        hashes = self.materials[0]["input_hashes"]
        self.assertEqual(len(hashes), 2)
        for relative, digest in hashes.items():
            self.assertEqual(hashlib.sha256((paired.ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashes[str(paired.RECORD.relative_to(paired.ROOT))], paired.EXPECTED_RECORD_SHA256)

    def test_all_ten_pairs_keep_204_numeric_leaves_and_exact_inverse(self):
        expected = paired.numeric_leaves(self.record["prompt_payload"])
        self.assertEqual(len(expected), 204)
        for material in self.materials:
            with self.subTest(pair=material["pair_index"]):
                labeled = material["arms"]["labeled"]["payload"]
                anonymous = material["arms"]["anonymous"]["payload"]
                inverse = {k: v for section in material["private_mapping"]["inverse"].values()
                           for k, v in section.items()}
                self.assertEqual(paired.numeric_leaves(labeled), expected)
                self.assertEqual(paired.numeric_leaves(anonymous), expected)
                self.assertEqual(paired.map_tree(anonymous, inverse), labeled)
                self.assertEqual(material["invariants"]["numeric_leaf_count"], 204)

    def test_alias_assignments_are_reproducible_pair_specific_permutations(self):
        self.assertEqual(paired.build_pair(0, 2026102001), self.materials[0])
        all_maps = []
        for material in self.materials:
            mapping = material["private_mapping"]["forward"]
            all_maps.append(paired.canonical_json(mapping))
            for family, prefix in (("fields", "F"), ("values", "V"), ("tasks", "T")):
                self.assertEqual(set(mapping[family].values()),
                                 {f"{prefix}{i:02d}" for i in range(1, len(mapping[family]) + 1)})
        self.assertEqual(len(set(all_maps)), 10)
        for family in ("fields", "values", "tasks"):
            sequences = [list(m["private_mapping"]["forward"][family].values()) for m in self.materials]
            self.assertTrue(any(s != sorted(s) for s in sequences), family)
        original = self.materials[0]["private_mapping"]["forward"]
        self.assertNotEqual(original, paired.build_pair(1, 2026102001)["private_mapping"]["forward"])
        self.assertNotEqual(original, paired.build_pair(0, 2026102002)["private_mapping"]["forward"])

    def test_only_two_neutral_messages_are_outbound_and_hashes_match(self):
        for material in self.materials:
            for arm in ("labeled", "anonymous"):
                data = material["arms"][arm]
                self.assertEqual(data["messages"][0], {"role": "system", "content": paired.SYSTEM})
                self.assertEqual(len(data["messages"]), 2)
                self.assertEqual(json.loads(data["messages"][1]["content"]), data["payload"])
                self.assertEqual(data["messages_sha256"], paired.json_sha256(data["messages"]))
                self.assertEqual(data["payload_sha256"], paired.json_sha256(data["payload"]))
                wire = paired.canonical_json(data["messages"])
                self.assertNotIn("private_mapping", wire)
                self.assertNotIn("care_knowledge_context", wire)
                self.assertNotIn("alias_seed", wire)
            wire = paired.canonical_json(material["arms"]["anonymous"]["messages"]).lower()
            self.assertFalse([term for term in paired.EXPLICIT_LABEL_DENYLIST if term in wire])

    def test_decoding_is_exact_for_both_arms_all_pairs_and_serialized_material(self):
        for material in self.materials:
            material = json.loads(json.dumps(material))
            for arm in ("labeled", "anonymous"):
                content = json.dumps(self.library_for(material, arm), allow_nan=False)
                self.assertEqual(paired.validate_and_decode(content, arm, material), self.library)
            content = json.dumps(self.library_for(material, "anonymous"))
            self.assertEqual(paired.validate_and_decode(content, "coded", material), self.library)

    def test_existing_executor_truth_and_features_match_all_2592_states(self):
        catalog = self.materials[0]["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"]
        fields = list(catalog)
        states = [dict(zip(fields, values)) for values in itertools.product(*(catalog[f] for f in fields))]
        self.assertEqual(len(states), 2592)
        # Include a conjunction to exercise the existing multi-condition grammar.
        library = copy.deepcopy(self.library)
        library["skills"][0]["rules"][0]["conditions"]["base_tpsa_bin"] = "tpsa_low"
        originals = semantic.normalize_skills(library, catalog, max_skills=5)
        self.assertEqual([len(s.rules) for s in originals], [2, 1, 1, 1, 1])
        comparisons = 0
        for material in self.materials:
            mapping = material["private_mapping"]["forward"]
            coded = copy.deepcopy(library)
            for skill in coded["skills"]:
                for rule in skill["rules"]:
                    rule["conditions"] = {mapping["fields"][f]: mapping["values"][v]
                                          for f, v in rule["conditions"].items()}
            coded_catalog = material["arms"]["anonymous"]["payload"]["target"]["public_semantic_fields"]
            anonymous = semantic.normalize_skills(coded, coded_catalog, max_skills=5)
            self.assertEqual(paired.validate_and_decode(json.dumps(coded), "anonymous", material), library)
            for state in states:
                coded_state = {mapping["fields"][f]: mapping["values"][v] for f, v in state.items()}
                candidate = SimpleNamespace(metadata=state, numeric_features=(0.2, 0.5, 0.8))
                coded_candidate = SimpleNamespace(metadata=coded_state, numeric_features=(0.2, 0.5, 0.8))
                for original, changed in zip(originals, anonymous):
                    self.assertEqual(semantic.feature_vector(original, candidate),
                                     semantic.feature_vector(changed, coded_candidate))
                    for rule, coded_rule in zip(original.rules, changed.rules):
                        self.assertEqual(semantic.rule_matches(rule, candidate),
                                         semantic.rule_matches(coded_rule, coded_candidate))
                        comparisons += 1
        self.assertEqual(comparisons, 155520)

    def test_nine_invalid_output_classes_fail_without_repair_in_both_arms(self):
        cases = ("too_few", "wrong_rule_count", "unknown_field", "nonfinite", "bad_condition",
                 "duplicate_id", "nonstring_id", "extra_field", "overlong_text")
        material = self.materials[0]
        for arm in ("labeled", "anonymous"):
            for case in cases:
                with self.subTest(arm=arm, case=case):
                    bad = self.library_for(material, arm)
                    skill = bad["skills"][0]
                    if case == "too_few":
                        bad["skills"].pop()
                    elif case == "wrong_rule_count":
                        skill["rules"].pop()
                    elif case == "unknown_field":
                        skill["rules"][0]["conditions"]["unknown_field"] = "bad"
                    elif case == "nonfinite":
                        skill["prior_scale"] = float("nan")
                    elif case == "bad_condition":
                        skill["rules"][0]["conditions"] = {}
                    elif case == "duplicate_id":
                        bad["skills"][1]["skill_id"] = skill["skill_id"]
                    elif case == "nonstring_id":
                        skill["skill_id"] = []
                    elif case == "extra_field":
                        skill["extra"] = "must not be discarded"
                    elif case == "overlong_text":
                        skill["hypothesis"] = "x" * 801
                    before = json.dumps(bad)
                    with self.assertRaises(paired.ValidationError):
                        paired.validate_and_decode(before, arm, material)
                    self.assertEqual(json.dumps(bad), before)

    def test_raw_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        invalid = [
            '{"skills":[],"skills":[]}', '{"skills":[],"\\u0073kills":[]}',
            '{"nested":{"a":1,"a":2}}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}',
            '{"x":1e999}', '{"x":-1e999}', '```json\n{}\n```', '{} trailing',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(paired.ValidationError):
                paired.parse_strict_json(text)
        # Duplicated condition keys must be rejected even when the values agree.
        content = json.dumps(self.library)
        rule = self.library["skills"][0]["rules"][0]
        field, value = next(iter(rule["conditions"].items()))
        token = json.dumps(field) + ": " + json.dumps(value)
        bad = content.replace(token, token + ", " + token, 1)
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(bad, "labeled", self.materials[0])

    def test_malformed_and_out_of_range_outputs_never_escape_as_type_errors(self):
        material = self.materials[0]
        mutations = [
            ("skill_id", {}), ("skill_id", True), ("skill_id", "UPPER"),
            ("prior_scale", True), ("prior_scale", "0.25"), ("prior_scale", -0.1),
            ("prior_scale", 10 ** 1000), ("rules", None), ("hypothesis", []),
        ]
        for key, value in mutations:
            with self.subTest(key=key, kind=type(value).__name__):
                bad = copy.deepcopy(self.library)
                bad["skills"][0][key] = value
                with self.assertRaises(paired.ValidationError):
                    paired.validate_and_decode(json.dumps(bad), "labeled", material)
        for value in (None, [], {}, True, "response", {"skills": [None] * 5}):
            with self.subTest(shape=type(value).__name__), self.assertRaises(paired.ValidationError):
                paired.validate_and_decode(json.dumps(value), "labeled", material)
        bad = copy.deepcopy(self.library)
        del bad["skills"][0]["ridge"]
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(bad), "labeled", material)

    def test_rules_reject_illegal_pairs_repeated_ids_and_wrong_shapes(self):
        material = self.materials[0]
        for conditions in ([], [["ligand_mw_bin", "mw_high"]], {"ligand_mw_bin": "bromide"},
                           {"ligand_mw_bin": {}}, {"ligand_mw_bin": True}):
            bad = copy.deepcopy(self.library)
            bad["skills"][0]["rules"][0]["conditions"] = conditions
            with self.subTest(conditions=conditions), self.assertRaises(paired.ValidationError):
                paired.validate_and_decode(json.dumps(bad), "labeled", material)
        for key, value in (("rule_id", []), ("weight", True), ("weight", 1.001),
                           ("rationale", "x" * 401)):
            bad = copy.deepcopy(self.library)
            bad["skills"][0]["rules"][0][key] = value
            with self.subTest(key=key), self.assertRaises(paired.ValidationError):
                paired.validate_and_decode(json.dumps(bad), "labeled", material)
        bad = copy.deepcopy(self.library)
        bad["skills"][0]["rules"][1]["rule_id"] = bad["skills"][0]["rules"][0]["rule_id"]
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(bad), "labeled", material)
        # An original label in anonymous output cannot be silently accepted.
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(self.library), "anonymous", material)

    def test_invalid_arguments_and_tampered_material_are_rejected(self):
        for index, seed in ((-1, 1), (10, 1), (True, 1), (0, True), (0, -1), (0, "seed")):
            with self.subTest(index=index, seed=seed), self.assertRaises(ValueError):
                paired.build_pair(index, seed)
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(self.library), "other", self.materials[0])
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(self.library, "labeled", self.materials[0])
        bad = copy.deepcopy(self.materials[0])
        bad["arms"]["anonymous"]["payload"]["target"]["candidate_count"] += 1
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(self.library), "labeled", bad)
        with self.assertRaises(paired.ValidationError):
            paired.validate_and_decode(json.dumps(self.library), "labeled", {})


if __name__ == "__main__":
    unittest.main()
