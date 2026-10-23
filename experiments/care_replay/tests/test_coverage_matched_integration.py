from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_matched_rule_run as audit
import run_matched_random_rule_controls as study


class CoverageIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.skill = study.semantic.SemanticSkill(
            skill_id="test", rules=(study.semantic.SemanticRule("r", (("a", "rare"),), 0.8, ""),),
            ridge=2.0, prior_scale=0.3, semantic_mass_start=0.3, semantic_mass_end=0.1,
            ucb_weight=0.5, gp_beta_start=1.5, gp_beta_end=1.0, gp_xi=0.01,
            confidence=0.7, hypothesis="test")
        self.candidates = [SimpleNamespace(candidate_id=str(i),
                                           metadata={"a": "rare" if i == 0 else "common"})
                           for i in range(10)]
        self.catalog = {"a": {"rare": 1, "common": 9}}
        self.config = {"null_replicates": 3, "random_seed": 71, "field_types": {"a": "binary"},
                       "coverage_matching": {"absolute_fraction_tolerance": 0.005,
                           "match_ordered_rule_support": True,
                           "match_joint_activation_histogram": True,
                           "sampling": "original_null_conditioned_on_public_coverage",
                           "identity_and_duplicates": "allowed"}}

    def test_generation_reproduction_and_diagnostic_tamper(self):
        diagnostics = {}
        banks = study.make_libraries((self.skill,), self.catalog, self.config,
                                     self.candidates, diagnostics)
        for bank in banks.values():
            self.assertEqual(len(bank), 1)
            self.assertEqual(bank[0].rules[0].conditions, self.skill.rules[0].conditions)
        manifest = {"libraries": {name: [asdict(skill) for skill in bank] for name, bank in banks.items()},
                    "config": self.config, "public_catalog": self.catalog,
                    "coverage_matching_diagnostics": diagnostics}
        self.assertEqual(audit.restore_libraries(manifest, study, self.candidates), banks)
        tampered = deepcopy(manifest)
        tampered["coverage_matching_diagnostics"]["r1"] = {}
        with self.assertRaisesRegex(audit.AuditError, "Coverage conditioning"):
            audit.restore_libraries(tampered, study, self.candidates)

    def test_coverage_requires_public_metadata_and_exact_protocol(self):
        with self.assertRaisesRegex(ValueError, "public candidate"):
            study.make_libraries((self.skill,), self.catalog, self.config)
        config = deepcopy(self.config)
        config["coverage_matching"]["match_ordered_rule_support"] = 1
        with self.assertRaisesRegex(ValueError, "Unsupported coverage"):
            study.make_libraries((self.skill,), self.catalog, config, self.candidates)

    def test_new_ranges_are_disjoint_and_prior_overlap_rejected(self):
        import json
        path = Path(__file__).resolve().parents[1] / "configs/bh_coverage_matched_rule_controls_v1.json"
        config = json.loads(path.read_text())
        cal, held = study.validate_config(config)
        self.assertEqual((len(cal), len(held)), (50, 100))
        for seed in (202610181000, 202610182000, 202610183000, 202610184000):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "prior experiment"):
                study.validate_config({**config, "heldout_seed_start": seed})


if __name__ == "__main__":
    unittest.main()
