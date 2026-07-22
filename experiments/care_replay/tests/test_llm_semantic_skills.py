from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import generate_llm_semantic_skills as generate  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


class LlmSemanticSkillsTest(unittest.TestCase):
    def test_normalizer_rejects_unknown_fields_and_values(self) -> None:
        skills = semantic.normalize_skills(
            {
                "skills": [{
                    "skill_id": "chemistry",
                    "rules": [
                        {"rule_id": "valid", "conditions": {"family": "oxide"}, "weight": 0.8},
                        {"rule_id": "bad-field", "conditions": {"hidden_target": "high"}, "weight": 1.0},
                        {"rule_id": "bad-value", "conditions": {"family": "secret"}, "weight": 1.0},
                        {"rule_id": "interaction", "conditions": {"family": "oxide", "metal": "no"}, "weight": 0.5},
                        {"rule_id": "compact", "conditions": {"exact_field": "family.halide"}, "weight": 0.4},
                    ],
                }]
            },
            {"family": {"oxide": 10, "halide": 5}, "metal": {"yes": 8, "no": 7}},
        )
        self.assertEqual(len(skills), 1)
        self.assertEqual([rule.rule_id for rule in skills[0].rules], ["valid", "interaction", "compact"])

    def test_rule_features_are_public_condition_matches(self) -> None:
        skill = semantic.SemanticSkill(
            skill_id="test",
            rules=(semantic.SemanticRule("oxide", (("family", "oxide"),), 0.8, "test"),),
            ridge=1.0,
            prior_scale=0.2,
            semantic_mass_start=0.3,
            semantic_mass_end=0.1,
            ucb_weight=0.5,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        candidate = replay.Candidate("c", "g", 0.2, 0.4, 0.6, 99.0, {"family": "oxide"})
        vector = semantic.feature_vector(skill, candidate)
        self.assertEqual(vector[-1], 1.0)
        candidate.metadata["family"] = "halide"
        self.assertEqual(semantic.feature_vector(skill, candidate)[-1], 0.0)

    def test_semantic_model_never_reads_unobserved_outcomes(self) -> None:
        skill = semantic.SemanticSkill(
            skill_id="test",
            rules=(semantic.SemanticRule("a", (("role", "a"),), 0.5, "test"),),
            ridge=1.0,
            prior_scale=0.2,
            semantic_mass_start=0.3,
            semantic_mass_end=0.1,
            ucb_weight=1.0,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        candidates = tuple(
            replay.Candidate(f"c{i}", "g", i / 10, 0.0, 0.0, float(i * 10), {"role": "a" if i % 2 else "b"})
            for i in range(7)
        )
        adapter = replay.DatasetAdapter("target", "target", "maximize", ("role",), "value", "group", (), "", candidates)
        observed = list(candidates[:5])
        scores_a, _ = semantic.semantic_model_scores(skill, adapter, observed, {c.candidate_id for c in observed}, 1.0)
        changed = (*candidates[:5], replace(candidates[5], objective_value=-9999.0), replace(candidates[6], objective_value=9999.0))
        changed_adapter = replace(adapter, candidates=changed)
        scores_b, _ = semantic.semantic_model_scores(skill, changed_adapter, observed, {c.candidate_id for c in observed}, 1.0)
        self.assertEqual(scores_a, scores_b)

    def test_chemlex_catalog_uses_public_smiles_descriptors(self) -> None:
        adapter = replay.real_chemlex_acidamine_adapter()
        catalog = semantic.semantic_field_catalog(adapter)
        self.assertIn("acid_smiles_length_bin", catalog)
        self.assertIn("amine_hetero_atom_bin", catalog)
        self.assertIn("acid_carbonyl_bin", catalog)
        self.assertIn("amine_nitrogen_bin", catalog)
        self.assertIn("reagent_coupling_family", catalog)
        self.assertNotIn(adapter.hidden_target, catalog)

    def test_chemlex_reagent_family_is_derived_from_public_smiles(self) -> None:
        self.assertEqual(
            replay.chemlex_reagent_family("CCN=C=NCCCN(C)C.Cl"),
            "carbodiimide_coupling",
        )
        self.assertEqual(
            replay.chemlex_reagent_family("CN(C)C(On1nnc2cccnc21)=[N+](C)C"),
            "aza_benzotriazole_uronium",
        )

    def test_source_schema_ablation_withholds_outcome_statistics(self) -> None:
        payload = generate.source_evidence_payload(
            "real_suzuki_miyaura",
            "real_buchwald_hartwig",
            512,
            0.65,
            "source_schema_only",
        )
        self.assertEqual(payload["evidence_mode"], "source_schema_only")
        self.assertTrue(payload["mapped_roles"])
        self.assertIsNone(payload["source_outcome_statistics"])
        self.assertEqual(payload["shared_value_priors"], [])
        self.assertNotIn("source_mean_abs_effect", str(payload))

    def test_target_only_ablation_withholds_source_identity(self) -> None:
        payload = generate.source_evidence_payload(
            "real_suzuki_miyaura",
            "real_buchwald_hartwig",
            512,
            0.65,
            "target_only",
        )
        self.assertEqual(payload["evidence_mode"], "target_only")
        self.assertIsNone(payload["source_dataset"])
        self.assertEqual(payload["mapped_roles"], [])
        self.assertEqual(payload["shared_value_priors"], [])


if __name__ == "__main__":
    unittest.main()
