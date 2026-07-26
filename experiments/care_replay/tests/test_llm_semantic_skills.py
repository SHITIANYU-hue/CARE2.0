from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import generate_llm_semantic_skills as generate  # noqa: E402
import run_calibrated_llm_semantic_selector as calibrated  # noqa: E402
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
        self.assertIn("acid_rdkit_acid_functional_class", catalog)
        self.assertIn("amine_rdkit_amine_functional_class", catalog)
        self.assertIn("acid_rdkit_ring_system_class", catalog)
        self.assertNotIn(adapter.hidden_target, catalog)

    def test_chemlex_uses_rdkit_numeric_reaction_representation(self) -> None:
        adapter = replay.real_chemlex_acidamine_adapter()
        candidate = adapter.candidates[0]
        self.assertEqual(len(candidate.numeric_features), 24)
        self.assertTrue(any(value > 0.0 for value in candidate.numeric_features))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in candidate.numeric_features))

    def test_expanded_skill_variants_include_target_calibrated_forms(self) -> None:
        skill = semantic.SemanticSkill(
            skill_id="base",
            rules=(semantic.SemanticRule("r", (("family", "oxide"),), 0.8, "test"),),
            ridge=1.0,
            prior_scale=0.5,
            semantic_mass_start=0.4,
            semantic_mass_end=0.2,
            ucb_weight=0.5,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        variants = calibrated.expand_skill_variants((skill,), "expanded")
        by_id = {item.skill_id: item for item in variants}
        self.assertEqual(by_id["base_noprior"].prior_scale, 0.0)
        self.assertLessEqual(by_id["base_conservative"].semantic_mass_start, 0.30)

    def test_direct_llm_prior_does_not_use_candidate_outcome(self) -> None:
        skill = semantic.SemanticSkill(
            skill_id="base",
            rules=(semantic.SemanticRule("r", (("family", "oxide"),), 0.8, "test"),),
            ridge=1.0,
            prior_scale=0.5,
            semantic_mass_start=0.4,
            semantic_mass_end=0.2,
            ucb_weight=0.5,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        left = replay.Candidate("a", "g", 0.0, 0.0, 0.0, -999.0, {"family": "oxide"})
        right = replace(left, objective_value=999.0)
        self.assertEqual(
            semantic.fixed_rule_score(skill, left),
            semantic.fixed_rule_score(skill, right),
        )

    def test_strategy_router_selects_stable_calibration_gain(self) -> None:
        rows = []
        for seed in range(10):
            rows.extend([
                {
                    "mode": "gp_ucb",
                    "seed": seed,
                    "final_best": 10.0,
                    "best_so_far_auc": 8.0,
                    "top10_hit": 0,
                },
                {
                    "mode": "llm_semantic_stable",
                    "seed": seed,
                    "final_best": 11.0,
                    "best_so_far_auc": 10.0,
                    "top10_hit": 1,
                },
                {
                    "mode": "llm_direct_prior_unstable",
                    "seed": seed,
                    "final_best": 9.0 if seed % 2 else 12.0,
                    "best_so_far_auc": 7.0 if seed % 2 else 10.0,
                    "top10_hit": 0,
                },
            ])
        selected, route = calibrated.select_strategy_route(
            rows,
            set(range(10)),
            (
                "gp_ucb",
                "llm_semantic_stable",
                "llm_direct_prior_unstable",
            ),
            "gp_ucb",
            0.25,
            0.8,
        )
        self.assertEqual(selected, "llm_semantic_stable")
        self.assertTrue(route["selected_llm_strategy"])
        self.assertFalse(
            route["diagnostics"]["llm_direct_prior_unstable"]["eligible"]
        )

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

    def test_hypothesis_compiler_freezes_execution_parameters(self) -> None:
        skills, compilation = semantic.compile_hypothesis_skills(
            {
                "hypotheses": [
                    {
                        "hypothesis_id": "acid_activation",
                        "claim": "A compatible acid motif should improve coupling.",
                        "mechanism": "The public functional group is a proxy for activation.",
                        "conditions": {"family": "oxide"},
                        "expected_direction": "positive",
                        "failure_conditions": ["Sparse support can make the proxy unreliable."],
                        "confidence": 0.8,
                    },
                    {
                        "hypothesis_id": "hidden_leak",
                        "claim": "This must not survive.",
                        "mechanism": "It uses a private field.",
                        "conditions": {"hidden_target": "high"},
                        "expected_direction": "positive",
                    },
                ]
            },
            {"family": {"oxide": 10, "halide": 5}},
        )
        self.assertEqual(len(skills), 1)
        self.assertTrue(compilation["fixed_execution_parameters"])
        self.assertEqual(compilation["accepted_hypotheses"], 1)
        self.assertEqual(compilation["rejected_hypotheses"], 1)
        skill = skills[0]
        self.assertEqual(skill.rules[0].weight, 0.5)
        self.assertEqual(skill.ridge, 2.0)
        self.assertEqual(skill.prior_scale, 0.25)
        self.assertEqual(skill.semantic_mass_start, 0.20)
        self.assertIn("Mechanism:", skill.hypothesis)

    def test_hypothesis_prompt_does_not_offer_llm_parameters(self) -> None:
        payload = generate.build_prompt_payload(
            "real_suzuki_miyaura",
            "real_buchwald_hartwig",
            64,
            0.65,
            3,
            "source_schema_only",
            proposal_mode="hypothesis_only",
        )
        self.assertEqual(payload["proposal_mode"], "hypothesis_only")
        self.assertIn("hypotheses", payload["output_contract"])
        self.assertNotIn("prior_scale", payload["output_contract"]["hypotheses"][0])
        self.assertNotIn("prior_scale", payload["allowed_ranges"])

    def test_prompt_contract_uses_real_catalog_field_placeholders(self) -> None:
        payload = generate.build_prompt_payload(
            "real_suzuki_miyaura",
            "real_buchwald_hartwig",
            512,
            0.65,
            8,
            "target_only",
        )
        conditions = payload["output_contract"]["skills"][0]["rules"][0]["conditions"]
        self.assertEqual(
            conditions,
            {"field_name_from_public_semantic_fields": "exact_catalog_value"},
        )
        self.assertEqual(
            payload["output_contract"]["skills"][0]["skill_id"],
            "descriptive_scientific_skill_id",
        )


if __name__ == "__main__":
    unittest.main()
