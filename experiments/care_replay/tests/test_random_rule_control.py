from __future__ import annotations

import sys
import unittest
from pathlib import Path
import random


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_random_rule_control as null_control  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


class RandomRuleControlTest(unittest.TestCase):
    def test_random_rule_preserves_shape_and_public_catalog(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_moleculenet_esol"]()
        catalog = semantic.semantic_field_catalog(adapter)
        skill = semantic.SemanticSkill(
            skill_id="source",
            rules=(
                semantic.SemanticRule(
                    "r1",
                    (("ring_bin", "ring_high"),),
                    0.8,
                    "test",
                ),
                semantic.SemanticRule(
                    "r2",
                    (("hbond_donor_bin", "hbd_low"),),
                    -0.4,
                    "test",
                ),
            ),
            ridge=2.0,
            prior_scale=0.3,
            semantic_mass_start=0.3,
            semantic_mass_end=0.1,
            ucb_weight=0.5,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        randomized = null_control.matched_random_skill(
            skill,
            catalog,
            random.Random(9),
            "random",
        )
        self.assertEqual(len(randomized.rules), len(skill.rules))
        self.assertEqual(
            [tuple(field for field, _ in rule.conditions) for rule in randomized.rules],
            [tuple(field for field, _ in rule.conditions) for rule in skill.rules],
        )
        for rule in randomized.rules:
            field, value = rule.conditions[0]
            self.assertIn(value, catalog[field])
        self.assertEqual(randomized.ridge, skill.ridge)
        self.assertEqual(randomized.semantic_mass_start, skill.semantic_mass_start)

    def test_random_rule_does_not_read_objective(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_moleculenet_esol"]()
        catalog = semantic.semantic_field_catalog(adapter)
        skill = semantic.SemanticSkill(
            skill_id="source",
            rules=(semantic.SemanticRule("r", (("ring_bin", "ring_high"),), 0.8, "test"),),
            ridge=1.0,
            prior_scale=0.3,
            semantic_mass_start=0.3,
            semantic_mass_end=0.1,
            ucb_weight=0.5,
            gp_beta_start=1.5,
            gp_beta_end=1.0,
            gp_xi=0.01,
            confidence=0.7,
            hypothesis="test",
        )
        left = null_control.matched_random_skill(skill, catalog, random.Random(1), "a")
        right = null_control.matched_random_skill(skill, catalog, random.Random(1), "b")
        self.assertEqual(left.rules, right.rules)


if __name__ == "__main__":
    unittest.main()
