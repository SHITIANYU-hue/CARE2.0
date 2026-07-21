from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_calibrated_llm_prior_selector as prior_selector  # noqa: E402
import run_llm_kernel_skill_evolution as evolution  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


class LlmPriorExecutionTest(unittest.TestCase):
    def test_reaction_prior_uses_semantic_descriptor_roles(self) -> None:
        role_map = prior_selector.prior_role_map_for(
            "real_suzuki_miyaura",
            "real_buchwald_hartwig",
        )

        self.assertNotIn("ligand", role_map)
        self.assertEqual(
            role_map["reactant_1_halide_type"],
            "aryl_halide_halide_type",
        )
        self.assertEqual(
            role_map["reagent_reagent_base_family"],
            "base_reagent_base_family",
        )

    def test_unaligned_reaction_pair_disables_source_prior(self) -> None:
        self.assertEqual(
            prior_selector.prior_role_map_for(
                "real_suzuki_miyaura",
                "real_chemlex_acidamine",
            ),
            {},
        )

    def test_exact_molecule_identity_overrides_descriptor_neighbor_prior(self) -> None:
        source = [
            replay.Candidate(
                candidate_id="source-low",
                group="g",
                x1=0.0,
                x2=0.0,
                x3=0.0,
                objective_value=10.0,
                metadata={"smiles": "CC", "role": "same"},
            ),
            replay.Candidate(
                candidate_id="source-high",
                group="g",
                x1=1.0,
                x2=0.0,
                x3=0.0,
                objective_value=90.0,
                metadata={"smiles": "CO", "role": "same"},
            ),
        ]
        target = replay.DatasetAdapter(
            dataset_id="real_moleculenet_freesolv",
            title="target",
            objective="maximize",
            decision_columns=("role",),
            hidden_target="value",
            group_column="group",
            preferred_groups=(),
            failure_note="",
            candidates=(
                replay.Candidate("matched", "g", 0.0, 0.0, 0.0, 0.0, {"smiles": "CO", "role": "same"}),
                replay.Candidate("unmatched", "g", 0.0, 0.0, 0.0, 0.0, {"smiles": "CN", "role": "same"}),
            ),
        )
        patch = evolution.KernelSkillPatch(
            patch_id="identity",
            scales=(0.0,),
            role_multipliers={"role": 1.0},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.4,
            source_neighbor_count=2,
            calibration_mode="signed",
            min_cv_gain=0.0,
            confidence=0.7,
            reason="test",
        )
        card = SimpleNamespace(
            source_dataset="real_moleculenet_esol",
            target_dataset="real_moleculenet_freesolv",
            roles=(),
        )

        prior, diagnostics = prior_selector.router.aligned_source_prior(
            source,
            target,
            card,  # type: ignore[arg-type]
            patch,
        )

        self.assertGreater(prior["matched"], 0.0)
        self.assertEqual(prior["unmatched"], 0.0)
        self.assertEqual(diagnostics["exact_match_count"], 1)

    def test_positive_only_patch_can_cold_start_exact_identity_prior(self) -> None:
        patch = evolution.KernelSkillPatch(
            patch_id="cold-start",
            scales=(0.0,),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.4,
            source_neighbor_count=2,
            calibration_mode="positive_only",
            min_cv_gain=0.1,
            confidence=0.75,
            reason="test",
        )
        observed = [
            replay.Candidate("seen", "g", 0.0, 0.0, 0.0, 50.0, {})
        ]

        adjustments, diagnostics = (
            prior_selector.router.exact_identity_cold_start_adjustments(
                {"seen": 0.0, "high": 1.5, "low": -1.5},
                {"seen", "high", "low"},
                observed,
                patch,
            )
        )

        self.assertGreater(adjustments["high"], 0.0)
        self.assertLess(adjustments["low"], 0.0)
        self.assertNotIn("seen", adjustments)
        self.assertTrue(diagnostics["active"])

    def test_patch_source_prior_is_calibrated_and_audited(self) -> None:
        candidates = tuple(
            replay.Candidate(
                candidate_id=f"c{index}",
                group="g",
                x1=index / 10.0,
                x2=0.0,
                x3=0.0,
                objective_value=float(index * 10),
                metadata={"role": f"v{index % 2}"},
            )
            for index in range(7)
        )
        adapter = replay.DatasetAdapter(
            dataset_id="target",
            title="target",
            objective="maximize",
            decision_columns=("role",),
            hidden_target="value",
            group_column="group",
            preferred_groups=(),
            failure_note="",
            candidates=candidates,
        )
        task = replay.TaskSpec(
            dataset_id="target",
            objective="maximize",
            decision_columns=("role",),
            hidden_target="value",
            initial_observations=5,
            reveal_budget=1,
            oracle_value=60.0,
        )
        patch = evolution.KernelSkillPatch(
            patch_id="prior_patch",
            scales=(0.0, 1.0),
            role_multipliers={"role": 1.0},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.2,
            source_similarity_temperature=0.4,
            source_neighbor_count=3,
            calibration_mode="signed",
            min_cv_gain=0.0,
            confidence=0.7,
            reason="test",
        )

        def fake_scores(
            _adapter: object,
            observed_ids: set[str],
            *_args: object,
            **_kwargs: object,
        ) -> tuple[dict[str, float], float, dict[str, object]]:
            scores = {
                candidate.candidate_id: float(index)
                for index, candidate in enumerate(candidates)
                if candidate.candidate_id not in observed_ids
            }
            return scores, 0.0, {"fake": True}

        with mock.patch.object(
            prior_selector.surrogate,
            "candidate_features",
            return_value=object(),
        ), mock.patch.object(
            prior_selector.bma,
            "score_scale_ensemble",
            side_effect=fake_scores,
        ), mock.patch.object(
                prior_selector.router,
                "aligned_source_prior",
                return_value=(
                    {candidate.candidate_id: 0.1 for candidate in candidates},
                    {"active": True},
                ),
            ) as aligned, mock.patch.object(
                prior_selector.router,
                "calibrated_prior_adjustments",
                return_value=(
                    {candidate.candidate_id: 0.05 for candidate in candidates},
                    {"active": True, "cv_gain": 0.2},
                ),
            ) as calibrated:
            _metrics, audit = prior_selector.run_prior_patch_policy(
                adapter,
                adapter,
                task,
                seed=0,
                source_observed=list(candidates[:5]),
                card=object(),  # type: ignore[arg-type]
                patch=patch,
                normalize=True,
                numeric_length_scale=0.35,
                categorical_length_scale=3.0,
                gp_noise=0.05,
            )

        aligned.assert_called_once()
        calibrated.assert_called_once()
        self.assertTrue(audit[0]["hypothesis_snapshot"]["prior_applied"])
        self.assertEqual(
            audit[0]["hypothesis_snapshot"]["llm_patch"]["source_prior_strength"],
            1.2,
        )


if __name__ == "__main__":
    unittest.main()
