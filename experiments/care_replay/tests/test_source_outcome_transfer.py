from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_calibrated_source_outcome_transfer as calibrated  # noqa: E402
import run_llm_kernel_skill_evolution as evolution  # noqa: E402
import run_llm_transfer_router as router  # noqa: E402
import run_surrogate_baselines as surrogate  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402
import run_transfer_ablation as transfer  # noqa: E402
import run_transfer_weighted_kernel as weighted  # noqa: E402
import upgrade_source_outcome_patch_record as upgrade  # noqa: E402


class SourceOutcomeTransferTests(unittest.TestCase):
    def test_canonical_transfer_bins_align_equivalent_vocabularies(self) -> None:
        self.assertEqual(
            router.canonical_transfer_value("ring_bin", "rings_high"),
            router.canonical_transfer_value("ring_token_bin", "ring_token_high"),
        )
        self.assertEqual(
            router.canonical_transfer_value("acid_rdkit_mw_bin", "mw_none_or_low"),
            router.canonical_transfer_value("aryl_halide_mw_bin", "mw_low"),
        )

    def test_reaction_transfer_card_map_includes_public_descriptors(self) -> None:
        role_map = transfer.descriptor_transfer_role_map_for(
            "real_chemlex_acidamine",
            "real_buchwald_hartwig",
        )
        self.assertEqual(role_map["acid"], "aryl_halide")
        self.assertEqual(role_map["acid_rdkit_mw_bin"], "aryl_halide_mw_bin")
        self.assertEqual(role_map["reagent_rdkit_tpsa_bin"], "base_tpsa_bin")

    def test_record_upgrade_preserves_llm_policy_and_adds_interactions(self) -> None:
        patch = {
            "patch_id": "frozen_llm",
            "scales": [0.0, 1.0],
            "role_multipliers": {"role": 1.3},
            "source_prior_strength": 0.8,
            "reason": "frozen",
        }
        upgraded = upgrade.upgrade_patch(patch)
        self.assertEqual(upgraded["scales"], patch["scales"])
        self.assertEqual(upgraded["role_multipliers"], patch["role_multipliers"])
        self.assertEqual(upgraded["reason"], "frozen")
        self.assertEqual(upgraded["source_interaction_strength"], 0.6)
        self.assertEqual(upgraded["source_interaction_min_support"], 5)
        self.assertTrue(upgraded["canonicalize_source_values"])

    def test_custom_target_anchor_reproduces_gp_ucb_action(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_matbench_expt_gap"]()
        task = replay.make_task(adapter, initial_observations=5, reveal_budget=2)
        seed = 17
        observed = calibrated.target_llm_initial_observations(
            adapter,
            task,
            seed,
            "gp_ucb",
            None,
        )
        observed_ids = {candidate.candidate_id for candidate in observed}
        features_by_id = {
            candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
            for candidate in adapter.candidates
        }
        scorer = calibrated.target_llm_anchor_scorer(
            adapter,
            task,
            "gp_ucb",
            None,
            gp_beta=1.5,
            gp_xi=0.01,
            numeric_length_scale=0.35,
            categorical_length_scale=3.0,
            gp_noise=0.05,
        )
        scores, _diagnostics = scorer(
            observed_ids,
            observed,
            features_by_id,
            0,
        )
        weights = (1.0,) * (len(adapter.decision_columns) + 1)
        _metrics, audit = weighted.run_policy(
            adapter,
            task,
            seed,
            "gp_ucb",
            weights,
            {"scale": 0.0, "weights": list(weights)},
            1.5,
            0.35,
            3.0,
            0.05,
        )
        self.assertEqual(
            replay.top_candidate(scores),
            audit[0]["selected_candidate"],
        )

    def test_material_source_interactions_produce_target_prior(self) -> None:
        source = replay.DATASET_BUILDERS["real_matbench_expt_gap"]()
        target = replay.DATASET_BUILDERS["real_matbench_dielectric"]()
        observed = transfer.source_observations(source, 0, 512)
        card = transfer.compile_transfer_card(
            source,
            target,
            observed,
            transfer.role_map_for(source.dataset_id, target.dataset_id),
            0.65,
            3,
        )
        patch = evolution.KernelSkillPatch(
            patch_id="test_interaction",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.7,
            reason="test",
            source_interaction_strength=1.0,
            source_interaction_min_support=4,
            canonicalize_source_values=True,
        )
        prior, diagnostics = router.source_interaction_prior(
            observed,
            target,
            card,
            patch,
        )
        self.assertTrue(diagnostics["active"])
        self.assertGreater(diagnostics["interaction_count"], 0)
        self.assertEqual(len(prior), len(target.candidates))
        self.assertGreater(len({round(value, 6) for value in prior.values()}), 1)

    def test_aligned_prior_uses_shared_numeric_descriptors_within_family(self) -> None:
        source = replay.DATASET_BUILDERS["real_moleculenet_esol"]()
        target = replay.DATASET_BUILDERS["real_moleculenet_lipophilicity"]()
        observed = transfer.source_observations(source, 0, 128)
        card = transfer.compile_transfer_card(
            source,
            target,
            observed,
            transfer.role_map_for(source.dataset_id, target.dataset_id),
            0.65,
            3,
        )
        patch = evolution.KernelSkillPatch(
            patch_id="numeric_prior",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.7,
            reason="test shared numeric descriptors",
            canonicalize_source_values=True,
        )
        prior, diagnostics = router.aligned_source_prior(
            observed,
            target,
            card,
            patch,
        )
        self.assertTrue(diagnostics["active"])
        self.assertEqual(diagnostics["numeric_descriptor_family"], "moleculenet")
        self.assertTrue(diagnostics["numeric_descriptor_active"])
        self.assertGreater(len({round(value, 6) for value in prior.values()}), 1)

        numpy_backend = surrogate.np
        if numpy_backend is not None:
            try:
                surrogate.np = None
                router._SOURCE_PRIOR_CACHE.clear()
                reference_prior, reference_diagnostics = router.aligned_source_prior(
                    observed,
                    target,
                    card,
                    patch,
                )
            finally:
                surrogate.np = numpy_backend
                router._SOURCE_PRIOR_CACHE.clear()
            self.assertEqual(reference_diagnostics["neighbor_backend"], "python_reference")
            self.assertEqual(prior.keys(), reference_prior.keys())
            for candidate_id in prior:
                self.assertAlmostEqual(prior[candidate_id], reference_prior[candidate_id])

    def test_cross_family_prior_does_not_reuse_numeric_coordinates(self) -> None:
        self.assertIsNone(
            router.shared_numeric_descriptor_family(
                "real_chemlex_acidamine",
                "real_buchwald_hartwig",
            )
        )

    def test_continuous_freesolv_adapter_preserves_order_without_clipping(self) -> None:
        adapter = replay.DATASET_BUILDERS[
            "real_moleculenet_freesolv_continuous"
        ]()
        values = [candidate.objective_value for candidate in adapter.candidates]
        self.assertLess(max(values), 100.0)
        self.assertGreater(len(set(round(value, 8) for value in values)), 450)
        ordered = sorted(
            adapter.candidates,
            key=lambda candidate: float(
                candidate.metadata["experimental_hydration_free_energy"]
            ),
        )
        self.assertGreater(ordered[0].objective_value, ordered[-1].objective_value)

    def test_source_initial_extremes_preserve_budget_and_ignore_target_outcomes(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_matbench_dielectric"]()
        matched = list(adapter.candidates[:5])
        candidate_ids = [candidate.candidate_id for candidate in adapter.candidates]
        prior = {
            candidate_id: float(index)
            for index, candidate_id in enumerate(candidate_ids)
        }
        patch = evolution.KernelSkillPatch(
            patch_id="initial_design",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.8,
            reason="test",
        )
        selected, diagnostics = router.source_informed_initial_observations(
            adapter,
            matched,
            {evolution.patch_mode(patch): prior},
            {evolution.patch_mode(patch): {}},
            {evolution.patch_mode(patch): {}},
            (patch,),
            5,
            "source_extremes",
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            [candidate.candidate_id for candidate in selected[:3]],
            [candidate.candidate_id for candidate in matched[:3]],
        )
        self.assertEqual(diagnostics["replaced_count"], 2)
        self.assertEqual(diagnostics["source_probe_ids"][0], candidate_ids[-1])
        self.assertGreater(
            diagnostics["source_probe_scores"][diagnostics["source_probe_ids"][0]],
            0.9,
        )
        self.assertLess(
            diagnostics["source_probe_scores"][diagnostics["source_probe_ids"][1]],
            -0.9,
        )
        changed_adapter = replace(
            adapter,
            candidates=[
                replace(candidate, objective_value=100.0 - candidate.objective_value)
                for candidate in adapter.candidates
            ],
        )
        changed_matched = list(changed_adapter.candidates[:5])
        changed_selected, _ = router.source_informed_initial_observations(
            changed_adapter,
            changed_matched,
            {evolution.patch_mode(patch): prior},
            {evolution.patch_mode(patch): {}},
            {evolution.patch_mode(patch): {}},
            (patch,),
            5,
            "source_extremes",
        )
        self.assertEqual(
            [candidate.candidate_id for candidate in selected],
            [candidate.candidate_id for candidate in changed_selected],
        )

    def test_source_negative_initial_design_selects_low_prior_probes(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_matbench_dielectric"]()
        candidate_ids = [candidate.candidate_id for candidate in adapter.candidates]
        prior = {
            candidate_id: float(index)
            for index, candidate_id in enumerate(candidate_ids)
        }
        patch = evolution.KernelSkillPatch(
            patch_id="negative_initial",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.8,
            reason="test",
        )
        selected, diagnostics = router.source_informed_initial_observations(
            adapter,
            list(adapter.candidates[:5]),
            {evolution.patch_mode(patch): prior},
            {evolution.patch_mode(patch): {}},
            {evolution.patch_mode(patch): {}},
            (patch,),
            5,
            "source_negative",
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(diagnostics["source_probe_ids"]), 2)
        self.assertTrue(all(
            diagnostics["source_probe_scores"][candidate_id] < -0.9
            for candidate_id in diagnostics["source_probe_ids"]
        ))

    def test_source_quantile_initial_design_avoids_absolute_tail(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_matbench_dielectric"]()
        candidate_ids = [candidate.candidate_id for candidate in adapter.candidates]
        prior = {
            candidate_id: float(index)
            for index, candidate_id in enumerate(candidate_ids)
        }
        patch = evolution.KernelSkillPatch(
            patch_id="quantile_initial",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.8,
            reason="test",
        )
        _selected, diagnostics = router.source_informed_initial_observations(
            adapter,
            [],
            {evolution.patch_mode(patch): prior},
            {evolution.patch_mode(patch): {}},
            {evolution.patch_mode(patch): {}},
            (patch,),
            2,
            "source_negative_quantile",
        )
        scores = [
            diagnostics["source_probe_scores"][candidate_id]
            for candidate_id in diagnostics["source_probe_ids"]
        ]
        self.assertTrue(all(score < 0.0 for score in scores))
        self.assertTrue(all(score > -1.0 for score in scores))

    def test_material_additive_source_prior_is_nonconstant(self) -> None:
        source = replay.DATASET_BUILDERS["real_matbench_expt_gap"]()
        target = replay.DATASET_BUILDERS["real_matbench_dielectric"]()
        observed = transfer.source_observations(source, 0, 512)
        card = transfer.compile_transfer_card(
            source,
            target,
            observed,
            transfer.role_map_for(source.dataset_id, target.dataset_id),
            0.65,
            3,
        )
        patch = evolution.KernelSkillPatch(
            patch_id="test_additive",
            scales=(0.0, 1.0),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.0,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=0.02,
            confidence=0.7,
            reason="test",
            canonicalize_source_values=True,
        )
        prior, diagnostics = router.source_additive_outcome_prior(
            observed,
            target,
            card,
            patch,
        )
        self.assertTrue(diagnostics["active"])
        self.assertEqual(len(prior), len(target.candidates))
        self.assertGreater(len({round(value, 6) for value in prior.values()}), 1)

    def test_calibrated_selector_falls_back_on_negative_transfer(self) -> None:
        rows = []
        for seed in range(10):
            for mode, final, auc in (
                ("gp_ucb", 10.0, 9.0),
                ("mixed_kernel_gp_ei", 9.0, 9.0),
                ("target_acquisition_portfolio", 9.5, 9.0),
                ("matched_target_only_llm", 10.5, 9.5),
                ("llm_transfer_router", 8.0, 8.0),
            ):
                rows.append({
                    "mode": mode,
                    "seed": seed,
                    "final_best": final,
                    "best_so_far_auc": auc,
                    "top10_hit": 0,
                })
        selected, diagnostics = calibrated.select_route(
            rows,
            set(range(10)),
            min_risk_adjusted_gain=0.25,
            min_positive_fold_rate=0.8,
            min_final_non_loss_rate=0.55,
        )
        self.assertEqual(selected, "matched_target_only_llm")
        self.assertFalse(diagnostics["selected_source_outcome_transfer"])

    def test_calibrated_selector_rejects_uncertain_positive_mean(self) -> None:
        rows = []
        deltas = [20.0, -18.0] * 5
        for seed, delta in enumerate(deltas):
            for mode, final, auc in (
                ("gp_ucb", 10.0, 10.0),
                ("mixed_kernel_gp_ei", 10.0, 10.0),
                ("target_acquisition_portfolio", 10.0, 10.0),
                ("matched_target_only_llm", 10.0, 10.0),
                ("llm_transfer_router", 10.0 + delta, 10.0 + delta),
            ):
                rows.append({
                    "mode": mode,
                    "seed": seed,
                    "final_best": final,
                    "best_so_far_auc": auc,
                    "top10_hit": 0,
                })
        selected, diagnostics = calibrated.select_route(
            rows,
            set(range(10)),
            min_risk_adjusted_gain=-100.0,
            min_positive_fold_rate=0.0,
            min_final_non_loss_rate=0.0,
            min_composite_ci_low=0.0,
        )
        self.assertEqual(selected, "matched_target_only_llm")
        comparison = diagnostics["source_outcome_diagnostics"][
            "matched_target_only_llm"
        ]
        self.assertLess(comparison["composite"]["normal_95ci_low"], 0.0)

    def test_prior_only_route_uses_target_anchor_residual_expert(self) -> None:
        source = replay.DATASET_BUILDERS["real_moleculenet_esol"]()
        target = replay.DATASET_BUILDERS["real_moleculenet_lipophilicity"]()
        source_observed = transfer.source_observations(source, 0, 128)
        card = transfer.compile_transfer_card(
            source,
            target,
            source_observed,
            transfer.role_map_for(source.dataset_id, target.dataset_id),
            0.65,
            3,
        )
        patch = evolution.KernelSkillPatch(
            patch_id="residual_expert",
            scales=(12.0,),
            role_multipliers={},
            gp_beta=1.5,
            gp_beta_end=1.5,
            source_prior_strength=1.0,
            source_similarity_temperature=0.35,
            source_neighbor_count=12,
            calibration_mode="signed",
            min_cv_gain=-1.0,
            confidence=0.8,
            reason="test source residual expert",
            source_interaction_strength=0.5,
            source_interaction_min_support=4,
            canonicalize_source_values=True,
        )
        task = replay.make_task(target, initial_observations=5, reveal_budget=1)
        initial = list(target.candidates[:5])
        _metrics, audit = router.run_router_policy(
            source,
            target,
            task,
            3,
            card,
            source_observed,
            (patch,),
            True,
            1.5,
            0.01,
            0.35,
            3.0,
            0.05,
            initial_observed=initial,
            router_min_observations=5,
            router_min_quality=0.0,
            router_max_transfer_mass=0.15,
            source_initial_strategy="matched",
        )
        diagnostics = audit[0]["hypothesis_snapshot"]["route_diagnostics"][
            evolution.patch_mode(patch)
        ]
        if diagnostics["active"] and diagnostics["source_outcome_quality"] > 0.0:
            self.assertEqual(
                diagnostics["expert_basis"],
                "target_anchor_plus_source_outcome_residual",
            )

    def test_fixed_data_only_patch_contains_no_llm_specific_role_choices(self) -> None:
        target = replay.DATASET_BUILDERS["real_matbench_expt_gap"]()
        patch = calibrated.fixed_data_only_patch(
            target,
            (0.5, 1.0, 2.0),
            1.5,
        )
        self.assertEqual(
            patch.role_multipliers,
            {field: 1.0 for field in target.decision_columns},
        )
        self.assertEqual(patch.scales, (0.0, 0.5, 1.0, 2.0))
        self.assertEqual(patch.confidence, 1.0)
        self.assertIn("Deterministic non-LLM control", patch.reason)

    def test_mechanism_attribution_detects_warmstart_only_gain(self) -> None:
        rows = []
        audits = {}
        for seed in range(5):
            for mode, final, auc in (
                ("matched_target_only_llm", 10.0, 9.0),
                (calibrated.WARMSTART_ONLY_MODE, 12.0, 11.0),
                (calibrated.DATA_ONLY_MODE, 12.5, 11.5),
                ("llm_transfer_router", 12.0, 11.0),
            ):
                rows.append({
                    "mode": mode,
                    "seed": seed,
                    "final_best": final,
                    "best_so_far_auc": auc,
                    "top10_hit": 0,
                })
            audits[("llm_transfer_router", seed)] = [{
                "hypothesis_snapshot": {
                    "source_initial_design": {
                        "source_outcome_active": True,
                        "replaced_count": 2,
                    },
                    "transfer_mass": 0.0,
                    "router_gate": {
                        "authorized": True,
                        "anchor_candidate": "a",
                        "selected_candidate": "a",
                    },
                }
            }]
        result = calibrated.mechanism_attribution(
            rows,
            audits,
            set(range(5)),
            selected_source_outcome_transfer=True,
            include_controls=True,
        )
        self.assertEqual(
            result["classification"],
            "source_informed_initial_design_only",
        )
        self.assertEqual(
            result["post_initialization_source_outcome_effect"][
                "exact_seed_match_rate"
            ],
            1.0,
        )
        self.assertEqual(
            result["full_route_action_diagnostics"][
                "source_initial_active_seed_rate"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
