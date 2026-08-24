from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_classical_transfer_baselines as classical  # noqa: E402
import calibrate_multisource_skill as calibration  # noqa: E402
import run_multisource_transfer_baselines as multisource  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402
import run_transfer_ablation as transfer  # noqa: E402


class MultiSourceTransferTests(unittest.TestCase):
    def test_frozen_protocol_is_task_disjoint(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "multisource_new_task_benchmark_v1.json").read_text(
                encoding="utf-8"
            )
        )
        multisource.validate_new_task_protocol(config)
        config["protocol"]["development_task_ids"].append(
            "real_reizman_suzuki_case_4"
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            multisource.validate_new_task_protocol(config)

    def test_rgpe_weights_cover_all_sources_and_target(self) -> None:
        source_adapters = [
            replay.real_reizman_suzuki_case_1_adapter(),
            replay.real_reizman_suzuki_case_2_adapter(),
        ]
        target = replay.real_reizman_suzuki_case_4_adapter()
        posteriors = []
        for adapter in source_adapters:
            observed = transfer.source_observations(adapter, 0, 20)
            posteriors.append(
                classical.build_source_posterior(
                    adapter,
                    target,
                    observed,
                    20,
                    0.35,
                    3.0,
                    0.05,
                )
            )
        target_features = classical.feature_arrays(target, target.candidates)
        observed_indices = [0, 1, 2]
        observed_y, _center, _scale = classical.normalized_outcomes(
            [target.candidates[index] for index in observed_indices]
        )
        _mean, _variance, diagnostics = classical.target_gp_posterior(
            target_features,
            observed_indices,
            observed_y,
            0.35,
            3.0,
            0.05,
        )
        weights, weight_diagnostics = multisource.multisource_rgpe_weights(
            posteriors,
            target_features,
            observed_indices,
            observed_y,
            diagnostics,
            16,
            7,
            0.35,
            3.0,
        )
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertEqual(len(weight_diagnostics["source_weights"]), 2)

    def test_one_seed_runs_without_target_calibration(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "multisource_new_task_benchmark_v1.json").read_text(
                encoding="utf-8"
            )
        )
        config["protocol"]["evaluation_seed_count"] = 1
        config["protocol"]["rgpe_draws"] = 8
        config["experiments"][0]["source_observations"] = [20, 20, 20]
        config["experiments"][0]["reveal_rounds"] = 2
        selection = {
            "config_fingerprint": multisource.config_fingerprint(config),
            "selected_skill_prior": {"mass_start": 0.25, "mass_end": 0.0},
            "selected_route": {"mode": "target_gp_ucb"},
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = multisource.run_experiment(
                config,
                config["experiments"][0],
                Path(directory),
                selection,
            )
        self.assertFalse(summary["target_task_calibration"])
        self.assertEqual(set(summary["aggregate"]), set(multisource.MODES))
        self.assertEqual(summary["comparisons"]["multisource_rgpe"]["seed_count"], 1)

    def test_fixed_initial_indices_are_used_verbatim(self) -> None:
        sources = [
            replay.real_reizman_suzuki_case_1_adapter(),
            replay.real_reizman_suzuki_case_2_adapter(),
        ]
        target = replay.real_reizman_suzuki_case_4_adapter()
        posteriors = []
        for source in sources:
            observed = transfer.source_observations(source, 0, 20)
            posteriors.append(
                classical.build_source_posterior(
                    source,
                    target,
                    observed,
                    20,
                    0.35,
                    3.0,
                    0.05,
                )
            )
        _metrics, audit = multisource.run_seed(
            sources,
            target,
            posteriors,
            seed=91,
            initial=3,
            rounds=1,
            mode="target_gp_ucb",
            gp_beta=1.5,
            numeric_length_scale=0.35,
            categorical_length_scale=3.0,
            gp_noise=0.05,
            rgpe_draws=8,
            rho_grid=(0.0,),
            bma_temperature=1.0,
            fixed_initial_indices=[1, 5, 8],
        )
        self.assertEqual(
            audit[0]["initial_candidate_ids"],
            [
                "reizman_suzuki_case4_001",
                "reizman_suzuki_case4_005",
                "reizman_suzuki_case4_008",
            ],
        )

    def test_skill_prior_is_selected_on_development_tasks_only(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "multisource_new_task_benchmark_v1.json").read_text(
                encoding="utf-8"
            )
        )
        config["protocol"]["development_seed_count"] = 1
        config["protocol"]["development_reveal_rounds"] = 2
        config["protocol"]["source_inducing_limit"] = 20
        config["protocol"]["skill_prior_candidates"] = [
            {"mass_start": 0.10, "mass_end": 0.00},
            {"mass_start": 0.25, "mass_end": 0.00},
        ]
        with tempfile.TemporaryDirectory() as directory:
            record = calibration.calibrate(config, Path(directory))
        self.assertFalse(record["target_task_calibration"])
        self.assertEqual(
            record["evaluation_task_ids_not_loaded"],
            ["real_reizman_suzuki_case_4"],
        )
        self.assertIn(record["selected_skill_prior"], config["protocol"]["skill_prior_candidates"])
        self.assertIn(
            record["selected_route"]["mode"],
            multisource.POLICY_MODES,
        )


if __name__ == "__main__":
    unittest.main()
