from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_source_outcome_falsification as falsification  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


class SourceOutcomeFalsificationTests(unittest.TestCase):
    def test_permutation_preserves_candidates_and_outcome_marginal(self) -> None:
        adapter = replay.DATASET_BUILDERS["real_moleculenet_esol"]()
        observed = list(adapter.candidates[:20])
        permuted = falsification.permuted_source_observations(observed, 17)
        self.assertEqual(
            [candidate.candidate_id for candidate in observed],
            [candidate.candidate_id for candidate in permuted],
        )
        self.assertEqual(
            sorted(candidate.objective_value for candidate in observed),
            sorted(candidate.objective_value for candidate in permuted),
        )
        self.assertTrue(any(
            left.objective_value != right.objective_value
            for left, right in zip(observed, permuted)
        ))
        self.assertEqual(
            [candidate.metadata for candidate in observed],
            [candidate.metadata for candidate in permuted],
        )

    def test_pair_summary_is_paired_by_target_seed(self) -> None:
        rows = []
        for target_seed, baseline, true, perm_a, perm_b in (
            (10, 1.0, 4.0, 2.0, 3.0),
            (11, 2.0, 6.0, 3.0, 4.0),
        ):
            for condition, value in (
                (falsification.BASELINE, baseline),
                (falsification.TRUE_OUTCOMES, true),
                (falsification.condition_name(101), perm_a),
                (falsification.condition_name(211), perm_b),
            ):
                rows.append({
                    "pair_id": "pair",
                    "target_seed": target_seed,
                    "condition": condition,
                    "final_best": value,
                    "best_so_far_auc": value,
                    "top10_hit": value,
                })
        summary = falsification.build_pair_summary(
            {
                "pair_id": "pair",
                "source_dataset": "source",
                "target_dataset": "target",
            },
            rows,
            [101, 211],
        )
        self.assertEqual(summary["target_seed_count"], 2)
        self.assertAlmostEqual(
            summary["true_outcomes_minus_target_only"]["best_so_far_auc"]["mean"],
            3.5,
        )
        self.assertAlmostEqual(
            summary["true_outcomes_minus_mean_permutation"]["best_so_far_auc"]["mean"],
            2.0,
        )

    def test_job_checkpoint_requires_matching_hash_and_conditions(self) -> None:
        permutation_seeds = [101, 211]
        rows = [
            {
                "pair_id": "pair",
                "target_seed": 10,
                "condition": condition,
                "final_best": 1.0,
                "best_so_far_auc": 1.0,
                "top10_hit": 0.0,
            }
            for condition in (
                falsification.BASELINE,
                falsification.TRUE_OUTCOMES,
                falsification.condition_name(101),
                falsification.condition_name(211),
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "job.json"
            falsification.write_job_checkpoint(
                checkpoint,
                config_sha256="config-hash",
                implementation_sha256="implementation-hash",
                pair_id="pair",
                target_seed=10,
                rows=rows,
                audits={},
            )
            loaded = falsification.load_job_checkpoint(
                checkpoint,
                config_sha256="config-hash",
                implementation_sha256="implementation-hash",
                pair_id="pair",
                target_seed=10,
                permutation_seeds=permutation_seeds,
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded[0], rows)

            checkpoint.write_text("{}\n", encoding="utf-8")
            self.assertIsNone(falsification.load_job_checkpoint(
                checkpoint,
                config_sha256="config-hash",
                implementation_sha256="implementation-hash",
                pair_id="pair",
                target_seed=10,
                permutation_seeds=permutation_seeds,
            ))

    def test_results_distinguish_seed_uncertainty_from_randomization_test(self) -> None:
        report = {
            "pairs": [
                {
                    "source_dataset": "source",
                    "target_dataset": "target",
                    "true_outcomes_minus_target_only": {
                        "best_so_far_auc": {
                            "mean": 2.0,
                            "normal_95ci_low": 1.0,
                            "normal_95ci_high": 3.0,
                        },
                    },
                    "true_outcomes_minus_mean_permutation": {
                        "best_so_far_auc": {
                            "mean": 1.5,
                            "normal_95ci_low": 1.5,
                            "normal_95ci_high": 1.5,
                        },
                    },
                    "randomization_test": {
                        "best_so_far_auc": {"one_sided_empirical_p": 0.15},
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "RESULTS.md"
            falsification.write_results(results_path, report)
            results = results_path.read_text(encoding="utf-8")
            normalized_results = " ".join(results.split())

        self.assertIn(
            "conditional on the tested outcome assignments",
            normalized_results,
        )
        self.assertIn("empirical randomization p-value", normalized_results)
        self.assertIn("none of the tested routes rejects", normalized_results)
        self.assertNotIn(
            "shows that the source feature-outcome association matters",
            normalized_results,
        )


if __name__ == "__main__":
    unittest.main()
