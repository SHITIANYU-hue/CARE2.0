from __future__ import annotations

import csv
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_classical_transfer_baselines as classical
import run_llm_initial_design_hypothesis as hypothesis
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay


class Flip2IredAdapterTests(unittest.TestCase):
    reference = "ACDEFGHIKLMNPQRSTVWY" * 2

    def mutate(self, *changes: tuple[int, str]) -> str:
        sequence = list(self.reference)
        for index, residue in changes:
            sequence[index] = residue
        return "".join(sequence)

    def fixture(self, path: Path, *, masked_test: bool = False) -> None:
        rows = [
            {"sequence": self.reference, "target": "0.0", "set": "train", "validation": "False"},
            {"sequence": self.mutate((0, "V")), "target": "2.0", "set": "train", "validation": "False"},
            {"sequence": self.mutate((1, "D")), "target": "1.0", "set": "train", "validation": "False"},
            {"sequence": self.mutate((0, "V"), (1, "D")), "target": "2.5", "set": "train", "validation": "True"},
            {
                "sequence": self.mutate((0, "V"), (1, "D"), (3, "F")),
                "target": "MASKED_TEST_OUTCOME" if masked_test else "3.0",
                "set": "test",
                "validation": "False",
            },
        ]
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sequence", "target", "set", "validation"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_official_partitions_share_mutation_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "ired.csv.gz"
            self.fixture(fixture)
            with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
                train = replay.real_flip2_ired_train_adapter()
                validation = replay.real_flip2_ired_validation_adapter()
                train_validation = replay.real_flip2_ired_train_validation_adapter()
                test = replay.real_flip2_ired_test_adapter()
        self.assertEqual((len(train.candidates), len(validation.candidates)), (3, 1))
        self.assertEqual(len(train_validation.candidates), 4)
        self.assertEqual(len(test.candidates), 1)
        self.assertEqual(len(test.candidates[0].numeric_features), 39)
        self.assertEqual(train.decision_columns, test.decision_columns)
        classical.validate_compatible_spaces(train, test)
        public_json = str(hypothesis.public_candidate(test.candidates[0], test)).lower()
        self.assertNotIn("measured_ired_activity", public_json)

    def test_additive_prior_uses_source_mutations_without_target_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "ired.csv.gz"
            self.fixture(fixture)
            with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
                train = replay.real_flip2_ired_train_adapter()
                test = replay.real_flip2_ired_test_adapter()
        scores, diagnostics = multisource.additive_mutation_prior(
            [train.candidates], test.candidates
        )
        self.assertEqual(scores.shape, (1,))
        self.assertFalse(diagnostics["target_outcomes_used"])
        self.assertGreater(
            diagnostics["sources"][0]["candidate_evidence_coverage"], 0.0
        )

    def test_source_adapter_does_not_parse_test_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "ired_masked.csv.gz"
            self.fixture(fixture, masked_test=True)
            with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
                train = replay.real_flip2_ired_train_adapter()
        self.assertEqual(len(train.candidates), 3)

    def test_partition_parser_prioritizes_validation_flag(self) -> None:
        self.assertEqual(
            replay.flip2_ired_partition(
                {"set": "train", "validation": "True"}
            ),
            "validation",
        )


if __name__ == "__main__":
    unittest.main()
