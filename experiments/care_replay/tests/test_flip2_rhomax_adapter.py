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
import run_multisource_warmstart as warmstart
import run_synthetic_suzuki as replay


class Flip2RhomaxAdapterTests(unittest.TestCase):
    def fixture(self, path: Path) -> None:
        rows = [
            {"sequence": "ACDEFGHIKLMNPQRSTVWY" * 10, "target": 510.0, "set": "train", "validation": "False"},
            {"sequence": "ACDEFGHIKLMNPQRSTVWY" * 11, "target": 520.0, "set": "train", "validation": "True"},
            {"sequence": "YWVTSRQPNMLKIHGFEDCA" * 10, "target": 530.0, "set": "validation", "validation": "False"},
            {"sequence": "AILMFWVYDEKRSTNQGP" * 10, "target": 540.0, "set": "test", "validation": "False"},
        ]
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sequence", "target", "set", "validation"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_official_partitions_share_public_sequence_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "rhomax.csv.gz"
            self.fixture(fixture)
            with mock.patch.object(
                replay, "ensure_public_data_file", return_value=fixture
            ):
                train = replay.real_flip2_rhomax_train_adapter()
                validation = replay.real_flip2_rhomax_validation_adapter()
                test = replay.real_flip2_rhomax_test_adapter()
        self.assertEqual(len(train.candidates), 1)
        self.assertEqual(len(validation.candidates), 2)
        self.assertEqual(len(test.candidates), 1)
        self.assertEqual(train.decision_columns, test.decision_columns)
        self.assertEqual(len(test.candidates[0].numeric_features), 26)
        self.assertEqual(warmstart.task_descriptor(test)["domain"], "protein_engineering")
        classical.validate_compatible_spaces(train, test)
        public = hypothesis.public_candidate(test.candidates[0], test)
        public_json = str(public).lower()
        self.assertNotIn("measured_peak_wavelength_nm", public_json)
        self.assertNotIn("normalized_peak_wavelength_score", public_json)

    def test_partition_parser_prioritizes_validation_flag(self):
        row = {"set": "train", "validation": "True"}
        self.assertEqual(replay.flip2_rhomax_partition(row), "validation")

    def test_source_adapter_does_not_parse_test_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "rhomax_masked.csv.gz"
            rows = [
                {
                    "sequence": "ACDEFGHIKLMNPQRSTVWY" * 10,
                    "target": "510.0",
                    "set": "train",
                    "validation": "False",
                },
                {
                    "sequence": "AILMFWVYDEKRSTNQGP" * 10,
                    "target": "MASKED_TEST_OUTCOME",
                    "set": "test",
                    "validation": "False",
                },
            ]
            with gzip.open(fixture, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sequence", "target", "set", "validation"],
                )
                writer.writeheader()
                writer.writerows(rows)
            with mock.patch.object(
                replay, "ensure_public_data_file", return_value=fixture
            ):
                train = replay.real_flip2_rhomax_train_adapter()
        self.assertEqual(len(train.candidates), 1)

    def test_public_features_reject_nonstandard_residues(self):
        with self.assertRaisesRegex(ValueError, "unsupported residues"):
            replay.flip2_rhomax_public_features("ACDX")


if __name__ == "__main__":
    unittest.main()
