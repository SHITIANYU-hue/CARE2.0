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


class Flip2HydroAdapterTests(unittest.TestCase):
    def fixture(self, path: Path) -> None:
        rows = []
        for backbone_id, length in replay.FLIP2_HYDRO_BACKBONES.items():
            base = ["A"] * length
            positions = tuple(range(0, 14, 2))
            for residue, target in (("F", -0.5), ("I", -1.5)):
                sequence = base.copy()
                for position in positions:
                    sequence[position] = residue
                rows.append(
                    {
                        "sequence": "".join(sequence),
                        "target": target,
                        "set": "test" if backbone_id == "P06241" else "train",
                        "validation": "False",
                    }
                )
        with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sequence", "target", "set", "validation"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_backbone_adapters_share_only_public_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "hydro.csv.gz"
            self.fixture(fixture)
            with mock.patch.object(
                replay, "ensure_public_data_file", return_value=fixture
            ):
                source = replay.real_flip2_hydro_p01053_adapter()
                target = replay.real_flip2_hydro_p06241_adapter()
        self.assertEqual(len(source.candidates), 2)
        self.assertEqual(len(target.candidates), 2)
        self.assertEqual(source.decision_columns, target.decision_columns)
        self.assertEqual(len(target.candidates[0].numeric_features), 5)
        self.assertEqual(target.candidates[0].metadata["core_sequence"], "FFFFFFF")
        self.assertEqual(
            target.candidates[0].metadata["variable_positions_1indexed"],
            [1, 3, 5, 7, 9, 11, 13],
        )
        self.assertEqual(warmstart.task_descriptor(target)["domain"], "protein_engineering")
        classical.validate_compatible_spaces(source, target)
        public = hypothesis.public_candidate(target.candidates[0], target)
        public_json = str(public).lower()
        self.assertNotIn("measured_stability_fitness", public_json)
        self.assertNotIn("normalized_stability_score", public_json)

    def test_variable_position_contract_rejects_non_hydrophobic_alphabet(self):
        records = []
        for length in replay.FLIP2_HYDRO_BACKBONES.values():
            for residue in ("F", "W"):
                sequence = ["A"] * length
                for position in range(0, 14, 2):
                    sequence[position] = residue
                records.append({"sequence": "".join(sequence)})
        with self.assertRaisesRegex(ValueError, "outside F/I/L/M/V"):
            replay.flip2_hydro_variable_positions(records)


if __name__ == "__main__":
    unittest.main()
