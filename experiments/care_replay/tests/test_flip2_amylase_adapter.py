from __future__ import annotations

import csv
import gzip
from pathlib import Path
import sys
import tempfile
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_classical_transfer_baselines as classical
import run_llm_initial_design_hypothesis as hypothesis
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay


REFERENCE = "ACDEFGHIKLMNPQRSTVWY" * 2


def mutate(*changes: tuple[int, str]) -> str:
    sequence = list(REFERENCE)
    for index, residue in changes:
        sequence[index] = residue
    return "".join(sequence)


def write_fixture(path: Path, *, test_target: str = "3.0") -> None:
    rows = [
        {"sequence": REFERENCE, "target": "0.0", "set": "train", "validation": "False"},
        {"sequence": mutate((0, "V")), "target": "2.0", "set": "train", "validation": "False"},
        {"sequence": mutate((1, "D")), "target": "1.0", "set": "train", "validation": "False"},
        {"sequence": mutate((3, "F")), "target": "1.5", "set": "train", "validation": "True"},
        {
            "sequence": mutate((0, "V"), (1, "D"), (3, "F")),
            "target": test_target,
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


def test_amylase_partitions_share_public_feature_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "amylase.csv.gz"
        write_fixture(fixture)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            train = replay.real_flip2_amylase_train_adapter()
            validation = replay.real_flip2_amylase_validation_adapter()
            train_validation = replay.real_flip2_amylase_train_validation_adapter()
            test = replay.real_flip2_amylase_test_adapter()
    assert (len(train.candidates), len(validation.candidates)) == (3, 1)
    assert len(train_validation.candidates) == 4
    assert len(test.candidates) == 1
    assert len(test.candidates[0].numeric_features) == 39
    assert train.decision_columns == test.decision_columns
    classical.validate_compatible_spaces(train, test)
    public_json = str(hypothesis.public_candidate(test.candidates[0], test)).lower()
    assert "measured_amylase_activity" not in public_json


def test_public_amylase_target_does_not_parse_or_store_outcomes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "amylase_masked.csv.gz"
        write_fixture(fixture, test_target="MASKED_TEST_OUTCOME")
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            public_test = replay.public_flip2_amylase_test_adapter()
    candidate = public_test.candidates[0]
    assert candidate.objective_value == 0.0
    assert "measured_amylase_activity" not in candidate.metadata
    assert public_test.hidden_target == "unavailable"


def test_amylase_structural_coverage_is_target_outcome_invariant() -> None:
    diagnostics = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, target in enumerate(("-999.0", "999.0")):
            fixture = Path(tmp) / f"amylase_{index}.csv.gz"
            write_fixture(fixture, test_target=target)
            with mock.patch.object(
                replay, "ensure_public_data_file", return_value=fixture
            ):
                source = replay.real_flip2_amylase_train_validation_adapter()
                public_test = replay.public_flip2_amylase_test_adapter()
            diagnostics.append(
                multisource.mutation_coverage_diagnostics(
                    [source.candidates], public_test.candidates
                )
            )
    assert diagnostics[0] == diagnostics[1]
    assert diagnostics[0]["target_outcomes_used"] is False
    assert diagnostics[0]["exact_token_coverage"] == 1.0
    assert diagnostics[0]["candidate_full_exact_coverage"] == 1.0
