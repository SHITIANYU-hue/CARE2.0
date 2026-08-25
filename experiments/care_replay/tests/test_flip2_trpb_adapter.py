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


def write_fixture(path: Path, *, masked_test: bool = False) -> None:
    rows = [
        {"sequence": REFERENCE, "target": "0.0", "set": "train", "validation": "False"},
        {"sequence": mutate((0, "V")), "target": "2.0", "set": "train", "validation": "False"},
        {"sequence": mutate((1, "D")), "target": "1.0", "set": "train", "validation": "False"},
        {"sequence": mutate((0, "V"), (1, "D")), "target": "2.5", "set": "train", "validation": "True"},
        {
            "sequence": mutate((0, "V"), (1, "D"), (3, "F")),
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


def test_trpb_partitions_share_public_feature_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "trpb.csv.gz"
        write_fixture(fixture)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            train = replay.real_flip2_trpb_train_adapter()
            validation = replay.real_flip2_trpb_validation_adapter()
            train_validation = replay.real_flip2_trpb_train_validation_adapter()
            test = replay.real_flip2_trpb_test_adapter()
    assert (len(train.candidates), len(validation.candidates)) == (3, 1)
    assert len(train_validation.candidates) == 4
    assert len(test.candidates) == 1
    assert len(test.candidates[0].numeric_features) == 39
    assert train.decision_columns == test.decision_columns
    classical.validate_compatible_spaces(train, test)
    public_json = str(hypothesis.public_candidate(test.candidates[0], test)).lower()
    assert "measured_trpb_fitness" not in public_json


def test_trpb_additive_prior_is_outcome_blind_to_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "trpb.csv.gz"
        write_fixture(fixture)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            train = replay.real_flip2_trpb_train_adapter()
            test = replay.real_flip2_trpb_test_adapter()
    scores, diagnostics = multisource.additive_mutation_prior(
        [train.candidates], test.candidates
    )
    assert scores.shape == (1,)
    assert diagnostics["target_outcomes_used"] is False
    assert diagnostics["sources"][0]["candidate_evidence_coverage"] > 0.0


def test_trpb_source_adapter_does_not_parse_test_outcomes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "trpb_masked.csv.gz"
        write_fixture(fixture, masked_test=True)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            train = replay.real_flip2_trpb_train_adapter()
    assert len(train.candidates) == 3


def test_trpb_sampling_is_sequence_hash_ordered() -> None:
    records = [
        {"sequence": mutate((index, "V")), "set": "test", "validation": "False"}
        for index in range(5)
    ]
    selected = replay._flip2_trpb_partition_rows(records, {"test"})
    keys = [
        replay.hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
        for _index, row in selected
    ]
    assert keys == sorted(keys)
