from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_llm_initial_design_hypothesis as hypothesis  # noqa: E402
import run_surrogate_baselines as surrogate  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


def write_fixture(path: Path) -> None:
    fields = [raw_name for raw_name, _ in replay.PHOTOCATALYSIS_FEATURES]
    rows = [
        {
            "P10-MIX1": "1.0",
            "L-Cysteine-100gL": "0.5",
            "NaCl-3M": "0.0",
            "NaOH-1M": "0.25",
            "Sodiumsilicate-1wt": "0.0",
            "AcidRed871_0gL": "0.0",
            "MethyleneB_250mgL": "0.0",
            "RhodamineB1_0gL": "0.0",
            "PVP-1wt": "0.0",
            "SDS-1wt": "0.0",
            "Target": "4.5",
        },
        {
            "P10-MIX1": "5.0",
            "L-Cysteine-100gL": "2.0",
            "NaCl-3M": "1.0",
            "NaOH-1M": "1.0",
            "Sodiumsilicate-1wt": "1.5",
            "AcidRed871_0gL": "0.5",
            "MethyleneB_250mgL": "0.0",
            "RhodamineB1_0gL": "0.0",
            "PVP-1wt": "0.5",
            "SDS-1wt": "0.0",
            "Target": "19.0",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fields, "Target"])
        writer.writeheader()
        writer.writerows(rows)


def test_photocatalysis_adapter_preserves_full_public_feature_space() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "data.csv"
        write_fixture(fixture)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            adapter = replay.real_photocatalytic_hydrogen_evolution_adapter()
    assert len(adapter.candidates) == 2
    assert len(adapter.candidates[0].numeric_features) == 10
    assert all(
        0.0 <= value <= 1.0
        for candidate in adapter.candidates
        for value in candidate.numeric_features
    )
    assert adapter.candidates[0].group == "dye_free"
    assert adapter.candidates[1].group == "dye_added"
    assert adapter.candidates[1].metadata[adapter.hidden_target] == 19.0
    numeric, categorical = surrogate.candidate_features(
        adapter, adapter.candidates[1]
    )
    assert len(numeric) == 10
    assert "dye_added" in categorical


def test_photocatalysis_public_views_exclude_hidden_outcome() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / "data.csv"
        write_fixture(fixture)
        with mock.patch.object(replay, "ensure_public_data_file", return_value=fixture):
            adapter = replay.real_photocatalytic_hydrogen_evolution_adapter()
    public = hypothesis.public_candidate(adapter.candidates[0], adapter)
    assert adapter.hidden_target not in str(public)
    catalog = semantic.semantic_field_catalog(adapter)
    assert "dye_presence" in catalog
    assert "p10_mix1_bin" in catalog
    assert adapter.hidden_target not in catalog
