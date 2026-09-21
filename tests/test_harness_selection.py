"""A calibrated deployment choice must keep the experiment it was selected for."""

import hashlib
import json
from pathlib import Path

import pytest

from care_harness.agents.care import load_pair_config
from care_harness.cli import freeze_selection, method_config, run_config


@pytest.fixture
def selection_files(tmp_path):
    pair = load_pair_config("materials_expt_gap_to_dielectric")
    execution = {
        key: pair[key] for key in (
            "initial_observations", "reveal_rounds", "discount", "min_source_support",
            "normalize_kernel_weights", "gp_beta", "gp_xi", "numeric_length_scale",
            "categorical_length_scale", "gp_noise", "router_min_observations",
            "router_min_quality", "router_max_transfer_mass", "source_initial_strategy",
        )
    }
    execution.update(
        source_observation_count=31, source_seed=91,
        target_anchor_mode=pair["target_llm_mode"],
        calibration_seed_start=100, calibration_seed_count=3,
        heldout_seed_start=200, heldout_seed_count=2,
    )
    hashes = {key: hashlib.sha256(Path(pair[key]).read_bytes()).hexdigest()
              for key in ("llm_record", "target_llm_record")}
    summary = {
        "source_dataset": pair["source_dataset"], "target_dataset": pair["target_dataset"],
        "selection": {"selected_mode": "matched_target_only_llm"},
        "calibration_seed_start": 100, "calibration_seed_count": 3,
        "offline_selection_cost": {"target_reveals": 45},
        "transfer_skill": {
            "execution": execution,
            "provenance": {"source_patch_record_sha256": hashes["llm_record"],
                           "target_anchor_record_sha256": hashes["target_llm_record"]},
        },
        "heldout": {"not_agent_visible": True},
    }
    source = tmp_path / "canonical-summary.json"
    source.write_text(json.dumps(summary))
    selection = tmp_path / "selection.json"
    artifact = freeze_selection(source, selection)
    return pair, execution, hashes, selection, artifact


def test_export_preserves_execution_and_records_not_heldout(selection_files):
    _, execution, hashes, _, artifact = selection_files
    assert artifact["execution"] == execution
    assert artifact["record_sha256"] == hashes
    assert artifact["deployment_mode"] == "matched_target_only_llm"
    assert "heldout" not in artifact


def test_selected_method_uses_frozen_execution(selection_files):
    pair, execution, _, path, _ = selection_files
    result = method_config({"care_pair": pair["pair_id"]},
                           {"type": "care", "selection_file": str(path)})
    assert result["source_observations"] == execution["source_observation_count"]
    assert result["source_seed"] == 91
    assert result["target_llm_mode"] == execution["target_anchor_mode"]
    assert result["deployment_mode"] == "matched_target_only_llm"
    assert result["router_max_transfer_mass"] == execution["router_max_transfer_mass"]
    assert result["calibration_status"] == "frozen_offline_selection"


def test_selected_method_rejects_changed_record(selection_files, tmp_path):
    pair, _, _, path, _ = selection_files
    changed = tmp_path / "changed-record.json"
    changed.write_text('{"normalized_patches": []}')
    with pytest.raises(ValueError, match="(?i)(record|hash|sha256)"):
        method_config({"care_pair": pair["pair_id"]},
                      {"type": "care", "selection_file": str(path), "llm_record": str(changed)})


@pytest.mark.parametrize("change", ["budget", "initialization", "calibration_seed"])
def test_selected_deployment_rejects_different_protocol(selection_files, tmp_path, monkeypatch, change):
    pair, execution, _, path, _ = selection_files
    protocol = {
        "initialization": "care", "seeds": [200],
        "initial_observations": execution["initial_observations"],
        "reveal_rounds": execution["reveal_rounds"],
    }
    if change == "budget":
        protocol["reveal_rounds"] += 1
    elif change == "initialization":
        protocol["initialization"] = "random"
    else:
        protocol["seeds"] = [101]
    config = {
        "protocol": protocol,
        "tasks": [{"care_pair": pair["pair_id"]}],
        "methods": {"care": {"type": "care", "selection_file": str(path)}},
    }
    def unexpected_load(_):
        pytest.fail("Protocol must be resolved before loading any target outcomes")
    monkeypatch.setattr("care_harness.cli.load_task", unexpected_load)
    with pytest.raises(ValueError):
        run_config(config, tmp_path / "run")
