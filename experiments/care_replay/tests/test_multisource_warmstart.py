from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_multisource_warmstart as warmstart  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402
import build_baumgartner_warmstart_skill_bank as build_skill  # noqa: E402


def frozen_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "baumgartner_multisource_warmstart_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_all_baumgartner_campaigns_load_with_compatible_spaces() -> None:
    adapters = [
        replay.real_baumgartner_cn_adapter(campaign_id)
        for campaign_id, _name in replay.BAUMGARTNER_CN_CAMPAIGNS
    ]
    assert len(adapters) == 13
    assert all(adapter.candidates for adapter in adapters)
    assert {adapter.decision_columns for adapter in adapters} == {("base",)}
    assert all(
        0.0 <= candidate.objective_value <= 100.0
        for adapter in adapters
        for candidate in adapter.candidates
    )


def test_frozen_protocol_is_task_disjoint_and_target_uncalibrated() -> None:
    config = frozen_config()
    warmstart.validate_protocol(config)
    config["protocol"]["evaluation_task_ids"].append(
        config["protocol"]["development_task_ids"][0]
    )
    with pytest.raises(ValueError, match="overlap"):
        warmstart.validate_protocol(config)


def test_source_diverse_initial_is_unique_and_deterministic() -> None:
    target = replay.real_baumgartner_cn_adapter("aniline_ephos")
    development = frozen_config()["protocol"]["development_task_ids"]
    source_ids = [task_id for task_id in development if task_id != target.dataset_id]
    protocol = frozen_config()["protocol"]
    selected_sources = warmstart.source_ids_for_scope(
        target, source_ids, "same_substrate_then_precatalyst"
    )
    prior, _source_ids = warmstart.build_source_consensus(
        target, selected_sources, protocol
    )
    first = warmstart.source_diverse_initial(target, prior, 3, 0.85)
    second = warmstart.source_diverse_initial(target, prior, 3, 0.85)
    assert first == second
    assert len(first) == len(set(first)) == 3


def test_source_quantile_keeps_diverse_candidates_in_admissible_region() -> None:
    target = replay.real_baumgartner_cn_adapter("aniline_ephos")
    prior = np.linspace(0.0, 1.0, len(target.candidates))
    selected = warmstart.source_diverse_initial(
        target, prior, 3, diversity_weight=1.0, source_quantile=0.5
    )
    threshold = float(np.quantile(prior, 0.5))
    assert all(float(prior[index]) >= threshold for index in selected)


def test_baumgartner_suzuki_campaigns_share_feature_contract() -> None:
    source = replay.real_baumgartner_suzuki_adapter("minlp1")
    target = replay.real_baumgartner_suzuki_adapter("minlp2")
    assert source.decision_columns == target.decision_columns == ("precatalyst",)
    assert source.candidates and target.candidates
    assert all(len(candidate.numeric_features) == 3 for candidate in source.candidates)
    assert all(len(candidate.numeric_features) == 3 for candidate in target.candidates)


def test_generic_task_descriptor_uses_domain_and_representation() -> None:
    source = replay.real_moleculenet_freesolv_continuous_adapter()
    target = replay.real_moleculenet_lipophilicity_adapter()
    source_descriptor = warmstart.task_descriptor(source)
    target_descriptor = warmstart.task_descriptor(target)
    assert source_descriptor["domain"] == target_descriptor["domain"] == "molecular_property"
    assert source_descriptor["representation"] == target_descriptor["representation"]
    assert warmstart.source_ids_for_scope(
        target,
        [source.dataset_id],
        "same_substrate_then_precatalyst",
    ) == [source.dataset_id]


def test_shared_esol_adapter_matches_molecule_transfer_contract() -> None:
    source = replay.real_moleculenet_freesolv_continuous_adapter()
    target = replay.real_moleculenet_esol_common_adapter()
    assert source.decision_columns == target.decision_columns
    assert target.candidates
    assert all(
        0.0 <= candidate.objective_value <= 100.0
        for candidate in target.candidates
    )


def test_bace_and_steels_holdouts_use_shared_transfer_contracts() -> None:
    molecule_source = replay.real_moleculenet_freesolv_continuous_adapter()
    molecule_target = replay.real_moleculenet_bace_common_adapter()
    material_source = replay.real_matbench_expt_gap_adapter()
    material_target = replay.real_matbench_steels_adapter()
    assert molecule_source.decision_columns == molecule_target.decision_columns
    assert material_source.decision_columns == material_target.decision_columns
    assert molecule_target.candidates and material_target.candidates


def test_small_calibration_and_confirmation_run_without_target_calibration(
    tmp_path: Path,
) -> None:
    config = frozen_config()
    protocol = config["protocol"]
    protocol["development_task_ids"] = protocol["development_task_ids"][:3]
    protocol["evaluation_task_ids"] = protocol["evaluation_task_ids"][:1]
    protocol["development_seed_count"] = 2
    protocol["evaluation_seed_count"] = 2
    protocol["initial_observations"] = 2
    protocol["reveal_rounds"] = 2
    protocol["source_inducing_limit"] = 12
    protocol["warmstart_candidates"] = [
        {
            "source_scope": "same_substrate_then_precatalyst",
            "diversity_weight": 0.85,
        }
    ]
    development_dir = tmp_path / "development"
    selection = warmstart.calibrate(config, development_dir)
    assert selection["target_task_calibration"] is False
    assert selection["evaluation_task_ids_not_loaded"] == protocol[
        "evaluation_task_ids"
    ]
    confirmation_dir = tmp_path / "confirmation"
    summary = warmstart.confirm(config, selection, confirmation_dir)
    assert summary["target_task_calibration"] is False
    if selection["selected_route"]["mode"] == warmstart.SOURCE_MODE:
        assert summary["evaluated_source_policy"] == selection["selected_route"]
    assert set(summary["aggregate_by_task"]) == set(
        protocol["evaluation_task_ids"]
    )
    assert (confirmation_dir / "confirmation_metrics.csv").exists()
    assert (confirmation_dir / "confirmation_summary.json").exists()


def test_confirmation_rejects_changed_frozen_config(tmp_path: Path) -> None:
    config = frozen_config()
    selection = {
        "config_fingerprint": warmstart.config_fingerprint(config),
        "best_source_policy": config["protocol"]["warmstart_candidates"][0],
        "selected_route": {"mode": warmstart.SPACE_FILLING_MODE},
    }
    config["protocol"]["reveal_rounds"] += 1
    with pytest.raises(ValueError, match="frozen config"):
        warmstart.confirm(config, selection, tmp_path)


def test_v2_skill_bank_keeps_external_confirmation_separate() -> None:
    config = json.loads(
        (ROOT / "configs" / "baumgartner_multisource_warmstart_v2.json").read_text()
    )
    result_root = (
        ROOT
        / "results"
        / "2026-08-10-baumgartner-warmstart-v2-external-confirmation"
    )
    selection = json.loads(
        (result_root / "development" / "selection_record.json").read_text()
    )
    confirmation = json.loads(
        (result_root / "confirmation" / "confirmation_summary.json").read_text()
    )
    bank = build_skill.build_bank(config, selection, confirmation)
    evidence = {item.evidence_id: item for item in bank.evidence}
    external = evidence["baumgartner_suzuki_external_confirmation_v2"]
    assert external.status == "candidate"
    assert "real_baumgartner_suzuki_minlp1" in bank.development_task_ids
    assert bank.evaluation_task_ids == ("real_baumgartner_suzuki_minlp2",)
