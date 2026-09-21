"""Offline policy parity: real descriptors and already-frozen LLM records."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from care_harness.agents.care import CareAgent, TargetGPAgent, load_pair_config
from care_harness.environment import ReplayEnvironment, load_task
from care_harness.legacy import import_module
from care_harness.runner import run_episode


@pytest.fixture
def small_pair(monkeypatch):
    replay = import_module("run_synthetic_suzuki")
    config = load_pair_config("materials_expt_gap_to_dielectric")
    config.update(source_observations=48, initial_observations=5, reveal_rounds=3)
    source = replay.DATASET_BUILDERS[config["source_dataset"]]()
    target = replay.DATASET_BUILDERS[config["target_dataset"]]()
    source = replace(source, candidates=source.candidates[::max(1, len(source.candidates) // 48)][:48])
    target = replace(target, candidates=target.candidates[::max(1, len(target.candidates) // 40)][:40])
    monkeypatch.setitem(replay.DATASET_BUILDERS, source.dataset_id, lambda: source)
    monkeypatch.setitem(replay.DATASET_BUILDERS, target.dataset_id, lambda: target)
    return config, source, target, load_task(config)


def legacy_run(config, source, target, seed, router=None):
    replay = import_module("run_synthetic_suzuki")
    calibrated = import_module("run_calibrated_source_outcome_transfer")
    evolution = import_module("run_llm_kernel_skill_evolution")
    transfer = import_module("run_transfer_ablation")
    router = router or import_module("run_llm_transfer_router")
    task = replay.make_task(target, config["initial_observations"], config["reveal_rounds"])
    record = json.loads(Path(config["llm_record"]).read_text())
    target_record = json.loads(Path(config["target_llm_record"]).read_text())
    patches = evolution.normalize_patches({"patches": record["normalized_patches"]}, target.decision_columns, 100)
    skill = calibrated.resolve_target_llm_skill(target_record, target, config["target_llm_mode"])
    initial = calibrated.target_llm_initial_observations(target, task, seed, config["target_llm_mode"], skill)
    scorer = calibrated.target_llm_anchor_scorer(
        target, task, config["target_llm_mode"], skill,
        config["gp_beta"], config["gp_xi"], config["numeric_length_scale"],
        config["categorical_length_scale"], config["gp_noise"],
    )
    observed = transfer.source_observations(source, config["source_seed"], config["source_observations"])
    card = transfer.compile_transfer_card(
        source, target, observed, transfer.descriptor_transfer_role_map_for(source.dataset_id, target.dataset_id),
        config["discount"], config["min_source_support"],
    )
    return router.run_router_policy(
        source, target, task, seed, card, observed, patches,
        config["normalize_kernel_weights"], config["gp_beta"], config["gp_xi"],
        config["numeric_length_scale"], config["categorical_length_scale"], config["gp_noise"],
        initial, scorer, config["target_llm_mode"], config["router_min_observations"],
        config["router_min_quality"], config["router_max_transfer_mass"], config["source_initial_strategy"],
    )


@pytest.mark.parametrize("seed", [17, 32])
def test_public_adapter_matches_legacy_router(small_pair, monkeypatch, seed):
    config, source, target, task = small_pair
    metrics, audit = legacy_run(config, source, target, seed)
    env = ReplayEnvironment(task, seed, 5, 3)
    agent = CareAgent(config)

    # Dataset loading belongs only to the environment, never to the adapter.
    replay = import_module("run_synthetic_suzuki")
    monkeypatch.setattr(replay, "DATASET_BUILDERS", {})
    initial_ids = agent.initial_candidate_ids(env.view())
    expected_initial = audit[0]["hypothesis_snapshot"]["source_initial_design"]["selected_initial_ids"]
    assert initial_ids == expected_initial
    result = run_episode(task, agent, seed, 5, 3, initial_ids)
    assert [event["candidate_id"] for event in result["trace"]] == [event["selected_candidate"] for event in audit]
    # Captured from the unmodified router at tangchao's a0a8c309, with this fixture.
    original_selections = {
        17: ["real_matbench_dielectric_00714", "real_matbench_dielectric_01428", "real_matbench_dielectric_00357"],
        32: ["real_matbench_dielectric_00119", "real_matbench_dielectric_03689", "real_matbench_dielectric_00476"],
    }
    assert [event["candidate_id"] for event in result["trace"]] == original_selections[seed]
    for metric in ("final_best", "best_so_far_auc", "simple_regret"):
        assert round(result["metrics"][metric], 4) == metrics[metric]
    assert result["usage"]["model_calls"] == 0
    assert result["trace"][0]["diagnostics"]["calibration_status"] == "not_run"


def test_initial_design_never_uses_target_outcomes(small_pair):
    config, _, _, task = small_pair
    first_view = ReplayEnvironment(task, 17, 5, 3).view()
    second_view = ReplayEnvironment(task, 17, 5, 3).view()
    second_view["observations"] = [{"candidate_id": row["candidate_id"], "outcome": -12345.0}
                                   for row in first_view["observations"]]
    assert CareAgent(config).initial_candidate_ids(first_view) == CareAgent(config).initial_candidate_ids(second_view)


def test_shared_random_start_matches_legacy_matched_policy(small_pair):
    config, source, target, task = small_pair
    view = ReplayEnvironment(task, 17, 5, 3).view()
    random_ids = [row["candidate_id"] for row in view["observations"]]
    assert CareAgent(config).initial_candidate_ids(view) != random_ids

    expected, audit = legacy_run({**config, "source_initial_strategy": "matched"}, source, target, 17)
    result = run_episode(task, CareAgent(config), 17, 5, 3)
    assert result["initial_ids"] == random_ids
    assert [row["candidate_id"] for row in result["trace"]] == [row["selected_candidate"] for row in audit]
    assert round(result["metrics"]["best_so_far_auc"], 4) == expected["best_so_far_auc"]


def test_calibrated_gp_fallback_is_target_policy(small_pair):
    config, _, _, task = small_pair
    config.update(deployment_mode="gp_ucb", calibration_status="frozen_selection")
    care = run_episode(task, CareAgent(config), 17, 5, 3)
    gp = run_episode(task, TargetGPAgent({"mode": "gp_ucb"}), 17, 5, 3)
    assert [event["candidate_id"] for event in care["trace"]] == [event["candidate_id"] for event in gp["trace"]]
    assert care["metrics"] == gp["metrics"]


def test_frozen_target_semantic_fallback_matches_existing_policy(small_pair):
    config, _, target, task = small_pair
    replay = import_module("run_synthetic_suzuki")
    calibrated = import_module("run_calibrated_source_outcome_transfer")
    semantic = import_module("llm_semantic_skills")
    record = json.loads(Path(config["target_llm_record"]).read_text())
    skill = calibrated.resolve_target_llm_skill(record, target, config["target_llm_mode"])
    expected, audit = semantic.run_direct_prior_skill(
        target, replay.make_task(target, 5, 3), 17, skill,
        config["numeric_length_scale"], config["categorical_length_scale"], config["gp_noise"],
    )
    config.update(deployment_mode="matched_target_only_llm", calibration_status="frozen_selection")
    result = run_episode(task, CareAgent(config), 17, 5, 3)
    assert [row["candidate_id"] for row in result["trace"]] == [row["selected_candidate"] for row in audit]
    assert round(result["metrics"]["best_so_far_auc"], 4) == expected["best_so_far_auc"]
