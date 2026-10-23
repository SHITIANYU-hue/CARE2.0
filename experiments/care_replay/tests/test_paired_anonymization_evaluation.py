"""Fail-closed handoff, unchanged execution, and paired inference safeguards."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_paired_anonymization_evaluation as paired


def payload():
    values = dict(ridge=1.25, prior_scale=0.4, semantic_mass_start=0.3,
                  semantic_mass_end=0.1, ucb_weight=0.6, gp_beta_start=1.7,
                  gp_beta_end=0.8, gp_xi=0.03, confidence=0.75)
    return {"skills": [
        {"skill_id": f"skill_{slot}", "hypothesis": "Hypothesis", **values,
         "rules": [{"rule_id": f"rule_{rule}",
                    "conditions": {"field": "value"},
                    "weight": -0.4 if rule else 0.6, "rationale": "Rationale"}
                   for rule in range(count)]}
        for slot, count in enumerate(paired.RULE_COUNTS)]}


CATALOG = {"field": {"value": 20}}


def libraries(pair_count=2):
    return {f"pair_{pair:02d}_{arm}": paired.normalize_without_loss(
                payload(), CATALOG, pair, arm)[0]
            for pair in range(pair_count) for arm in paired.ARMS}


def synthetic_rows(banks, seeds, split="heldout"):
    result = []
    for seed_index, seed in enumerate(sorted(seeds)):
        for mode in paired.matched.BASELINES:
            result.append(dict(mode=mode, seed=seed, split=split,
                               final_best=10.0, best_so_far_auc=10.0, top10_hit=0))
        for bank, skills in banks.items():
            pair = int(bank.split("_")[1])
            # The same start effect appears across both paired-generation rows.
            delta = (1 + 2 * pair + (8 if seed_index >= len(seeds) // 2 else 0)
                     if bank.endswith("_labeled") else 0)
            for skill in skills:
                result.append(dict(mode=paired.matched.semantic.skill_mode(skill),
                                   seed=seed, split=split, final_best=20.0 + delta,
                                   best_so_far_auc=20.0 + delta, top10_hit=1))
    return result


def frozen_choices(banks):
    return {name: {"ungated_calibration_winner": paired.matched.semantic.skill_mode(skills[0]),
                   "selected_mode": paired.matched.semantic.skill_mode(skills[0]),
                   "selection": {"target_anchor_mode": paired.matched.BASELINES[0]}}
            for name, skills in banks.items()}


def test_identical_raw_ids_across_libraries_do_not_collide_and_numbers_are_unchanged():
    banks = libraries(10)
    assert len({paired.matched.semantic.skill_mode(skill)
                for bank in banks.values() for skill in bank}) == 100
    source = payload()
    for bank in banks.values():
        for slot, skill in enumerate(bank):
            assert tuple(rule.weight for rule in skill.rules) == tuple(
                rule["weight"] for rule in source["skills"][slot]["rules"])
            for field in paired.PARAMETER_RANGES:
                assert getattr(skill, field) == source["skills"][slot][field]
    _, aliases = paired.matched.execution_plan(
        tuple(skill for bank in banks.values() for skill in bank))
    assert len(aliases) == 100
    assert len(set(aliases.values())) == 2  # one- versus two-rule executable template


@pytest.mark.parametrize("mutation", [
    lambda value: value["skills"][0]["rules"][0]["conditions"].update(unknown="value"),
    lambda value: value["skills"][0].update(ridge=25),
    lambda value: value["skills"][0].update(confidence=True),
    lambda value: value["skills"][0]["rules"].pop(),
    lambda value: value["skills"][0]["rules"][0].update(weight=float("inf")),
    lambda value: value["skills"][0].update(extra=3),
    lambda value: value["skills"].pop(),
])
def test_repairable_or_incomplete_payloads_are_rejected(mutation):
    value = payload()
    mutation(value)
    with pytest.raises(ValueError):
        paired.normalize_without_loss(value, CATALOG, 0, "labeled")


def test_normalizer_cannot_silently_change_numbers_or_drop_conditions():
    original = paired.matched.semantic.normalize_skills

    def changed(*args):
        skills = original(*args)
        return (replace(skills[0], prior_scale=0.41), *skills[1:])

    with patch.object(paired.matched.semantic, "normalize_skills", side_effect=changed):
        with pytest.raises(ValueError, match="changed an executable number"):
            paired.normalize_without_loss(payload(), CATALOG, 0, "labeled")


def test_duplicate_json_keys_fail_before_payload_interpretation(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"skills": [], "skills": []}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        paired.strict_json(path)


def test_paired_analysis_keeps_shared_seed_axis_and_separates_descriptive_blocks():
    banks, seeds = libraries(), {31, 32, 33, 34}
    config = dict(pair_count=2, bootstrap_seed=9, bootstrap_replicates=10000, heldout_blocks=2)
    result = paired.analyze(synthetic_rows(banks, seeds), banks, frozen_choices(banks), seeds, config)
    primary = result["primary_ungated_auc"]
    assert primary["delta_matrix"] == [[1, 1, 9, 9], [3, 3, 11, 11]]
    assert primary["mean_delta"] == 6
    assert primary["pair_mean_deltas"] == [5, 7]
    assert [block["mean_delta"] for block in primary["ordered_seed_blocks"]] == [2, 10]
    rng = np.random.default_rng(9)
    starts = rng.multinomial(4, np.full(4, 0.25), size=10000)
    pairs = rng.multinomial(2, np.full(2, 0.5), size=10000)
    # Explicit row/column reweighting checks that the arms/cells are not resampled independently.
    matrix = np.array([[1, 1, 9, 9], [3, 3, 11, 11]])
    explicit = np.array([sum(matrix[i, j] * pairs[b, i] * starts[b, j]
                             for i in range(2) for j in range(4)) / 8
                         for b in range(10000)])
    assert primary["interval"]["lower"] == np.quantile(explicit, 0.025)
    assert primary["interval"]["upper"] == np.quantile(explicit, 0.975)
    assert result["bootstrap"]["primary_comparisons"] == 1


def test_positive_primary_interval_does_not_hide_a_negative_descriptive_block():
    banks, seeds = libraries(), set(range(100))
    rows = synthetic_rows(banks, seeds)
    for row in rows:
        if "_labeled_" in row["mode"]:
            row["best_so_far_auc"] = 20 + (-0.1 if row["seed"] < 20 else 10)
    result = paired.analyze(rows, banks, frozen_choices(banks), seeds,
                            dict(pair_count=2, bootstrap_seed=9,
                                 bootstrap_replicates=10000, heldout_blocks=5))
    assert result["primary_mean_advantage"] is True
    assert result["stable_advantage"] is False
    assert result["primary_ungated_auc"]["ordered_seed_blocks"][0]["mean_delta"] < 0


@pytest.mark.parametrize("fault", ["missing_cell", "duplicate_cell", "missing_library",
                                   "cross_library_selection", "different_anchor"])
def test_incomplete_or_unpaired_analysis_fails(fault):
    banks, seeds = libraries(), {31, 32, 33, 34}
    rows, choices = synthetic_rows(banks, seeds), frozen_choices(banks)
    if fault == "missing_cell":
        rows.pop()
    elif fault == "duplicate_cell":
        rows.append(deepcopy(rows[0]))
    elif fault == "missing_library":
        banks.pop("pair_00_anonymous")
    elif fault == "cross_library_selection":
        choices["pair_00_labeled"]["ungated_calibration_winner"] = choices["pair_01_labeled"]["ungated_calibration_winner"]
    else:
        choices["pair_00_anonymous"]["selection"]["target_anchor_mode"] = paired.matched.BASELINES[1]
    with pytest.raises(ValueError):
        paired.analyze(rows, banks, choices, seeds,
                       dict(pair_count=2, bootstrap_seed=9, bootstrap_replicates=20, heldout_blocks=2))


def test_holdout_cannot_start_until_calibration_selection_is_written_and_verified(tmp_path):
    banks = libraries()
    calibration, heldout = set(range(10, 15)), set(range(31, 35))
    config = dict(pair_count=2, bootstrap_seed=9, bootstrap_replicates=20, heldout_blocks=2,
                  min_risk_adjusted_gain=0.25, min_positive_fold_rate=0.8)
    paired.matched.save_json(tmp_path / "manifest.json", {"frozen": True})
    steps = []

    def phase(split, seeds, _config, _skills, _workers, output):
        steps.append(split)
        if split == "heldout":
            lock = paired.strict_json(output / "selection_lock.json")
            assert lock["heldout_started"] is False
            assert set(lock["selections"]) == set(banks)
            assert steps == ["calibration", "verify", "heldout"]
        rows = synthetic_rows(banks, seeds, split)
        paired.matched.nulls.write_csv(output / f"{split}_metrics.csv", rows)
        return rows, 0.0

    def verify():
        assert (tmp_path / "selection_lock.json").exists()
        assert not (tmp_path / "heldout_start.json").exists()
        steps.append("verify")

    with patch.object(paired.matched, "run_phase", side_effect=phase):
        result, _ = paired.evaluate_phases(config, banks, calibration, heldout, 1, tmp_path,
                                           "protocol", verify)
    assert result["primary_ungated_auc"]["mean_delta"] == 6


def test_incomplete_calibration_never_starts_holdout(tmp_path):
    banks = libraries()
    calibration = set(range(10, 15))
    rows = synthetic_rows(banks, calibration, "calibration")[:-1]
    called = []

    def phase(split, *args):
        called.append(split)
        return rows, 0.0

    with patch.object(paired.matched, "run_phase", side_effect=phase):
        with pytest.raises(ValueError, match="Incomplete"):
            paired.evaluate_phases(
                dict(min_risk_adjusted_gain=0.25, min_positive_fold_rate=0.8),
                banks, calibration, {31, 32}, 1, tmp_path, "protocol", lambda: None)
    assert called == ["calibration"]
    assert not (tmp_path / "selection_lock.json").exists()


def handoff(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    scripts = root / "experiments/care_replay/scripts"
    scripts.mkdir(parents=True)
    (scripts / "one.py").write_text("# frozen code\n")
    (root / "data.txt").write_text("frozen data\n")
    (root / "historical.json").write_text("{}")
    monkeypatch.setattr(paired, "ROOT", root)
    config = {**paired.FIXED_PARAMETERS, "calibration_seed_start": 202610201000,
              "heldout_seed_start": 202610202000, "bootstrap_seed": 2026102003,
              "data_files": ["data.txt"]}
    config_path, lock_path = root / "config.json", root / "lock.json"
    paired.matched.save_json(config_path, config)
    generation = tmp_path / "generation"
    (generation / "libraries").mkdir(parents=True)
    messages, materials = {}, {}
    for subdir, names, destination in (
        ("messages", paired.library_ids(10), messages),
        ("pairs", [f"pair_{i:02d}" for i in range(10)], materials),
    ):
        (generation / "input_snapshot" / subdir).mkdir(parents=True)
        for name in names:
            path = generation / "input_snapshot" / subdir / (name + ".json")
            paired.matched.save_json(path, {"frozen": name})
            destination[name] = paired.matched.file_digest(path)
    library_hashes = {}
    for name in paired.library_ids(10):
        path = generation / "libraries" / (name + ".json")
        paired.matched.save_json(path, payload())
        library_hashes[name] = paired.matched.file_digest(path)
    lock = {
        "protocol_id": config["protocol_id"], "status": "frozen_before_generation",
        "model_api_calls_at_freeze": 0, "config_sha256": paired.matched.file_digest(config_path),
        "source_sha256": {"experiments/care_replay/scripts/one.py":
                          paired.matched.file_digest(scripts / "one.py")},
        "data_sha256": {"data.txt": paired.matched.file_digest(root / "data.txt")},
        "input_sha256": {"historical.json": paired.matched.file_digest(root / "historical.json")},
        "messages_sha256": messages, "pair_material_sha256": materials,
    }
    paired.matched.save_json(lock_path, lock)
    completion = {"status": "complete", "config_sha256": lock["config_sha256"],
                  "protocol_lock_sha256": paired.matched.file_digest(lock_path),
                  "libraries_sha256": library_hashes, "messages_sha256": messages,
                  "pair_material_sha256": materials}
    paired.matched.save_json(generation / "completion.json", completion)
    return root, config_path, lock_path, generation, config


@pytest.mark.parametrize("fault", ["none", "failed", "missing", "changed_library",
                                   "changed_source", "changed_data", "changed_message", "changed_lock"])
def test_generation_handoff_requires_exact_complete_frozen_bytes(tmp_path, monkeypatch, fault):
    root, config_path, lock_path, generation, config = handoff(tmp_path, monkeypatch)
    if fault == "failed":
        completion = paired.strict_json(generation / "completion.json")
        completion["status"] = "failed"
        paired.matched.save_json(generation / "completion.json", completion)
    elif fault == "missing":
        (generation / "libraries/pair_00_labeled.json").unlink()
    elif fault == "changed_library":
        (generation / "libraries/pair_00_labeled.json").write_text("{}")
    elif fault == "changed_source":
        (root / "experiments/care_replay/scripts/one.py").write_text("# changed\n")
    elif fault == "changed_data":
        (root / "data.txt").write_text("changed")
    elif fault == "changed_message":
        (generation / "input_snapshot/messages/pair_00_labeled.json").write_text("{}")
    elif fault == "changed_lock":
        with lock_path.open("a") as handle:
            handle.write("\n")
    if fault == "none":
        paired.verify_frozen_inputs(config_path, lock_path, generation, config)
    else:
        with pytest.raises(ValueError):
            paired.verify_frozen_inputs(config_path, lock_path, generation, config)


def test_real_protocol_schema_and_fresh_seed_boundaries():
    config_path = SCRIPTS.parent / "configs/bh_semantic_anonymization_v1.json"
    config = paired.strict_json(config_path)
    calibration, heldout = paired.validate_config(config)
    assert calibration == set(range(202610201000, 202610201050))
    assert heldout == set(range(202610202000, 202610202100))
    assert config["bootstrap_seed"] == 2026102003
    changed = deepcopy(config)
    changed["analysis"]["primary_policy"] = "selected_mode"
    with pytest.raises(ValueError, match="Analysis contract"):
        paired.validate_config(changed)


def test_complete_handoff_to_snapshots_selection_and_analysis_without_replay_or_api(tmp_path, monkeypatch):
    root, config_path, lock_path, generation, config = handoff(tmp_path, monkeypatch)
    # Use the real formal schema while substituting only fixture data paths.
    formal = paired.strict_json(SCRIPTS.parent / "configs/bh_semantic_anonymization_v1.json")
    formal["data_files"] = config["data_files"]
    paired.matched.save_json(config_path, formal)
    lock = paired.strict_json(lock_path)
    lock["config_sha256"] = paired.matched.file_digest(config_path)
    paired.matched.save_json(lock_path, lock)
    completion = paired.strict_json(generation / "completion.json")
    completion.update(config_sha256=lock["config_sha256"],
                      protocol_lock_sha256=paired.matched.file_digest(lock_path))
    paired.matched.save_json(generation / "completion.json", completion)
    from types import SimpleNamespace
    monkeypatch.setitem(paired.matched.replay.DATASET_BUILDERS, "real_buchwald_hartwig",
                        lambda: SimpleNamespace(candidates=[None] * 3955))
    monkeypatch.setattr(paired.matched.semantic, "semantic_field_catalog", lambda _: CATALOG)
    calls = []

    def phase(split, seeds, _config, skills, _workers, output):
        calls.append(split)
        if split == "heldout":
            assert (output / "selection_lock.json").exists()
            assert (output / "heldout_start.json").exists()
        banks = {name: tuple(skill for skill in skills if skill.skill_id.startswith(name + "_slot_"))
                 for name in paired.library_ids(10)}
        rows = synthetic_rows(banks, seeds, split)
        paired.matched.nulls.write_csv(output / f"{split}_metrics.csv", rows)
        (output / f"{split}_traces.jsonl").write_text("")
        return rows, 0.0

    monkeypatch.setattr(paired.matched, "run_phase", phase)
    output = tmp_path / "evaluated"
    result = paired.run(config_path, lock_path, generation, output, workers=1)
    assert calls == ["calibration", "heldout"]
    assert result["status"] == "complete"
    assert result["model_api_calls"] == 0
    assert result["completed_logical_trajectories"] == 15450
    assert result["primary_mean_auc_delta"] == 14
    assert (output / "source_snapshot/experiments/care_replay/scripts/one.py").exists()
    assert len(list((output / "generation_snapshot/libraries").glob("*.json"))) == 20
    assert len(list((output / "generation_snapshot/input_snapshot/messages").glob("*.json"))) == 20
    assert len(list((output / "generation_snapshot/input_snapshot/pairs").glob("*.json"))) == 10
    integrity = paired.strict_json(output / "integrity.json")
    assert integrity["selection_lock.json"] == paired.matched.file_digest(output / "selection_lock.json")
    assert not (output / "failure.json").exists()
