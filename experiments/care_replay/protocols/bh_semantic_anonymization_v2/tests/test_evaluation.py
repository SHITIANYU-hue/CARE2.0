"""Synthetic-only tests: no model requests or real BH calibration/heldout runs."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import shutil
import socket
import sys

import numpy as np
import pytest

PROTOCOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOCOL))
import evaluation as e

FORMAL_CONFIG = PROTOCOL.parents[1] / "configs/bh_semantic_anonymization_v2.json"
CATALOG = {"field": {"value": 20}}


def formal():
    return e.common.load(FORMAL_CONFIG)


def payload():
    values = dict(ridge=1.25, prior_scale=0.4, semantic_mass_start=0.3,
                  semantic_mass_end=0.1, ucb_weight=0.6, gp_beta_start=1.7,
                  gp_beta_end=0.8, gp_xi=0.03, confidence=0.75)
    return {"skills": [
        {"skill_id": f"skill_{slot}", "hypothesis": "Hypothesis", **values,
         "rules": [{"rule_id": f"rule_{rule}", "conditions": {"field": "value"},
                    "weight": -0.4 if rule else 0.6, "rationale": "Rationale"}
                   for rule in range(count)]}
        for slot, count in enumerate((2, 1, 1, 1, 1))]}


def banks_and_outcomes(valid=()):
    valid = set(valid)
    banks, outcomes = {}, {}
    for pair in range(10):
        for arm in e.ARMS:
            identifier = f"pair_{pair:02d}_{arm}"
            status = e.VALID if identifier in valid else e.INVALID
            outcomes[identifier] = dict(identifier=identifier, pair_index=pair, arm=arm,
                                        status=status, reason_codes=[] if status == e.VALID
                                        else ["invalid_library_contract", "field is illegal"])
            if status == e.VALID:
                banks[identifier] = e.v1.normalize_without_loss(payload(), CATALOG, pair, arm)[0]
    return banks, outcomes


def synthetic_rows(banks, seeds, split="heldout", offsets=None):
    rows = []
    for seed_index, seed in enumerate(sorted(seeds)):
        for mode in e.matched.BASELINES:
            rows.append(dict(mode=mode, seed=seed, split=split, final_best=10.0,
                             best_so_far_auc=10.0, top10_hit=0))
        for name, bank in banks.items():
            delta = offsets[name] if offsets and name in offsets else 5
            if callable(delta):
                delta = delta(seed_index, seed)
            for skill in bank:
                rows.append(dict(mode=e.matched.semantic.skill_mode(skill), seed=seed,
                                 split=split, final_best=10.0 + delta,
                                 best_so_far_auc=10.0 + delta, top10_hit=int(delta > 0)))
    return rows


def choices_for(banks, outcomes, config=None, offsets=None):
    return e.freeze_choices(synthetic_rows(banks, range(50), "calibration", offsets),
                            banks, outcomes, set(range(50)), config or formal())[0]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    e.common.save(path, value)
    return e.common.file_hash(path)


def test_formal_configuration_exact_budget_thresholds_and_fresh_seeds():
    config = formal()
    calibration, heldout = e.validate_config(config, 4)
    assert calibration == set(range(202610205000, 202610205050))
    assert heldout == set(range(202610206000, 202610206100))
    assert config["selection"]["strategies_per_library"] == 5
    assert config["min_risk_adjusted_gain"] == 0.25
    assert config["min_positive_fold_rate"] == 0.8
    assert config["bootstrap_seed"] == 2026102004
    for key, value in (("pair_count", 9), ("calibration_seed_count", 49),
                       ("heldout_seed_count", 99), ("bootstrap_replicates", 1000)):
        altered = deepcopy(config)
        altered[key] = value
        with pytest.raises(ValueError):
            e.validate_config(altered, 4)
    with pytest.raises(ValueError, match="four workers"):
        e.validate_config(config, 2)


def test_all_invalid_uses_one_calibration_anchor_zero_by_construction_not_equivalence():
    banks, outcomes = banks_and_outcomes()
    config = formal()
    choices = choices_for(banks, outcomes, config)
    anchor = max(e.matched.BASELINES)
    assert {choice["primary_mode"] for choice in choices.values()} == {anchor}
    assert {choice["selected_mode"] for choice in choices.values()} == {anchor}
    result = e.analyze(synthetic_rows(banks, range(100)), banks, outcomes, choices, set(range(100)), config)
    primary = result["primary_all_pairs_auc"]
    assert primary["delta_matrix"] == [[0.0] * 100 for _ in range(10)]
    assert primary["interval"]["lower"] == primary["interval"]["upper"] == 0
    assert result["all_invalid_constructed_zero"] is True
    assert "not equivalence" in result["degenerate_interpretation"]
    assert result["both_valid_pairs_descriptive"]["available"] is False
    assert "descriptive_auc" not in result["both_valid_pairs_descriptive"]
    assert result["pair_validity_counts"]["both_invalid"] == 10
    assert result["gate_accepted_library_counts_by_arm"] == {"labeled": 0, "anonymous": 0}
    assert result["stable_advantage"] is result["primary_mean_advantage"] is False


def test_mixed_validity_retains_all_ten_pairs_and_both_valid_subset_is_descriptive_only():
    banks, outcomes = banks_and_outcomes(
        ["pair_00_labeled", "pair_01_anonymous", "pair_02_labeled", "pair_02_anonymous"])
    offsets = {"pair_00_labeled": 5, "pair_01_anonymous": 3,
               "pair_02_labeled": 7, "pair_02_anonymous": 2}
    choices = choices_for(banks, outcomes, offsets=offsets)
    result = e.analyze(synthetic_rows(banks, range(100), offsets=offsets),
                       banks, outcomes, choices, set(range(100)), formal())
    assert result["primary_all_pairs_auc"]["pair_mean_deltas"] == [5, -3, 5] + [0] * 7
    assert result["primary_all_pairs_auc"]["mean_delta"] == 0.7
    assert result["valid_library_counts_by_arm"] == {"labeled": 2, "anonymous": 2}
    assert result["pair_validity_counts"] == dict(both_valid=1, both_invalid=7,
                                                 labeled_invalid_only=1, anonymous_invalid_only=1)
    subset = result["both_valid_pairs_descriptive"]
    assert subset["pair_ids"] == ["pair_02"]
    assert subset["descriptive_auc"]["mean_delta"] == 5
    assert "interval" not in subset["descriptive_auc"]
    assert result["gate_accepted_library_counts_by_arm"] == {"labeled": 2, "anonymous": 2}


def test_primary_valid_raw_choice_survives_gate_rejection_and_holdout_never_reselects():
    banks, outcomes = banks_and_outcomes(["pair_00_labeled"])
    choices = choices_for(banks, outcomes, offsets={"pair_00_labeled": -5})
    selected = choices["pair_00_labeled"]
    assert selected["primary_mode"] != selected["selected_mode"]
    assert selected["selected_mode"] == selected["selection"]["target_anchor_mode"]
    frozen = deepcopy(choices)
    rows = synthetic_rows(banks, range(100), offsets={"pair_00_labeled": -2})
    # A different candidate becomes attractive on heldout, but primary remains frozen.
    other = next(e.matched.semantic.skill_mode(s) for s in banks["pair_00_labeled"]
                 if e.matched.semantic.skill_mode(s) != selected["primary_mode"])
    for row in rows:
        if row["mode"] == other:
            row["best_so_far_auc"] = 90
    result = e.analyze(rows, banks, outcomes, choices, set(range(100)), formal())
    assert choices == frozen
    assert result["primary_all_pairs_auc"]["mean_delta"] == -0.2
    assert result["secondary_gated_auc"]["mean_delta"] == 0


def test_crossed_bootstrap_resamples_pairs_and_shared_starts_not_independent_cells():
    banks, outcomes = banks_and_outcomes(["pair_00_labeled", "pair_01_labeled"])
    choices = choices_for(banks, outcomes)
    seeds = {31, 32, 33, 34}
    config = formal()
    config.update(heldout_blocks=2)
    offsets = {"pair_00_labeled": lambda i, _: 1 + (8 if i >= 2 else 0),
               "pair_01_labeled": lambda i, _: 3 + (8 if i >= 2 else 0)}
    result = e.analyze(synthetic_rows(banks, seeds, offsets=offsets), banks, outcomes, choices, seeds, config)
    matrix = np.array([[1, 1, 9, 9], [3, 3, 11, 11]] + [[0] * 4] * 8)
    assert result["primary_all_pairs_auc"]["delta_matrix"] == matrix.tolist()
    rng = np.random.default_rng(config["bootstrap_seed"])
    starts = rng.multinomial(4, np.full(4, 0.25), size=10000)
    pairs = rng.multinomial(10, np.full(10, 0.1), size=10000)
    explicit = np.einsum("bi,ij,bj->b", pairs, matrix, starts) / 40
    interval = result["primary_all_pairs_auc"]["interval"]
    assert interval["lower"] == np.quantile(explicit, 0.025)
    assert interval["upper"] == np.quantile(explicit, 0.975)
    assert result["bootstrap"]["primary_hypotheses"] == 1


def test_stability_requires_blocks_but_not_every_pair_mean_positive():
    valid = [f"pair_{p:02d}_labeled" for p in range(10)]
    banks, outcomes = banks_and_outcomes(valid)
    choices = choices_for(banks, outcomes)
    offsets = {name: (10 if name != "pair_09_labeled" else -0.1) for name in valid}
    result = e.analyze(synthetic_rows(banks, range(100), offsets=offsets), banks, outcomes,
                       choices, set(range(100)), formal())
    assert result["primary_mean_advantage"] is result["stable_advantage"] is True
    assert result["primary_all_pairs_auc"]["pair_mean_deltas"][-1] < 0
    offsets = {name: lambda i, _: -0.1 if i < 20 else 10 for name in valid}
    result = e.analyze(synthetic_rows(banks, range(100), offsets=offsets), banks, outcomes,
                       choices, set(range(100)), formal())
    assert result["primary_mean_advantage"] is True
    assert result["stable_advantage"] is False


@pytest.mark.parametrize("fault", ["missing_cell", "duplicate_cell", "missing_outcome", "halt",
                                  "partial_invalid", "cross_bank", "different_anchor", "missing_choice"])
def test_analysis_fails_closed_for_missing_invalid_or_unpaired_inputs(fault):
    banks, outcomes = banks_and_outcomes(["pair_00_labeled", "pair_01_labeled"])
    choices = choices_for(banks, outcomes)
    rows = synthetic_rows(banks, range(10))
    if fault == "missing_cell":
        rows.pop()
    elif fault == "duplicate_cell":
        rows.append(deepcopy(rows[0]))
    elif fault == "missing_outcome":
        outcomes.pop("pair_09_anonymous")
    elif fault == "halt":
        outcomes["pair_09_anonymous"]["status"] = "HALT"
    elif fault == "partial_invalid":
        banks["pair_09_anonymous"] = banks["pair_00_labeled"]
    elif fault == "cross_bank":
        choices["pair_00_labeled"]["primary_mode"] = choices["pair_01_labeled"]["primary_mode"]
    elif fault == "different_anchor":
        choices["pair_09_anonymous"]["primary_mode"] = e.matched.BASELINES[0]
    else:
        choices.pop("pair_09_anonymous")
    with pytest.raises(ValueError):
        e.analyze(rows, banks, outcomes, choices, set(range(10)), formal())


def test_all_invalid_phase_order_and_450_baseline_trajectories(tmp_path, monkeypatch):
    banks, outcomes = banks_and_outcomes()
    config = formal()
    calibration, heldout = e.validate_config(config, 4)
    save(tmp_path / "manifest.json", {"frozen": True})
    steps = []

    def phase(split, seeds, _config, skills, workers, output):
        assert skills == ()
        assert workers == 4
        if split == "heldout":
            assert steps == ["calibration", "verify"]
            lock = e.common.load(output / "selection_lock.json")
            assert lock["heldout_started"] is False
            assert len(lock["choices"]) == 20
            assert (output / "heldout_start.json").exists()
        steps.append(split)
        rows = synthetic_rows(banks, seeds, split)
        e.matched.nulls.write_csv(output / f"{split}_metrics.csv", rows)
        return rows, 0.0

    def verify():
        assert (tmp_path / "selection_lock.json").exists()
        assert not (tmp_path / "heldout_start.json").exists()
        steps.append("verify")

    monkeypatch.setattr(e.matched, "run_phase", phase)
    result, execution = e.evaluate_phases(config, banks, outcomes, calibration, heldout,
                                          tmp_path, "protocol", verify)
    assert steps == ["calibration", "verify", "heldout"]
    assert execution["completed_logical_trajectories"] == 450
    assert result["all_invalid_constructed_zero"] is True


def test_incomplete_calibration_cannot_begin_holdout(tmp_path, monkeypatch):
    banks, outcomes = banks_and_outcomes()
    calls = []

    def phase(split, seeds, *_):
        calls.append(split)
        return synthetic_rows(banks, seeds, split)[:-1], 0.0

    monkeypatch.setattr(e.matched, "run_phase", phase)
    with pytest.raises(ValueError, match="Incomplete"):
        e.evaluate_phases(formal(), banks, outcomes, set(range(50)), set(range(100, 200)),
                          tmp_path, "protocol", lambda: None)
    assert calls == ["calibration"]
    assert not (tmp_path / "heldout_start.json").exists()


def handoff(tmp_path, monkeypatch, valid=()):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(e, "ROOT", root)
    config = formal()
    config.update(generation_output_directory="generation", evaluation_output_directory="evaluation")
    config_path, lock_path = root / "config.json", root / "lock.json"
    save(config_path, config)
    base = root / "generation"
    for folder in ("libraries", "outcomes", "receipts"):
        (base / folder).mkdir(parents=True)
    lock = {"config_sha256": e.common.file_hash(config_path), "source_sha256": {},
            "data_sha256": {}, "input_sha256": {}, "messages_sha256": {}, "pair_material_sha256": {}}
    for group, folder, relative in (("source_sha256", "source_snapshot", "scripts/frozen.py"),
                                     ("data_sha256", "input_snapshot", "data.json"),
                                     ("input_sha256", "input_snapshot", "prior_ledger.json")):
        lock[group][relative] = save(root / relative, {"frozen": relative})
        destination = base / folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    for name in e.names(config):
        lock["messages_sha256"][name] = save(base / f"input_snapshot/messages/{name}.json",
                                              [{"role": "user", "content": name}])
    for pair in range(10):
        name = f"pair_{pair:02d}"
        lock["pair_material_sha256"][name] = save(base / f"input_snapshot/pairs/{name}.json", {"pair_index": pair})
    save(lock_path, lock)
    shutil.copyfile(config_path, base / "config.json")
    shutil.copyfile(lock_path, base / "protocol_lock.json")
    _, outcomes = banks_and_outcomes(valid)
    library_hashes, outcome_hashes, attempts = {}, {}, []
    for slot_number, slot in enumerate(config["generation_order"]):
        identifier = f"pair_{slot['pair_index']:02d}_{slot['arm']}"
        receipt = base / "receipts" / identifier
        request = {"model": config["model"]["id"],
                   "messages": e.common.load(base / f"input_snapshot/messages/{identifier}.json"),
                   "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_tokens"],
                   "response_format": {"type": "json_object"}, "stream": False, "provider": config["api"]["provider"]}
        status = outcomes[identifier]["status"]
        response_id = f"gen-fixture-{slot_number}"
        response = {"id": response_id, "fixture_status": status, "model": config["model"]["id"],
                    "provider": "OpenAI", "usage": {"cost": None, "is_byok": False}}
        save(receipt / "cost_lookup_0.json", {"data": {
            "id": response_id, "model": response["model"], "provider_name": response["provider"],
            "total_cost": "0.001", "is_byok": False}})
        library_hash = save(base / f"libraries/{identifier}.json", payload()) if status == e.VALID else None
        if library_hash:
            library_hashes[identifier] = library_hash
        outcome = outcomes[identifier]
        outcome.update(errors=outcome["reason_codes"][1:], response_id=response_id, cost_usd="0.001",
                       request_sha256=save(receipt / "request.json", request),
                       response_sha256=save(receipt / "response.json", response),
                       billing_sha256=save(receipt / "billing.json", {"source": "generation_lookup",
                           "response_id": response_id, "cost_usd": "0.001", "is_byok": False}),
                       library_sha256=library_hash)
        outcome_hashes[identifier] = save(base / f"outcomes/{identifier}.json", outcome)
        attempts.append({"slot": slot_number, **outcome, "outcome_sha256": outcome_hashes[identifier]})
    totals = {"prior_cost_usd": "0.0125145", "known_new_cost_usd": "0.020",
              "cumulative_known_cost_usd": "0.0325145", "actual_post_attempts": 20}
    ledger_hash = save(base / "ledger.json", {"attempts": attempts, **totals})
    completion = {"status": "complete", "config_sha256": lock["config_sha256"],
                  "protocol_lock_sha256": e.common.file_hash(lock_path), "ledger_sha256": ledger_hash,
                  "messages_sha256": lock["messages_sha256"], "pair_material_sha256": lock["pair_material_sha256"],
                  "outcomes_sha256": outcome_hashes, "libraries_sha256": library_hashes,
                  "valid_library_count": len(library_hashes), "invalid_library_count": 20 - len(library_hashes), **totals}
    save(base / "completion.json", completion)

    def classify(response, arm, material, cost):
        status = response["fixture_status"]
        return SimpleNamespace(status=status, arm=arm, cost_usd=str(cost),
                               reasons=() if status == e.VALID else ("invalid_library_contract", "field is illegal"),
                               library=payload() if status == e.VALID else None)

    monkeypatch.setattr(e, "replay_classification", classify)
    return SimpleNamespace(root=root, config_path=config_path, lock_path=lock_path,
                           base=base, config=config, lock=lock, completion=completion)


def check_handoff(f):
    return e.verify_generation(f.config_path, f.lock_path, f.base, f.config, f.lock)


def update_outcome(f, identifier, callback):
    path = f.base / f"outcomes/{identifier}.json"
    value = e.common.load(path)
    callback(value)
    f.completion["outcomes_sha256"][identifier] = save(path, value)
    save(f.base / "completion.json", f.completion)


def test_receipt_handoff_accepts_twenty_terminal_slots_and_settled_get_billing(tmp_path, monkeypatch):
    f = handoff(tmp_path, monkeypatch, valid=["pair_00_labeled"])
    outcomes, payloads, hashes = check_handoff(f)
    assert len(outcomes) == 20
    assert set(payloads) == {"pair_00_labeled"}
    assert "input_snapshot/prior_ledger.json" in hashes
    assert "input_snapshot/data.json" in hashes
    assert "source_snapshot/scripts/frozen.py" in hashes
    assert len([key for key in hashes if key.endswith("/billing.json")]) == 20


@pytest.mark.parametrize("fault", ["missing_outcome", "halt", "failure_present", "changed_response",
                                  "changed_billing", "changed_data_snapshot", "changed_input_snapshot",
                                  "changed_source_snapshot", "changed_message", "partial_invalid",
                                  "missing_valid", "changed_ledger", "classification_mismatch",
                                  "request_mismatch", "billing_byok", "unknown_cost"])
def test_provenance_handoff_rejects_missing_or_changed_evidence(tmp_path, monkeypatch, fault):
    f = handoff(tmp_path, monkeypatch, valid=["pair_00_labeled"])
    identifier = "pair_00_anonymous"
    receipt = f.base / "receipts" / identifier
    if fault == "missing_outcome":
        (f.base / f"outcomes/{identifier}.json").unlink()
    elif fault == "halt":
        update_outcome(f, identifier, lambda value: value.update(status="HALT"))
    elif fault == "failure_present":
        save(f.base / "failure.json", {"status": "HALT"})
    elif fault.startswith("changed_"):
        paths = {"changed_response": receipt / "response.json", "changed_billing": receipt / "billing.json",
                 "changed_data_snapshot": f.base / "input_snapshot/data.json",
                 "changed_input_snapshot": f.base / "input_snapshot/prior_ledger.json",
                 "changed_source_snapshot": f.base / "source_snapshot/scripts/frozen.py",
                 "changed_message": f.base / f"input_snapshot/messages/{identifier}.json",
                 "changed_ledger": f.base / "ledger.json"}
        paths[fault].write_text("{}")
    elif fault == "partial_invalid":
        save(f.base / f"libraries/{identifier}.json", payload())
    elif fault == "missing_valid":
        (f.base / "libraries/pair_00_labeled.json").unlink()
    elif fault == "classification_mismatch":
        update_outcome(f, identifier, lambda value: value.update(reason_codes=["different"], errors=[]))
    elif fault == "request_mismatch":
        path = receipt / "request.json"
        value = e.common.load(path)
        value["temperature"] = 0
        digest = save(path, value)
        update_outcome(f, identifier, lambda value: value.update(request_sha256=digest))
    else:
        path = receipt / "billing.json"
        value = e.common.load(path)
        value.update({"is_byok": True} if fault == "billing_byok" else {"cost_usd": None})
        digest = save(path, value)
        update_outcome(f, identifier, lambda value: value.update(billing_sha256=digest))
    with pytest.raises(ValueError):
        check_handoff(f)


@pytest.mark.parametrize("fault", ["order", "slot_number", "request_hash", "response_id",
                                  "outcome_hash", "total", "post_count", "status"])
def test_completed_ledger_requires_order_provenance_and_accounting_even_after_rehash(tmp_path, monkeypatch, fault):
    f = handoff(tmp_path, monkeypatch)
    ledger = e.common.load(f.base / "ledger.json")
    if fault == "order":
        ledger["attempts"][0], ledger["attempts"][1] = ledger["attempts"][1], ledger["attempts"][0]
    elif fault == "slot_number":
        ledger["attempts"][0]["slot"] = 1
    elif fault == "request_hash":
        ledger["attempts"][0]["request_sha256"] = "0" * 64
    elif fault == "response_id":
        ledger["attempts"][0]["response_id"] = "different"
    elif fault == "outcome_hash":
        ledger["attempts"][0]["outcome_sha256"] = "0" * 64
    elif fault == "total":
        ledger["cumulative_known_cost_usd"] = "0.03"
    elif fault == "post_count":
        ledger["actual_post_attempts"] = 21
    else:
        ledger["attempts"][0]["status"] = "HALT"
    f.completion["ledger_sha256"] = save(f.base / "ledger.json", ledger)
    save(f.base / "completion.json", f.completion)
    with pytest.raises(ValueError):
        check_handoff(f)


@pytest.mark.parametrize("valid", [[], ["pair_00_labeled", "pair_01_anonymous"]])
def test_full_offline_flow_fake_phase_snapshots_integrity_and_early_selection(tmp_path, monkeypatch, valid):
    f = handoff(tmp_path, monkeypatch, valid=valid)
    monkeypatch.setattr(e.common, "verify_freeze", lambda *_: (f.config, f.lock))
    monkeypatch.setitem(e.matched.replay.DATASET_BUILDERS, "real_buchwald_hartwig",
                        lambda: SimpleNamespace(candidates=[None] * 3955))
    monkeypatch.setattr(e.matched.semantic, "semantic_field_catalog", lambda _: CATALOG)
    banks, _ = banks_and_outcomes(valid)
    steps = []

    def phase(split, seeds, config, skills, workers, output):
        assert workers == 4
        assert len(skills) == 5 * len(valid)
        with pytest.raises(RuntimeError):
            socket.create_connection(("example.invalid", 443))
        if split == "heldout":
            assert (output / "selection_lock.json").is_file()
            assert (output / "heldout_start.json").is_file()
            assert steps == ["calibration"]
        steps.append(split)
        rows = synthetic_rows(banks, seeds, split)
        e.matched.nulls.write_csv(output / f"{split}_metrics.csv", rows)
        (output / f"{split}_traces.jsonl").write_text("")
        return rows, 0.0

    monkeypatch.setattr(e.matched, "run_phase", phase)
    output = f.root / "evaluation"
    result = e.run(f.config_path, f.lock_path, f.root / "freeze", f.base, output)
    assert steps == ["calibration", "heldout"]
    assert result["status"] == "complete"
    assert result["model_api_calls"] == 0
    assert result["completed_logical_trajectories"] == (3 + 5 * len(valid)) * 150
    assert result["all_invalid_constructed_zero"] == (not valid)
    assert len(list((output / "generation_snapshot/outcomes").glob("*.json"))) == 20
    assert len(list((output / "generation_snapshot/receipts").glob("*/billing.json"))) == 20
    assert (output / "generation_snapshot/input_snapshot/prior_ledger.json").is_file()
    assert (output / "source_snapshot/scripts/frozen.py").is_file()
    integrity = e.common.load(output / "integrity.json")
    assert integrity["selection_lock.json"] == e.common.file_hash(output / "selection_lock.json")
    assert not (output / "failure.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        e.run(f.config_path, f.lock_path, f.root / "freeze", f.base, output)


def test_halt_blocks_before_data_loading_or_calibration(tmp_path, monkeypatch):
    f = handoff(tmp_path, monkeypatch)
    save(f.base / "failure.json", {"status": "HALT"})
    monkeypatch.setattr(e.common, "verify_freeze", lambda *_: (f.config, f.lock))

    def forbidden(*_):
        pytest.fail("Neither data loading nor calibration may start on HALT")

    monkeypatch.setitem(e.matched.replay.DATASET_BUILDERS, "real_buchwald_hartwig", forbidden)
    monkeypatch.setattr(e.matched, "run_phase", forbidden)
    with pytest.raises(ValueError, match="halted"):
        e.run(f.config_path, f.lock_path, f.root / "freeze", f.base, f.root / "evaluation")
    assert not (f.root / "evaluation").exists()


def test_billing_reconciliation_requires_original_inline_or_lookup_evidence(tmp_path):
    response = {"id": "gen-one", "model": "openai/gpt-4o-mini", "provider": "OpenAI",
                "usage": {"cost": "0.001", "is_byok": False}}
    billing = {"source": "response_usage", "response_id": "gen-one",
               "cost_usd": "0.001", "is_byok": False}
    e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.001"))
    with pytest.raises(ValueError, match="original response"):
        e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.002"))
    response["usage"] = {"cost": "0.002"}  # Cost known, BYOK status only settles via GET.
    billing["source"] = "generation_lookup"
    save(tmp_path / "cost_lookup_0_error.json", {"http_status": 404})
    lookup = {"data": {"id": "gen-one", "model": response["model"], "provider_name": "OpenAI",
                       "is_byok": False, "total_cost": "0.001"}}
    save(tmp_path / "cost_lookup_1.json", lookup)
    e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.002"))
    with pytest.raises(ValueError, match="original lookup"):
        e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.001"))
    lookup["data"]["provider_name"] = "Different"
    save(tmp_path / "cost_lookup_1.json", lookup)
    with pytest.raises(ValueError, match="provenance"):
        e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.002"))


@pytest.mark.parametrize("fault", ["missing", "wrong_id", "byok", "negative", "nan", "unsettled",
                                  "fatal_error", "ambiguous", "usage_byok", "wrong_source"])
def test_billing_reconciliation_rejects_unsettled_or_unsupported_evidence(tmp_path, fault):
    response = {"id": "gen-one", "model": "openai/gpt-4o-mini", "provider": "OpenAI",
                "usage": {"cost": None}}
    billing = {"source": "generation_lookup"}
    lookup = {"data": {"id": "gen-one", "model": response["model"], "provider_name": "OpenAI",
                       "is_byok": False, "total_cost": "0.001"}}
    if fault == "wrong_id":
        lookup["data"]["id"] = "other"
    elif fault == "byok":
        lookup["data"]["is_byok"] = True
    elif fault == "negative":
        lookup["data"]["total_cost"] = -1
    elif fault == "nan":
        lookup["data"]["total_cost"] = "NaN"
    elif fault == "unsettled":
        lookup["data"]["is_byok"] = None
    elif fault == "usage_byok":
        response["usage"]["is_byok"] = True
    elif fault == "wrong_source":
        billing["source"] = "response_usage"
    if fault not in {"missing", "fatal_error"}:
        save(tmp_path / "cost_lookup_0.json", lookup)
    if fault in {"fatal_error", "ambiguous"}:
        save(tmp_path / "cost_lookup_0_error.json", {"http_status": 500})
    with pytest.raises(ValueError):
        e.verify_billing_evidence(response, billing, tmp_path, Decimal("0.001"))
