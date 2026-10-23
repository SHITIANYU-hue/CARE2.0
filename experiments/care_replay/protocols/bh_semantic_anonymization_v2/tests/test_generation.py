"""Offline production-generation tests. All network and credential reads are forbidden."""
from copy import deepcopy
from decimal import Decimal
import builtins
import io
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import generation as g
REAL_ROOT = g.ROOT


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline generation tests prohibit network and credentials")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(g.legacy, "load_key", forbidden)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open")):
        original = getattr(module, name)
        def guarded(path, *args, _original=original, **kwargs):
            if not isinstance(path, int) and Path(os.fsdecode(path)).name.startswith(".env"):
                return forbidden()
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(module, name, guarded)


@pytest.fixture(scope="module")
def materials():
    return {i: g.contract.build_pair(i, 202610207000 + i) for i in range(10)}


@pytest.fixture(scope="module")
def library():
    result = {"skills": deepcopy(json.loads(g.contract.RECORD.read_text())["normalized_skills"])}
    for skill in result["skills"]:
        for rule in skill["rules"]:
            rule["conditions"] = dict(rule["conditions"])
    return result


def envelope(library, material, arm, index=0):
    content = deepcopy(library)
    if arm == "anonymous":
        mapping = material["private_mapping"]["forward"]
        for skill in content["skills"]:
            for rule in skill["rules"]:
                rule["conditions"] = {mapping["fields"][f]: mapping["values"][v]
                                      for f, v in rule["conditions"].items()}
    return {"id": f"offline-generation-{index}", "object": "chat.completion",
            "model": "openai/gpt-4o-mini", "provider": "OpenAI",
            "usage": {"cost": 0.001, "is_byok": False},
            "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(content)}}]}


def classify(response, material, arm):
    return g.classify_generation(response, arm, material,
             response_sha256=g.object_sha256(response), material_sha256=g.object_sha256(material),
             settled_cost=Decimal("0.001"))


@pytest.mark.parametrize("arm", g.ARMS)
def test_classification_preserves_valid_library_and_rejects_whole_invalid(library, materials, arm):
    material = materials[0]
    response = envelope(library, material, arm)
    assert classify(response, material, arm).library == library
    payload = json.loads(response["choices"][0]["message"]["content"])
    payload["skills"][0]["rules"][0]["conditions"] = {"unknown_field": "bad"}
    response["choices"][0]["message"]["content"] = json.dumps(payload)
    before = deepcopy(response)
    decision = classify(response, material, arm)
    assert decision.status == g.INVALID and decision.library is None
    assert response == before and decision.reasons[0] == "invalid_library_contract"


@pytest.mark.parametrize("case", ["length", "content_filter", "refusal", "tool_calls", "function_call", "empty", "duplicate_json", "nonfinite_json"])
@pytest.mark.parametrize("arm", g.ARMS)
def test_known_model_failures_fallback_symmetrically(case, arm, library, materials):
    response = envelope(library, materials[0], arm)
    choice = response["choices"][0]
    if case in ("length", "content_filter"):
        choice["finish_reason"] = case
    elif case == "refusal": choice["message"]["refusal"] = "offline refusal"
    elif case == "tool_calls":
        choice["finish_reason"] = case
        choice["message"]["tool_calls"] = [{"type": "function", "function": {"name": "unused", "arguments": "{}"}}]
    elif case == "function_call":
        choice["finish_reason"] = case
        choice["message"]["function_call"] = {"name": "unused", "arguments": "{}"}
    else:
        choice["message"]["content"] = {"empty": " ", "duplicate_json": '{"skills":[],"skills":[]}', "nonfinite_json": '{"skills":NaN}'}[case]
    decision = classify(response, materials[0], arm)
    assert decision.status == g.INVALID and decision.library is None


@pytest.mark.parametrize("case", ["unknown_finish", "bad_choices", "bad_message", "bad_model", "bad_provider", "bad_function", "bare_tools"])
def test_bad_envelope_or_unknown_provenance_is_halt(case, library, materials):
    response = envelope(library, materials[0], "labeled")
    choice = response["choices"][0]
    if case == "unknown_finish": choice["finish_reason"] = "unexpected"
    elif case == "bad_choices": response["choices"] = []
    elif case == "bad_message": choice["message"] = []
    elif case == "bad_model": response["model"] = "wrong"
    elif case == "bad_provider": response["provider"] = "wrong"
    elif case == "bad_function": choice["message"]["function_call"] = []
    else: choice["finish_reason"] = "tool_calls"
    with pytest.raises(g.Halt): classify(response, materials[0], "labeled")


def test_validator_internal_exception_is_halt_not_fallback(monkeypatch, library, materials):
    response = envelope(library, materials[0], "labeled")
    def bug(*args): raise RuntimeError("validator internal failure")
    monkeypatch.setattr(g.contract, "strict_validate_library", bug)
    with pytest.raises(g.Halt, match="contract-validator"):
        classify(response, materials[0], "labeled")


@pytest.fixture
def run_fixture(tmp_path, monkeypatch, materials):
    config = json.loads((REAL_ROOT / "experiments/care_replay/configs/bh_semantic_anonymization_v2.json").read_text())
    config["generation_output_directory"] = "generation"
    cfg, lockpath, freeze = tmp_path / "config.json", tmp_path / "lock.json", tmp_path / "freeze"
    (freeze / "messages").mkdir(parents=True)
    (freeze / "private_pairs").mkdir()
    g.save(cfg, config)
    lock = {"config_sha256": g.file_hash(cfg), "messages_sha256": {}, "pair_material_sha256": {},
            "source_sha256": {}, "data_sha256": {}, "input_sha256": {}}
    for pair, material in materials.items():
        name = f"pair_{pair:02d}"
        path = freeze / "private_pairs" / (name + ".json")
        g.save(path, material)
        lock["pair_material_sha256"][name] = g.file_hash(path)
        for arm in g.ARMS:
            name = f"pair_{pair:02d}_{arm}"
            path = freeze / "messages" / (name + ".json")
            g.save(path, material["arms"][arm]["messages"])
            lock["messages_sha256"][name] = g.file_hash(path)
    for group, relative in (("source_sha256", "src/frozen.py"), ("data_sha256", "data/test.json"),
                            ("input_sha256", "evidence/prior_ledger.json")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        g.save(path, {"offline": True})
        lock[group][relative] = g.file_hash(path)
    g.save(lockpath, lock)
    monkeypatch.setattr(g, "ROOT", tmp_path)
    def verify(*args):
        assert g.file_hash(cfg) == lock["config_sha256"]
        for group in ("source_sha256", "data_sha256", "input_sha256"):
            for relative, expected in lock[group].items():
                if g.file_hash(tmp_path / relative) != expected:
                    raise ValueError("Frozen source/input changed")
        return config, lock
    monkeypatch.setattr(g.common, "verify_freeze", verify)
    return config, lock, cfg, lockpath, freeze, tmp_path / "generation"


class FakeAPI:
    key = "OFFLINE_SECRET_SENTINEL"
    def __init__(self, fixture, library, materials, mutate=None):
        self.fixture, self.library, self.materials, self.mutate = fixture, library, materials, mutate
        self.posts, self.gets, self.sleep_calls = [], [], []
    def request(self, path, payload=None):
        if path == "/models":
            self.gets.append(path)
            return {"data": [{"id": "openai/gpt-4o-mini", "context_length": 128000,
                "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0"}}]}
        if path == "/key":
            self.gets.append(path)
            return {"data": {"limit_remaining": 1}}
        if path.startswith("/generation?id="):
            self.gets.append(path)
            raise g.TransportError(404, "offline metadata unavailable")
        assert path == "/chat/completions"
        config, lock, cfg, lockpath, freeze, output = self.fixture
        index = len(self.posts)
        slot = config["generation_order"][index]
        identifier = f"pair_{slot['pair_index']:02d}_{slot['arm']}"
        ledger = g.load(output / "ledger.json")
        assert len(ledger["attempts"]) == index + 1
        assert ledger["attempts"][-1]["status"] == "RESERVED_BEFORE_POST"
        request_path = output / "receipts" / identifier / "request.json"
        assert g.load(request_path) == payload
        assert ledger["attempts"][-1]["request_sha256"] == g.file_hash(request_path)
        assert "n" not in payload and "service_tier" not in payload
        self.posts.append(deepcopy(payload))
        response = envelope(self.library, self.materials[slot["pair_index"]], slot["arm"], index)
        if self.mutate is not None: self.mutate(index, response, self)
        return response


def run(fixture, api):
    return g.execute(*fixture[2:], api=api, sleep=lambda _: None)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_twenty_fresh_slots_preserve_invalids_without_extra_posts_and_account_old_spend(run_fixture, library, materials, all_invalid):
    def mutate(i, response, api):
        if all_invalid or i % 3 == 0:
            response["choices"][0]["message"]["content"] = "bad JSON"
    api = FakeAPI(run_fixture, library, materials, mutate)
    result = run(run_fixture, api)
    output = run_fixture[-1]
    expected_invalid = 20 if all_invalid else 7
    assert len(api.posts) == 20 and api.gets == ["/models", "/key"]
    assert result["actual_post_attempts"] == 20
    assert result["invalid_library_count"] == expected_invalid
    assert len(result["outcomes_sha256"]) == 20
    assert len(result["libraries_sha256"]) == 20 - expected_invalid
    assert Decimal(result["known_new_cost_usd"]) == Decimal("0.020")
    assert Decimal(result["cumulative_known_cost_usd"]) == Decimal("0.0325145")
    ledger = g.load(output / "ledger.json")
    assert result["ledger_sha256"] == g.file_hash(output / "ledger.json")
    for attempt in ledger["attempts"]:
        name = attempt["identifier"]
        outcome = g.load(output / "outcomes" / (name + ".json"))
        assert attempt["status"] == outcome["status"]
        assert (output / "libraries" / (name + ".json")).exists() == (outcome["status"] == g.VALID)
    for group in ("data_sha256", "input_sha256"):
        for relative, expected in run_fixture[1][group].items():
            assert g.file_hash(output / "input_snapshot" / relative) == expected
    with pytest.raises(g.Halt, match="already exists"): run(run_fixture, api)
    assert len(api.posts) == 20


@pytest.mark.parametrize("case", ["unknown_finish", "unknown_cost", "byok", "wrong_model", "cost_over_reserve", "transport", "snapshot_changed", "internal"])
def test_fail_closed_after_one_attempt_no_retry_or_fallback(case, run_fixture, library, materials, monkeypatch):
    def mutate(i, response, api):
        if case == "unknown_finish": response["choices"][0]["finish_reason"] = "unknown"
        elif case == "unknown_cost": response["usage"] = {}
        elif case == "byok": response["usage"]["is_byok"] = True
        elif case == "wrong_model": response["model"] = "wrong"
        elif case == "cost_over_reserve": response["usage"]["cost"] = 0.046
        elif case == "transport": raise g.TransportError(503, api.key + " injected fixture failure")
        elif case == "snapshot_changed":
            path = run_fixture[-1] / "input_snapshot/pairs/pair_00.json"
            path.write_text("{}")
        elif case == "internal":
            def bug(*args): raise RuntimeError(api.key + " injected decoder fault")
            monkeypatch.setattr(g.contract, "validate_and_decode", bug)
    api = FakeAPI(run_fixture, library, materials, mutate)
    with pytest.raises(g.Halt) as error: run(run_fixture, api)
    output = run_fixture[-1]
    assert len(api.posts) == 1
    assert not (output / "completion.json").exists()
    assert list((output / "outcomes").glob("*.json")) == []
    failure = g.load(output / "failure.json")
    assert failure["status"] == "HALT" and failure["retry_performed"] is False
    assert api.key not in str(error.value)
    for path in output.rglob("*.json"): assert api.key not in path.read_text()
    if case in ("unknown_cost", "byok", "wrong_model", "transport"):
        assert failure["unresolved_reservation_usd"] == "0.045"
    if case == "unknown_cost": assert len([p for p in api.gets if p.startswith("/generation")]) == 3


def test_final_freeze_drift_blocks_completion_after_twenty_calls(run_fixture, library, materials):
    def mutate(i, response, api):
        if i == 19: (g.ROOT / "src/frozen.py").write_text("changed after generation began")
    api = FakeAPI(run_fixture, library, materials, mutate)
    with pytest.raises(g.Halt, match="Frozen source/input changed"): run(run_fixture, api)
    assert len(api.posts) == 20
    assert not (run_fixture[-1] / "completion.json").exists()
    assert g.load(run_fixture[-1] / "failure.json")["status"] == "HALT"


@pytest.mark.parametrize("field,value", [("generation_enabled", False), ("new_model_calls_approved", False),
    ("new_invalid_output_policy_approved", False), ("maximum_new_post_attempts", 21),
    ("prior_known_cost_usd", "0"), ("proposed_cumulative_max_cost_usd", "2")])
def test_authorization_and_budget_changes_fail_before_api(field, value, run_fixture, library, materials):
    run_fixture[0]["resource_authorization"][field] = value
    api = FakeAPI(run_fixture, library, materials)
    with pytest.raises(g.Halt): run(run_fixture, api)
    assert api.posts == [] and api.gets == []


def test_alternative_output_cannot_repeat_approved_protocol(run_fixture, library, materials):
    api = FakeAPI(run_fixture, library, materials)
    with pytest.raises(g.Halt, match="unique approved"):
        g.execute(*run_fixture[2:-1], run_fixture[-1].with_name("alternative"), api=api)
    assert api.posts == [] and api.gets == []


def test_default_cli_verifies_without_credentials_or_transport(monkeypatch, run_fixture, capsys):
    _, _, cfg, lock, freeze, _ = run_fixture
    monkeypatch.setattr(sys, "argv", ["generation.py", "--config", str(cfg), "--protocol-lock", str(lock), "--freeze-dir", str(freeze)])
    g.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified_offline" and result["api_calls"] == 0
    assert result["credentials_read"] is False


@pytest.mark.parametrize("cost", [None, True, -1, "NaN", "Infinity"])
def test_invalid_or_unknown_cost_never_becomes_zero(tmp_path, library, materials, cost):
    response = envelope(library, materials[0], "labeled")
    response["usage"]["cost"] = cost
    class API:
        def request(self, path, payload=None):
            assert payload is None and path.startswith("/generation?id=")
            raise g.TransportError(404, "offline not found")
    with pytest.raises(g.Halt): g.response_cost(response, API(), tmp_path, sleep=lambda _: None)
    assert not (tmp_path / "billing.json").exists()


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_bounded_generation_lookup_settles_only_same_id_and_preserves_response(tmp_path, library, materials, wrong_identity):
    response = envelope(library, materials[0], "labeled")
    response["usage"].pop("is_byok")
    before = deepcopy(response)
    class API:
        calls = 0
        def request(self, path, payload=None):
            assert payload is None and path.startswith("/generation?id=")
            self.calls += 1
            if self.calls == 1: raise g.TransportError(404, "offline delayed metadata")
            return {"data": {"id": "wrong" if wrong_identity else response["id"],
                     "model": response["model"], "provider_name": response["provider"],
                     "is_byok": False, "total_cost": 0.0012}}
    api = API()
    if wrong_identity:
        with pytest.raises(g.Halt, match="provenance"):
            g.response_cost(response, api, tmp_path, sleep=lambda _: None)
        assert not (tmp_path / "billing.json").exists()
    else:
        assert g.response_cost(response, api, tmp_path, sleep=lambda _: None) == Decimal("0.0012")
        billing = g.load(tmp_path / "billing.json")
        assert billing["source"] == "generation_lookup" and billing["response_id"] == response["id"]
        assert billing["is_byok"] is False
    assert api.calls == 2 and response == before
