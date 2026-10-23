"""Offline safety regressions for the one-shot paired generation runner.

All credentials, responses, catalogs and frozen inputs below are synthetic.
An autouse network guard makes an accidental real HTTP request fail locally.
"""
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_paired_anonymization_generation as runner


MODEL = {
    "id": "openai/gpt-4o-mini",
    "context_length": 128000,
    "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0"},
    "supported_parameters": ["temperature", "max_tokens", "response_format"],
}
PROVIDER = {
    "only": ["openai"], "order": ["openai"], "allow_fallbacks": False,
    "require_parameters": True,
    "max_price": {"prompt": 0.15, "completion": 0.60, "request": 0},
}
FAKE_SECRET = "sk-or-v1-NOT-A-REAL-KEY-test-secret"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def valid_response(index=0):
    return {
        "id": f"gen-offline-{index}", "model": MODEL["id"], "provider": "OpenAI",
        "choices": [{"finish_reason": "stop",
                     "message": {"role": "assistant", "content": '{"skills": []}'}}],
        "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 100},
    }


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in offline runner tests")
    monkeypatch.setattr(runner.urllib.request, "urlopen", blocked)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A complete, genuinely hashed 10-pair fixture; no real dataset/key reads."""
    root = tmp_path / "root"
    root.mkdir()
    freeze = root / "freeze"
    (freeze / "messages").mkdir(parents=True)
    (freeze / "private_pairs").mkdir()
    config = {
        "protocol_id": "offline_test",
        "pair_count": 10,
        "data_files": ["data.csv"],
        "llm_record": "replay_config.json",
        "model": {"id": MODEL["id"], "temperature": 0.45, "max_tokens": 5000,
                  "response_format": "json_object", "sampling_seed": None},
        "api": {"max_cost_usd": 1, "reservation_per_attempt_usd": 0.045,
                "context_token_bound": 128000, "timeout_seconds": 120,
                "base_url": runner.BASE_URL, "transport_retries": 0,
                "provider": deepcopy(PROVIDER)},
        "resource_authorization": {
            "generation_enabled": True, "approved_pair_count": 10,
            "approved_logical_requests": 20, "approved_max_cost_usd": 1,
        },
        "generation_order": [
            {"pair_index": pair, "arm": arm}
            for pair in range(10) for arm in ("labeled", "anonymous")
        ],
    }
    config_path, lock_path = root / "config.json", freeze / "protocol_lock.json"
    write_json(config_path, config)
    lock = {"status": "frozen_before_generation", "model_api_calls_at_freeze": 0,
            "config_sha256": runner.file_hash(config_path),
            "source_sha256": {}, "data_sha256": {}, "input_sha256": {},
            "messages_sha256": {}, "pair_material_sha256": {}}
    required_sources = [
        "experiments/care_replay/scripts/run_paired_anonymization_generation.py",
        "experiments/care_replay/scripts/paired_semantic_anonymization.py",
        "experiments/care_replay/scripts/run_paired_anonymization_evaluation.py",
    ]
    inventories = {
        "source_sha256": required_sources,
        "data_sha256": ["data.csv"],
        "input_sha256": ["replay_config.json", "uv.lock", "pyproject.toml"],
    }
    for group, names in inventories.items():
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen input\n", encoding="utf-8")
            lock[group][name] = runner.file_hash(path)
    for pair in range(10):
        material = {"pair_index": pair, "private_mapping": "DO_NOT_SEND_PRIVATE_MAPPING"}
        path = freeze / "private_pairs" / f"pair_{pair:02d}.json"
        write_json(path, material)
        lock["pair_material_sha256"][f"pair_{pair:02d}"] = runner.file_hash(path)
        for arm in ("labeled", "anonymous"):
            identifier = f"pair_{pair:02d}_{arm}"
            path = freeze / "messages" / f"{identifier}.json"
            write_json(path, [{"role": "user", "content": f"public {pair} {arm}"}])
            lock["messages_sha256"][identifier] = runner.file_hash(path)
    write_json(lock_path, lock)
    monkeypatch.setattr(runner, "ROOT", root)
    original_cost = runner.response_cost
    monkeypatch.setattr(runner, "response_cost",
                        lambda *args, **kwargs: original_cost(
                            *args, **{**kwargs, "sleep": lambda _: None}))
    return SimpleNamespace(root=root, freeze=freeze, config=config, lock=lock,
                           config_path=config_path, lock_path=lock_path,
                           output=root / "generated",
                           source_path=root / required_sources[0])


def refreeze_config(frozen):
    write_json(frozen.config_path, frozen.config)
    frozen.lock["config_sha256"] = runner.file_hash(frozen.config_path)
    write_json(frozen.lock_path, frozen.lock)


class FakeAPI:
    def __init__(self, output, response=None, post_error=None, metadata=None):
        self.output = output
        self.response = response
        self.post_error = post_error
        self.metadata = metadata
        self.calls, self.posts = [], []
        self.key = FAKE_SECRET

    def request(self, path, payload=None):
        self.calls.append(path)
        if path == "/models":
            return {"data": [deepcopy(MODEL)]}
        if path == "/key":
            return {"data": {"limit": 1, "limit_remaining": 1, "usage": 0}}
        if path.startswith("/generation?id="):
            if isinstance(self.metadata, BaseException):
                raise self.metadata
            if self.metadata is not None:
                return deepcopy(self.metadata)
            return {"data": {"is_byok": False, "total_cost": 0.001,
                             "model": MODEL["id"], "provider_name": "OpenAI",
                             "native_tokens_prompt": 100, "native_tokens_completion": 100}}
        assert path == "/chat/completions", f"Unexpected fake API path: {path}"
        # This assertion runs INSIDE the network boundary, before any fake response.
        ledger = read_json(self.output / "ledger.json")
        assert len(ledger["attempts"]) == len(self.posts) + 1
        last = ledger["attempts"][-1]
        assert last["status"] == "reserved_before_post"
        assert Decimal(last["reservation_usd"]) == Decimal("0.045")
        request_path = self.output / "receipts" / last["identifier"] / "request.json"
        assert last["request_sha256"] == runner.file_hash(request_path)
        assert read_json(request_path) == payload
        assert "DO_NOT_SEND_PRIVATE_MAPPING" not in json.dumps(payload)
        self.posts.append(deepcopy(payload))
        if self.post_error is not None:
            raise self.post_error
        return deepcopy(self.response if self.response is not None
                        else valid_response(len(self.posts) - 1))


def execute(frozen, api, validator=None):
    return runner.execute(
        frozen.config_path, frozen.lock_path, frozen.freeze, frozen.output,
        api=api, validator=validator or (lambda text, arm, material: json.loads(text)),
    )


def assert_stopped_once(frozen, api, unresolved=False):
    assert len(api.posts) == 1
    assert not (frozen.output / "completion.json").exists()
    failure = read_json(frozen.output / "failure.json")
    assert failure["model_api_calls"] == 1
    assert failure["retry_performed"] is False
    assert failure["automatic_repair_performed"] is False
    assert failure["incomplete_pairs_not_evaluated"] is True
    if unresolved:
        assert Decimal(failure["unresolved_reservation_usd"]) == Decimal("0.045")


def test_real_freeze_accepts_complete_untampered_inputs(frozen):
    config, lock = runner.verify_freeze(
        frozen.config_path, frozen.lock_path, frozen.freeze)
    assert config == frozen.config
    assert lock == frozen.lock


@pytest.mark.parametrize("kind", ["config", "source", "data", "input", "message", "pair"])
def test_real_freeze_rejects_each_hashed_input_tamper(frozen, kind):
    paths = {
        "config": frozen.config_path, "source": frozen.source_path,
        "data": frozen.root / "data.csv", "input": frozen.root / "replay_config.json",
        "message": frozen.freeze / "messages/pair_00_labeled.json",
        "pair": frozen.freeze / "private_pairs/pair_00.json",
    }
    with paths[kind].open("a", encoding="utf-8") as handle:
        handle.write("\n ")
    api = FakeAPI(frozen.output)
    with pytest.raises(ValueError):
        execute(frozen, api)
    assert api.calls == [], "Freeze verification must precede every API operation"
    assert not frozen.output.exists()


@pytest.mark.parametrize("group", [
    "source_sha256", "data_sha256", "input_sha256",
    "messages_sha256", "pair_material_sha256",
])
def test_freeze_cannot_bypass_verification_by_erasing_manifest(frozen, group):
    frozen.lock[group] = {}
    write_json(frozen.lock_path, frozen.lock)
    with pytest.raises(ValueError):
        runner.verify_freeze(frozen.config_path, frozen.lock_path, frozen.freeze)


@pytest.mark.parametrize("field,value", [
    ("status", "draft"), ("model_api_calls_at_freeze", 1),
])
def test_protocol_must_have_been_frozen_before_generation(frozen, field, value):
    frozen.lock[field] = value
    write_json(frozen.lock_path, frozen.lock)
    with pytest.raises(ValueError):
        runner.verify_freeze(frozen.config_path, frozen.lock_path, frozen.freeze)


@pytest.mark.parametrize("defect", ["short", "long", "duplicate", "arm", "pair", "bool_pair"])
def test_generation_order_must_be_exactly_ten_complete_pairs(frozen, defect):
    order = frozen.config["generation_order"]
    if defect == "short":
        order.pop()
    elif defect == "long":
        order.append({"pair_index": 10, "arm": "labeled"})
    elif defect == "duplicate":
        order[-1] = deepcopy(order[0])
    elif defect == "arm":
        order[-1]["arm"] = "third_arm"
    elif defect == "pair":
        order[-1]["pair_index"] = 10
    else:
        order[0]["pair_index"] = False
    refreeze_config(frozen)
    api = FakeAPI(frozen.output)
    with pytest.raises((ValueError, KeyError)):
        execute(frozen, api)
    assert api.posts == [], "Malformed order must be rejected before first POST"


@pytest.mark.parametrize("reserve", [0.046, 0.05, 0, -0.045])
def test_total_reservation_must_be_positive_and_at_most_ninety_cents(frozen, reserve):
    frozen.config["api"]["reservation_per_attempt_usd"] = reserve
    with pytest.raises(ValueError):
        runner.validate_budget(frozen.config, deepcopy(MODEL), {"limit_remaining": 1})


def test_catalog_bound_and_total_reservation_are_decimal_exact(frozen):
    check = runner.validate_budget(frozen.config, deepcopy(MODEL), {"limit_remaining": 1})
    assert Decimal(check["per_request_catalog_bound_usd"]) == Decimal("0.0222")
    assert Decimal(check["whole_batch_reservation_usd"]) == Decimal("0.90")


@pytest.mark.parametrize("change", [
    {"id": "other/model"}, {"context_length": 128001},
    {"pricing": {"prompt": "0.000000151", "completion": "0.0000006"}},
    {"pricing": {"prompt": "0.00000015", "completion": "0.00000061"}},
    {"pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0.01"}},
    {"pricing": {"prompt": "-1", "completion": "0.0000006"}},
])
def test_uncovered_catalog_prices_or_model_identity_block_generation(frozen, change):
    model = {**deepcopy(MODEL), **change}
    with pytest.raises(ValueError):
        runner.validate_budget(frozen.config, model, {"limit_remaining": 1})


def test_key_limit_must_cover_whole_batch_reservation(frozen):
    with pytest.raises(ValueError):
        runner.validate_budget(frozen.config, deepcopy(MODEL), {"limit_remaining": 0.899})


@pytest.mark.parametrize("field,value", [
    ("allow_fallbacks", True), ("only", ["azure"]), ("order", ["openai", "azure"]),
    ("require_parameters", False),
    ("max_price", {"prompt": 10, "completion": 10, "request": 0}),
])
def test_routing_cannot_relax_frozen_single_provider_budget_contract(frozen, field, value):
    frozen.config["api"]["provider"][field] = value
    with pytest.raises(ValueError):
        runner.validate_budget(frozen.config, deepcopy(MODEL), {"limit_remaining": 1})


def test_complete_run_has_twenty_posts_and_one_library_per_frozen_slot(frozen):
    api = FakeAPI(frozen.output)
    result = execute(frozen, api)
    assert result["model_api_calls"] == len(api.posts) == 20
    assert len(result["libraries_sha256"]) == 20
    assert Decimal(result["known_cost_usd"]) == Decimal("0.020")
    assert read_json(frozen.output / "completion.json") == result
    assert not (frozen.output / "failure.json").exists()
    assert all(post["provider"] == PROVIDER for post in api.posts)
    assert all(post["model"] == MODEL["id"] and post["temperature"] == 0.45
               and post["max_tokens"] == 5000 and post["stream"] is False
               and post["response_format"] == {"type": "json_object"}
               and "n" not in post and "service_tier" not in post
               for post in api.posts)
    assert len([p for p in api.calls if p.startswith("/generation?id=")]) >= 20


def test_later_freeze_changes_never_reach_post_or_decoder(frozen):
    """After copying verified snapshots, use them or stop on a later mutation."""
    seen_material = []
    class MutatingAPI(FakeAPI):
        def request(self, path, payload=None):
            result = super().request(path, payload)
            if path == "/chat/completions" and len(self.posts) == 1:
                write_json(frozen.freeze / "messages/pair_00_anonymous.json",
                           [{"role": "user", "content": "LATE_MUTATION"}])
                write_json(frozen.freeze / "private_pairs/pair_00.json",
                           {"pair_index": 0, "private_mapping": "LATE_MUTATION"})
            return result
    def validator(text, arm, material):
        seen_material.append(deepcopy(material))
        return json.loads(text)
    api = MutatingAPI(frozen.output)
    try:
        execute(frozen, api, validator)
    except (ValueError, RuntimeError):
        pass  # Rehashing and failing closed is also acceptable.
    assert all("LATE_MUTATION" not in json.dumps(post) for post in api.posts)
    assert all("LATE_MUTATION" not in json.dumps(material) for material in seen_material)

def test_existing_output_is_not_automatically_resumed(frozen):
    frozen.output.mkdir()
    api = FakeAPI(frozen.output)
    with pytest.raises(ValueError):
        execute(frozen, api)
    assert api.calls == []


def test_disabled_generation_never_contacts_api(frozen):
    frozen.config["resource_authorization"]["generation_enabled"] = False
    refreeze_config(frozen)
    api = FakeAPI(frozen.output)
    with pytest.raises(ValueError):
        execute(frozen, api)
    assert api.calls == []


def test_first_invalid_library_stops_without_repair_or_replacement(frozen):
    api = FakeAPI(frozen.output)
    def invalid(*args):
        raise ValueError("Schema-invalid offline fixture")
    with pytest.raises(RuntimeError):
        execute(frozen, api, invalid)
    assert_stopped_once(frozen, api)
    assert list((frozen.output / "libraries").iterdir()) == []
    assert Decimal(read_json(frozen.output / "failure.json")["known_cost_usd"]) == Decimal(".001")


@pytest.mark.parametrize("status", [None, 408, 429, 500, 502])
def test_transport_failure_does_not_retry_or_release_reservation(frozen, status):
    api = FakeAPI(frozen.output, post_error=runner.TransportError(status, "offline error"))
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api, unresolved=True)


@pytest.mark.parametrize("reason", ["length", "error", "content_filter", "tool_calls", None])
def test_non_stop_finish_reason_stops_after_first_response(frozen, reason):
    response = valid_response()
    response["choices"][0]["finish_reason"] = reason
    api = FakeAPI(frozen.output, response=response)
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api)


@pytest.mark.parametrize("location", ["top", "choice"])
def test_http_200_error_payload_is_not_accepted_as_library(frozen, location):
    response = valid_response()
    target = response if location == "top" else response["choices"][0]
    target["error"] = {"code": 500, "message": "upstream failure"}
    api = FakeAPI(frozen.output, response=response)
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api)


@pytest.mark.parametrize("cost", [None, True, -1, "NaN", "Infinity"])
def test_unknown_or_invalid_billing_never_becomes_zero(frozen, cost):
    response = valid_response()
    response["usage"]["cost"] = cost
    api = FakeAPI(frozen.output, response=response,
                  metadata={"data": {"is_byok": False}})
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api, unresolved=True)


@pytest.mark.parametrize("flag", [True, None, "false", 0])
def test_byok_or_unproven_billing_route_stops_before_second_post(frozen, flag):
    api = FakeAPI(frozen.output,
                  metadata={"data": {"total_cost": 0.001, "is_byok": flag}})
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api)


def test_generation_metadata_not_found_stops_without_second_post(frozen):
    api = FakeAPI(frozen.output, metadata=runner.TransportError(404, "not settled"))
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api)
    assert 1 <= len([p for p in api.calls if p.startswith("/generation?id=")]) <= 5


def test_reported_cost_exceeding_reservation_stops_immediately(frozen):
    response = valid_response()
    response["usage"]["cost"] = 0.046
    api = FakeAPI(frozen.output, response=response,
                  metadata={"data": {"is_byok": False, "total_cost": 0.046}})
    with pytest.raises(RuntimeError):
        execute(frozen, api)
    assert_stopped_once(frozen, api)


def test_cost_lookup_retries_only_read_only_404_and_reconciles(frozen):
    receipt = frozen.root / "receipt"
    receipt.mkdir()
    class DelayedMetadata:
        def __init__(self):
            self.paths = []
        def request(self, path, payload=None):
            self.paths.append(path)
            assert payload is None and path.startswith("/generation?id=")
            if len(self.paths) == 1:
                raise runner.TransportError(404, "not settled")
            return {"data": {"total_cost": 0.002, "is_byok": False}}
    api = DelayedMetadata()
    response = valid_response()
    response["usage"] = {}
    assert runner.response_cost(response, api, receipt) == Decimal(".002")
    assert len(api.paths) == 2


def test_generation_id_is_quoted_as_one_query_value(frozen):
    receipt = frozen.root / "receipt"
    receipt.mkdir()
    response = valid_response()
    response["id"] = "gen/&next=evil?plus +fragment#"
    api = FakeAPI(frozen.output)
    runner.response_cost(response, api, receipt)
    assert api.calls == ["/generation?id=gen%2F%26next%3Devil%3Fplus%20%2Bfragment%23"]


@pytest.mark.parametrize("invalid", [-1, True, "NaN", "Infinity"])
@pytest.mark.parametrize("source", ["response", "lookup"])
def test_invalid_cost_field_is_not_masked_by_other_valid_endpoint(frozen, invalid, source):
    receipt = frozen.root / "receipt"
    receipt.mkdir()
    response = valid_response()
    metadata = {"data": {"is_byok": False, "total_cost": 0.001}}
    if source == "response":
        response["usage"]["cost"] = invalid
    else:
        metadata["data"]["total_cost"] = invalid
    api = FakeAPI(frozen.output, metadata=metadata)
    with pytest.raises((ValueError, ArithmeticError)):
        runner.response_cost(response, api, receipt)


@pytest.mark.parametrize("kind", ["http_error", "network_error", "invalid_json"])
def test_transport_never_includes_fake_key_in_error_or_body(monkeypatch, kind):
    def fake_urlopen(request, **kwargs):
        if kind == "http_error":
            raise urllib.error.HTTPError(
                request.full_url, 500, "error", {}, io.BytesIO(FAKE_SECRET.encode()))
        if kind == "network_error":
            raise OSError("transport included " + FAKE_SECRET)
        return io.BytesIO(("not-json " + FAKE_SECRET).encode())
    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
    api = runner.OpenRouter(FAKE_SECRET)
    with pytest.raises(runner.TransportError) as captured:
        api.request("/chat/completions", {"model": MODEL["id"]})
    assert FAKE_SECRET not in str(captured.value)
    assert FAKE_SECRET not in captured.value.body


def test_decoded_response_cannot_reintroduce_json_escaped_fake_key(monkeypatch):
    # Redacting the raw JSON bytes alone misses JSON Unicode escapes.
    body = json.dumps({"error": {"message": FAKE_SECRET}})
    body = body.replace("-", "\\u002d").encode()
    monkeypatch.setattr(runner.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(body))
    response = runner.OpenRouter(FAKE_SECRET).request("/chat/completions", {})
    assert FAKE_SECRET not in json.dumps(response)


def test_failure_artifact_and_raised_exception_do_not_leak_fake_key(frozen, capsys):
    api = FakeAPI(frozen.output)
    def invalid(*args):
        raise ValueError("untrusted validator detail " + FAKE_SECRET)
    with pytest.raises(Exception) as captured:
        execute(frozen, api, invalid)
    assert_stopped_once(frozen, api)
    assert FAKE_SECRET not in str(captured.value)
    assert FAKE_SECRET not in (frozen.output / "failure.json").read_text()
    assert FAKE_SECRET not in capsys.readouterr().out
