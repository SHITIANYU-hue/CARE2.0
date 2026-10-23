#!/usr/bin/env python3
"""Frozen paired generation. One POST per slot; any failure stops the run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "https://openrouter.ai/api/v1"
REQUIRED_SOURCES = {
    "experiments/care_replay/scripts/run_paired_anonymization_generation.py",
    "experiments/care_replay/scripts/paired_semantic_anonymization.py",
    "experiments/care_replay/scripts/run_paired_anonymization_evaluation.py",
}
EXPECTED_PROVIDER = {
    "only": ["openai"], "order": ["openai"], "allow_fallbacks": False,
    "require_parameters": True,
    "max_price": {"prompt": 0.15, "completion": 0.60, "request": 0},
}


def redact(value, api):
    secret = getattr(api, "key", None)
    return str(value).replace(secret, "[REDACTED]") if secret else str(value)


def redact_object(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [redact_object(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact_object(key, secret): redact_object(item, secret)
                for key, item in value.items()}
    return value


def expected_slots(config):
    if config.get("pair_count") != 10:
        raise ValueError("Exactly ten pairs were authorized")
    expected = {(pair, arm) for pair in range(10) for arm in ("labeled", "anonymous")}
    order = config.get("generation_order", [])
    if len(order) != 20 or any(set(slot) != {"pair_index", "arm"} or
                              type(slot["pair_index"]) is not int for slot in order):
        raise ValueError("Expected exactly twenty predefined generation slots")
    if {(slot["pair_index"], slot["arm"]) for slot in order} != expected:
        raise ValueError("Generation slots are incomplete or duplicated")
    return {f"pair_{pair:02d}_{arm}" for pair, arm in expected}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_key(path):
    """Read only the one requested variable; never execute dotenv content."""
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "OPENROUTER_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            entries.append(value)
    if len(entries) != 1 or not entries[0] or any(c.isspace() for c in entries[0]):
        raise ValueError("Expected one nonempty OPENROUTER_API_KEY assignment")
    return entries[0]


class TransportError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"OpenRouter transport failed (HTTP {status})")
        self.status, self.body = status, body


class OpenRouter:
    def __init__(self, key, timeout=120):
        self.key, self.timeout = key, timeout

    def request(self, path, payload=None):
        if path not in ("/models", "/key", "/chat/completions") and not path.startswith("/generation?id="):
            raise ValueError("Unexpected API path")
        headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(BASE_URL + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace").replace(self.key, "[REDACTED]")
            raise TransportError(exc.code, raw) from None
        except Exception as exc:
            # Exception strings can contain request details; only retain the type.
            raise TransportError(None, type(exc).__name__) from None
        raw = raw.replace(self.key, "[REDACTED]")
        try:
            return redact_object(json.loads(raw), self.key)
        except ValueError:
            raise TransportError("invalid_json", raw) from None


def verify_freeze(config_path, lock_path, freeze_dir):
    config = json.loads(Path(config_path).read_text())
    lock = json.loads(Path(lock_path).read_text())
    if lock["status"] != "frozen_before_generation" or lock["model_api_calls_at_freeze"] != 0:
        raise ValueError("Protocol was not frozen before generation")
    if lock["config_sha256"] != file_hash(config_path):
        raise ValueError("Config changed after freeze")
    identifiers = expected_slots(config)
    if set(lock["messages_sha256"]) != identifiers or set(lock["pair_material_sha256"]) != {f"pair_{i:02d}" for i in range(10)}:
        raise ValueError("Incomplete frozen message or private-map inventory")
    if not REQUIRED_SOURCES <= set(lock["source_sha256"]):
        raise ValueError("Required protocol executors are not frozen")
    current_sources = {p.relative_to(ROOT).as_posix() for p in
                       (ROOT / "experiments/care_replay/scripts").glob("*.py")}
    if set(lock["source_sha256"]) != current_sources:
        raise ValueError("Frozen source inventory does not match the execution tree")
    if set(lock["data_sha256"]) != set(config["data_files"]):
        raise ValueError("Frozen data inventory differs from configuration")
    if not {config["llm_record"], "uv.lock", "pyproject.toml"} <= set(lock["input_sha256"]):
        raise ValueError("Frozen generation/dependency inputs are incomplete")
    for group in ("source_sha256", "data_sha256", "input_sha256"):
        for relative, expected in lock[group].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f"Frozen input changed: {relative}")
    for group, folder in (("messages_sha256", "messages"), ("pair_material_sha256", "private_pairs")):
        for identifier, expected in lock[group].items():
            if file_hash(Path(freeze_dir) / folder / (identifier + ".json")) != expected:
                raise ValueError(f"Frozen {group} changed: {identifier}")
    return config, lock


def validate_budget(config, model, key_metadata):
    expected_slots(config)
    if (config["model"]["id"] != "openai/gpt-4o-mini" or
            config["model"]["temperature"] != 0.45 or config["model"]["max_tokens"] != 5000 or
            config["api"]["context_token_bound"] != 128000 or
            config["api"]["provider"] != EXPECTED_PROVIDER or
            config["api"].get("base_url") != BASE_URL or
            config["api"].get("transport_retries") != 0):
        raise ValueError("Frozen model, routing or retry limits differ from authorization")
    budget = Decimal(str(config["api"]["max_cost_usd"]))
    reserve = Decimal(str(config["api"]["reservation_per_attempt_usd"]))
    planned = config["pair_count"] * 2
    if planned != 20 or budget != 1 or reserve != Decimal("0.045") or reserve * planned > Decimal("0.90"):
        raise ValueError("Approved generation count or budget changed")
    if (model["id"] != config["model"]["id"] or type(model["context_length"]) is not int or
            not 0 < model["context_length"] <= config["api"]["context_token_bound"]):
        raise ValueError("Model identity/context differs from frozen budget assumptions")
    pricing = model["pricing"]
    prompt = Decimal(str(pricing["prompt"]))
    completion = Decimal(str(pricing["completion"]))
    if any(not value.is_finite() or value < 0 for value in (prompt, completion)):
        raise ValueError("Catalog prices must be finite and nonnegative")
    if prompt > Decimal("0.00000015") or completion > Decimal("0.0000006"):
        raise ValueError("Current catalog price exceeds the frozen price caps")
    if Decimal(str(pricing.get("request", 0))) != 0:
        raise ValueError("A non-token request charge is not covered by this protocol")
    bound = Decimal(config["api"]["context_token_bound"]) * prompt + Decimal(config["model"]["max_tokens"]) * completion
    if bound > reserve:
        raise ValueError("Per-request cost bound exceeds the reservation")
    remaining = key_metadata.get("limit_remaining")
    if remaining is not None:
        remaining = Decimal(str(remaining))
        if not remaining.is_finite() or remaining < reserve * planned:
            raise ValueError("The key limit cannot cover the batch reservation")
    return {"per_request_catalog_bound_usd": str(bound),
            "per_attempt_reservation_usd": str(reserve),
            "whole_batch_reservation_usd": str(reserve * planned)}


def extract_completion(response):
    if response.get("error"):
        raise ValueError("API returned a top-level generation error")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one response choice")
    choice = choices[0]
    if choice.get("error"):
        raise ValueError("API returned a choice-level generation error")
    if choice.get("finish_reason") != "stop":
        raise ValueError("Unexpected finish reason")
    message = choice.get("message", {})
    if message.get("refusal") or message.get("tool_calls"):
        raise ValueError("Refusal/tool response is not an executable library")
    if not isinstance(message.get("content"), str):
        raise ValueError("Missing string completion")
    return message["content"]


def response_cost(response, api, receipt_dir, sleep=time.sleep):
    """Reconcile billing and confirm the request did not use BYOK."""
    cost = response.get("usage", {}).get("cost")
    info_data = None
    if isinstance(response.get("id"), str):
        # Reconcile an already-existing request only; never retry generation.
        from urllib.parse import quote
        for attempt in range(3):
            sleep(2 ** attempt)
            try:
                info = api.request("/generation?id=" + quote(response["id"], safe=""))
                save(receipt_dir / f"cost_lookup_{attempt}.json", info, exclusive=True)
                info_data = info.get("data", {})
                if info_data.get("total_cost") is not None and type(info_data.get("is_byok")) is bool:
                    break
            except TransportError as exc:
                save(receipt_dir / f"cost_lookup_{attempt}_error.json",
                     {"http_status": exc.status, "error": exc.body}, exclusive=True)
                if exc.status != 404:
                    break
    if not info_data or info_data.get("is_byok") is not False:
        raise ValueError("BYOK status is true or unknown; stop for billing review")
    values = [value for value in (cost, info_data.get("total_cost")) if value is not None]
    if not values or any(isinstance(value, bool) for value in values):
        raise ValueError("Billing cost is unknown; stop without another generation")
    values = [Decimal(str(value)) for value in values]
    if any(not value.is_finite() or value < 0 for value in values):
        raise ValueError("Invalid billing cost")
    # Use the larger valid observation if accounting updates between endpoints.
    return max(values)


def execute(config_path, lock_path, freeze_dir, output, api=None, validator=None):
    config, lock = verify_freeze(config_path, lock_path, freeze_dir)
    freeze_dir, output = Path(freeze_dir), Path(output)
    if output.exists():
        raise ValueError("Generation output already exists; never resume or repeat POSTs automatically")
    if config["resource_authorization"]["generation_enabled"] is not True:
        raise ValueError("Generation is not enabled by this frozen protocol")
    if validator is None:
        from paired_semantic_anonymization import validate_and_decode
        validator = validate_and_decode
    if api is None:
        api = OpenRouter(load_key(ROOT / ".env"), config["api"]["timeout_seconds"])
    model = next(m for m in api.request("/models")["data"] if m["id"] == config["model"]["id"])
    key_metadata = api.request("/key")["data"]
    budget_check = validate_budget(config, model, key_metadata)
    output.mkdir(parents=True)
    (output / "receipts").mkdir()
    (output / "libraries").mkdir()
    save(output / "preflight.json", {"checked_at": utc(), "budget": budget_check,
         "catalog_model": model, "key_authenticated": True, "secret_values_recorded": False})
    shutil.copyfile(config_path, output / "config.json")
    shutil.copyfile(lock_path, output / "protocol_lock.json")
    for source_group, folder in (("messages_sha256", "messages"), ("pair_material_sha256", "pairs")):
        destination = output / "input_snapshot" / folder
        destination.mkdir(parents=True)
        source_folder = "private_pairs" if folder == "pairs" else folder
        for identifier, expected in lock[source_group].items():
            path = destination / f"{identifier}.json"
            shutil.copyfile(freeze_dir / source_folder / path.name, path)
            if file_hash(path) != expected:
                raise ValueError("Frozen generation input changed while taking snapshot")
    for relative, expected in lock["source_sha256"].items():
        path = output / "source_snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
        if file_hash(path) != expected:
            raise ValueError("Source changed while taking snapshot")
    reserve = Decimal(str(config["api"]["reservation_per_attempt_usd"]))
    known_cost = Decimal(0)
    attempts, library_hashes = [], {}
    try:
        for slot in config["generation_order"]:
            if len(attempts) >= 20 or reserve * (len(attempts) + 1) > Decimal("0.90"):
                raise ValueError("The whole-batch attempt reservation is exhausted")
            identifier = f"pair_{slot['pair_index']:02d}_{slot['arm']}"
            messages_path = output / "input_snapshot" / "messages" / f"{identifier}.json"
            if file_hash(messages_path) != lock["messages_sha256"][identifier]:
                raise ValueError("Frozen message snapshot changed before POST")
            messages = json.loads(messages_path.read_text())
            # Model inputs exclude the local alias map.
            request = {"model": config["model"]["id"], "messages": messages,
                       "temperature": config["model"]["temperature"],
                       "max_tokens": config["model"]["max_tokens"],
                       "response_format": {"type": "json_object"},
                       "stream": False,
                       "provider": config["api"]["provider"]}
            if len(json.dumps(messages, ensure_ascii=False).encode()) + 4096 + config["model"]["max_tokens"] > config["api"]["context_token_bound"]:
                raise ValueError("Conservative input-byte bound exceeds the frozen context cap")
            if known_cost + reserve > Decimal(str(config["api"]["max_cost_usd"])):
                raise ValueError("Insufficient remaining approved budget")
            receipt_dir = output / "receipts" / identifier
            receipt_dir.mkdir()
            save(receipt_dir / "request.json", request, exclusive=True)
            attempt = {"slot": len(attempts), "identifier": identifier, "started_at": utc(),
                       "reservation_usd": str(reserve), "status": "reserved_before_post",
                       "request_sha256": file_hash(receipt_dir / "request.json")}
            attempts.append(attempt)
            # Durable reservation precedes the non-idempotent network operation.
            save(output / "ledger.json", {"attempts": attempts, "known_cost_usd": str(known_cost)})
            try:
                response = api.request("/chat/completions", request)
            except TransportError as exc:
                save(receipt_dir / "transport_error.json",
                     {"http_status": exc.status, "response": exc.body}, exclusive=True)
                attempt["status"] = "transport_failed_cost_unknown"
                raise
            save(receipt_dir / "response.json", response, exclusive=True)
            cost = response_cost(response, api, receipt_dir)
            known_cost += cost
            attempt.update(cost_usd=str(cost), response_id=response.get("id"),
                           response_model=response.get("model"), response_provider=response.get("provider"),
                           usage=response.get("usage"), status="response_received")
            save(output / "ledger.json", {"attempts": attempts, "known_cost_usd": str(known_cost)})
            if cost > reserve or known_cost > Decimal(str(config["api"]["max_cost_usd"])):
                raise ValueError("Reported cost exceeds the frozen reservation; stop")
            content = extract_completion(response)
            pair_id = f"pair_{slot['pair_index']:02d}"
            material_path = output / "input_snapshot" / "pairs" / f"{pair_id}.json"
            if file_hash(material_path) != lock["pair_material_sha256"][pair_id]:
                raise ValueError("Private pair snapshot changed before decoding")
            material = json.loads(material_path.read_text())
            decoded = validator(content, slot["arm"], material)
            library_path = output / "libraries" / f"{identifier}.json"
            save(library_path, decoded, exclusive=True)
            library_hashes[identifier] = file_hash(library_path)
            attempt.update(status="valid", completed_at=utc(), library_sha256=library_hashes[identifier])
            save(output / "ledger.json", {"attempts": attempts, "known_cost_usd": str(known_cost)})
            print(json.dumps({"identifier": identifier, "status": "valid", "calls": len(attempts),
                              "known_cost_usd": str(known_cost)}), flush=True)
        if len(library_hashes) != 20:
            raise ValueError("Incomplete paired libraries")
        result = {"status": "complete", "completed_at": utc(), "config_sha256": file_hash(config_path),
                  "protocol_lock_sha256": file_hash(lock_path), "libraries_sha256": library_hashes,
                  "messages_sha256": lock["messages_sha256"],
                  "pair_material_sha256": lock["pair_material_sha256"],
                  "model_api_calls": len(attempts), "known_cost_usd": str(known_cost)}
        save(output / "completion.json", result, exclusive=True)
        return result
    except BaseException as exc:
        failure = {"status": "stopped", "stopped_at": utc(), "model_api_calls": len(attempts),
                   "valid_library_count": len(library_hashes), "known_cost_usd": str(known_cost),
                   "error_type": type(exc).__name__, "error": redact(exc, api),
                   "retry_performed": False, "automatic_repair_performed": False,
                   "incomplete_pairs_not_evaluated": True}
        if attempts and attempts[-1]["status"] != "valid":
            attempts[-1]["status"] = "stopped_after_response" if "cost_usd" in attempts[-1] else "stopped_cost_unknown"
            failure["unresolved_reservation_usd"] = "0" if "cost_usd" in attempts[-1] else str(reserve)
        save(output / "ledger.json", {"attempts": attempts, "known_cost_usd": str(known_cost)})
        save(output / "failure.json", failure, exclusive=True)
        raise RuntimeError("Generation stopped: " + redact(exc, api)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config, args.protocol_lock, args.freeze_dir, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
