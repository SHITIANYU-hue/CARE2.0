#!/usr/bin/env python3
"""V2 frozen generation: one fresh POST per slot; whole invalid libraries fall back."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import shutil
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/care_replay/scripts"))
import paired_semantic_anonymization as contract
import run_paired_anonymization_generation as legacy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

file_hash, save, load, expected_slots = common.file_hash, common.save, common.load, common.expected_slots
TransportError, OpenRouter = legacy.TransportError, legacy.OpenRouter
PRIOR_COST = Decimal("0.0125145")
RESERVE = Decimal("0.045")
BUDGET = Decimal("1")

VALID = "VALID"
INVALID = "INVALID_ANCHOR_FALLBACK"
ARMS = ("labeled", "anonymous")


class Halt(RuntimeError):
    """A protocol, provenance, billing, material or internal failure; stop."""


def object_sha256(value: Any) -> str:
    """Canonical object hash; distinct from the original receipt byte hash."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Decision:
    status: str
    arm: str
    cost_usd: str
    reasons: tuple[str, ...]
    # INVALID always has library=None; no partial or repaired library survives.
    library: dict[str, Any] | None


def _pin(value: Any, expected: str, label: str) -> None:
    try:
        if not isinstance(expected, str) or len(expected) != 64 or object_sha256(value) != expected:
            raise ValueError("canonical object hash mismatch")
    except Exception as exc:
        raise Halt(f"{label} integrity failure") from exc


def _check_material(material: dict[str, Any], expected_sha256: str) -> None:
    _pin(material, expected_sha256, "pair material")
    try:
        # This checks all mappings, frozen record/code hashes and both catalogs,
        # before any model-content failure can be labeled INVALID.
        rebuilt = contract.build_pair(material["pair_index"], material["alias_seed"])
        if object_sha256(rebuilt) != object_sha256(material):
            raise ValueError("material differs from deterministic builder")
    except Exception as exc:
        raise Halt("pair material or builder failed validation") from exc


def classify_generation(response: Any, arm: str, pair_material: dict[str, Any], *,
                        response_sha256: str, material_sha256: str,
                        settled_cost: Decimal, expected_model: str = "openai/gpt-4o-mini",
                        expected_provider: str = "OpenAI") -> Decision:
    """VALID, whole-library INVALID, or Halt; never infer a repair or retry.

    Source/billing/material checks precede model-output classification. A strict
    parser/contract rejection is INVALID. Once those checks pass, *any* failure
    in the final decoder is internal/protocol failure and is HALT, including its
    ValidationError class, which also represents material errors in the v1 API.
    """
    if arm not in ARMS:
        raise Halt("unknown arm")
    _pin(response, response_sha256, "response")
    if (not isinstance(response, dict) or response.get("object") != "chat.completion"
            or not isinstance(response.get("id"), str) or not response["id"]
            or response.get("model") != expected_model
            or response.get("provider") != expected_provider or response.get("error")):
        raise Halt("response provenance/envelope cannot be established")
    cost = money(settled_cost)
    if cost > RESERVE:
        raise Halt("settled cost exceeds frozen reservation")
    _check_material(pair_material, material_sha256)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise Halt("response must have exactly one well-formed choice")
    choice = choices[0]
    message = choice.get("message")
    if (choice.get("error") or not isinstance(message, dict)
            or message.get("role") != "assistant"):
        raise Halt("malformed choice/message envelope")
    reason = choice.get("finish_reason")
    if reason not in ("stop", "length", "content_filter", "tool_calls", "function_call"):
        raise Halt("unknown or unsupported finish_reason")
    refusal, tools, content = message.get("refusal"), message.get("tool_calls"), message.get("content")
    if refusal is not None and not isinstance(refusal, str):
        raise Halt("malformed refusal envelope")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
        raise Halt("malformed tool-call envelope")
    function = message.get("function_call")
    if function is not None and (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                                 or not isinstance(function.get("arguments"), str)):
        raise Halt("malformed function-call envelope")
    if content is not None and not isinstance(content, str):
        raise Halt("malformed text-content envelope")
    if reason == "tool_calls" and not tools:
        raise Halt("tool-call termination without returned tool calls")
    if reason == "function_call" and not function:
        raise Halt("function-call termination without returned function call")

    def invalid(*reasons: str) -> Decision:
        return Decision(INVALID, arm, str(cost), tuple(reasons), None)

    if reason in ("length", "content_filter"):
        return invalid("finish_reason:" + reason)
    if refusal:
        return invalid("model_refusal")
    if tools or function:
        return invalid("returned_tool_calls")
    if content is None or not content.strip():
        return invalid("empty_text")
    try:
        payload = contract.parse_strict_json(content)
    except contract.ValidationError as exc:
        return invalid("invalid_json", *exc.errors)
    except Exception as exc:
        raise Halt("unexpected parser failure") from exc
    try:
        catalog = pair_material["arms"][arm]["payload"]["target"]["public_semantic_fields"]
        errors = contract.strict_validate_library(payload, catalog)
        if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
            raise TypeError("invalid validator return type")
    except Exception as exc:
        raise Halt("unexpected contract-validator failure") from exc
    if errors:
        return invalid("invalid_library_contract", *errors)
    try:
        decoded = contract.validate_and_decode(content, arm, pair_material)
        # A second explicit check prevents an internal decoder from dropping data.
        labeled_catalog = pair_material["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"]
        if contract.strict_validate_library(decoded, labeled_catalog):
            raise ValueError("decoded library failed its contract")
        expected = deepcopy(payload)
        if arm == "anonymous":
            inverse = pair_material["private_mapping"]["inverse"]
            for skill in expected["skills"]:
                for rule in skill["rules"]:
                    rule["conditions"] = {inverse["fields"][f]: inverse["values"][v]
                                          for f, v in rule["conditions"].items()}
        if decoded != expected:
            raise ValueError("decoder changed output beyond condition-label decoding")
    except Exception as exc:
        raise Halt("decoder/internal validation failed after valid output checks") from exc
    return Decision(VALID, arm, str(cost), (), decoded)


def money(value):
    if value is None or isinstance(value, bool):
        raise Halt("Missing or invalid cost")
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise ValueError("not finite/nonnegative")
        return result
    except Exception:
        raise Halt("Invalid finite nonnegative cost") from None


def utc():
    return datetime.now(timezone.utc).isoformat()


def validate_authorization(config):
    expected_slots(config)
    auth = config["resource_authorization"]
    if any(auth.get(key) is not True for key in (
            "generation_enabled", "new_model_calls_approved", "new_invalid_output_policy_approved")):
        raise Halt("V2 generation, new calls and invalid fallback must all be explicitly authorized")
    if (auth.get("approved_new_logical_requests") != 20 or
            auth.get("approved_logical_requests") != 20 or
            auth.get("approved_pair_count") != 10 or
            money(auth.get("approved_max_cost_usd")) != BUDGET or
            not isinstance(auth.get("authorization_reference"), str) or not auth["authorization_reference"] or
            auth.get("maximum_new_post_attempts") != 20 or
            auth.get("proposed_new_logical_requests") != 20 or
            auth.get("prior_experiment_actual_post_attempts") != 11 or
            money(auth.get("prior_known_cost_usd")) != PRIOR_COST or
            money(auth.get("proposed_cumulative_max_cost_usd")) != BUDGET or
            money(auth.get("per_attempt_reservation_usd")) != RESERVE or
            money(config["api"].get("cumulative_prior_spend_usd")) != PRIOR_COST or
            money(config["api"].get("max_cost_usd")) != BUDGET or
            money(config["api"].get("reservation_per_attempt_usd")) != RESERVE):
        raise Halt("V2 approved count, prior spend or cumulative budget differs")
    if PRIOR_COST + 20 * RESERVE > BUDGET:
        raise Halt("Prior spend plus all new reservations exceeds cumulative budget")
    if not isinstance(config.get("generation_output_directory"), str) or not config["generation_output_directory"]:
        raise Halt("A unique approved generation output is required")
    # Validate exact model, routing, parameters and price caps without network.
    legacy.validate_budget(config, {"id": "openai/gpt-4o-mini", "context_length": 128000,
                           "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0"}}, {})


def approved_output(config, output):
    output = Path(output)
    if output.resolve() != (ROOT / config["generation_output_directory"]).resolve():
        raise Halt("Output differs from the unique approved V2 destination")
    if output.exists():
        raise Halt("Generation output already exists; no automatic resume or repeat POSTs")
    return output


def response_cost(response, api, receipt_dir, sleep=time.sleep):
    """Settle one existing completion. Only ancillary GETs may be retried."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise Halt("Malformed usage envelope")
    if usage.get("is_byok") is True:
        raise Halt("Response reports BYOK")
    inline = usage.get("cost")
    inline_cost = money(inline) if inline is not None else None
    if inline_cost is not None and usage.get("is_byok") is False:
        result = {"source": "response_usage", "response_id": response["id"],
                  "cost_usd": str(inline_cost), "is_byok": False}
        save(receipt_dir / "billing.json", result, exclusive=True)
        return inline_cost
    from urllib.parse import quote
    info_data = None
    for attempt in range(3):
        sleep(2 ** attempt)
        try:
            info = api.request("/generation?id=" + quote(response["id"], safe=""))
            save(receipt_dir / f"cost_lookup_{attempt}.json", info, exclusive=True)
            if not isinstance(info, dict) or not isinstance(info.get("data"), dict):
                raise Halt("Malformed generation billing metadata")
            info_data = info["data"]
            if (info_data.get("id") != response["id"] or
                    info_data.get("model") != response["model"] or
                    info_data.get("provider_name") != response["provider"]):
                raise Halt("Generation billing provenance differs from the completion")
            if info_data.get("is_byok") is True:
                raise Halt("Generation metadata reports BYOK")
            if info_data.get("total_cost") is not None and info_data.get("is_byok") is False:
                break
        except TransportError as exc:
            save(receipt_dir / f"cost_lookup_{attempt}_error.json",
                 {"http_status": exc.status, "error": legacy.redact(exc.body, api)}, exclusive=True)
            if exc.status != 404:
                break
    if not info_data or info_data.get("is_byok") is not False or info_data.get("total_cost") is None:
        raise Halt("Cost or non-BYOK status remains unknown")
    lookup_cost = money(info_data["total_cost"])
    cost = max(lookup_cost, inline_cost) if inline_cost is not None else lookup_cost
    save(receipt_dir / "billing.json", {"source": "generation_lookup",
         "response_id": response["id"], "cost_usd": str(cost), "is_byok": False}, exclusive=True)
    return cost


def provenance(response, config):
    if (not isinstance(response, dict) or response.get("object") != "chat.completion" or
            not isinstance(response.get("id"), str) or not response["id"] or
            response.get("model") != config["model"]["id"] or
            response.get("provider") != "OpenAI" or response.get("error")):
        raise Halt("Response provenance/envelope cannot be established")


def execute(config_path, lock_path, freeze_dir, output, *, api=None, sleep=time.sleep):
    config, lock = common.verify_freeze(config_path, lock_path, freeze_dir)
    validate_authorization(config)
    output, freeze_dir = approved_output(config, output), Path(freeze_dir)
    output.mkdir(parents=True, exist_ok=False)  # Single durable claim before any request.
    for folder in ("receipts", "libraries", "outcomes"):
        (output / folder).mkdir()
    attempts, outcomes, libraries = [], {}, {}
    known_new = Decimal(0)

    def write_ledger():
        save(output / "ledger.json", {"attempts": attempts,
             "actual_post_attempts": len(attempts), "prior_cost_usd": str(PRIOR_COST),
             "known_new_cost_usd": str(known_new),
             "cumulative_known_cost_usd": str(PRIOR_COST + known_new)})

    try:
        shutil.copyfile(config_path, output / "config.json")
        shutil.copyfile(lock_path, output / "protocol_lock.json")
        for group, folder, source_folder in (("messages_sha256", "messages", "messages"),
                                            ("pair_material_sha256", "pairs", "private_pairs")):
            destination = output / "input_snapshot" / folder
            destination.mkdir(parents=True)
            for identifier, expected in lock[group].items():
                path = destination / (identifier + ".json")
                shutil.copyfile(freeze_dir / source_folder / path.name, path)
                if file_hash(path) != expected:
                    raise Halt("Frozen input changed while copying snapshot")
        for relative, expected in lock["source_sha256"].items():
            path = output / "source_snapshot" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
            if file_hash(path) != expected:
                raise Halt("Frozen source changed while copying snapshot")
        for group in ("data_sha256", "input_sha256"):
            for relative, expected in lock[group].items():
                path = output / "input_snapshot" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
                if file_hash(path) != expected:
                    raise Halt("Frozen data/dependency changed while copying snapshot")
        if file_hash(output / "config.json") != lock["config_sha256"]:
            raise Halt("Config changed while copying snapshot")
        if api is None:
            api = OpenRouter(legacy.load_key(ROOT / ".env"), config["api"]["timeout_seconds"])
        model = next(m for m in api.request("/models")["data"] if m["id"] == config["model"]["id"])
        key_metadata = api.request("/key")["data"]
        budget_check = legacy.validate_budget(config, model, key_metadata)
        save(output / "preflight.json", {"checked_at": utc(), "budget": budget_check,
             "prior_cost_usd": str(PRIOR_COST), "cumulative_reserved_usd": str(PRIOR_COST + 20 * RESERVE),
             "catalog_model": model, "key_authenticated": True, "secret_values_recorded": False}, exclusive=True)
        write_ledger()
        seen_response_ids = set()
        for slot in config["generation_order"]:
            if len(attempts) >= 20 or PRIOR_COST + RESERVE * (len(attempts) + 1) > BUDGET:
                raise Halt("New POST count or cumulative reservation exhausted")
            if PRIOR_COST + known_new + RESERVE > BUDGET:
                raise Halt("Insufficient cumulative budget before new POST")
            identifier = f"pair_{slot['pair_index']:02d}_{slot['arm']}"
            pair_id = f"pair_{slot['pair_index']:02d}"
            message_path = output / "input_snapshot/messages" / (identifier + ".json")
            material_path = output / "input_snapshot/pairs" / (pair_id + ".json")
            if (file_hash(message_path) != lock["messages_sha256"][identifier] or
                    file_hash(material_path) != lock["pair_material_sha256"][pair_id]):
                raise Halt("Frozen message or pair snapshot changed before POST")
            messages, material = load(message_path), load(material_path)
            if (material.get("pair_index") != slot["pair_index"] or
                    material.get("alias_seed") != config["paired_design"]["alias_seed_by_pair"][pair_id]):
                raise Halt("Pair material identity differs from frozen slot")
            _check_material(material, object_sha256(material))
            request = {"model": config["model"]["id"], "messages": messages,
                       "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_tokens"],
                       "response_format": {"type": "json_object"}, "stream": False,
                       "provider": config["api"]["provider"]}
            if len(json.dumps(messages, ensure_ascii=False).encode()) + 4096 + config["model"]["max_tokens"] > config["api"]["context_token_bound"]:
                raise Halt("Input byte bound exceeds frozen model context")
            receipt = output / "receipts" / identifier
            receipt.mkdir()
            save(receipt / "request.json", request, exclusive=True)
            attempt = {"slot": len(attempts), "identifier": identifier, **slot,
                       "started_at": utc(), "reservation_usd": str(RESERVE),
                       "status": "RESERVED_BEFORE_POST", "request_sha256": file_hash(receipt / "request.json")}
            attempts.append(attempt)
            write_ledger()  # Durable request/reservation precedes the only POST.
            try:
                response = api.request("/chat/completions", request)
            except TransportError as exc:
                save(receipt / "transport_error.json", {"http_status": exc.status,
                     "response": legacy.redact(exc.body, api)}, exclusive=True)
                raise
            save(receipt / "response.json", response, exclusive=True)
            attempt.update(response_sha256=file_hash(receipt / "response.json"),
                           response_id=response.get("id") if isinstance(response, dict) else None,
                           usage=response.get("usage") if isinstance(response, dict) else None)
            write_ledger()
            provenance(response, config)
            if response["id"] in seen_response_ids:
                raise Halt("Duplicate generation identity; fresh calls required")
            seen_response_ids.add(response["id"])
            cost = response_cost(response, api, receipt, sleep=sleep)
            known_new += cost
            attempt.update(cost_usd=str(cost), status="ACCOUNTED_RESPONSE")
            write_ledger()
            if cost > RESERVE or PRIOR_COST + known_new > BUDGET:
                raise Halt("Settled response cost exceeds reservation or cumulative budget")
            # Recheck on-disk snapshots and receipts immediately before interpreting content.
            if (file_hash(material_path) != lock["pair_material_sha256"][pair_id] or
                    file_hash(message_path) != lock["messages_sha256"][identifier] or
                    file_hash(receipt / "request.json") != attempt["request_sha256"] or
                    file_hash(receipt / "response.json") != attempt["response_sha256"]):
                raise Halt("Pinned generation evidence changed after POST")
            decision = classify_generation(response, slot["arm"], material,
                        response_sha256=object_sha256(response), material_sha256=object_sha256(material),
                        settled_cost=cost, expected_model=config["model"]["id"])
            library_sha = None
            if decision.status == VALID:
                save(output / "libraries" / (identifier + ".json"), decision.library, exclusive=True)
                library_sha = file_hash(output / "libraries" / (identifier + ".json"))
                libraries[identifier] = library_sha
            outcome = {"identifier": identifier, **slot, "status": decision.status,
                       "reason_codes": list(decision.reasons), "errors": list(decision.reasons[1:]),
                       "request_sha256": attempt["request_sha256"], "response_sha256": attempt["response_sha256"],
                       "response_id": response["id"], "cost_usd": str(cost), "library_sha256": library_sha,
                       "billing_sha256": file_hash(receipt / "billing.json")}
            save(output / "outcomes" / (identifier + ".json"), outcome, exclusive=True)
            outcomes[identifier] = file_hash(output / "outcomes" / (identifier + ".json"))
            attempt.update(status=decision.status, outcome_sha256=outcomes[identifier],
                           library_sha256=library_sha, completed_at=utc())
            write_ledger()
            print(json.dumps({"identifier": identifier, "status": decision.status,
                              "actual_post_attempts": len(attempts), "cumulative_known_cost_usd": str(PRIOR_COST + known_new)}), flush=True)
        if set(outcomes) != expected_slots(config) or len(attempts) != 20:
            raise Halt("Missing terminal slots; no completed generation run")
        common.verify_freeze(config_path, lock_path, freeze_dir)
        if (file_hash(output / "config.json") != file_hash(config_path) or
                file_hash(output / "protocol_lock.json") != file_hash(lock_path)):
            raise Halt("Frozen protocol changed before completion")
        result = {"status": "complete", "completed_at": utc(), "config_sha256": file_hash(config_path),
                  "protocol_lock_sha256": file_hash(lock_path), "messages_sha256": lock["messages_sha256"],
                  "pair_material_sha256": lock["pair_material_sha256"], "outcomes_sha256": outcomes,
                  "libraries_sha256": libraries, "actual_post_attempts": len(attempts),
                  "ledger_sha256": file_hash(output / "ledger.json"),
                  "valid_library_count": len(libraries), "invalid_library_count": 20 - len(libraries),
                  "known_new_cost_usd": str(known_new), "cumulative_known_cost_usd": str(PRIOR_COST + known_new),
                  "prior_cost_usd": str(PRIOR_COST)}
        save(output / "completion.json", result, exclusive=True)
        return result
    except BaseException as exc:
        unresolved = RESERVE if attempts and "cost_usd" not in attempts[-1] else Decimal(0)
        if attempts and attempts[-1]["status"] not in (VALID, INVALID):
            attempts[-1]["status"] = "HALT"
        write_ledger()
        error = legacy.redact(exc, api)
        save(output / "failure.json", {"status": "HALT", "stopped_at": utc(),
             "actual_post_attempts": len(attempts), "terminal_outcome_count": len(outcomes),
             "valid_library_count": len(libraries), "invalid_library_count": len(outcomes) - len(libraries),
             "prior_cost_usd": str(PRIOR_COST), "known_new_cost_usd": str(known_new),
             "cumulative_known_cost_usd": str(PRIOR_COST + known_new),
             "unresolved_reservation_usd": str(unresolved), "error_type": type(exc).__name__, "error": error,
             "retry_performed": False, "automatic_repair_performed": False,
             "incomplete_run_not_evaluated": True}, exclusive=True)
        raise Halt("V2 generation halted: " + error) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    config, _ = common.verify_freeze(args.config, args.protocol_lock, args.freeze_dir)
    validate_authorization(config)
    if not args.execute_approved:
        print(json.dumps({"status": "verified_offline", "api_calls": 0, "credentials_read": False,
                          "prior_cost_usd": str(PRIOR_COST), "maximum_new_post_attempts": 20}))
        return
    if args.output is None:
        parser.error("--execute-approved requires the unique --output")
    execute(args.config, args.protocol_lock, args.freeze_dir, args.output)


if __name__ == "__main__":
    main()
