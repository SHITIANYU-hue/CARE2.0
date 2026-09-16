#!/usr/bin/env python3
"""Run transport-robust confirmation with context-preserving JSON repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import run_online_llm_scientist as online
import run_repeated_online_llm_confirmation_v2 as v2
import run_repeated_online_llm_confirmation_v5 as v5


SCHEMA_VERSION = "care.repeated_online_confirmation/v6"
SCRIPT_PATH = Path(__file__).resolve()


def validate_suite(suite: Mapping[str, Any], *, check_paths: bool = True) -> None:
    v5.validate_suite(suite, check_paths=check_paths)
    repair = suite.get("repair_policy", {})
    if repair.get("mode") != "context_preserving_schema_repair_v1":
        raise ValueError("repair_policy.mode must be context_preserving_schema_repair_v1")
    if repair.get("original_scientific_context") != "retained":
        raise ValueError("Schema repair must retain original scientific context.")
    if repair.get("semantic_replanning") != "forbidden":
        raise ValueError("Schema repair must forbid semantic replanning.")


def context_preserving_validated_llm_json(
    config: Any,
    messages: Sequence[Mapping[str, str]],
    normalizer: Callable[[Mapping[str, Any]], dict[str, Any]],
    repair_context: Mapping[str, Any],
    repair_attempts: int,
) -> tuple[
    dict[str, Any],
    str,
    Mapping[str, Any],
    Mapping[str, Any],
    list[dict[str, Any]],
]:
    """Repair schema errors while retaining the complete scientific prompt."""
    if repair_attempts < 0:
        raise ValueError("repair_attempts must be non-negative.")
    original_messages = [dict(message) for message in messages]
    current_messages = list(original_messages)
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(repair_attempts + 1):
        content, metadata = online.replay.chat_completion_text(
            config,
            current_messages,
        )
        parsed: Mapping[str, Any] = {}
        try:
            parsed = online.replay.extract_json_object(content)
            normalized = normalizer(parsed)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "valid": True,
                    "usage": metadata.get("usage", {}),
                    "raw_response": content,
                }
            )
            return normalized, content, metadata, parsed, attempts
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "valid": False,
                    "usage": metadata.get("usage", {}),
                    "raw_response": content,
                    "validation_error": str(exc),
                }
            )
            if attempt_index >= repair_attempts:
                raise online.LLMValidationError(str(exc), attempts) from exc
            repair_prompt = {
                "task": "Repair only the response format and schema.",
                "validation_error": str(exc),
                "previous_response": content or "",
                "constraints": dict(repair_context),
                "repair_rules": [
                    "Use the original scientific evidence and candidate menu above.",
                    "Preserve the intended scientific decision when it is valid.",
                    "Do not claim that evidence is missing when it appears above.",
                    "Return one JSON object only, with no file wrapper or markdown.",
                ],
            }
            repair_messages: list[dict[str, str]] = []
            if content:
                repair_messages.append(
                    {
                        "role": "assistant",
                        "content": str(content),
                    }
                )
            repair_messages.append(
                {
                    "role": "user",
                    "content": json.dumps(repair_prompt, ensure_ascii=False),
                }
            )
            current_messages = original_messages + repair_messages
    raise AssertionError("Unreachable validated LLM loop.")


def protocol_lock(
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    lock = v5.protocol_lock(suite_config, suite)
    lock["schema_version"] = SCHEMA_VERSION
    lock["artifacts"]["context_preserving_repair_runner"] = {
        "path": str(SCRIPT_PATH),
        "sha256": v2.base.file_sha256(SCRIPT_PATH),
    }
    return lock


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv or sys.argv)
    suite = v2.base.load_json(v5._suite_path(arguments))
    validate_suite(suite)
    online.replay.extract_json_object = v5.extract_provider_transport_object
    online.validated_llm_json = context_preserving_validated_llm_json
    v2.validate_suite = validate_suite
    v2.protocol_lock = protocol_lock
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.main()


if __name__ == "__main__":
    main()
