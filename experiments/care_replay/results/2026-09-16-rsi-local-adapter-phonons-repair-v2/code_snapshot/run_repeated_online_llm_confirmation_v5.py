#!/usr/bin/env python3
"""Run confirmation with a strict provider transport-envelope adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_online_llm_scientist as online
import run_repeated_online_llm_confirmation_v2 as v2


SCHEMA_VERSION = "care.repeated_online_confirmation/v5"
SCRIPT_PATH = Path(__file__).resolve()
_ORIGINAL_EXTRACT_JSON_OBJECT = online.replay.extract_json_object
_ORIGINAL_PROTOCOL_LOCK = v2.protocol_lock
_ORIGINAL_VALIDATE_SUITE = v2.validate_suite


def _unwrap_answer(parsed: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(parsed) == {"answer"} and isinstance(parsed.get("answer"), Mapping):
        return dict(parsed["answer"])
    return parsed


def extract_provider_transport_object(content: str) -> Mapping[str, Any]:
    """Unwrap only declared JSON transport envelopes without semantic edits."""
    parsed = _ORIGINAL_EXTRACT_JSON_OBJECT(content)
    parsed = _unwrap_answer(parsed)
    allowed_artifact_keys = {"content"} | ({"file_path"} if "file_path" in parsed else set())
    if set(parsed) == allowed_artifact_keys and isinstance(parsed.get("content"), str):
        nested = _ORIGINAL_EXTRACT_JSON_OBJECT(str(parsed["content"]))
        return _unwrap_answer(nested)
    return parsed


def validate_suite(suite: Mapping[str, Any], *, check_paths: bool = True) -> None:
    _ORIGINAL_VALIDATE_SUITE(suite, check_paths=check_paths)
    adapter = suite.get("provider_schema_adapter", {})
    if adapter.get("mode") != "strict_transport_envelopes_v1":
        raise ValueError("provider_schema_adapter.mode must be strict_transport_envelopes_v1")
    if adapter.get("semantic_field_mutation") != "forbidden":
        raise ValueError("Transport adapter must forbid semantic field mutation.")


def protocol_lock(
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    lock = _ORIGINAL_PROTOCOL_LOCK(suite_config, suite)
    lock["schema_version"] = SCHEMA_VERSION
    lock["artifacts"]["provider_transport_adapter_runner"] = {
        "path": str(SCRIPT_PATH),
        "sha256": v2.base.file_sha256(SCRIPT_PATH),
    }
    return lock


def _suite_path(argv: Sequence[str]) -> Path:
    try:
        index = argv.index("--suite-config")
        return Path(argv[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--suite-config is required") from exc


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv or sys.argv)
    suite = v2.base.load_json(_suite_path(arguments))
    validate_suite(suite)
    online.replay.extract_json_object = extract_provider_transport_object
    v2.validate_suite = validate_suite
    v2.protocol_lock = protocol_lock
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.main()


if __name__ == "__main__":
    main()
