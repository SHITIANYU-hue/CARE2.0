#!/usr/bin/env python3
"""Run provider-aware confirmation with a transparent JSON envelope adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_online_llm_scientist as online
import run_repeated_online_llm_confirmation_v2 as v2
import run_repeated_online_llm_confirmation_v3 as v3


SCHEMA_VERSION = "care.repeated_online_confirmation/v4"
SCRIPT_PATH = Path(__file__).resolve()
_ORIGINAL_EXTRACT_JSON_OBJECT = online.replay.extract_json_object


def extract_provider_json_object(content: str) -> Mapping[str, Any]:
    """Remove only a single provider-added ``answer`` object envelope."""
    parsed = _ORIGINAL_EXTRACT_JSON_OBJECT(content)
    if set(parsed) == {"answer"} and isinstance(parsed.get("answer"), Mapping):
        return dict(parsed["answer"])
    return parsed


def protocol_lock(
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    lock = v3.protocol_lock(suite_config, suite)
    lock["schema_version"] = SCHEMA_VERSION
    lock["artifacts"]["provider_schema_adapter_runner"] = {
        "path": str(SCRIPT_PATH),
        "sha256": v2.base.file_sha256(SCRIPT_PATH),
    }
    return lock


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(argv or sys.argv)
    suite = v2.base.load_json(v3._suite_path(arguments))
    v3.validate_suite(suite)
    client = v3.ProviderAwareChatClient(suite["rate_control"])
    online.replay.chat_completion_text = client
    online.replay.extract_json_object = extract_provider_json_object
    v2.validate_suite = v3.validate_suite
    v2.protocol_lock = protocol_lock
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.main()


if __name__ == "__main__":
    main()
