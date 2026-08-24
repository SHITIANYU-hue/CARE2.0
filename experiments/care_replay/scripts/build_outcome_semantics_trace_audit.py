#!/usr/bin/env python3
"""Audit whether transformed replay scores are incorrectly given raw units."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "care.outcome_semantics_trace_audit/v1"
RESPONSE_EVENTS = {
    "llm_round_proposal_response",
    "llm_round_critic_response",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)


def audit_trace(path: Path, raw_unit: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])(?:\d+(?:\.\d+)?|\.\d+)\s*{re.escape(raw_unit)}\b",
        re.IGNORECASE,
    )
    responses = []
    violations = []
    for event in load_trace(path):
        if event.get("event") not in RESPONSE_EVENTS:
            continue
        response = event.get("parsed_response", {})
        responses.append(event)
        for text in strings(response):
            for match in pattern.finditer(text):
                violations.append(
                    {
                        "round_index": int(event["round_index"]),
                        "event": str(event["event"]),
                        "matched_value_unit": match.group(0),
                        "text": text,
                    }
                )
    return {
        "trace": str(path),
        "trace_sha256": sha256(path),
        "response_events": len(responses),
        "invalid_score_unit_couplings": len(violations),
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-trace", required=True, type=Path)
    parser.add_argument("--after-trace", required=True, type=Path)
    parser.add_argument("--raw-unit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    before = audit_trace(args.before_trace, args.raw_unit)
    after = audit_trace(args.after_trace, args.raw_unit)
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_question": (
            "Does an LLM response attach the target's raw physical unit to a "
            "numeric transformed replay score?"
        ),
        "raw_unit": args.raw_unit,
        "scope": (
            "Parsed proposal and critic responses only; prompts and hidden target "
            "values are excluded."
        ),
        "before_outcome_scale_contract": before,
        "after_outcome_scale_contract": after,
        "observed_change": {
            "invalid_score_unit_couplings": (
                after["invalid_score_unit_couplings"]
                - before["invalid_score_unit_couplings"]
            ),
            "post_contract_pass": after["invalid_score_unit_couplings"] == 0,
        },
        "claim_boundary": (
            "This development audit verifies one response-level semantic failure "
            "mode on one material route. It does not establish general scientific "
            "reasoning accuracy or confirm the transfer effect."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sha256(args.output)
    print(json.dumps(report["observed_change"], ensure_ascii=False))


if __name__ == "__main__":
    main()
