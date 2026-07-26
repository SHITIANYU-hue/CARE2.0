#!/usr/bin/env python3
"""Index auditable LLM and replay traces without exposing API credentials.

The repository stores structured prompt/response records and replay audit
events. This index makes those records discoverable. It does not claim to
contain hidden model chain-of-thought; the saved artifact is an auditable
decision trace: request metadata, model output, compiled skill, and replay
events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_jsonl(path: Path) -> tuple[int, dict[str, Any]]:
    count = 0
    first: dict[str, Any] = {}
    last: dict[str, Any] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            events = row.get("events") if isinstance(row, dict) else None
            if isinstance(events, list):
                count += len(events)
                if events and not first:
                    first = events[0]
                if events:
                    last = events[-1]
            else:
                count += 1
                if not first:
                    first = row
                last = row
    metadata = {
        "event_count": count,
        "first_seed": first.get("seed"),
        "last_seed": last.get("seed"),
        "first_mode": first.get("mode"),
        "last_mode": last.get("mode"),
        "first_round": first.get("round_index"),
        "last_round": last.get("round_index"),
    }
    return count, metadata


def model_call_entry(path: Path, repo_root: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    api = record.get("api_configuration", {})
    return {
        "type": "llm_call",
        "path": str(path.relative_to(repo_root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "source_dataset": record.get("source_dataset"),
        "target_dataset": record.get("target_dataset"),
        "model": record.get("model"),
        "llm_generation_call_count": record.get("llm_generation_call_count"),
        "usage": record.get("usage", {}),
        "api_mode": api.get("api_mode"),
        "structured_mode": api.get("structured_mode"),
        "temperature": api.get("temperature"),
        "normalized_skill_count": len(record.get("normalized_skills", [])),
        "normalized_patch_count": len(record.get("normalized_patches", [])),
        "has_prompt_payload": bool(record.get("prompt_payload")),
        "has_raw_response": bool(record.get("raw_response")),
        "has_parsed_response": bool(record.get("parsed_response")),
    }


def trace_entry(path: Path, repo_root: Path, trace_type: str) -> dict[str, Any]:
    count, metadata = count_jsonl(path)
    entry = {
        "type": trace_type,
        "path": str(path.relative_to(repo_root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    entry.update(metadata)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-outcome-root", type=Path, required=True)
    parser.add_argument("--model-call-root", type=Path, action="append", default=[])
    parser.add_argument("--zero-shot-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.source_outcome_root.resolve()

    entries: list[dict[str, Any]] = []
    model_dir = source_root / "model_calls"
    for path in sorted(model_dir.glob("*.json")):
        entries.append(model_call_entry(path, repo_root))
    for root in args.model_call_root:
        root = root.resolve()
        for path in sorted(root.glob("*.json")):
            entries.append(model_call_entry(path, repo_root))
    trace_dir = source_root / "reasoning_traces"
    for path in sorted(trace_dir.glob("*.jsonl")):
        entries.append(trace_entry(path, repo_root, "replay_reasoning_trace"))
    for root in args.zero_shot_root:
        root = root.resolve()
        for path in sorted(root.glob("audits.jsonl")):
            entries.append(trace_entry(path, repo_root, "zero_shot_replay_audit"))

    summary = {
        "protocol": (
            "Structured LLM request/response records and replay decisions. "
            "This is an audit trace, not hidden chain-of-thought."
        ),
        "source_outcome_root": str(source_root.relative_to(repo_root)),
        "entry_count": len(entries),
        "llm_call_count": sum(entry["type"] == "llm_call" for entry in entries),
        "replay_trace_count": sum(entry["type"] != "llm_call" for entry in entries),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("entry_count", "llm_call_count", "replay_trace_count")}, indent=2))


if __name__ == "__main__":
    main()
