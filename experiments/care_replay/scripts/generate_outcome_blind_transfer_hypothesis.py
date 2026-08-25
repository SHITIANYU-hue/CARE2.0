#!/usr/bin/env python3
"""Generate and archive an outcome-blind external-transfer hypothesis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib import request


ALLOWED_SKILLS = {"source_additive_mutation_prior", "abstain"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response contains no JSON object.")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    required = {
        "hypothesis",
        "mechanism",
        "recommended_skill",
        "transfer_scope",
        "failure_conditions",
        "rationale",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Model response is missing fields: {missing}")
    if payload["recommended_skill"] not in ALLOWED_SKILLS:
        raise ValueError("Model selected a skill outside the frozen menu.")
    if not isinstance(payload["failure_conditions"], list):
        raise ValueError("failure_conditions must be a list.")
    return payload


def generate(
    spec_path: Path,
    output_path: Path,
    *,
    base_url: str,
    model: str,
) -> dict[str, object]:
    api_key = os.environ.get("COMMONSTACK_API_KEY", "").strip()
    if not api_key:
        raise ValueError("COMMONSTACK_API_KEY is required.")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    system = (
        "You are the semantic transfer planner in a preregistered scientific "
        "optimization protocol. You may use only the supplied public task "
        "metadata. You have no experimental target values. Decide whether the "
        "bounded additive-mutation skill is scientifically justified. Return "
        "JSON only and never invent observed performance."
    )
    user = json.dumps(
        {
            "task": spec,
            "skill_menu": {
                "source_additive_mutation_prior": (
                    "Estimate shrunk per-mutation effects from the completed "
                    "low-order source experiments defined by the supplied "
                    "split, sum available effects for a higher-order target "
                    "variant, and blend that source rank with target-only "
                    "GP-UCB under a fixed decaying authority."
                ),
                "abstain": "Use target-only GP-UCB and transfer no source effect.",
            },
            "required_json_schema": {
                "hypothesis": "string",
                "mechanism": "string",
                "recommended_skill": "source_additive_mutation_prior or abstain",
                "transfer_scope": "string",
                "failure_conditions": ["string"],
                "rationale": "string",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    request_payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    raw_response = response_payload["choices"][0]["message"]["content"]
    parsed = extract_json(str(raw_response))
    record = {
        "schema_version": "care.outcome_blind_transfer_hypothesis/v1",
        "status": "generated_before_dataset_download",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "temperature": 0,
        "public_task_spec": spec,
        "target_outcomes_available": False,
        "prompt": {"system": system, "user": user},
        "request_sha256": sha256_bytes(body),
        "raw_response": raw_response,
        "parsed_hypothesis": parsed,
        "provider_response_id": response_payload.get("id"),
        "finish_reason": response_payload["choices"][0].get("finish_reason"),
        "system_fingerprint": response_payload.get("system_fingerprint"),
        "usage": response_payload.get("usage", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    output_path.write_bytes(encoded)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{sha256_bytes(encoded)}  {output_path.name}\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.commonstack.ai/v1")
    parser.add_argument("--model", default="anthropic/claude-opus-4-8")
    args = parser.parse_args()
    record = generate(
        args.spec,
        args.output,
        base_url=args.base_url,
        model=args.model,
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "model": record["model"],
                "recommended_skill": record["parsed_hypothesis"][
                    "recommended_skill"
                ],
                "request_sha256": record["request_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
