#!/usr/bin/env python3
"""Upgrade a frozen LLM kernel record with the generic outcome-interaction operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def upgrade_patch(patch: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(patch)
    source_strength = max(0.0, float(upgraded.get("source_prior_strength", 0.0)))
    if "source_interaction_strength" not in upgraded:
        upgraded["source_interaction_strength"] = round(
            min(1.5, max(0.25, 0.75 * source_strength)),
            4,
        )
    upgraded.setdefault("source_interaction_min_support", 5)
    upgraded.setdefault("canonicalize_source_values", True)
    return upgraded


def upgrade_record(
    record: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    patches = record.get("normalized_patches", [])
    if not isinstance(patches, list) or not patches:
        raise ValueError("The input record contains no normalized_patches.")
    upgraded = dict(record)
    upgraded["normalized_patches"] = [
        upgrade_patch(patch)
        for patch in patches
        if isinstance(patch, dict)
    ]
    upgraded["source_outcome_operator_upgrade"] = {
        "source_record": str(source_path),
        "operator": (
            "Generic mapped source-neighbor outcome prior plus pairwise source residual "
            "interaction prior, both calibrated prequentially on revealed target outcomes."
        ),
        "target_outcomes_used": False,
        "llm_regenerated": False,
        "rule": (
            "Preserve every frozen LLM scale, role multiplier, beta schedule, confidence, "
            "and reason. Add one dataset-agnostic interaction strength derived only from "
            "the LLM source-prior strength, with fixed support and canonicalization defaults."
        ),
    }
    return upgraded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the generic source-outcome interaction operator to a frozen LLM record."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    upgraded = upgrade_record(record, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
