#!/usr/bin/env python3
"""Freeze config, local inputs, sources and public-metadata libraries before replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import run_matched_random_rule_controls as study


def freeze(config_path: Path, output: Path):
    if output.exists():
        raise ValueError("Protocol lock already exists; create a new version")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    study.validate_config(config)
    record_path = study.ROOT / config["llm_record"]
    if study.file_digest(record_path) != config["llm_record_sha256"]:
        raise ValueError("Frozen LLM record hash mismatch")
    with study.offline_guard():
        adapter = study.replay.DATASET_BUILDERS[config["target_dataset"]]()
        catalog = study.semantic.semantic_field_catalog(adapter)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        skills = study.semantic.normalize_skills({"skills": record["normalized_skills"]}, catalog, 100)
        if [skill.skill_id for skill in skills] != config["expected_skill_ids"]:
            raise ValueError("Frozen LLM skill IDs changed")
        if len(adapter.candidates) != config["expected_candidate_count"]:
            raise ValueError("Candidate pool changed")
        diagnostics = {}
        libraries = study.make_libraries(skills, catalog, config, adapter.candidates, diagnostics)
        all_skills = tuple(skill for bank in libraries.values() for skill in bank)
        representatives, aliases = study.execution_plan(all_skills, config.get("exact_execution_reuse", False))
    lock = {
        "protocol_id": config["protocol_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "statement": "Local freeze before any calibration or held-out replay for this protocol; not an external preregistration.",
        "config_sha256": study.file_digest(config_path),
        "source_sha256": {path.relative_to(study.ROOT).as_posix(): study.file_digest(path)
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "data_sha256": {name: study.file_digest(study.ROOT / name) for name in config["data_files"]},
        "libraries_sha256": study.digest({name: [asdict(skill) for skill in bank]
                                            for name, bank in libraries.items()}),
        "execution_aliases_sha256": study.digest(aliases),
    }
    if "coverage_matching" in config:
        lock["coverage_matching_sha256"] = study.digest(diagnostics)
    output.parent.mkdir(parents=True, exist_ok=True)
    study.save_json(output, lock)
    return {"protocol_lock": str(output), "sha256": study.file_digest(output),
            "logical_skills": len(all_skills), "unique_execution_skills": len(representatives),
            "coverage_matching_diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
