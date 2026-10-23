#!/usr/bin/env python3
"""Freeze both arms, private alias maps, source and data before any generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

from paired_semantic_anonymization import build_pair
from run_paired_anonymization_generation import expected_slots, file_hash, save, verify_freeze

ROOT = Path(__file__).resolve().parents[3]


def freeze(config_path, lock_path, output, validation_log):
    config_path, lock_path, output = (Path(path).resolve() for path in (config_path, lock_path, output))
    validation_log = Path(validation_log).resolve()
    if output.exists() or lock_path.exists():
        raise ValueError("Never overwrite an existing freeze")
    config = json.loads(config_path.read_text())
    expected_slots(config)
    calibration = set(range(config["calibration_seed_start"],
                            config["calibration_seed_start"] + config["calibration_seed_count"]))
    heldout = set(range(config["heldout_seed_start"],
                        config["heldout_seed_start"] + config["heldout_seed_count"]))
    if len(calibration) != 50 or len(heldout) != 100 or calibration & heldout:
        raise ValueError("Expected disjoint 50 calibration and 100 heldout seeds")
    for start, end in config["excluded_prior_seed_ranges"]:
        if any(start <= seed <= end for seed in calibration | heldout):
            raise ValueError("Fresh replay seeds overlap a prior BH run")
    validation_text = Path(validation_log).read_text()
    if "passed" not in validation_text or " failed" in validation_text or "ERRORS " in validation_text:
        raise ValueError("A passing offline validation log is required")
    output.mkdir(parents=True)
    (output / "messages").mkdir()
    (output / "private_pairs").mkdir()
    message_hashes, pair_hashes, invariant_report = {}, {}, {}
    for pair in range(10):
        identifier = f"pair_{pair:02d}"
        material = build_pair(pair, config["paired_design"]["alias_seed_by_pair"][identifier])
        private_path = output / "private_pairs" / f"{identifier}.json"
        save(private_path, material, exclusive=True)
        pair_hashes[identifier] = file_hash(private_path)
        invariant_report[identifier] = material["invariants"]
        for arm in ("labeled", "anonymous"):
            path = output / "messages" / f"{identifier}_{arm}.json"
            save(path, material["arms"][arm]["messages"], exclusive=True)
            message_hashes[f"{identifier}_{arm}"] = file_hash(path)
    source_files = sorted((ROOT / "experiments/care_replay/scripts").glob("*.py"))
    input_files = [ROOT / config["llm_record"], ROOT / "uv.lock", ROOT / "pyproject.toml"]
    source_hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files}
    data_hashes = {p: file_hash(ROOT / p) for p in config["data_files"]}
    input_hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in input_files}
    for group, files in (("source_snapshot", source_files),
                         ("input_snapshot", input_files + [ROOT / p for p in config["data_files"]])):
        for path in files:
            destination = output / group / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if file_hash(destination) != file_hash(path):
                raise ValueError("Snapshot does not match its input")
    shutil.copyfile(config_path, output / "config.json")
    shutil.copyfile(validation_log, output / "offline_validation.log")
    save(output / "pair_invariants.json", invariant_report, exclusive=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "pandas", "matplotlib", "rdkit", "pytest")}
    lock = {
        "protocol_id": config["protocol_id"], "status": "frozen_before_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_api_calls_at_freeze": 0,
        "config_sha256": file_hash(config_path), "source_sha256": source_hashes,
        "data_sha256": data_hashes, "input_sha256": input_hashes,
        "messages_sha256": message_hashes, "pair_material_sha256": pair_hashes,
        "offline_validation_log_sha256": file_hash(validation_log),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": versions,
                        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "freeze_directory": output.relative_to(ROOT).as_posix(),
        "evidence_boundary": config["analysis"]["scope"],
    }
    save(lock_path, lock, exclusive=True)
    shutil.copyfile(lock_path, output / "protocol_lock.json")
    verify_freeze(config_path, lock_path, output)
    print(json.dumps({"status": "frozen", "lock": str(lock_path), "lock_sha256": file_hash(lock_path),
                      "messages": 20, "private_pairs": 10, "source_files": len(source_files),
                      "model_api_calls": 0}, indent=2))
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-log", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.config, args.protocol_lock, args.output, args.validation_log)


if __name__ == "__main__":
    main()
