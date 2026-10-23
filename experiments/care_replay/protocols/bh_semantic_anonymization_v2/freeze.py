"""Freeze V2 inputs after completed offline tests; never call a model API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import re
import sys

import common


def build_pair(pair_index, alias_seed):
    # Keep the original generator and frozen prompt construction untouched.
    scripts = str(common.ROOT / "experiments/care_replay/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from paired_semantic_anonymization import build_pair as original_builder
    return original_builder(pair_index, alias_seed)


def validate_log(path, sources):
    path = common.repo_path(common.relative_path(path))
    text = path.read_text(encoding="utf-8")
    if (re.search(r"\b[1-9][0-9]* passed\b", text) is None
            or re.search(r"\b[1-9][0-9]* (?:failed|errors?)\b|Traceback \(most recent call last\)|^ERROR ", text, re.MULTILINE)):
        raise ValueError("Validation log must confirm passed pytest checks without failures")
    if any(common.repo_path(name).stat().st_mtime_ns > path.stat().st_mtime_ns for name in sources):
        raise ValueError("Validation log predates a current source file; rerun checks")
    return path


def freeze_protocol(config_path, protocol_lock, output, validation_log):
    """Create one new frozen directory and lock, refusing every overwrite.

    Only ROOT invokes this for the approved production run after all tests.
    Tests use an isolated repository and inject a fake builder into this module.
    """
    config_name = common.relative_path(config_path)
    if config_name != common.CONFIG_REL:
        raise ValueError("Freeze requires the formal approved V2 config")
    config_path = common.repo_path(config_name)
    lock_path = common.repo_path(common.relative_path(protocol_lock), must_exist=False)
    output = common.repo_path(common.relative_path(output), must_exist=False)
    if output.exists() or lock_path.exists():
        raise FileExistsError("Refusing to overwrite an existing freeze directory or protocol lock")
    if lock_path.is_relative_to(output) or output.is_relative_to(lock_path):
        raise ValueError("Protocol lock and freeze directory must be separate paths")
    config = common.load(config_path)
    common.validate_config(config)
    parent, _ = common.prior_context(config)
    sources = common.source_inventory()
    log_path = validate_log(validation_log, sources)
    if common.repo_path(config["generation_output_directory"], must_exist=False).exists():
        raise FileExistsError("V2 generation output already exists; cannot authorize a second run")
    hashes = {
        "source_sha256": {name: common.file_hash(common.repo_path(name)) for name in sources},
        "data_sha256": {name: common.file_hash(common.repo_path(name)) for name in config["data_files"]},
        "input_sha256": {name: common.file_hash(common.repo_path(name)) for name in common.input_inventory(config)},
    }
    config_hash = common.file_hash(config_path)
    materials = {f"pair_{p:02d}": build_pair(p, config["paired_design"]["alias_seed_by_pair"][f"pair_{p:02d}"])
                 for p in range(10)}
    # Every new labeled prompt must equal the original V1 prompt. Anonymous
    # prompts are generated only by fresh reversible code assignments.
    previous_dir = common.repo_path(parent["freeze_directory"], must_exist=False)
    for pair, material in materials.items():
        if material.get("pair_index") != int(pair[-2:]) or material.get("alias_seed") != config["paired_design"]["alias_seed_by_pair"][pair]:
            raise ValueError("Builder returned a wrong pair or seed")
        prior_name = pair + "_labeled"
        previous_path = common.repo_path(common.relative_path(previous_dir / "messages" / (prior_name + ".json")))
        if common.file_hash(previous_path) != parent["messages_sha256"][prior_name]:
            raise ValueError("Original V1 prompt changed")
        if material["arms"]["labeled"]["messages"] != common.load(previous_path):
            raise ValueError("V2 labeled prompt must exactly preserve V1")
        if not material.get("private_mapping", {}).get("never_send_to_model") is True:
            raise ValueError("Private mapping must not be sent to the model")
    output.mkdir(parents=True, exist_ok=False)
    (output / "messages").mkdir()
    (output / "private_pairs").mkdir()
    for group, folder in (("source_sha256", "source_snapshot"),
                          ("data_sha256", "input_snapshot"), ("input_sha256", "input_snapshot")):
        for relative, expected_hash in hashes[group].items():
            destination = output / folder / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(common.repo_path(relative).read_bytes())
                handle.flush()
                common.os.fsync(handle.fileno())
            if common.file_hash(destination) != expected_hash:
                raise ValueError("Frozen input changed while snapshotting: " + relative)
    messages_hashes, pairs_hashes = {}, {}
    for pair, material in materials.items():
        path = output / "private_pairs" / (pair + ".json")
        common.save(path, material, exclusive=True)
        pairs_hashes[pair] = common.file_hash(path)
        for arm in common.ARMS:
            name = pair + "_" + arm
            path = output / "messages" / (name + ".json")
            common.save(path, material["arms"][arm]["messages"], exclusive=True)
            messages_hashes[name] = common.file_hash(path)
    copied_log = output / "offline_validation.log"
    with copied_log.open("xb") as handle:
        handle.write(log_path.read_bytes())
        handle.flush()
        common.os.fsync(handle.fileno())
    with (output / "config.json").open("xb") as handle:
        handle.write(config_path.read_bytes())
        handle.flush()
        common.os.fsync(handle.fileno())
    # Catch edits that occurred while material snapshots were being constructed.
    if sources != common.source_inventory() or common.file_hash(config_path) != config_hash:
        raise ValueError("Source/config changed during freeze")
    for mapping in hashes.values():
        for name, digest in mapping.items():
            if common.file_hash(common.repo_path(name)) != digest:
                raise ValueError("Input changed during freeze: " + name)
    if common.file_hash(copied_log) != common.file_hash(log_path):
        raise ValueError("Validation log changed during freeze")
    versions = {}
    for name in ("numpy", "scipy", "pandas", "pytest", "rdkit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    lock = {
        "protocol_id": config["protocol_id"], "status": "frozen_before_generation",
        "model_api_calls_at_freeze": 0, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash, "freeze_directory": common.relative_path(output), **hashes,
        "messages_sha256": messages_hashes, "pair_material_sha256": pairs_hashes,
        "offline_validation_log_sha256": common.file_hash(copied_log),
        "authorization_reference": config["resource_authorization"]["authorization_reference"],
        "parent_v1_protocol_lock": config["parent_v1_protocol_lock"],
        "parent_v1_protocol_lock_sha256": common.file_hash(common.repo_path(config["parent_v1_protocol_lock"])),
        "prior_spend_ledger": config["prior_spend_ledger"],
        "prior_spend_ledger_sha256": common.file_hash(common.repo_path(config["prior_spend_ledger"])),
        "prior_known_cost_usd": str(common.PRIOR_COST), "prior_model_api_calls": 11,
        "new_authorized_model_calls": 20, "cumulative_max_cost_usd": "1",
        "evidence_boundary": config["analysis"]["scope"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    common.save(lock_path, lock, exclusive=True)
    with (output / "protocol_lock.json").open("xb") as handle:
        handle.write(lock_path.read_bytes())
        handle.flush()
        common.os.fsync(handle.fileno())
    common.verify_freeze(config_path, lock_path, output)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-log", type=Path, required=True)
    args = parser.parse_args()
    lock = freeze_protocol(args.config, args.protocol_lock, args.output, args.validation_log)
    print(common.json.dumps({"status": lock["status"], "model_api_calls": 0,
                             "message_count": len(lock["messages_sha256"]),
                             "pair_count": len(lock["pair_material_sha256"])}))


if __name__ == "__main__":
    main()
