#!/usr/bin/env python3
"""Write an outcome-blind preregistration lock before dataset download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import run_source_only_family_gate as gate
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(config: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    gate.validate_protocol(config)
    protocol = config["protocol"]
    if protocol.get("analysis_status") != "prospective_preregistered_before_test_outcomes":
        raise ValueError("Only a prospective protocol can be preregistered here.")
    filename = str(protocol["dataset"]["local_filename"])
    raw_path = replay.RAW_DATA / filename
    if raw_path.exists():
        raise ValueError(
            "Refusing prospective freeze because the deployment dataset is already present."
        )
    declared_lock = ROOT / str(protocol["preregistration"]["lock_path"])
    if output_path.resolve() != declared_lock.resolve():
        raise ValueError("Output path does not match the protocol declaration.")
    implementation_files = (
        Path(__file__).resolve(),
        *gate.IMPLEMENTATION_FILES,
    )
    return {
        "schema_version": "care.prospective_family_gate_preregistration/v1",
        "status": "preregistered_before_dataset_download",
        "protocol_version": protocol["version"],
        "frozen_on": protocol["frozen_on"],
        "canonical_config_sha256": gate.canonical_sha256(config),
        "dataset": protocol["dataset"],
        "deployment_target_task_id": protocol["deployment"]["target_task_id"],
        "deployment_target_outcomes_available_at_freeze": False,
        "raw_dataset_present_at_freeze": False,
        "implementation_files_sha256": {
            str(path.relative_to(ROOT.parent)): sha256_path(path)
            for path in dict.fromkeys(implementation_files)
        },
        "gate": protocol["gate"],
        "calibration_routes": protocol["calibration_routes"],
        "deployment": protocol["deployment"],
        "constraints": protocol["constraints"],
        "claim_boundary": protocol["claim_boundary"],
    }


def write_lock(config: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    payload = build_lock(config, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Existing preregistration lock differs from this protocol.")
    else:
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
        sidecar.write_text(
            f"{sha256_path(output_path)}  {output_path.name}\n",
            encoding="utf-8",
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(write_lock(config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
