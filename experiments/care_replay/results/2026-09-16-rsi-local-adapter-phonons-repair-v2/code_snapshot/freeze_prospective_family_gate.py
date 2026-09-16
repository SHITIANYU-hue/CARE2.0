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


def verify_public_task_spec(
    hypothesis: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[Path, str]:
    public_spec_path = ROOT / str(hypothesis["public_task_spec_path"])
    if not public_spec_path.exists():
        raise ValueError("The declared public task specification is missing.")
    public_spec_sha256 = sha256_path(public_spec_path)
    if public_spec_sha256 != hypothesis.get("public_task_spec_sha256"):
        raise ValueError("The public task specification hash does not match the config.")
    public_spec = json.loads(public_spec_path.read_text(encoding="utf-8"))
    if record.get("public_task_spec") != public_spec:
        raise ValueError("The LLM record embeds a different public task specification.")
    return public_spec_path, public_spec_sha256


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
    hypothesis_lock: dict[str, Any] | None = None
    hypothesis = protocol.get("semantic_hypothesis")
    if hypothesis:
        record_path = ROOT / str(hypothesis["record_path"])
        if not record_path.exists():
            raise ValueError("The declared outcome-blind LLM hypothesis is missing.")
        record_sha256 = sha256_path(record_path)
        if record_sha256 != hypothesis.get("record_sha256"):
            raise ValueError("The LLM hypothesis record hash does not match the config.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "generated_before_dataset_download":
            raise ValueError("The LLM hypothesis record is not outcome-blind.")
        if record.get("target_outcomes_available") is not False:
            raise ValueError("The LLM hypothesis record does not exclude target outcomes.")
        recommended = record.get("parsed_hypothesis", {}).get("recommended_skill")
        if recommended != hypothesis.get("recommended_skill"):
            raise ValueError("The parsed LLM recommendation differs from the config.")
        if recommended == "abstain":
            if protocol["gate"]["fallback_policy"] != "target_gp_ucb":
                raise ValueError(
                    "An LLM abstention requires the frozen target-only fallback."
                )
        elif recommended not in protocol["gate"]["candidate_methods"]:
            raise ValueError("The LLM-recommended skill is absent from the gate menu.")
        public_spec_path, public_spec_sha256 = verify_public_task_spec(
            hypothesis, record
        )
        hypothesis_lock = {
            "record_path": str(record_path.relative_to(ROOT)),
            "record_sha256": record_sha256,
            "public_task_spec_path": str(public_spec_path.relative_to(ROOT)),
            "public_task_spec_sha256": public_spec_sha256,
            "model": record.get("model"),
            "recommended_skill": recommended,
            "decision_authority": hypothesis.get("decision_authority"),
            "target_outcomes_available": False,
        }
    implementation_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("generate_outcome_blind_transfer_hypothesis.py"),
        Path(__file__).with_name("run_ired_source_outcome_falsification.py"),
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
        "semantic_hypothesis": hypothesis_lock,
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
