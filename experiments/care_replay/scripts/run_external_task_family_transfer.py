#!/usr/bin/env python3
"""Run frozen strong baselines on a previously unused task family."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import run_classical_transfer_baselines as classical
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
MODES = multisource.POLICY_MODES
IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    Path(replay.__file__).resolve(),
    Path(transfer.__file__).resolve(),
    Path(classical.__file__).resolve(),
    Path(multisource.__file__).resolve(),
)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_protocol_lock(
    config: Mapping[str, Any],
    output_dir: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    protocol = config["protocol"]
    implementation_hashes = {
        str(path.relative_to(ROOT.parent)): sha256_path(path)
        for path in IMPLEMENTATION_FILES
    }
    payload = {
        "schema_version": "care.external_task_family_protocol_lock/v1",
        "status": "locked_before_outcome_replay",
        "protocol_version": protocol["version"],
        "frozen_on": protocol["frozen_on"],
        "canonical_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "dataset": {
            **protocol["dataset"],
            "local_filename": dataset_path.name,
            "raw_file_sha256": sha256_path(dataset_path),
        },
        "implementation_files_sha256": implementation_hashes,
        "constraints": protocol["constraints"],
    }
    lock_path = output_dir / "protocol_lock.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Existing protocol lock differs from the current config, data, "
                "or implementation. Use a new output directory."
            )
    else:
        write_json(lock_path, payload)
        write_sha256(lock_path)
    return payload


def validate_protocol(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("External task-family protocol must be frozen before execution.")
    if protocol.get("target_task_calibration", True):
        raise ValueError("External task-family evaluation forbids target calibration.")
    source_ids = [str(item) for item in protocol["source_task_ids"]]
    target_id = str(protocol["target_task_id"])
    if target_id in source_ids:
        raise ValueError("The target task cannot also be a source task.")
    unknown = [
        task_id
        for task_id in [*source_ids, target_id]
        if task_id not in replay.DATASET_BUILDERS
    ]
    if unknown:
        raise ValueError(f"Unknown task IDs: {unknown}")
    if len(source_ids) != len(protocol["source_observations"]):
        raise ValueError("source_observations must contain one count per source task.")


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_protocol(config)
    protocol = config["protocol"]
    kernel = protocol["kernel"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = replay.ensure_public_data_file("flip2_hydro_to_P06241.csv.gz")
    protocol_lock = write_protocol_lock(config, output_dir, dataset_path)
    source_ids = [str(item) for item in protocol["source_task_ids"]]
    sources = [replay.DATASET_BUILDERS[task_id]() for task_id in source_ids]
    target = replay.DATASET_BUILDERS[str(protocol["target_task_id"])]()
    for source in sources:
        classical.validate_compatible_spaces(source, target)

    source_posteriors = []
    for source, observation_count in zip(
        sources, protocol["source_observations"]
    ):
        observed = transfer.source_observations(
            source,
            int(protocol["source_seed"]),
            int(observation_count),
        )
        source_posteriors.append(
            classical.build_source_posterior(
                source,
                target,
                observed,
                int(protocol["source_inducing_limit"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )

    rows: list[dict[str, Any]] = []
    audits: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    seed_start = int(protocol["heldout_seed_start"])
    seed_count = int(protocol["heldout_seed_count"])
    for seed in range(seed_start, seed_start + seed_count):
        for mode in MODES:
            metrics, audit = multisource.run_seed(
                sources,
                target,
                source_posteriors,
                seed,
                int(protocol["initial_observations"]),
                int(protocol["reveal_rounds"]),
                mode,
                float(kernel["gp_beta"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
                int(protocol["rgpe_draws"]),
                tuple(float(value) for value in protocol["multitask_rho_grid"]),
                float(protocol["bma_temperature"]),
                float(protocol["skill_prior_mass_start"]),
                float(protocol["skill_prior_mass_end"]),
            )
            rows.append(metrics)
            audits[mode].extend(audit)

    metrics_path = output_dir / "heldout_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit_paths = []
    for mode, events in audits.items():
        path = output_dir / f"{mode}_audit.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        audit_paths.append(path)

    summary = {
        "schema_version": "care.external_task_family_transfer/v1",
        "protocol_version": protocol["version"],
        "protocol_lock_sha256": sha256_path(output_dir / "protocol_lock.json"),
        "raw_dataset_sha256": protocol_lock["dataset"]["raw_file_sha256"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "evidence_class": protocol["evidence_class"],
        "source_tasks": source_ids,
        "target_task": target.dataset_id,
        "target_task_calibration": False,
        "heldout_seed_count": seed_count,
        "aggregate": multisource.aggregate(rows),
        "comparisons_vs_target_gp_ucb": {
            mode: classical.paired_comparison(rows, mode, "target_gp_ucb")
            for mode in MODES
            if mode != "target_gp_ucb"
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    for path in [metrics_path, summary_path, *audit_paths]:
        write_sha256(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
