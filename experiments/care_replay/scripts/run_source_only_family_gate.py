#!/usr/bin/env python3
"""Develop a family-level negative-transfer gate from source tasks only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import run_external_task_family_transfer as external
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    Path(__file__).resolve(),
    Path(external.__file__).resolve(),
    *external.IMPLEMENTATION_FILES[1:],
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def validate_protocol(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("Source-only gate protocol must be frozen before execution.")
    deployment_target = str(protocol["deployment"]["target_task_id"])
    calibration_tasks: set[str] = set()
    route_ids: set[str] = set()
    for route in protocol["calibration_routes"]:
        route_id = str(route["route_id"])
        if route_id in route_ids:
            raise ValueError(f"Duplicate calibration route: {route_id}")
        route_ids.add(route_id)
        pseudo_target = str(route["pseudo_target_task_id"])
        sources = {str(item) for item in route["source_task_ids"]}
        if pseudo_target in sources:
            raise ValueError("A pseudo-target cannot also be its route source.")
        calibration_tasks.add(pseudo_target)
        calibration_tasks.update(sources)
    if deployment_target in calibration_tasks:
        raise ValueError(
            "Deployment target must be absent from source-only calibration routes."
        )
    declared = calibration_tasks | {
        deployment_target,
        *(str(item) for item in protocol["deployment"]["source_task_ids"]),
    }
    unknown = sorted(task for task in declared if task not in replay.DATASET_BUILDERS)
    if unknown:
        raise ValueError(f"Unknown task IDs: {unknown}")
    methods = [str(item) for item in protocol["gate"]["candidate_methods"]]
    unknown_methods = sorted(set(methods) - set(external.MODES))
    if unknown_methods or "target_gp_ucb" in methods:
        raise ValueError(f"Invalid gate candidate methods: {unknown_methods}")
    if protocol["gate"]["fallback_policy"] != "target_gp_ucb":
        raise ValueError("The source-only safety fallback must be target_gp_ucb.")


def verify_preregistration(config: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = config["protocol"].get("preregistration")
    if not spec:
        return None
    lock_path = ROOT / str(spec["lock_path"])
    if not lock_path.exists():
        raise ValueError(f"Required preregistration lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "preregistered_before_dataset_download":
        raise ValueError("Preregistration lock has an invalid status.")
    if lock.get("canonical_config_sha256") != canonical_sha256(config):
        raise ValueError("Preregistration config hash does not match this run.")
    if lock.get("deployment_target_outcomes_available_at_freeze") is not False:
        raise ValueError("Preregistration does not prove an outcome-blind freeze.")
    return lock


def build_inner_config(
    config: Mapping[str, Any],
    route: Mapping[str, Any],
    *,
    version_suffix: str,
    evidence_class: str,
    claim_boundary: str,
) -> dict[str, Any]:
    protocol = config["protocol"]
    source_ids = [str(item) for item in route["source_task_ids"]]
    target_id = str(
        route.get("pseudo_target_task_id", route.get("target_task_id"))
    )
    return {
        "protocol": {
            "version": f"{protocol['version']}_{version_suffix}",
            "status": "frozen_before_execution",
            "frozen_on": protocol["frozen_on"],
            "evidence_class": evidence_class,
            "dataset": {
                **protocol["dataset"],
                "official_split": version_suffix,
            },
            "source_task_ids": source_ids,
            "target_task_id": target_id,
            "target_task_calibration": False,
            "source_seed": int(protocol["source_seed"]),
            "source_observations": [
                int(protocol["source_observation_count_per_task"])
            ]
            * len(source_ids),
            "source_inducing_limit": int(protocol["source_inducing_limit"]),
            "heldout_seed_start": int(route["heldout_seed_start"]),
            "heldout_seed_count": int(route["heldout_seed_count"]),
            "initial_observations": int(protocol["initial_observations"]),
            "reveal_rounds": int(protocol["reveal_rounds"]),
            "rgpe_draws": int(protocol["rgpe_draws"]),
            "multitask_rho_grid": list(protocol["multitask_rho_grid"]),
            "bma_temperature": float(protocol["bma_temperature"]),
            "skill_prior_mass_start": float(protocol["skill_prior_mass_start"]),
            "skill_prior_mass_end": float(protocol["skill_prior_mass_end"]),
            "kernel": dict(protocol["kernel"]),
            "constraints": {
                "new_domain_not_used_in_controller_development": True,
                "official_wild_type_split_preserved": True,
                "target_outcomes_hidden_until_selection": True,
                "no_target_task_parameter_tuning": True,
                "same_initial_observations_within_seed": True,
                "same_target_reveal_budget": True,
                "all_heldout_seeds_retained": True,
            },
            "claim_boundary": claim_boundary,
        }
    }


def select_source_only_policy(
    calibration_summaries: Mapping[str, Mapping[str, Any]],
    candidate_methods: Sequence[str],
    fallback_policy: str = "target_gp_ucb",
) -> dict[str, Any]:
    if not calibration_summaries:
        raise ValueError("At least one source-only calibration route is required.")
    method_diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for method in candidate_methods:
        route_effects = []
        for route_id, summary in calibration_summaries.items():
            effect = summary["comparisons_vs_target_gp_ucb"][method][
                "best_so_far_auc"
            ]
            route_effects.append(
                {
                    "route_id": route_id,
                    "mean_delta": float(effect["mean_delta"]),
                    "ci95_low": float(effect["normal_95ci_low"]),
                    "ci95_high": float(effect["normal_95ci_high"]),
                }
            )
        route_equal_mean = mean(item["mean_delta"] for item in route_effects)
        is_eligible = all(item["ci95_low"] > 0.0 for item in route_effects)
        if is_eligible:
            eligible.append(method)
        method_diagnostics[method] = {
            "route_effects": route_effects,
            "route_equal_mean_auc_delta": round(route_equal_mean, 6),
            "eligible": is_eligible,
        }
    order = {method: index for index, method in enumerate(candidate_methods)}
    ungated = max(
        candidate_methods,
        key=lambda method: (
            method_diagnostics[method]["route_equal_mean_auc_delta"],
            -order[method],
        ),
    )
    deployed = (
        max(
            eligible,
            key=lambda method: (
                method_diagnostics[method]["route_equal_mean_auc_delta"],
                -order[method],
            ),
        )
        if eligible
        else fallback_policy
    )
    return {
        "candidate_methods": list(candidate_methods),
        "method_diagnostics": method_diagnostics,
        "eligible_methods": eligible,
        "ungated_source_only_selection": ungated,
        "deployed_policy": deployed,
        "fallback_triggered": deployed == fallback_policy,
    }


def write_outer_lock(
    config: Mapping[str, Any], output_dir: Path, dataset_path: Path
) -> dict[str, Any]:
    protocol = config["protocol"]
    payload = {
        "schema_version": "care.source_only_family_gate_protocol_lock/v1",
        "status": "locked_before_replay",
        "protocol_version": protocol["version"],
        "analysis_status": protocol["analysis_status"],
        "canonical_config_sha256": canonical_sha256(config),
        "raw_dataset_sha256": sha256_path(dataset_path),
        "implementation_files_sha256": {
            str(path.relative_to(ROOT.parent)): sha256_path(path)
            for path in dict.fromkeys(IMPLEMENTATION_FILES)
        },
        "constraints": protocol["constraints"],
    }
    path = output_dir / "protocol_lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Existing source-only gate protocol lock differs.")
    else:
        write_json(path, payload)
        write_sha256(path)
    return payload


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_protocol(config)
    preregistration = verify_preregistration(config)
    protocol = config["protocol"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = str(
        protocol["dataset"].get(
            "local_filename", "flip2_hydro_to_P06241.csv.gz"
        )
    )
    dataset_path = replay.ensure_public_data_file(dataset_filename)
    lock = write_outer_lock(config, output_dir, dataset_path)

    calibration_summaries: dict[str, dict[str, Any]] = {}
    for route in protocol["calibration_routes"]:
        route_id = str(route["route_id"])
        inner = build_inner_config(
            config,
            route,
            version_suffix=f"calibration_{route_id}",
            evidence_class="retrospective_source_only_pseudo_target_calibration",
            claim_boundary=(
                "This route uses completed source-family tasks only and is "
                "development evidence for the family gate."
            ),
        )
        calibration_summaries[route_id] = external.run(
            inner, output_dir / "calibration" / route_id
        )

    gate = select_source_only_policy(
        calibration_summaries,
        [str(item) for item in protocol["gate"]["candidate_methods"]],
        str(protocol["gate"]["fallback_policy"]),
    )
    deployment_spec = protocol["deployment"]
    deployment_config = build_inner_config(
        config,
        deployment_spec,
        version_suffix="deployment_p06241",
        evidence_class="external_family_gate_application",
        claim_boundary=protocol["claim_boundary"],
    )
    deployment_summary = external.run(
        deployment_config, output_dir / "deployment"
    )
    ungated = gate["ungated_source_only_selection"]
    deployed = gate["deployed_policy"]
    deployment_comparisons = deployment_summary["comparisons_vs_target_gp_ucb"]
    ungated_delta = float(
        deployment_comparisons[ungated]["best_so_far_auc"]["mean_delta"]
    )
    deployed_delta = (
        0.0
        if deployed == "target_gp_ucb"
        else float(
            deployment_comparisons[deployed]["best_so_far_auc"]["mean_delta"]
        )
    )

    rows = []
    for method, diagnostics in gate["method_diagnostics"].items():
        for effect in diagnostics["route_effects"]:
            rows.append(
                {
                    "stage": "source_only_calibration",
                    "route_id": effect["route_id"],
                    "method": method,
                    "mean_auc_delta": effect["mean_delta"],
                    "ci95_low": effect["ci95_low"],
                    "ci95_high": effect["ci95_high"],
                    "used_for_gate_selection": 1,
                }
            )
        deployment_effect = deployment_comparisons[method]["best_so_far_auc"]
        rows.append(
            {
                "stage": "external_deployment",
                "route_id": "p01053_p0a9x9_to_p06241",
                "method": method,
                "mean_auc_delta": deployment_effect["mean_delta"],
                "ci95_low": deployment_effect["normal_95ci_low"],
                "ci95_high": deployment_effect["normal_95ci_high"],
                "used_for_gate_selection": 0,
            }
        )
    metrics_path = output_dir / "route_effects.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "care.source_only_family_gate/v1",
        "protocol_version": protocol["version"],
        "analysis_status": protocol["analysis_status"],
        "protocol_lock_sha256": sha256_path(output_dir / "protocol_lock.json"),
        "preregistration_lock_sha256": (
            sha256_path(ROOT / str(protocol["preregistration"]["lock_path"]))
            if preregistration is not None
            else None
        ),
        "raw_dataset_sha256": lock["raw_dataset_sha256"],
        "calibration_route_count": len(calibration_summaries),
        "calibration_target_excludes_deployment_target": True,
        "gate": gate,
        "deployment": {
            "target_task": deployment_spec["target_task_id"],
            "heldout_seed_count": deployment_spec["heldout_seed_count"],
            "method_auc_deltas_vs_target_gp": {
                method: round(
                    float(
                        deployment_comparisons[method]["best_so_far_auc"][
                            "mean_delta"
                        ]
                    ),
                    6,
                )
                for method in gate["candidate_methods"]
            },
            "ungated_source_only_selection": ungated,
            "ungated_auc_delta_vs_target_gp": round(ungated_delta, 6),
            "deployed_policy": deployed,
            "deployed_auc_delta_vs_target_gp": round(deployed_delta, 6),
            "negative_transfer_auc_avoided": round(
                deployed_delta - ungated_delta, 6
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    summary_path = output_dir / "source_only_gate_summary.json"
    write_json(summary_path, summary)
    for path in (metrics_path, summary_path):
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
