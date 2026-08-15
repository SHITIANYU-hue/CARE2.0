#!/usr/bin/env python3
"""Task-disjoint multi-source initial-design transfer for real wet-lab tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import run_classical_transfer_baselines as classical
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
RANDOM_MODE = "target_gp_random_initial"
SPACE_FILLING_MODE = "target_gp_space_filling"
SOURCE_MODE = "multisource_diverse_warmstart"
DEPLOYED_MODE = "development_selected_route"


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def validate_protocol(config: Mapping[str, Any]) -> None:
    protocol = config["protocol"]
    development = set(protocol["development_task_ids"])
    evaluation = set(protocol["evaluation_task_ids"])
    overlap = development & evaluation
    if overlap:
        raise ValueError(f"Development and evaluation tasks overlap: {sorted(overlap)}")
    if protocol.get("target_task_calibration", True):
        raise ValueError("Warm-start confirmation forbids target-task calibration.")
    declared_tasks = development | evaluation
    for case in protocol.get("evaluation_cases", []):
        declared_tasks.add(str(case["target_task_id"]))
        declared_tasks.update(str(item) for item in case["source_task_ids"])
    missing = [
        task_id
        for task_id in declared_tasks
        if task_id not in replay.DATASET_BUILDERS
    ]
    if missing:
        raise ValueError(f"Unknown benchmark tasks: {sorted(missing)}")
    if int(protocol["initial_observations"]) <= 0:
        raise ValueError("initial_observations must be positive.")
    if int(protocol["reveal_rounds"]) <= 0:
        raise ValueError("reveal_rounds must be positive.")


def task_descriptor(adapter: replay.DatasetAdapter) -> dict[str, str]:
    metadata = adapter.candidates[0].metadata
    if adapter.dataset_id.startswith("real_moleculenet_"):
        domain = "molecular_property"
    elif adapter.dataset_id.startswith("real_matbench_"):
        domain = "materials_property"
    elif "suzuki" in adapter.dataset_id or "buchwald" in adapter.dataset_id:
        domain = "reaction_optimization"
    elif "chemlex" in adapter.dataset_id:
        domain = "reaction_optimization"
    else:
        domain = "scientific_optimization"
    representation = "decision_schema:" + "|".join(adapter.decision_columns)
    return {
        "domain": domain,
        "representation": representation,
        # Keep the legacy keys for frozen reaction-task routing. Generic tasks
        # fall back to domain and representation compatibility.
        "substrate": str(metadata.get("substrate", domain)),
        "precatalyst": str(metadata.get("precatalyst", representation)),
    }


def source_ids_for_scope(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    scope: str,
) -> list[str]:
    target_descriptor = task_descriptor(target)
    descriptors = {
        source_id: task_descriptor(replay.DATASET_BUILDERS[source_id]())
        for source_id in source_ids
    }
    same_substrate = [
        source_id
        for source_id in source_ids
        if descriptors[source_id]["substrate"] == target_descriptor["substrate"]
    ]
    same_precatalyst = [
        source_id
        for source_id in source_ids
        if descriptors[source_id]["precatalyst"]
        == target_descriptor["precatalyst"]
    ]
    if scope == "all":
        return list(source_ids)
    if scope == "same_substrate_then_precatalyst":
        return same_substrate or same_precatalyst or list(source_ids)
    if scope == "same_precatalyst_then_substrate":
        return same_precatalyst or same_substrate or list(source_ids)
    raise ValueError(f"Unknown source scope: {scope}")


def build_source_consensus(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    kernel = protocol["kernel"]
    posteriors = []
    for source_id in source_ids:
        source = replay.DATASET_BUILDERS[source_id]()
        classical.validate_compatible_spaces(source, target)
        posteriors.append(
            classical.build_source_posterior(
                source,
                target,
                list(source.candidates),
                int(protocol["source_inducing_limit"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
    return multisource.consensus_prior(posteriors), list(source_ids)


def space_filling_initial(
    adapter: replay.DatasetAdapter,
    count: int,
) -> list[int]:
    features = classical.feature_arrays(adapter, adapter.candidates)
    center = np.full(features.numeric.shape[1], 0.5, dtype=np.float64)
    first = int(np.argmin(np.sum((features.numeric - center) ** 2, axis=1)))
    selected = [first]
    while len(selected) < count:
        min_distance = np.full(len(adapter.candidates), np.inf, dtype=np.float64)
        for index in selected:
            min_distance = np.minimum(
                min_distance,
                classical.mixed_distance_to_point(features, index),
            )
        min_distance[np.asarray(selected, dtype=int)] = -1.0
        selected.append(int(np.argmax(min_distance)))
    return selected


def source_diverse_initial(
    adapter: replay.DatasetAdapter,
    source_prior: np.ndarray,
    count: int,
    diversity_weight: float,
    source_quantile: float = 0.0,
) -> list[int]:
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must lie in [0, 1].")
    if not 0.0 <= source_quantile < 1.0:
        raise ValueError("source_quantile must lie in [0, 1).")
    features = classical.feature_arrays(adapter, adapter.candidates)
    selected = [int(np.argmax(source_prior))]
    eligible = source_prior >= np.quantile(source_prior, source_quantile)
    while len(selected) < count:
        min_distance = np.full(len(adapter.candidates), np.inf, dtype=np.float64)
        for index in selected:
            min_distance = np.minimum(
                min_distance,
                classical.mixed_distance_to_point(features, index),
            )
        finite_max = max(float(np.max(min_distance[np.isfinite(min_distance)])), 1e-9)
        normalized_distance = min_distance / finite_max
        scores = (
            (1.0 - diversity_weight) * source_prior
            + diversity_weight * normalized_distance
        )
        scores[~eligible] = -np.inf
        scores[np.asarray(selected, dtype=int)] = -np.inf
        if not np.isfinite(scores).any():
            raise ValueError(
                "source_quantile leaves too few candidates for the initial design."
            )
        selected.append(int(np.argmax(scores)))
    return selected


def random_initial(
    adapter: replay.DatasetAdapter,
    count: int,
    seed: int,
) -> list[int]:
    indices = list(range(len(adapter.candidates)))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def public_conditions(candidate: replay.Candidate) -> dict[str, Any]:
    return {
        "base": candidate.metadata.get("base"),
        "precatalyst": candidate.metadata.get("precatalyst"),
        "base_equivalents": candidate.metadata.get("base_equivalents"),
        "temperature_celsius": candidate.metadata.get("temperature_celsius"),
        "residence_time_minutes": candidate.metadata.get("residence_time_minutes"),
        "residence_time_seconds": candidate.metadata.get("residence_time_seconds"),
        "precatalyst_loading_mol_percent": candidate.metadata.get(
            "precatalyst_loading_mol_percent"
        ),
        "precatalyst_fraction": candidate.metadata.get("precatalyst_fraction"),
    }


def run_target_gp(
    adapter: replay.DatasetAdapter,
    initial_indices: Sequence[int],
    rounds: int,
    mode: str,
    seed: int,
    kernel: Mapping[str, Any],
    source_ids: Sequence[str] = (),
    policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = adapter.candidates
    features = classical.feature_arrays(adapter, pool)
    observed = [int(index) for index in initial_indices]
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        raise ValueError("Initial-design candidate indices must be unique.")
    audit = [
        {
            "event": "initial_design_revealed",
            "target_dataset": adapter.dataset_id,
            "mode": mode,
            "seed": seed,
            "source_datasets": list(source_ids),
            "policy": dict(policy or {}),
            "candidate_ids": [pool[index].candidate_id for index in observed],
            "public_conditions": [public_conditions(pool[index]) for index in observed],
            "revealed_values": [pool[index].objective_value for index in observed],
        }
    ]
    best_trace: list[float] = []
    for round_index in range(rounds):
        observed_y, _center, _scale = classical.normalized_outcomes(
            [pool[index] for index in observed]
        )
        posterior_mean, posterior_variance, _diagnostics = (
            classical.target_gp_posterior(
                features,
                observed,
                observed_y,
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
        scores = posterior_mean + float(kernel["gp_beta"]) * np.sqrt(
            posterior_variance
        )
        selected_index = classical.top_unobserved(scores, observed_set)
        selected = pool[selected_index]
        observed.append(selected_index)
        observed_set.add(selected_index)
        best_so_far = max(pool[index].objective_value for index in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "event": "target_gp_reveal",
                "target_dataset": adapter.dataset_id,
                "mode": mode,
                "seed": seed,
                "round_index": round_index,
                "selected_candidate": selected.candidate_id,
                "public_conditions": public_conditions(selected),
                "selected_score": round(float(scores[selected_index]), 6),
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
            }
        )
    oracle = max(candidate.objective_value for candidate in pool)
    final_best = max(pool[index].objective_value for index in observed)
    top10 = {
        index
        for index, _candidate in sorted(
            enumerate(pool),
            key=lambda item: item[1].objective_value,
            reverse=True,
        )[: min(10, len(pool))]
    }
    return {
        "target_dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "initial_observations": len(initial_indices),
        "reveal_rounds": rounds,
        "initial_candidate_ids": ";".join(
            pool[index].candidate_id for index in initial_indices
        ),
        "source_datasets": ";".join(source_ids),
        "final_best": round(final_best, 6),
        "best_so_far_auc": round(float(np.mean(best_trace)), 6),
        "simple_regret": round(oracle - final_best, 6),
        "top10_hit": int(bool(set(observed) & top10)),
    }, audit


def policy_id(policy: Mapping[str, Any]) -> str:
    scope = str(policy["source_scope"])
    weight = float(policy["diversity_weight"])
    quantile = float(policy.get("source_quantile", 0.0))
    suffix = f"_q{quantile:.2f}" if quantile > 0.0 else ""
    return f"warmstart_{scope}_{weight:.2f}{suffix}"


def mean_for(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    mode: str,
    field: str,
) -> float:
    values = [
        float(row[field])
        for row in rows
        if row["target_dataset"] == target and row["mode"] == mode
    ]
    if not values:
        raise ValueError(f"No {mode} rows for {target}")
    return float(np.mean(values))


def normal_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean_value = float(np.mean(array))
    if len(array) <= 1:
        return mean_value, mean_value, mean_value
    half_width = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return mean_value, mean_value - half_width, mean_value + half_width


def task_level_comparison(
    rows: Sequence[Mapping[str, Any]],
    challenger: str,
    task_ids: Sequence[str],
    references: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "task_count": len(task_ids),
        "references": list(references),
        "reference_selection": "strongest_mean_per_task",
    }
    for field in ("final_best", "best_so_far_auc", "top10_hit"):
        deltas = []
        reference_modes = []
        per_task = {}
        for task_id in task_ids:
            challenger_value = mean_for(rows, task_id, challenger, field)
            reference_values = {
                reference: mean_for(rows, task_id, reference, field)
                for reference in references
            }
            reference_mode = max(reference_values, key=reference_values.get)
            delta = challenger_value - reference_values[reference_mode]
            deltas.append(delta)
            reference_modes.append(reference_mode)
            per_task[task_id] = {
                "challenger": round(challenger_value, 6),
                "reference_mode": reference_mode,
                "reference": round(reference_values[reference_mode], 6),
                "delta": round(delta, 6),
            }
        mean_delta, ci_low, ci_high = normal_interval(deltas)
        output[field] = {
            "task_mean_delta": round(mean_delta, 6),
            "task_95ci_low": round(ci_low, 6),
            "task_95ci_high": round(ci_high, 6),
            "task_win_rate": round(float(np.mean(np.asarray(deltas) > 0.0)), 6),
            "task_nonloss_rate": round(
                float(np.mean(np.asarray(deltas) >= 0.0)), 6
            ),
            "per_task": per_task,
        }
    return output


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metrics table.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_audits(path: Path, events: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")


def calibrate(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_protocol(config)
    protocol = config["protocol"]
    development = list(protocol["development_task_ids"])
    policies = [dict(item) for item in protocol["warmstart_candidates"]]
    initial_count = int(protocol["initial_observations"])
    rounds = int(protocol["reveal_rounds"])
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for target_id in development:
        target = replay.DATASET_BUILDERS[target_id]()
        source_pool = [task_id for task_id in development if task_id != target_id]
        for seed in range(
            int(protocol["development_seed_start"]),
            int(protocol["development_seed_start"])
            + int(protocol["development_seed_count"]),
        ):
            metrics, audit = run_target_gp(
                target,
                random_initial(target, initial_count, seed),
                rounds,
                RANDOM_MODE,
                seed,
                protocol["kernel"],
            )
            rows.append(metrics)
            events.extend(audit)
        metrics, audit = run_target_gp(
            target,
            space_filling_initial(target, initial_count),
            rounds,
            SPACE_FILLING_MODE,
            -1,
            protocol["kernel"],
        )
        rows.append(metrics)
        events.extend(audit)
        priors: dict[str, tuple[np.ndarray, list[str]]] = {}
        for policy in policies:
            scope = str(policy["source_scope"])
            if scope not in priors:
                selected_sources = source_ids_for_scope(target, source_pool, scope)
                priors[scope] = build_source_consensus(
                    target, selected_sources, protocol
                )
            source_prior, selected_sources = priors[scope]
            mode = policy_id(policy)
            metrics, audit = run_target_gp(
                target,
                source_diverse_initial(
                    target,
                    source_prior,
                    initial_count,
                    float(policy["diversity_weight"]),
                    float(policy.get("source_quantile", 0.0)),
                ),
                rounds,
                mode,
                -1,
                protocol["kernel"],
                selected_sources,
                policy,
            )
            rows.append(metrics)
            events.extend(audit)
    comparisons = {
        policy_id(policy): task_level_comparison(
            rows,
            policy_id(policy),
            development,
            (RANDOM_MODE, SPACE_FILLING_MODE),
        )
        for policy in policies
    }
    metric = str(protocol["selection_metric"])
    required_ci_low = float(protocol["selection_required_ci_low"])
    required_nonloss = float(protocol["selection_required_task_nonloss_rate"])
    eligible = [
        mode
        for mode, comparison in comparisons.items()
        if comparison[metric]["task_95ci_low"] > required_ci_low
        and comparison[metric]["task_nonloss_rate"] >= required_nonloss
    ]
    best_source_mode = max(
        comparisons,
        key=lambda mode: (
            comparisons[mode][metric]["task_95ci_low"],
            comparisons[mode][metric]["task_mean_delta"],
        ),
    )
    best_source_policy = next(
        policy for policy in policies if policy_id(policy) == best_source_mode
    )
    baseline_means = {
        mode: float(
            np.mean(
                [mean_for(rows, task_id, mode, metric) for task_id in development]
            )
        )
        for mode in (RANDOM_MODE, SPACE_FILLING_MODE)
    }
    if eligible:
        selection_priority = str(
            protocol.get("eligible_selection_priority", "confidence_then_mean")
        )
        if selection_priority == "mean_then_confidence":
            selection_key = lambda mode: (
                comparisons[mode][metric]["task_mean_delta"],
                comparisons[mode][metric]["task_95ci_low"],
            )
        elif selection_priority == "confidence_then_mean":
            selection_key = lambda mode: (
                comparisons[mode][metric]["task_95ci_low"],
                comparisons[mode][metric]["task_mean_delta"],
            )
        else:
            raise ValueError(
                f"Unknown eligible_selection_priority: {selection_priority}"
            )
        selected_mode = max(
            eligible,
            key=selection_key,
        )
        selected_policy = next(
            policy for policy in policies if policy_id(policy) == selected_mode
        )
        selected_route = {
            "mode": SOURCE_MODE,
            "policy_id": selected_mode,
            **selected_policy,
        }
    else:
        selected_route = {"mode": max(baseline_means, key=baseline_means.get)}
        selected_mode = selected_route["mode"]
    record = {
        "schema_version": "care.multisource_warmstart_selection/v1",
        "selection_scope": "development_tasks_only",
        "config_fingerprint": config_fingerprint(config),
        "development_task_ids": development,
        "evaluation_task_ids_not_loaded": list(protocol["evaluation_task_ids"]),
        "candidate_comparisons": comparisons,
        "baseline_development_means": baseline_means,
        "selection_metric": metric,
        "selection_required_ci_low": required_ci_low,
        "selection_required_task_nonloss_rate": required_nonloss,
        "eligible_selection_priority": protocol.get(
            "eligible_selection_priority", "confidence_then_mean"
        ),
        "best_source_mode": best_source_mode,
        "best_source_policy": best_source_policy,
        "selected_mode": selected_mode,
        "selected_route": selected_route,
        "target_task_calibration": False,
    }
    write_rows(output_dir / "development_metrics.csv", rows)
    write_audits(output_dir / "development_audits.jsonl", events)
    (output_dir / "selection_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def clone_as_deployed(
    metrics: Mapping[str, Any],
    audit: Sequence[Mapping[str, Any]],
    selected_route: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cloned_metrics = dict(metrics)
    cloned_metrics["mode"] = DEPLOYED_MODE
    cloned_audit = []
    for event in audit:
        cloned = dict(event)
        cloned["policy_mode"] = event["mode"]
        cloned["mode"] = DEPLOYED_MODE
        cloned["development_selected_route"] = dict(selected_route)
        cloned_audit.append(cloned)
    return cloned_metrics, cloned_audit


def aggregate_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    task_ids = sorted({str(row["target_dataset"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows})
    for task_id in task_ids:
        output[task_id] = {}
        for mode in modes:
            selected = [
                row
                for row in rows
                if row["target_dataset"] == task_id and row["mode"] == mode
            ]
            if not selected:
                continue
            output[task_id][mode] = {
                field: round(float(np.mean([float(row[field]) for row in selected])), 6)
                for field in (
                    "final_best",
                    "best_so_far_auc",
                    "simple_regret",
                    "top10_hit",
                )
            }
    return output


def confirm(
    config: dict[str, Any],
    selection: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol(config)
    if selection.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Selection record does not match the frozen config.")
    protocol = config["protocol"]
    evaluation = list(protocol["evaluation_task_ids"])
    development = list(protocol["development_task_ids"])
    evaluation_sources = {
        str(case["target_task_id"]): [str(item) for item in case["source_task_ids"]]
        for case in protocol.get("evaluation_cases", [])
    }
    selected_route = dict(selection["selected_route"])
    source_policy = (
        selected_route
        if selected_route["mode"] == SOURCE_MODE
        else dict(selection["best_source_policy"])
    )
    initial_count = int(protocol["initial_observations"])
    rounds = int(protocol["reveal_rounds"])
    rows: list[dict[str, Any]] = []
    audit_groups: dict[str, list[dict[str, Any]]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for target_id in evaluation:
        target = replay.DATASET_BUILDERS[target_id]()
        random_runs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for seed in range(
            int(protocol["evaluation_seed_start"]),
            int(protocol["evaluation_seed_start"])
            + int(protocol["evaluation_seed_count"]),
        ):
            result = run_target_gp(
                target,
                random_initial(target, initial_count, seed),
                rounds,
                RANDOM_MODE,
                seed,
                protocol["kernel"],
            )
            random_runs.append(result)
            rows.append(result[0])
            audit_groups.setdefault(f"{target_id}_{RANDOM_MODE}", []).extend(result[1])
        space_result = run_target_gp(
            target,
            space_filling_initial(target, initial_count),
            rounds,
            SPACE_FILLING_MODE,
            -1,
            protocol["kernel"],
        )
        rows.append(space_result[0])
        audit_groups[f"{target_id}_{SPACE_FILLING_MODE}"] = space_result[1]
        source_pool = evaluation_sources.get(target_id, development)
        source_ids = source_ids_for_scope(
            target, source_pool, str(source_policy["source_scope"])
        )
        source_prior, source_ids = build_source_consensus(
            target, source_ids, protocol
        )
        source_result = run_target_gp(
            target,
            source_diverse_initial(
                target,
                source_prior,
                initial_count,
                float(source_policy["diversity_weight"]),
                float(source_policy.get("source_quantile", 0.0)),
            ),
            rounds,
            SOURCE_MODE,
            -1,
            protocol["kernel"],
            source_ids,
            source_policy,
        )
        rows.append(source_result[0])
        audit_groups[f"{target_id}_{SOURCE_MODE}"] = source_result[1]
        if selected_route["mode"] == SOURCE_MODE:
            deployed_source = source_result
            deployed = clone_as_deployed(
                deployed_source[0], deployed_source[1], selected_route
            )
            rows.append(deployed[0])
            audit_groups[f"{target_id}_{DEPLOYED_MODE}"] = deployed[1]
        elif selected_route["mode"] == SPACE_FILLING_MODE:
            deployed = clone_as_deployed(
                space_result[0], space_result[1], selected_route
            )
            rows.append(deployed[0])
            audit_groups[f"{target_id}_{DEPLOYED_MODE}"] = deployed[1]
        else:
            for metrics, audit in random_runs:
                deployed = clone_as_deployed(metrics, audit, selected_route)
                rows.append(deployed[0])
                audit_groups.setdefault(
                    f"{target_id}_{DEPLOYED_MODE}", []
                ).extend(deployed[1])
    for name, events in audit_groups.items():
        write_audits(audit_dir / f"{name}.jsonl", events)
    write_rows(output_dir / "confirmation_metrics.csv", rows)
    summary = {
        "experiment": "care2_baumgartner_multisource_warmstart_confirmation",
        "protocol_version": protocol["version"],
        "target_task_calibration": False,
        "development_selection": dict(selection),
        "evaluated_source_policy": dict(source_policy),
        "aggregate_by_task": aggregate_by_task(rows),
        "comparisons": {
            SOURCE_MODE: task_level_comparison(
                rows,
                SOURCE_MODE,
                evaluation,
                (RANDOM_MODE, SPACE_FILLING_MODE),
            ),
            DEPLOYED_MODE: task_level_comparison(
                rows,
                DEPLOYED_MODE,
                evaluation,
                (RANDOM_MODE, SPACE_FILLING_MODE),
            ),
        },
    }
    (output_dir / "confirmation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "confirm"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--config",
            type=Path,
            default=ROOT / "configs" / "baumgartner_multisource_warmstart_v1.json",
        )
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "confirm":
            child.add_argument("--selection-record", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "calibrate":
        result = calibrate(config, args.output_dir)
    else:
        selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
        result = confirm(config, selection, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
