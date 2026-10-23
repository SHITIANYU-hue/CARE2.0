#!/usr/bin/env python3
"""Offline, two-stage comparison of frozen LLM, R1 and R2 rule libraries.

Each random replicate has the same templates and selection budget as the LLM
library. Calibration finishes and selections are saved before held-out replay.
The historical single-null runner remains available for old experiments.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time
from typing import Any

import llm_semantic_skills as semantic
import run_calibrated_frozen_llm_selector as statistics
import run_calibrated_llm_semantic_selector as calibrated
import run_random_rule_control as nulls
import run_synthetic_suzuki as replay

ROOT = Path(__file__).resolve().parents[3]
BASELINES = tuple(statistics.TARGET_MODES)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


@contextmanager
def offline_guard():
    """Fail closed if an adapter or future dependency attempts a network call."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def reject(*args, **kwargs):
        raise RuntimeError("Network access is disabled for the offline rule-control study")

    socket.socket.connect = reject
    socket.socket.connect_ex = reject
    socket.create_connection = reject
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create


def validate_config(config: dict[str, Any]) -> tuple[set[int], set[int]]:
    for key in ("calibration_seed_count", "heldout_seed_count", "null_replicates",
                "initial_observations", "reveal_rounds"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["calibration_seed_count"] < 5:
        raise ValueError("The canonical five-fold selector needs at least five calibration seeds")
    if config["heldout_seed_count"] < 2:
        raise ValueError("At least two held-out seeds are required")
    for key in ("calibration_seed_start", "heldout_seed_start", "random_seed"):
        if type(config.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    calibration = set(range(config["calibration_seed_start"],
                            config["calibration_seed_start"] + config["calibration_seed_count"]))
    heldout = set(range(config["heldout_seed_start"],
                        config["heldout_seed_start"] + config["heldout_seed_count"]))
    if calibration & heldout:
        raise ValueError("Calibration and held-out seeds must not overlap")
    for start, end in config.get("excluded_prior_seed_ranges", []):
        if any(start <= seed <= end for seed in calibration | heldout):
            raise ValueError("New seeds overlap an excluded prior experiment")
    if config["target_dataset"] not in replay.DATASET_BUILDERS:
        raise ValueError("Unknown target dataset")
    return calibration, heldout


def make_libraries(skills, catalog, config, candidates=None, coverage_diagnostics=None):
    pools = None
    if "coverage_matching" in config:
        import coverage_matched_rule_controls as coverage
        expected = {
            "absolute_fraction_tolerance": 0.005,
            "match_ordered_rule_support": True,
            "match_joint_activation_histogram": True,
            "sampling": "original_null_conditioned_on_public_coverage",
            "identity_and_duplicates": "allowed",
        }
        if digest(config["coverage_matching"]) != digest(expected):
            raise ValueError("Unsupported coverage matching definition")
        if candidates is None:
            raise ValueError("Coverage matching requires public candidate metadata")
        pools = coverage.build_pools(skills, catalog, config["field_types"], candidates,
                                     tolerance=expected["absolute_fraction_tolerance"])
        if coverage_diagnostics is not None:
            coverage_diagnostics.update({
                family: {name: pool.diagnostics for name, pool in family_pools.items()}
                for family, family_pools in pools.items()
            })
    libraries = {"llm": skills}
    for family in ("r1", "r2"):
        for replicate in range(config["null_replicates"]):
            # Independent named streams keep R1 unchanged if R2 or workers change.
            seed = int(digest([config["random_seed"], family, replicate]), 16)
            rng = random.Random(seed)
            library_id = f"{family}_{replicate + 1:03d}"
            libraries[library_id] = tuple(
                coverage.draw_skill(pools[family][skill.skill_id], rng,
                                    f"{library_id}_{index + 1:02d}_{skill.skill_id}")
                if pools is not None else nulls.matched_random_skill(
                    skill, catalog, rng, f"{library_id}_{index + 1:02d}_{skill.skill_id}",
                    null_type=family, field_types=config["field_types"],
                ) for index, skill in enumerate(skills)
            )
    return libraries


def skill_execution_signature(skill):
    """Exact execution inputs; ordered rules/conditions and all numbers remain."""
    value = asdict(skill)
    for key in ("skill_id", "hypothesis"):
        del value[key]
    for rule in value["rules"]:
        for key in ("rule_id", "rationale"):
            del rule[key]
    return digest(value)


def execution_plan(skills, enabled=True):
    representatives, signatures, aliases = [], {}, {}
    for skill in skills:
        mode = semantic.skill_mode(skill)
        signature = skill_execution_signature(skill) if enabled else mode
        if signature not in signatures:
            signatures[signature] = mode
            representatives.append(skill)
        if mode in aliases:
            raise ValueError("Duplicate skill mode")
        aliases[mode] = signatures[signature]
    return tuple(representatives), aliases


def expand_metric_aliases(rows, aliases):
    by_mode = {row["mode"]: row for row in rows}
    if len(by_mode) != len(rows):
        raise ValueError("Expected a single seed with unique representative modes")
    return ([row for row in rows if row["mode"] in BASELINES] +
            [dict(by_mode[representative], mode=mode) for mode, representative in aliases.items()])


def public_rule_diagnostics(skills, candidates):
    """Only public metadata is inspected; objective values are never accessed."""
    result = {}
    for skill in skills:
        rules = []
        for rule in skill.rules:
            active = [candidate.candidate_id for candidate in candidates
                      if semantic.rule_matches(rule, candidate)]
            rules.append({"rule_id": rule.rule_id, "conditions": rule.conditions,
                          "support_count": len(active),
                          "support_fraction": len(active) / len(candidates),
                          "activation_sha256": digest(sorted(active))})
        signatures = [rule["activation_sha256"] for rule in rules]
        result[semantic.skill_mode(skill)] = {
            "rules": rules, "unique_activation_count": len(set(signatures)),
            "zero_support_count": sum(rule["support_count"] == 0 for rule in rules),
        }
    return result


def validate_rows(rows, modes, seeds, split):
    expected = {(mode, seed) for mode in modes for seed in seeds}
    if any(type(row["seed"]) is not int for row in rows):
        raise ValueError(f"Noninteger seed in {split} rows")
    if any(not math.isfinite(float(row[key])) for row in rows
           for key in ("final_best", "best_so_far_auc", "top10_hit")):
        raise ValueError(f"Nonfinite metric in {split} rows")
    actual = [(str(row["mode"]), int(row["seed"])) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete or duplicate {split} cells")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Wrong split in {split} rows")


def freeze_selection(rows, libraries, seeds, config):
    # Reject held-out input entirely, rather than only trusting seed filtering.
    modes = (*BASELINES, *(semantic.skill_mode(skill)
                           for bank in libraries.values() for skill in bank))
    validate_rows(rows, modes, seeds, "calibration")
    selections = {}
    for library_id, bank in libraries.items():
        selected, details = calibrated.select_skill(
            rows, seeds, bank, config["min_risk_adjusted_gain"],
            config["min_positive_fold_rate"],
        )
        family = library_id.split("_")[0]
        details["selected_rule"] = details["selected_llm_skill"]
        details["selected_llm_skill"] = family == "llm" and details["selected_rule"]
        # This ungated calibration choice is a diagnostic, not deployed policy.
        diagnostics = details["skill_diagnostics"]
        raw_best = max(diagnostics, key=lambda mode: (
            diagnostics[mode]["risk_adjusted_composite_gain"],
            diagnostics[mode]["best_so_far_auc"]["mean"], mode))
        selections[library_id] = {"library_family": family, "selected_mode": selected,
                                  "ungated_calibration_winner": raw_best,
                                  "selection": details}
    return selections


def evaluate_seed(seed, split, config, skills):
    started = time.perf_counter()
    with offline_guard():
        try:
            rows, audits = calibrated.evaluate_seed(
                seed=seed, split=split, target_dataset=config["target_dataset"],
                skills=skills, initial=config["initial_observations"],
                rounds=config["reveal_rounds"], gp_beta=config["gp_beta"],
                gp_xi=config["gp_xi"], numeric_length_scale=config["numeric_length_scale"],
                categorical_length_scale=config["categorical_length_scale"],
                gp_noise=config["gp_noise"], include_llm_baselines=False,
            )
        finally:
            # The legacy cache owns feature dictionaries. Release between seeds
            # so a worker's memory does not grow with the number of starts.
            calibrated.surrogate._FEATURE_ARRAY_CACHE.clear()
    # Keep all choices/reveals, without duplicating each full skill in every round.
    traces = []
    for (mode, trace_seed), events in audits.items():
        for event in events:
            trace = {key: event[key] for key in (
                "mode", "seed", "round_index", "selected_candidate",
                "revealed_value", "best_so_far")}
            # The historical portfolio audit omits this count; it measures one
            # new candidate per round, exactly like the other executors.
            trace["public_observed_count"] = event.get(
                "public_observed_count", config["initial_observations"] + event["round_index"])
            traces.append(trace)
    return rows, traces, time.perf_counter() - started


def run_phase(split, seeds, config, skills, workers, output):
    compressed = config.get("exact_execution_reuse", False)
    representatives, aliases = execution_plan(skills, enabled=compressed)
    worker = partial(evaluate_seed, split=split, config=config, skills=representatives)
    rows = []
    durations = []
    trace_path = output / f"{split}_traces.jsonl{'.gz' if compressed else ''}"
    opener = gzip.open if compressed else open
    with opener(trace_path, "xt", encoding="utf-8") as handle:
        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                batches = pool.map(worker, sorted(seeds))
                for seed, batch in zip(sorted(seeds), batches):
                    seed_rows, traces, duration = batch
                    rows.extend(expand_metric_aliases(seed_rows, aliases))
                    durations.append(duration)
                    for event in traces:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")
                    handle.flush()
                    print(f"{split} seed {seed}: {len(seed_rows)} executed, "
                          f"{len(BASELINES) + len(aliases)} logical policies complete", flush=True)
        else:
            for seed in sorted(seeds):
                seed_rows, traces, duration = worker(seed)
                rows.extend(expand_metric_aliases(seed_rows, aliases))
                durations.append(duration)
                for event in traces:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                print(f"{split} seed {seed}: {len(seed_rows)} policies complete", flush=True)
    modes = (*BASELINES, *(semantic.skill_mode(skill) for skill in skills))
    validate_rows(rows, modes, seeds, split)
    nulls.write_csv(output / f"{split}_metrics.csv", rows)
    return rows, sum(durations)


def contrast(rows, treatment, baseline, seeds):
    if isinstance(rows, dict):
        return {metric: statistics.delta_summary([
            float(rows[(treatment, seed)][metric]) - float(rows[(baseline, seed)][metric])
            for seed in sorted(seeds)])
            for metric in ("best_so_far_auc", "final_best", "top10_hit")}
    return {metric: statistics.delta_summary(statistics.paired_deltas(
        rows, treatment, baseline, seeds, metric))
        for metric in ("best_so_far_auc", "final_best", "top10_hit")}


def summarize(rows, libraries, selections, seeds):
    # Reuse the exact same paired values, avoiding thousands of full-table
    # scans in the large study. This does not change interval calculations.
    rows = {(str(row["mode"]), int(row["seed"])): row for row in rows}
    llm_mode = selections["llm"]["selected_mode"]
    anchor = selections["llm"]["selection"]["target_anchor_mode"]
    result = {}
    for library_id, bank in libraries.items():
        selected = selections[library_id]["selected_mode"]
        raw = selections[library_id]["ungated_calibration_winner"]
        result[library_id] = {
            "library_family": selections[library_id]["library_family"],
            "selected_mode": selected, "target_anchor_mode": anchor,
            "ungated_calibration_winner": raw,
            "deployed_vs_anchor": contrast(rows, selected, anchor, seeds),
            "ungated_winner_vs_anchor": contrast(rows, raw, anchor, seeds),
            "llm_deployed_minus_library": contrast(rows, llm_mode, selected, seeds),
        }
    all_modes = [semantic.skill_mode(skill) for bank in libraries.values() for skill in bank]
    all_effects = {mode: contrast(rows, mode, anchor, seeds) for mode in all_modes}
    return {"libraries": result, "all_skill_effects": all_effects,
            "inference": "Paired normal 95% intervals across held-out starts, conditional on "
                         "this known BH pool and these frozen libraries. Random-library repeats "
                         "share starts, templates and baseline; they are not independent tasks. "
                         "All skill effects are descriptive, not used to reselect the winner."}


def run(config_path: Path, output: Path, workers: int, protocol_lock: Path | None = None):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    calibration_seeds, heldout_seeds = validate_config(config)
    if config.get("require_protocol_lock") and protocol_lock is None:
        raise ValueError("This protocol requires a pre-run frozen protocol lock")
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists():
        raise ValueError("Output directory already exists; use a new run ID")
    record_path = ROOT / config["llm_record"]
    if file_digest(record_path) != config["llm_record_sha256"]:
        raise ValueError("Frozen LLM record hash mismatch")
    with offline_guard():
        adapter = replay.DATASET_BUILDERS[config["target_dataset"]]()
        catalog = semantic.semantic_field_catalog(adapter)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("target_dataset") != config["target_dataset"]:
            raise ValueError("Record target dataset does not match protocol")
        skills = semantic.normalize_skills({"skills": record["normalized_skills"]}, catalog, 100)
        if [skill.skill_id for skill in skills] != config["expected_skill_ids"]:
            raise ValueError("Frozen skill library changed or normalization dropped a skill")
        if len(adapter.candidates) != config["expected_candidate_count"]:
            raise ValueError("Candidate pool size changed")
        if config["initial_observations"] + config["reveal_rounds"] > len(adapter.candidates):
            raise ValueError("Measurement budget exceeds the candidate pool")
        coverage_diagnostics = {}
        libraries = make_libraries(skills, catalog, config, adapter.candidates, coverage_diagnostics)
        all_skills = tuple(skill for bank in libraries.values() for skill in bank)
        representatives, aliases = execution_plan(all_skills, config.get("exact_execution_reuse", False))
        diagnostics = public_rule_diagnostics(all_skills, adapter.candidates)
    output.mkdir(parents=True)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    raw_files = [ROOT / path for path in config["data_files"]]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "config_sha256": file_digest(config_path), "python": sys.version,
        "frozen_protocol_sha256": file_digest(protocol_lock) if protocol_lock is not None else None,
        "numerical_thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "workers": workers, "model_api_calls": 0, "network_policy": "socket connections blocked",
        "source_sha256": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in source_files},
        "data_sha256": {path.relative_to(ROOT).as_posix(): file_digest(path) for path in raw_files},
        "public_catalog": catalog,
        "libraries": {name: [asdict(skill) for skill in bank] for name, bank in libraries.items()},
        "execution_aliases": aliases,
        "execution_reuse": "Exact ordered execution inputs; only IDs and explanatory text ignored. "
                           "Metrics retain all sampled modes and library membership. "
                           "Traces store representatives; aliases define identical trajectories.",
        "initial_candidates": {},
        "planned_trajectories": len(calibration_seeds | heldout_seeds) * (len(BASELINES) + len(all_skills)),
        "planned_physical_trajectories": len(calibration_seeds | heldout_seeds) * (len(BASELINES) + len(representatives)),
        "evidence_boundary": "Offline retrospective comparison on a known measured finite pool. "
                             "Reuses a historical LLM library, with no new model proposals. "
                             "Same five-skill selection budget per library; no best-of-replicates selection. "
                             "The data adapter holds labels for replay; only revealed observations feed "
                             "the shared executor. No new-task or pretraining-independence claim.",
    }
    if "coverage_matching" in config:
        manifest["coverage_matching_diagnostics"] = coverage_diagnostics
    for seed in sorted(calibration_seeds | heldout_seeds):
        pool = list(adapter.candidates)
        random.Random(seed).shuffle(pool)
        manifest["initial_candidates"][str(seed)] = [c.candidate_id for c in pool[:config["initial_observations"]]]
    save_json(output / "manifest.json", manifest)
    save_json(output / "public_rule_diagnostics.json", diagnostics)
    try:
        if protocol_lock is not None:
            frozen = json.loads(protocol_lock.read_text(encoding="utf-8"))
            expected = {key: manifest[key] for key in ("config_sha256", "source_sha256", "data_sha256")}
            expected.update(libraries_sha256=digest(manifest["libraries"]),
                            execution_aliases_sha256=digest(aliases))
            if "coverage_matching" in config:
                expected["coverage_matching_sha256"] = digest(coverage_diagnostics)
            if any(frozen.get(key) != value for key, value in expected.items()):
                raise ValueError("Frozen protocol does not match config, source, data or libraries")
            (output / "frozen_protocol.json").write_bytes(protocol_lock.read_bytes())
        (output / "config.json").write_bytes(config_path.read_bytes())
        # Save the exact checked bytes before replay so later source edits do
        # not remove the ability to audit a completed historical protocol.
        snapshots = {}
        for source in source_files:
            destination = output / "source_snapshot" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            if file_digest(destination) != manifest["source_sha256"][source.relative_to(ROOT).as_posix()]:
                raise RuntimeError("Source changed while taking the frozen snapshot")
        for source in [*raw_files, record_path]:
            destination = output / "input_snapshot" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            expected_hash = (config["llm_record_sha256"] if source == record_path
                             else manifest["data_sha256"][source.relative_to(ROOT).as_posix()])
            if file_digest(destination) != expected_hash:
                raise RuntimeError("Input changed while taking the frozen snapshot")
            snapshots[source.relative_to(ROOT).as_posix()] = {
                "snapshot": destination.relative_to(output).as_posix(), "sha256": expected_hash}
        save_json(output / "input_snapshot_map.json", snapshots)
        calibration_rows, calibration_seconds = run_phase(
            "calibration", calibration_seeds, config, all_skills, workers, output)
        selections = freeze_selection(calibration_rows, libraries, calibration_seeds, config)
        lock = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": file_digest(output / "manifest.json"),
                "frozen_protocol_sha256": manifest["frozen_protocol_sha256"],
                "calibration_metrics_sha256": file_digest(output / "calibration_metrics.csv"),
                "heldout_started": False, "selections": selections}
        save_json(output / "selection_lock.json", lock)
        lock_hash = file_digest(output / "selection_lock.json")
        print("Selection locked; starting held-out evaluation", flush=True)
        heldout_rows, heldout_seconds = run_phase(
            "heldout", heldout_seeds, config, all_skills, workers, output)
        if file_digest(output / "selection_lock.json") != lock_hash:
            raise RuntimeError("Selection lock changed during held-out evaluation")
        summary = summarize(heldout_rows, libraries, selections, heldout_seeds)
        summary.update({"status": "complete", "model_api_calls": 0,
                        "completed_trajectories": len(calibration_rows) + len(heldout_rows),
                        "completed_physical_trajectories": manifest["planned_physical_trajectories"],
                        "unique_execution_skills": len(representatives),
                        "wall_seconds": time.perf_counter() - started,
                        "summed_seed_worker_seconds": calibration_seconds + heldout_seconds,
                        "selection_lock_sha256": lock_hash})
        save_json(output / "summary.json", summary)
        save_json(output / "integrity.json", {
            path.name: file_digest(path) for path in sorted(output.iterdir()) if path.is_file()})
        print(json.dumps({"status": "complete", "output_dir": str(output),
                          "completed_trajectories": summary["completed_trajectories"],
                          "model_api_calls": 0}, indent=2), flush=True)
        return summary
    except BaseException as exc:
        save_json(output / "failure.json", {"status": "failed", "error": repr(exc),
                                           "model_api_calls": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.workers, args.protocol_lock)


if __name__ == "__main__":
    main()
