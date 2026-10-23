#!/usr/bin/env python3
"""Audit recorded offline rule-control runs without rerunning acquisition.

Supports the original uncompressed pilot and compressed representative traces
with execution aliases. Run this CLI in a fresh process so historical snapshots
can supply the exact selector and summarizer recorded by an older run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys


class AuditError(ValueError):
    """The recorded run cannot be reconstructed or a frozen input changed."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes its recorded root: {relative}")
    return path


def verify_sources(repo, run, hashes):
    """Allow a verified historical snapshot, while reporting current changes."""
    paths, current_changed = {}, []
    for name, expected in hashes.items():
        current = checked_path(repo, name)
        if current.is_file() and file_digest(current) == expected:
            paths[name] = current
            continue
        current_changed.append(name)
        candidates = (checked_path(run, f"source_snapshot/{name}"),
                      checked_path(run, f"source_snapshot/{Path(name).name}"))
        snapshot = next((path for path in candidates
                         if path.is_file() and file_digest(path) == expected), None)
        require(snapshot is not None, f"Source hash mismatch without a matching snapshot: {name}")
        paths[name] = snapshot
    return paths, current_changed


def load_recorded_runner(repo, verified_sources, expected_hashes):
    """Import only after resolving verified historical/current source paths."""
    scripts = repo / "experiments/care_replay/scripts"
    directories = list(dict.fromkeys(
        str(path.parent) for path in verified_sources.values() if path.parent != scripts))
    sys.path[:0] = [*directories, str(scripts)]
    module = importlib.import_module("run_matched_random_rule_controls")
    # A snapshot folder must not silently shadow an otherwise verified module.
    for name, expected in expected_hashes.items():
        loaded = sys.modules.get(Path(name).stem)
        if loaded is not None and getattr(loaded, "__file__", None):
            require(file_digest(Path(loaded.__file__)) == expected,
                    f"Imported source differs from recorded source: {name}; use a fresh process")
    return module


def restore_libraries(manifest, runner, candidates=None):
    libraries = {}
    for name, bank in manifest["libraries"].items():
        skills = []
        for raw in bank:
            value = dict(raw)
            value["rules"] = tuple(runner.semantic.SemanticRule(
                **{**rule, "conditions": tuple(tuple(condition) for condition in rule["conditions"])}
            ) for rule in raw["rules"])
            skills.append(runner.semantic.SemanticSkill(**value))
        libraries[name] = tuple(skills)
    require("llm" in libraries and libraries["llm"], "Missing frozen LLM library")
    require(all(len(bank) == len(libraries["llm"]) for bank in libraries.values()),
            "Unequal library selection budgets")
    require(len(libraries) == 1 + 2 * manifest["config"]["null_replicates"],
            "Unexpected number of rule libraries")
    if "coverage_matching" in manifest["config"]:
        diagnostics = {}
        generated = runner.make_libraries(libraries["llm"], manifest["public_catalog"],
                                          manifest["config"], candidates, diagnostics)
        require(digest(diagnostics) == digest(manifest.get("coverage_matching_diagnostics")),
                "Coverage conditioning pools cannot be reproduced")
    else:
        generated = runner.make_libraries(libraries["llm"], manifest["public_catalog"], manifest["config"])
    require(digest({name: [asdict(skill) for skill in bank] for name, bank in generated.items()})
            == digest(manifest["libraries"]), "Frozen library generation cannot be reproduced")
    return libraries


def execution_signature(skill):
    value = asdict(skill)
    for key in ("skill_id", "hypothesis"):
        del value[key]
    for rule in value["rules"]:
        for key in ("rule_id", "rationale"):
            del rule[key]
    return digest(value)


def validate_aliases(manifest, libraries, runner):
    skills = {runner.semantic.skill_mode(skill): skill
              for bank in libraries.values() for skill in bank}
    require(len(skills) == sum(map(len, libraries.values())), "Duplicate logical skill modes")
    aliases = manifest.get("execution_aliases", {mode: mode for mode in skills})
    require(set(aliases) == set(skills), "Aliases do not cover exactly the logical skill modes")
    for mode, representative in aliases.items():
        require(representative in skills, f"Unknown execution representative: {representative}")
        require(aliases[representative] == representative, f"Noncanonical alias chain for {mode}")
        require(execution_signature(skills[mode]) == execution_signature(skills[representative]),
                f"Alias changes execution inputs: {mode}")
    return aliases


def verify_frozen_protocol(run, manifest, lock):
    path = run / "frozen_protocol.json"
    required = manifest["config"].get("require_protocol_lock", False)
    declared_hash = manifest.get("frozen_protocol_sha256")
    if not path.exists():
        require(not required and declared_hash is None, "Required frozen protocol is absent")
        return False
    protocol = load_json(path)
    expected = {key: manifest[key] for key in ("config_sha256", "source_sha256", "data_sha256")}
    expected.update(libraries_sha256=digest(manifest["libraries"]),
                    execution_aliases_sha256=digest(manifest.get("execution_aliases", {})))
    if "coverage_matching" in manifest["config"]:
        expected["coverage_matching_sha256"] = digest(manifest.get("coverage_matching_diagnostics"))
    for key, value in expected.items():
        require(protocol.get(key) == value, f"Frozen protocol differs: {key}")
    protocol_hash = file_digest(path)
    if required or declared_hash is not None:
        require(declared_hash == protocol_hash, "Manifest frozen protocol hash mismatch")
        require(lock.get("frozen_protocol_sha256") == protocol_hash,
                "Selection lock frozen protocol hash mismatch")
    return True


def read_metrics(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw_seed = row["seed"]
        try:
            row["seed"] = int(raw_seed)
        except (ValueError, TypeError) as exc:
            raise AuditError(f"Invalid CSV seed: {raw_seed!r}") from exc
        require(str(row["seed"]) == raw_seed, f"Noncanonical CSV seed: {raw_seed!r}")
    return rows


def read_traces(run, split):
    plain, compressed = run / f"{split}_traces.jsonl", run / f"{split}_traces.jsonl.gz"
    require(plain.exists() != compressed.exists(), f"Expected exactly one {split} trace artifact")
    path, opener = (compressed, gzip.open) if compressed.exists() else (plain, open)
    traces = defaultdict(list)
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            require(type(event["seed"]) is int, "Noninteger trace seed")
            require(type(event["round_index"]) is int, "Noninteger trace round")
            traces[(event["mode"], event["seed"])].append(event)
    return traces


def close(actual, expected, message):
    try:
        matched = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-8)
    except (ValueError, TypeError):
        matched = False
    require(matched, message)


def reconstruct_trajectory(events, initial, candidates, rounds, top10):
    require(len(events) == rounds, "Wrong number of reveal events")
    seen = set(initial)
    require(len(initial) == len(seen), "Duplicate initial candidates")
    require(seen <= set(candidates), "Unknown initial candidate")
    best = max(candidates[candidate].objective_value for candidate in initial)
    best_trace = []
    for index, event in enumerate(sorted(events, key=lambda item: item["round_index"])):
        require(event["round_index"] == index, "Missing or duplicate reveal round")
        require(type(event["public_observed_count"]) is int and
                event["public_observed_count"] == len(seen), "Incorrect public observation count")
        candidate = event["selected_candidate"]
        require(candidate in candidates, f"Unknown revealed candidate: {candidate}")
        require(candidate not in seen, f"Duplicate candidate query: {candidate}")
        seen.add(candidate)
        value = candidates[candidate].objective_value
        close(event["revealed_value"], value, f"Reveal value mismatch: {candidate}")
        best = max(best, value)
        close(event["best_so_far"], best, f"Best-so-far mismatch in round {index}")
        best_trace.append(best)
    return {"final_best": round(best, 4), "best_so_far_auc": round(mean(best_trace), 4),
            "top10_hit": int(bool(seen & top10))}


def audit_run(repo, run, *, runner=None):
    """Return an audit report; never alter the run or rerun any acquisition."""
    repo, run = Path(repo).resolve(), Path(run).resolve()
    manifest, summary, lock = (load_json(run / name) for name in
                               ("manifest.json", "summary.json", "selection_lock.json"))
    require(summary["status"] == "complete", "Run is not complete")
    require(manifest.get("model_api_calls") == summary.get("model_api_calls") == 0,
            "Run does not declare zero model calls")
    require(file_digest(run / "selection_lock.json") == summary["selection_lock_sha256"],
            "Selection lock hash mismatch")
    require(file_digest(run / "manifest.json") == lock["manifest_sha256"], "Manifest hash mismatch")
    require(file_digest(run / "calibration_metrics.csv") == lock["calibration_metrics_sha256"],
            "Calibration metrics hash mismatch")
    require(lock.get("heldout_started") is False, "Selection was not declared frozen before held-out evaluation")
    integrity = load_json(run / "integrity.json")
    for name, expected in integrity.items():
        require(file_digest(checked_path(run, name)) == expected, f"Artifact hash mismatch: {name}")
    verified_sources, changed_sources = verify_sources(repo, run, manifest["source_sha256"])
    for name, expected in manifest["data_sha256"].items():
        require(file_digest(checked_path(repo, name)) == expected, f"Data hash mismatch: {name}")
    frozen_protocol_verified = verify_frozen_protocol(run, manifest, lock)
    config = manifest["config"]
    if (run / "config.json").exists():
        require(file_digest(run / "config.json") == manifest["config_sha256"], "Config file hash mismatch")
        require(load_json(run / "config.json") == config, "Config content differs from manifest")
    if "llm_record" in config:
        require(file_digest(checked_path(repo, config["llm_record"])) == config["llm_record_sha256"],
                "Frozen LLM record hash mismatch")
    if runner is None:
        runner = load_recorded_runner(repo, verified_sources, manifest["source_sha256"])
    calibration, heldout = runner.validate_config(config)
    with runner.offline_guard():
        adapter = runner.replay.DATASET_BUILDERS[config["target_dataset"]]()
    candidates = {candidate.candidate_id: candidate for candidate in adapter.candidates}
    require(len(candidates) == len(adapter.candidates) == config["expected_candidate_count"],
            "Candidate pool size or unique IDs changed")
    require(all(math.isfinite(candidate.objective_value) for candidate in adapter.candidates),
            "Nonfinite candidate objective")
    require(runner.semantic.semantic_field_catalog(adapter) == manifest["public_catalog"],
            "Public candidate metadata catalog changed")
    require(set(manifest["initial_candidates"]) == {str(seed) for seed in calibration | heldout},
            "Initial candidates do not cover exactly the planned seeds")
    for seed in calibration | heldout:
        pool = list(adapter.candidates)
        random.Random(seed).shuffle(pool)
        expected = [candidate.candidate_id for candidate in pool[:config["initial_observations"]]]
        require(manifest["initial_candidates"][str(seed)] == expected,
                f"Initial candidates differ from the seeded shared start: {seed}")
    libraries = restore_libraries(manifest, runner, adapter.candidates)
    aliases = validate_aliases(manifest, libraries, runner)
    logical_modes = (*runner.BASELINES, *aliases)
    physical_modes = set(runner.BASELINES) | set(aliases.values())
    require(not (set(runner.BASELINES) & set(aliases)), "Skill modes collide with baselines")
    top10 = {candidate.candidate_id for candidate in sorted(
        adapter.candidates, key=lambda candidate: candidate.objective_value, reverse=True)[:10]}
    oracle = max(candidate.objective_value for candidate in adapter.candidates)
    logical_count = physical_count = 0
    for split, seeds in (("calibration", calibration), ("heldout", heldout)):
        rows = read_metrics(run / f"{split}_metrics.csv")
        runner.validate_rows(rows, logical_modes, seeds, split)
        traces = read_traces(run, split)
        require(set(traces) == {(mode, seed) for mode in physical_modes for seed in seeds},
                f"Missing, duplicate or unexpected physical {split} trajectories")
        reconstructed = {}
        for (mode, seed), events in traces.items():
            reconstructed[(mode, seed)] = reconstruct_trajectory(
                events, manifest["initial_candidates"][str(seed)], candidates,
                config["reveal_rounds"], top10)
        for row in rows:
            representative = aliases.get(row["mode"], row["mode"])
            expected = reconstructed[(representative, row["seed"])]
            for key, value in expected.items():
                close(row[key], value, f"Metric mismatch: {split}/{row['mode']}/{row['seed']}/{key}")
            if "simple_regret" in row:
                # Use unrounded final best from the verified events, as the executor does.
                final = max(candidates[c].objective_value for c in manifest["initial_candidates"][str(row["seed"])]
                            + [event["selected_candidate"] for event in traces[(representative, row["seed"])]] )
                close(row["simple_regret"], round(oracle - final, 4), "Simple regret mismatch")
        logical_count += len(rows)
        physical_count += len(traces)
        if split == "calibration":
            rebuilt = runner.freeze_selection(rows, libraries, calibration, config)
            require(rebuilt == lock["selections"], "Selection reconstruction differs from frozen calibration")
        else:
            rebuilt = runner.summarize(rows, libraries, lock["selections"], heldout)
            for key, value in rebuilt.items():
                require(summary.get(key) == value, f"Summary reconstruction differs: {key}")
    require(logical_count == manifest["planned_trajectories"] == summary["completed_trajectories"],
            "Logical trajectory total mismatch")
    require(physical_count == manifest.get("planned_physical_trajectories", manifest["planned_trajectories"]),
            "Physical trajectory total mismatch")
    if "completed_physical_trajectories" in summary:
        require(physical_count == summary["completed_physical_trajectories"],
                "Completed physical trajectory total mismatch")
    return {"status": "passed", "trajectories_reconstructed": logical_count,
            "physical_trajectories_reconstructed": physical_count,
            "decisions_reconstructed": logical_count * config["reveal_rounds"],
            "physical_decisions_reconstructed": physical_count * config["reveal_rounds"],
            "duplicate_queries": 0, "metric_mismatches": 0,
            "selection_reconstructed_from_calibration_only": True,
            "summary_reconstructed": True, "frozen_protocol_verified": frozen_protocol_verified,
            "source_and_data_hashes_unchanged": not changed_sources,
            "current_source_files_changed": changed_sources,
            "historical_source_snapshots_verified": changed_sources,
            "recorded_source_and_data_hashes_verified": True,
            "execution_aliases_verified": True, "model_api_calls": 0,
            "boundary": "Checks recorded choices, outcomes, metrics and calibration selection. "
                        "Does not rerun acquisition ranking or establish scientific generalization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new report file; existing files are not overwritten")
    args = parser.parse_args()
    report = audit_run(args.repo, args.run)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
