#!/usr/bin/env python3
"""Offline, locked five-select-one evaluation of paired BH generation libraries."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

import analyze_matched_rule_stability as stability
import run_matched_random_rule_controls as matched

ROOT = matched.ROOT
ARMS = ("labeled", "anonymous")
RULE_COUNTS = (2, 1, 1, 1, 1)
PARAMETER_RANGES = {
    "ridge": (0.05, 20.0), "prior_scale": (0.0, 1.5),
    "semantic_mass_start": (0.0, 0.85), "semantic_mass_end": (0.0, 0.85),
    "ucb_weight": (0.0, 1.0), "gp_beta_start": (0.2, 4.0),
    "gp_beta_end": (0.2, 4.0), "gp_xi": (0.0, 0.2),
    "confidence": (0.05, 1.0),
}
FIXED_PARAMETERS = {
    "protocol_id": "bh_semantic_anonymization_v1",
    "target_dataset": "real_buchwald_hartwig", "expected_candidate_count": 3955,
    "pair_count": 10, "calibration_seed_count": 50, "heldout_seed_count": 100,
    "initial_observations": 5, "reveal_rounds": 10,
    "gp_beta": 1.5, "gp_xi": 0.01, "numeric_length_scale": 0.35,
    "categorical_length_scale": 3.0, "gp_noise": 0.05,
    "min_risk_adjusted_gain": 0.25, "min_positive_fold_rate": 0.8,
    "bootstrap_replicates": 10000, "heldout_blocks": 5,
}


def strict_json(path):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique, parse_constant=reject)


def library_ids(pair_count):
    return [f"pair_{pair:02d}_{arm}" for pair in range(pair_count) for arm in ARMS]


def validate_config(config):
    for key, expected in FIXED_PARAMETERS.items():
        if config.get(key) != expected or isinstance(config.get(key), bool):
            raise ValueError(f"Frozen evaluation parameter changed: {key}")
    for key in ("calibration_seed_start", "heldout_seed_start", "bootstrap_seed"):
        if type(config.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    calibration = set(range(config["calibration_seed_start"],
                            config["calibration_seed_start"] + config["calibration_seed_count"]))
    heldout = set(range(config["heldout_seed_start"],
                        config["heldout_seed_start"] + config["heldout_seed_count"]))
    if calibration & heldout:
        raise ValueError("Calibration and heldout seeds overlap")
    for begin, end in config.get("excluded_prior_seed_ranges", []):
        if any(begin <= seed <= end for seed in calibration | heldout):
            raise ValueError("Seeds overlap an excluded prior experiment")
    if "ordered_atomic_rule_counts" in config:
        if tuple(config["ordered_atomic_rule_counts"]) != RULE_COUNTS:
            raise ValueError("Rule-count contract changed")
    if "paired_design" in config:
        if tuple(config["paired_design"].get("ordered_rule_counts", [])) != RULE_COUNTS:
            raise ValueError("Paired rule-count contract changed")
    if "selection" in config:
        for key, expected in (("strategies_per_library", 5), ("expand_variants", False),
                              ("across_library_selection", False),
                              ("lock_selection_before_any_heldout", True)):
            if config["selection"].get(key) != expected:
                raise ValueError(f"Selection contract differs from implementation: {key}")
    if "analysis" in config:
        for key, expected in (("primary_metric", "best_so_far_auc"),
                              ("primary_policy", "ungated_calibration_winner"),
                              ("primary_contrast", "labeled_minus_anonymous"),
                              ("confidence_level", 0.95), ("primary_hypotheses", 1),
                              ("no_heldout_reselection", True)):
            if config["analysis"].get(key) != expected:
                raise ValueError(f"Analysis contract differs from implementation: {key}")
    return calibration, heldout


def checked_repo_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Frozen input must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
        raise ValueError(f"Frozen input is missing or outside repository: {relative}")
    return resolved


def verify_frozen_inputs(config_path, protocol_lock, generation_output, config):
    """Verify current executable/input bytes and the completed generation handoff."""
    frozen = strict_json(protocol_lock)
    completion = strict_json(generation_output / "completion.json")
    if (frozen.get("protocol_id") != config["protocol_id"]
            or frozen.get("status") != "frozen_before_generation"
            or frozen.get("model_api_calls_at_freeze") != 0):
        raise ValueError("A pre-generation frozen protocol is required")
    if frozen.get("config_sha256") != matched.file_digest(config_path):
        raise ValueError("Config differs from frozen protocol")
    if strict_json(config_path) != config:
        raise ValueError("Config changed after it was loaded")
    source_hashes = frozen.get("source_sha256")
    expected_sources = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "experiments/care_replay/scripts").glob("*.py")
    }
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_sources:
        raise ValueError("Frozen source inventory differs from current scripts")
    if set(frozen.get("data_sha256", {})) != set(config["data_files"]):
        raise ValueError("Frozen data inventory differs from config")
    for section in ("source_sha256", "data_sha256", "input_sha256"):
        mapping = frozen.get(section)
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Missing frozen {section}")
        for relative, expected_hash in mapping.items():
            if matched.file_digest(checked_repo_path(relative)) != expected_hash:
                raise ValueError(f"Frozen file hash mismatch: {relative}")
    if completion.get("status") != "complete":
        raise ValueError("Generation must be complete; incomplete pairs cannot be excluded")
    if completion.get("config_sha256") != frozen["config_sha256"]:
        raise ValueError("Generation config hash mismatch")
    if completion.get("protocol_lock_sha256") != matched.file_digest(protocol_lock):
        raise ValueError("Generation protocol-lock hash mismatch")
    names = library_ids(config["pair_count"])
    if set(completion.get("libraries_sha256", {})) != set(names):
        raise ValueError("Generation must include all twenty libraries")
    actual_files = {path.name for path in (generation_output / "libraries").iterdir()}
    if actual_files != {name + ".json" for name in names}:
        raise ValueError("Generation library files differ from the complete fixed inventory")
    for name in names:
        path = generation_output / "libraries" / (name + ".json")
        if not path.is_file() or path.is_symlink():
            raise ValueError("Library must be an ordinary JSON file")
        if matched.file_digest(path) != completion["libraries_sha256"][name]:
            raise ValueError(f"Completed library hash mismatch: {name}")
    for section, subdirectory, expected_names in (
        ("messages_sha256", "messages", set(names)),
        ("pair_material_sha256", "pairs",
         {f"pair_{pair:02d}" for pair in range(config["pair_count"])}),
    ):
        mapping = frozen.get(section)
        if not isinstance(mapping, dict) or set(mapping) != expected_names:
            raise ValueError(f"Frozen generation inventory mismatch: {section}")
        if completion.get(section) != mapping:
            raise ValueError(f"Generation handoff differs from frozen {section}")
        for name, expected_hash in mapping.items():
            path = generation_output / "input_snapshot" / subdirectory / (name + ".json")
            if not path.is_file() or path.is_symlink() or matched.file_digest(path) != expected_hash:
                raise ValueError(f"Frozen generation material mismatch: {section}/{name}")
    return frozen, completion


def normalize_without_loss(payload, catalog, pair, arm):
    """Reject normalization repairs; rename only metadata IDs after exact checks."""
    if not isinstance(payload, dict) or set(payload) != {"skills"}:
        raise ValueError("Library must contain only the skills key")
    raw_skills = payload["skills"]
    if not isinstance(raw_skills, list) or len(raw_skills) != 5:
        raise ValueError("Exactly five skills are required")
    seen = set()
    for slot, raw in enumerate(raw_skills):
        if not isinstance(raw, dict) or set(raw) != {"skill_id", "rules", "hypothesis",
                                                    *PARAMETER_RANGES}:
            raise ValueError("Unexpected or missing skill fields")
        identifier = raw["skill_id"]
        if (not isinstance(identifier, str)
                or re.fullmatch(r"[a-z0-9_]{1,64}", identifier) is None or identifier in seen):
            raise ValueError("Invalid or duplicate skill ID")
        seen.add(identifier)
        if not isinstance(raw["hypothesis"], str) or len(raw["hypothesis"]) > 800:
            raise ValueError("Invalid hypothesis")
        for key, (lower, upper) in PARAMETER_RANGES.items():
            value = raw[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid executable parameter: {key}")
        if not isinstance(raw["rules"], list) or len(raw["rules"]) != RULE_COUNTS[slot]:
            raise ValueError("Ordered atomic rule-count contract changed")
        rule_ids = set()
        for rule in raw["rules"]:
            if not isinstance(rule, dict) or set(rule) != {"rule_id", "conditions", "weight", "rationale"}:
                raise ValueError("Unexpected or missing rule fields")
            identifier = rule["rule_id"]
            if (not isinstance(identifier, str) or re.fullmatch(r"[a-z0-9_]{1,64}", identifier) is None
                    or identifier in rule_ids):
                raise ValueError("Invalid or duplicate rule ID")
            rule_ids.add(identifier)
            conditions = rule["conditions"]
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError("Nonempty conditions object required")
            if any(field not in catalog or not isinstance(value, str) or value not in catalog[field]
                   for field, value in conditions.items()):
                raise ValueError("Unknown executable field/value")
            weight = rule["weight"]
            if type(weight) not in (int, float) or not math.isfinite(weight) or not -1 <= weight <= 1:
                raise ValueError("Invalid rule weight")
            if not isinstance(rule["rationale"], str) or len(rule["rationale"]) > 400:
                raise ValueError("Invalid rationale")
    normalized = matched.semantic.normalize_skills(payload, catalog, 5)
    if len(normalized) != 5:
        raise ValueError("Normalizer dropped a skill")
    result, provenance = [], []
    for slot, (raw, skill) in enumerate(zip(raw_skills, normalized)):
        if len(skill.rules) != len(raw["rules"]):
            raise ValueError("Normalizer dropped a rule")
        if any(getattr(skill, key) != raw[key] for key in PARAMETER_RANGES):
            raise ValueError("Normalizer changed an executable number")
        if skill.hypothesis != raw["hypothesis"]:
            raise ValueError("Normalizer truncated hypothesis")
        for raw_rule, rule in zip(raw["rules"], skill.rules):
            if (rule.conditions != tuple(sorted(raw_rule["conditions"].items()))
                    or rule.weight != raw_rule["weight"] or rule.rationale != raw_rule["rationale"]):
                raise ValueError("Normalizer changed a rule")
        identifier = f"pair_{pair:02d}_{arm}_slot_{slot:02d}"
        renamed = replace(skill, skill_id=identifier, rules=tuple(
            replace(rule, rule_id=f"{identifier}_rule_{index:02d}")
            for index, rule in enumerate(skill.rules)))
        if matched.skill_execution_signature(renamed) != matched.skill_execution_signature(skill):
            raise ValueError("ID assignment changed executable semantics")
        result.append(renamed)
        provenance.append({"slot": slot, "raw_skill_id": raw["skill_id"],
                           "normalized_skill_id": skill.skill_id, "execution_skill_id": identifier,
                           "execution_signature": matched.skill_execution_signature(skill)})
    return tuple(result), provenance


def pair_delta_matrix(index, selections, pair_count, seeds, key, metric, arm=None):
    values = []
    for pair in range(pair_count):
        labeled = selections[f"pair_{pair:02d}_labeled"]
        anonymous = selections[f"pair_{pair:02d}_anonymous"]
        left = labeled[key] if arm is None else selections[f"pair_{pair:02d}_{arm}"][key]
        right = anonymous[key] if arm is None else labeled["selection"]["target_anchor_mode"]
        values.append([float(index[(left, seed)][metric]) - float(index[(right, seed)][metric])
                       for seed in seeds])
    return np.asarray(values, dtype=float)


def analyze(heldout_rows, libraries, selections, seeds, config):
    """Single prespecified primary comparison; no policy selection occurs here."""
    names = library_ids(config["pair_count"])
    if set(libraries) != set(names) or set(selections) != set(names):
        raise ValueError("Missing paired library or frozen selection")
    modes = (*matched.BASELINES, *(matched.semantic.skill_mode(skill)
                                  for name in names for skill in libraries[name]))
    ordered_seeds = sorted(seeds)
    matched.validate_rows(heldout_rows, modes, seeds, "heldout")
    anchors = set()
    for name in names:
        bank_modes = {matched.semantic.skill_mode(skill) for skill in libraries[name]}
        choice = selections[name]
        if choice["ungated_calibration_winner"] not in bank_modes:
            raise ValueError("Raw selection is not a member of its five-skill library")
        if choice["selected_mode"] not in bank_modes | set(matched.BASELINES):
            raise ValueError("Deployed selection is outside its library and target anchors")
        anchors.add(choice["selection"]["target_anchor_mode"])
    if len(anchors) != 1 or not anchors <= set(matched.BASELINES):
        raise ValueError("Expected one shared calibration-selected target anchor")
    index = {(row["mode"], row["seed"]): row for row in heldout_rows}
    rng = np.random.default_rng(config["bootstrap_seed"])
    draws = config["bootstrap_replicates"]
    seed_counts = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=draws)
    pair_counts = rng.multinomial(config["pair_count"],
                                 np.full(config["pair_count"], 1 / config["pair_count"]), size=draws)

    def describe(matrix, primary=False):
        result = stability.describe_deltas(matrix, ordered_seeds, config["heldout_blocks"])
        result["pair_mean_deltas"] = result.pop("bank_mean_deltas")
        result["pair_mean_quantiles"] = result.pop("bank_mean_quantiles")
        for sign in ("positive", "zero", "negative"):
            result[sign + "_pair_fraction"] = result.pop(sign + "_bank_fraction")
        result["interval"] = stability.percentile_interval(
            stability.crossed_bootstrap_means(matrix, pair_counts, seed_counts), 0.95)
        result["interval"]["method"] = "crossed paired-generation/shared-start percentile bootstrap"
        result["delta_matrix"] = matrix.tolist()
        result["role"] = "single primary comparison" if primary else "exploratory secondary comparison"
        result["block_interpretation"] = "Five consecutive equal seed blocks are descriptive, not independent replications."
        return result

    primary = describe(pair_delta_matrix(index, selections, config["pair_count"], ordered_seeds,
                                        "ungated_calibration_winner", "best_so_far_auc"), True)
    secondary = describe(pair_delta_matrix(index, selections, config["pair_count"], ordered_seeds,
                                          "selected_mode", "best_so_far_auc"))
    anchor_comparisons = {
        f"{arm}_{label}_vs_anchor": describe(pair_delta_matrix(
            index, selections, config["pair_count"], ordered_seeds, key, "best_so_far_auc", arm))
        for arm in ARMS
        for label, key in (("ungated", "ungated_calibration_winner"), ("deployed", "selected_mode"))
    }
    other_metrics = {
        label: {
            metric: stability.describe_deltas(
                pair_delta_matrix(index, selections, config["pair_count"], ordered_seeds, key, metric),
                ordered_seeds, config["heldout_blocks"])
            for metric in ("final_best", "top10_hit")
        }
        for label, key in (("ungated", "ungated_calibration_winner"), ("deployed", "selected_mode"))
    }
    return {
        "direction": "labeled minus anonymous; positive AUC difference favours semantic labels",
        "primary_estimand": "Mean paired heldout AUC difference of the separately calibration-selected ungated five-select-one winners.",
        "auc_units": "Original yield percentage points averaged across reveal-round best-so-far values.",
        "pair_ids": [f"pair_{pair:02d}" for pair in range(config["pair_count"])],
        "heldout_seeds": ordered_seeds,
        "bootstrap": {"seed": config["bootstrap_seed"], "replicates": draws,
                      "confidence_level": 0.95, "primary_comparisons": 1,
                      "units": "Independent paired-generation rows and shared heldout-start columns",
                      "resampling": "Product multinomial weights; both arms remain paired and one seed resample is shared across all pairs and endpoints."},
        "primary_ungated_auc": primary, "secondary_deployed_auc": secondary,
        "secondary_anchor_auc": anchor_comparisons,
        "descriptive_other_metrics": other_metrics,
        "primary_mean_advantage": bool(primary["interval"]["lower"] > 0),
        "stable_advantage": bool(primary["interval"]["lower"] > 0
                                 and primary["all_block_means_positive"]),
        "stability_interpretation": "Descriptive BH reporting convention: the one primary 95% interval is positive and all five consecutive seed-block means are positive. Blocks are not extra confirmatory tests; pair means do not enter this flag.",
        "target_anchor_mode": next(iter(anchors)),
        "evidence_boundary": "Conditional on this BH pool, model/prompt protocol and frozen calibration choices. No calibration bootstrap or heldout reselection. The 1000 cells are not independent observations, and the ten pairs are not ten scientific tasks.",
        "numpy_version": np.__version__,
    }


def evaluate_phases(config, libraries, calibration_seeds, heldout_seeds, workers, output,
                    protocol_hash, verify_before_holdout):
    all_skills = tuple(skill for bank in libraries.values() for skill in bank)
    calibration_rows, calibration_seconds = matched.run_phase(
        "calibration", calibration_seeds, config, all_skills, workers, output)
    selections = matched.freeze_selection(calibration_rows, libraries, calibration_seeds, config)
    for name, choice in selections.items():
        # The old helper reserves library ID "llm" for this metadata flag.
        # Both present arms are generated LLM libraries; choices/gates stay exact.
        choice["library_family"] = name.rsplit("_", 1)[1]
        choice["pair_id"] = name.rsplit("_", 1)[0]
        choice["selection"]["selected_llm_skill"] = choice["selection"]["selected_rule"]
    selection_lock = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": matched.file_digest(output / "manifest.json"),
        "protocol_lock_sha256": protocol_hash,
        "calibration_metrics_sha256": matched.file_digest(output / "calibration_metrics.csv"),
        "heldout_started": False, "selections": selections,
        "heldout_seeds": sorted(heldout_seeds),
    }
    matched.save_json(output / "selection_lock.json", selection_lock)
    selection_hash = matched.file_digest(output / "selection_lock.json")
    verify_before_holdout()
    matched.save_json(output / "heldout_start.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_lock_sha256": selection_hash,
    })
    heldout_rows, heldout_seconds = matched.run_phase(
        "heldout", heldout_seeds, config, all_skills, workers, output)
    if matched.file_digest(output / "selection_lock.json") != selection_hash:
        raise RuntimeError("Selection lock changed during heldout evaluation")
    if (matched.file_digest(output / "manifest.json") != selection_lock["manifest_sha256"]
            or matched.file_digest(output / "calibration_metrics.csv")
            != selection_lock["calibration_metrics_sha256"]):
        raise RuntimeError("Locked calibration evidence changed during heldout evaluation")
    result = analyze(heldout_rows, libraries, selections, heldout_seeds, config)
    return result, {
        "selection_lock_sha256": selection_hash,
        "completed_logical_trajectories": len(calibration_rows) + len(heldout_rows),
        "summed_seed_worker_seconds": calibration_seconds + heldout_seconds,
    }


def run(config_path, protocol_lock, generation_output, output, workers=4):
    config_path, protocol_lock, generation_output, output = map(
        Path, (config_path, protocol_lock, generation_output, output))
    started = time.perf_counter()
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing evaluation directory")
    config = strict_json(config_path)
    calibration, heldout = validate_config(config)
    frozen, completion = verify_frozen_inputs(config_path, protocol_lock, generation_output, config)
    protocol_hash = matched.file_digest(protocol_lock)
    completion_hash = matched.file_digest(generation_output / "completion.json")
    with matched.offline_guard():
        adapter = matched.replay.DATASET_BUILDERS[config["target_dataset"]]()
        if len(adapter.candidates) != config["expected_candidate_count"]:
            raise ValueError("Candidate pool size changed")
        catalog = matched.semantic.semantic_field_catalog(adapter)
        libraries, provenance = {}, {}
        for pair in range(config["pair_count"]):
            for arm in ARMS:
                name = f"pair_{pair:02d}_{arm}"
                libraries[name], provenance[name] = normalize_without_loss(
                    strict_json(generation_output / "libraries" / (name + ".json")),
                    catalog, pair, arm)
        all_skills = tuple(skill for bank in libraries.values() for skill in bank)
        representatives, aliases = matched.execution_plan(
            all_skills, config.get("exact_execution_reuse", False))
    output.mkdir(parents=True, exist_ok=False)
    try:
        snapshots = {}
        for section, destination_base in (("source_sha256", "source_snapshot"),
                                          ("data_sha256", "input_snapshot"),
                                          ("input_sha256", "input_snapshot")):
            for relative, expected_hash in frozen[section].items():
                destination = output / destination_base / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checked_repo_path(relative), destination)
                if matched.file_digest(destination) != expected_hash:
                    raise RuntimeError("Frozen file changed during snapshot")
                snapshots[relative] = {"path": destination.relative_to(output).as_posix(),
                                       "sha256": expected_hash}
        for name in library_ids(config["pair_count"]):
            source = generation_output / "libraries" / (name + ".json")
            destination = output / "generation_snapshot" / "libraries" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if matched.file_digest(destination) != completion["libraries_sha256"][name]:
                raise RuntimeError("Generation changed during snapshot")
        for section, directory in (("messages_sha256", "messages"), ("pair_material_sha256", "pairs")):
            for name, expected_hash in frozen[section].items():
                source = generation_output / "input_snapshot" / directory / (name + ".json")
                destination = output / "generation_snapshot" / "input_snapshot" / directory / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if matched.file_digest(destination) != expected_hash:
                    raise RuntimeError("Generation material changed during snapshot")
        for source, name, expected_hash in (
            (config_path, "config.json", frozen["config_sha256"]),
            (protocol_lock, "frozen_protocol.json", protocol_hash),
            (generation_output / "completion.json", "generation_completion.json", completion_hash),
        ):
            shutil.copyfile(source, output / name)
            if matched.file_digest(output / name) != expected_hash:
                raise RuntimeError("Protocol handoff changed during snapshot")
        manifest = {
            "config": config, "config_sha256": frozen["config_sha256"],
            "protocol_lock_sha256": protocol_hash, "generation_completion_sha256": completion_hash,
            "source_sha256": frozen["source_sha256"], "data_sha256": frozen["data_sha256"],
            "input_sha256": frozen["input_sha256"], "snapshots": snapshots,
            "libraries": {name: [asdict(skill) for skill in bank] for name, bank in libraries.items()},
            "generation_libraries_sha256": completion["libraries_sha256"],
            "normalization_and_id_provenance": provenance, "execution_aliases": aliases,
            "unique_execution_skills": len(representatives),
            "planned_logical_trajectories": (len(matched.BASELINES) + len(all_skills)) * (len(calibration) + len(heldout)),
            "planned_physical_trajectories": (len(matched.BASELINES) + len(representatives)) * (len(calibration) + len(heldout)),
            "model_api_calls_in_evaluation": 0,
            "tie_breaking": "Canonical selector ranks risk-adjusted composite, AUC, then deterministic pair/arm/slot mode; within each library this final tie-break follows slot order.",
        }
        matched.save_json(output / "manifest.json", manifest)

        def verify_unchanged():
            verify_frozen_inputs(config_path, protocol_lock, generation_output, config)
            if (matched.file_digest(protocol_lock) != protocol_hash
                    or matched.file_digest(generation_output / "completion.json") != completion_hash):
                raise RuntimeError("Protocol or generation completion changed during evaluation")

        with matched.offline_guard():
            analysis, execution = evaluate_phases(
                config, libraries, calibration, heldout, workers, output, protocol_hash,
                verify_unchanged)
            verify_unchanged()
        matched.save_json(output / "analysis.json", analysis)
        summary = {
            "status": "complete", "model_api_calls": 0, "generation_pair_count": config["pair_count"],
            "library_count": len(libraries), "logical_skill_count": len(all_skills),
            "unique_execution_skills": len(representatives), "wall_seconds": time.perf_counter() - started,
            **execution, "primary_mean_auc_delta": analysis["primary_ungated_auc"]["mean_delta"],
            "primary_95_interval": analysis["primary_ungated_auc"]["interval"],
            "primary_mean_advantage": analysis["primary_mean_advantage"],
            "stable_advantage": analysis["stable_advantage"],
            "scope": "One prespecified paired labeled-minus-anonymous comparison; all secondary comparisons and five seed blocks are descriptive/exploratory.",
        }
        matched.save_json(output / "summary.json", summary)
        matched.save_json(output / "integrity.json", {
            path.relative_to(output).as_posix(): matched.file_digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()})
        return summary
    except BaseException as error:
        matched.save_json(output / "failure.json",
                          {"status": "failed", "error": repr(error), "model_api_calls": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--generation-output", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.protocol_lock, args.generation_output,
                         args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
