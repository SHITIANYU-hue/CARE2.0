#!/usr/bin/env python3
"""Offline BH v2 evaluation: every scheduled slot, including invalid-library fallback."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import shutil
import sys
import time

import numpy as np

import common

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/care_replay/scripts"))
import run_matched_random_rule_controls as matched
import run_paired_anonymization_evaluation as v1
import analyze_matched_rule_stability as stability

ARMS = ("labeled", "anonymous")
VALID = "VALID"
INVALID = "INVALID_ANCHOR_FALLBACK"
FIXED = {
    "protocol_id": "bh_semantic_anonymization_v2", "target_dataset": "real_buchwald_hartwig",
    "expected_candidate_count": 3955, "pair_count": 10,
    "calibration_seed_start": 202610205000, "calibration_seed_count": 50,
    "heldout_seed_start": 202610206000, "heldout_seed_count": 100,
    "bootstrap_seed": 2026102004, "bootstrap_replicates": 10000, "heldout_blocks": 5,
    "initial_observations": 5, "reveal_rounds": 10, "gp_beta": 1.5, "gp_xi": 0.01,
    "numeric_length_scale": 0.35, "categorical_length_scale": 3.0, "gp_noise": 0.05,
    "min_risk_adjusted_gain": 0.25, "min_positive_fold_rate": 0.8,
    "exact_execution_reuse": True,
}


def money(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Malformed cost") from None
    if not result.is_finite() or result < 0:
        raise ValueError("Cost must be finite and nonnegative")
    return result


def names(config):
    return [f"pair_{pair:02d}_{arm}" for pair in range(config["pair_count"]) for arm in ARMS]


def validate_config(config, workers):
    if type(workers) is not int or workers != 4:
        raise ValueError("The v2 runtime freezes exactly four workers")
    for key, expected in FIXED.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError(f"Frozen v2 evaluation parameter changed: {key}")
    if config["paired_design"]["ordered_rule_counts"] != [2, 1, 1, 1, 1]:
        raise ValueError("Rule-count contract changed")
    selection = config["selection"]
    for key, expected in (("strategies_per_library", 5), ("expand_variants", False),
                          ("across_library_selection", False),
                          ("lock_selection_before_any_heldout", True)):
        if selection.get(key) != expected:
            raise ValueError(f"Selection contract changed: {key}")
    if tuple(selection["target_anchor"]["modes"]) != matched.BASELINES:
        raise ValueError("Target anchor inventory changed")
    for key, expected in (
        ("primary_metric", "best_so_far_auc"),
        ("primary_policy", "ungated_calibration_winner_if_valid_else_shared_target_anchor"),
        ("primary_contrast", "labeled_minus_anonymous"), ("confidence_level", 0.95),
        ("primary_hypotheses", 1), ("no_heldout_reselection", True),
    ):
        if config["analysis"].get(key) != expected:
            raise ValueError(f"Analysis contract changed: {key}")
    terminal = config["generation_completion_contract"]
    if (set(terminal["allowed_terminal_states"]) != {VALID, INVALID}
            or terminal["required_slots"] != 20 or terminal["halt_never_counts_as_invalid"] is not True):
        raise ValueError("Terminal-state contract changed")
    for key in ("generation_enabled", "new_model_calls_approved", "new_invalid_output_policy_approved"):
        if config["resource_authorization"].get(key) is not True:
            raise ValueError("v2 requires explicit approved generation and fallback policy")
    calibration = set(range(config["calibration_seed_start"],
                            config["calibration_seed_start"] + config["calibration_seed_count"]))
    heldout = set(range(config["heldout_seed_start"],
                        config["heldout_seed_start"] + config["heldout_seed_count"]))
    if calibration & heldout:
        raise ValueError("Calibration and heldout overlap")
    for start, end in config["excluded_prior_seed_ranges"]:
        if any(start <= seed <= end for seed in calibration | heldout):
            raise ValueError("A new replay seed overlaps an excluded prior protocol")
    return calibration, heldout


def checked(path, expected_hash=None):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Required ordinary file missing: {path.name}")
    digest = common.file_hash(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"File hash mismatch: {path.name}")
    return digest


def replay_classification(response, arm, material, cost):
    """Use the production classifier only as a pure function, never its API runner."""
    import generation
    return generation.classify_generation(
        response, arm, material, response_sha256=generation.object_sha256(response),
        material_sha256=generation.object_sha256(material), settled_cost=cost)


def verify_billing_evidence(response, billing, receipt, cost):
    """Reproduce the generator's billing decision using retained receipts only."""
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("is_byok") is True:
        raise ValueError("Original response usage is missing/malformed or reports BYOK")
    inline = money(usage["cost"]) if usage.get("cost") is not None else None
    if inline is not None and usage.get("is_byok") is False:
        if billing["source"] != "response_usage" or cost != inline:
            raise ValueError("Billing differs from the settled original response usage")
        return
    if billing["source"] != "generation_lookup":
        raise ValueError("Missing original non-BYOK evidence requires a generation lookup")
    for attempt in range(3):
        path = receipt / f"cost_lookup_{attempt}.json"
        error_path = receipt / f"cost_lookup_{attempt}_error.json"
        if path.exists() == error_path.exists():
            raise ValueError("Lookup attempts must have exactly one retained response or error")
        if error_path.exists():
            error = common.load(error_path)
            if error.get("http_status") != 404:
                raise ValueError("Non-retryable billing lookup error cannot have a completed outcome")
            continue
        info = common.load(path)
        data = info.get("data") if isinstance(info, dict) else None
        if (not isinstance(data, dict) or data.get("id") != response["id"]
                or data.get("model") != response.get("model")
                or data.get("provider_name") != response.get("provider")
                or data.get("is_byok") is True):
            raise ValueError("Original billing lookup provenance/BYOK mismatch")
        if data.get("total_cost") is not None and data.get("is_byok") is False:
            lookup = money(data["total_cost"])
            expected = max(lookup, inline) if inline is not None else lookup
            if cost != expected:
                raise ValueError("Settled billing differs from original lookup and inline cost")
            return
    raise ValueError("Retained billing lookups never settle the response cost and non-BYOK status")


def verify_generation(config_path, lock_path, generation_output, config, lock):
    """Require twenty settled, traceable terminal outcomes before any calibration."""
    base = Path(generation_output)
    if (base / "failure.json").exists():
        raise ValueError("A halted generation directory cannot be evaluated")
    completion = common.load(base / "completion.json")
    if completion.get("status") != "complete" or completion.get("actual_post_attempts") != 20:
        raise ValueError("Twenty completed new POST slots are required; HALT/missing slots cannot be evaluated")
    slots = names(config)
    expected = set(slots)
    if expected != common.expected_slots(config):
        raise ValueError("Unexpected slot inventory")
    hashes = {"completion.json": checked(base / "completion.json")}
    if (completion.get("config_sha256") != checked(config_path)
            or completion.get("protocol_lock_sha256") != checked(lock_path)):
        raise ValueError("Completion config/protocol mismatch")
    hashes["config.json"] = checked(base / "config.json", checked(config_path))
    hashes["protocol_lock.json"] = checked(base / "protocol_lock.json", checked(lock_path))
    if set(completion.get("outcomes_sha256", {})) != expected:
        raise ValueError("All twenty outcome hashes are required")
    for directory in ("outcomes", "libraries", "receipts"):
        if not (base / directory).is_dir() or (base / directory).is_symlink():
            raise ValueError(f"Required directory missing: {directory}")
    if {p.name for p in (base / "outcomes").iterdir()} != {slot + ".json" for slot in slots}:
        raise ValueError("Outcome files must match all twenty slots exactly")
    if {p.name for p in (base / "receipts").iterdir()} != expected:
        raise ValueError("Receipt directories must match all twenty slots exactly")
    for section, folder, identifiers in (
        ("messages_sha256", "messages", expected),
        ("pair_material_sha256", "pairs", {f"pair_{i:02d}" for i in range(config["pair_count"])}),
    ):
        if set(lock.get(section, {})) != identifiers or completion.get(section) != lock[section]:
            raise ValueError(f"Completion differs from frozen {section}")
        for identifier, digest in lock[section].items():
            relative = f"input_snapshot/{folder}/{identifier}.json"
            hashes[relative] = checked(base / relative, digest)
    for relative, digest in lock["source_sha256"].items():
        destination = "source_snapshot/" + relative
        hashes[destination] = checked(base / destination, digest)
    for group in ("data_sha256", "input_sha256"):
        for relative, digest in lock[group].items():
            destination = "input_snapshot/" + relative
            hashes[destination] = checked(base / destination, digest)

    outcomes, payloads, response_ids = {}, {}, set()
    cost_total = Decimal("0")
    for pair in range(config["pair_count"]):
        material = common.load(base / f"input_snapshot/pairs/pair_{pair:02d}.json")
        for arm in ARMS:
            identifier = f"pair_{pair:02d}_{arm}"
            relative = f"outcomes/{identifier}.json"
            hashes[relative] = checked(base / relative, completion["outcomes_sha256"][identifier])
            outcome = common.load(base / relative)
            if (outcome.get("identifier") != identifier or outcome.get("pair_index") != pair
                    or outcome.get("arm") != arm or outcome.get("status") not in {VALID, INVALID}):
                raise ValueError("Wrong slot identity or nonterminal/HALT outcome")
            receipt = base / "receipts" / identifier
            if receipt.is_symlink() or not receipt.is_dir():
                raise ValueError("Receipt must be an ordinary directory")
            for filename, digest in (("request.json", outcome.get("request_sha256")),
                                     ("response.json", outcome.get("response_sha256"))):
                if not isinstance(digest, str):
                    raise ValueError("Request/response provenance hash missing")
                key = f"receipts/{identifier}/{filename}"
                hashes[key] = checked(base / key, digest)
            request = common.load(receipt / "request.json")
            response = common.load(receipt / "response.json")
            billing = common.load(receipt / "billing.json")
            if not isinstance(outcome.get("billing_sha256"), str):
                raise ValueError("Billing provenance hash missing")
            hashes[f"receipts/{identifier}/billing.json"] = checked(
                receipt / "billing.json", outcome["billing_sha256"])
            expected_request = {
                "model": config["model"]["id"],
                "messages": common.load(base / f"input_snapshot/messages/{identifier}.json"),
                "temperature": config["model"]["temperature"], "max_tokens": config["model"]["max_tokens"],
                "response_format": {"type": "json_object"}, "stream": False,
                "provider": config["api"]["provider"],
            }
            if request != expected_request:
                raise ValueError("Request differs from its frozen message/parameters")
            response_id = response.get("id") if isinstance(response, dict) else None
            if (not isinstance(response_id, str) or not response_id or response_id in response_ids
                    or outcome.get("response_id") != response_id or billing.get("response_id") != response_id):
                raise ValueError("Missing, duplicate or inconsistent response identity")
            response_ids.add(response_id)
            if (billing.get("is_byok") is not False
                    or billing.get("source") not in {"response_usage", "generation_lookup"}):
                raise ValueError("Every outcome requires explicitly settled non-BYOK billing")
            cost = money(billing.get("cost_usd"))
            if cost != money(outcome.get("cost_usd")) or cost > money(config["api"]["reservation_per_attempt_usd"]):
                raise ValueError("Outcome billing mismatch or reservation exceeded")
            verify_billing_evidence(response, billing, receipt, cost)
            decision = replay_classification(response, arm, material, cost)
            if (decision.status != outcome["status"]
                    or list(decision.reasons) != outcome.get("reason_codes")
                    or list(decision.reasons[1:]) != outcome.get("errors")):
                raise ValueError("Recorded terminal state/reasons differ from deterministic classification")
            if money(decision.cost_usd) != cost:
                raise ValueError("Classifier billing mismatch")
            if outcome["status"] == VALID:
                if not isinstance(outcome.get("library_sha256"), str):
                    raise ValueError("Valid library hash missing")
                key = f"libraries/{identifier}.json"
                hashes[key] = checked(base / key, outcome["library_sha256"])
                payloads[identifier] = common.load(base / key)
                if payloads[identifier] != decision.library:
                    raise ValueError("Valid library differs from the decoded complete response")
            elif (outcome.get("library_sha256") is not None or decision.library is not None
                  or not outcome.get("reason_codes")):
                raise ValueError("Invalid slots must retain reasons and no partial library")
            outcomes[identifier] = outcome
            cost_total += cost
            # Preserve extra provider reconciliation receipts as immutable audit inputs.
            for path in receipt.rglob("*"):
                if path.is_file():
                    hashes[path.relative_to(base).as_posix()] = checked(path)
                elif path.is_symlink():
                    raise ValueError("Receipt links are forbidden")
    valid_ids = set(payloads)
    if set(completion.get("libraries_sha256", {})) != valid_ids:
        raise ValueError("Completion libraries must equal the VALID subset")
    if {p.name for p in (base / "libraries").iterdir()} != {name + ".json" for name in valid_ids}:
        raise ValueError("Invalid slots cannot retain partial libraries; valid files cannot be missing")
    if any(completion["libraries_sha256"][name] != outcomes[name]["library_sha256"] for name in valid_ids):
        raise ValueError("Outcome/completion library hashes differ")
    if (completion.get("valid_library_count") != len(valid_ids)
            or completion.get("invalid_library_count") != len(slots) - len(valid_ids)):
        raise ValueError("Completion validity counts differ from terminal outcomes")
    ledger = common.load(base / "ledger.json")
    if not isinstance(completion.get("ledger_sha256"), str):
        raise ValueError("Completed ledger hash missing")
    hashes["ledger.json"] = checked(base / "ledger.json", completion["ledger_sha256"])
    attempts = ledger.get("attempts")
    if (not isinstance(attempts, list) or len(attempts) != 20
            or ledger.get("actual_post_attempts") != 20
            or {a.get("identifier") for a in attempts} != expected):
        raise ValueError("Ledger must contain exactly twenty unique new attempts")
    for slot_number, (attempt, scheduled) in enumerate(zip(attempts, config["generation_order"], strict=True)):
        identifier = f"pair_{scheduled['pair_index']:02d}_{scheduled['arm']}"
        outcome = outcomes[identifier]
        if (attempt.get("slot") != slot_number or attempt.get("identifier") != identifier
                or attempt.get("pair_index") != scheduled["pair_index"]
                or attempt.get("arm") != scheduled["arm"]):
            raise ValueError("Ledger slot order differs from frozen generation order")
        for key in ("request_sha256", "response_sha256", "response_id", "library_sha256"):
            if attempt.get(key) != outcome[key]:
                raise ValueError(f"Ledger/outcome provenance mismatch: {key}")
        if attempt.get("outcome_sha256") != completion["outcomes_sha256"][identifier]:
            raise ValueError("Ledger outcome hash differs from completion")
        if attempt.get("status") != outcome["status"] or money(attempt.get("cost_usd")) != money(outcome["cost_usd"]):
            raise ValueError("Ledger/outcome mismatch")
    prior = money(config["resource_authorization"]["prior_known_cost_usd"])
    cumulative = prior + cost_total
    if cumulative > money(config["api"]["max_cost_usd"]):
        raise ValueError("Cumulative experimental spend exceeds the frozen cap")
    for record in (ledger, completion):
        for key, expected_cost in (("prior_cost_usd", prior), ("known_new_cost_usd", cost_total),
                                    ("cumulative_known_cost_usd", cumulative)):
            if money(record.get(key)) != expected_cost:
                raise ValueError(f"Cost ledger mismatch: {key}")
    return outcomes, payloads, hashes


def freeze_choices(rows, libraries, outcomes, seeds, config):
    if (set(outcomes) != set(names(config))
            or any(outcome["status"] not in {VALID, INVALID} for outcome in outcomes.values())
            or set(libraries) != {name for name, outcome in outcomes.items() if outcome["status"] == VALID}):
        raise ValueError("Selection requires all terminal outcomes and exactly the valid libraries")
    modes = (*matched.BASELINES, *(matched.semantic.skill_mode(skill)
                                  for bank in libraries.values() for skill in bank))
    matched.validate_rows(rows, modes, seeds, "calibration")
    summary = matched.statistics.mode_means(rows, matched.BASELINES, seeds)
    anchor = max(matched.BASELINES, key=lambda mode: (summary[mode]["composite"], mode))
    valid_choices = matched.freeze_selection(rows, libraries, seeds, config) if libraries else {}
    choices = {}
    for identifier in names(config):
        outcome = outcomes[identifier]
        if outcome["status"] == VALID:
            choice = valid_choices[identifier]
            if choice["selection"]["target_anchor_mode"] != anchor:
                raise ValueError("A valid library selected a different target-only anchor")
            choice["primary_mode"] = choice["ungated_calibration_winner"]
            choice["selection"]["selected_llm_skill"] = choice["selection"]["selected_rule"]
        else:
            choice = {
                "primary_mode": anchor, "selected_mode": anchor, "ungated_calibration_winner": None,
                "selection": {"target_anchor_mode": anchor, "target_anchor_summary": summary,
                              "selected_rule": False, "selected_llm_skill": False,
                              "fallback": "prespecified_invalid_whole_library"},
            }
        choice.update(status=outcome["status"], pair_index=outcome["pair_index"],
                      library_family=outcome["arm"], reason_codes=outcome["reason_codes"])
        choices[identifier] = choice
    return choices, anchor, summary


def describe(matrix, seeds, blocks):
    result = stability.describe_deltas(matrix, seeds, blocks)
    for old, new in (("bank_mean_deltas", "pair_mean_deltas"),
                     ("bank_mean_quantiles", "pair_mean_quantiles"),
                     ("positive_bank_fraction", "positive_pair_fraction"),
                     ("zero_bank_fraction", "zero_pair_fraction"),
                     ("negative_bank_fraction", "negative_pair_fraction")):
        result[new] = result.pop(old)
    return result


def analyze(rows, libraries, outcomes, choices, seeds, config):
    slots = names(config)
    if set(outcomes) != set(slots) or set(choices) != set(slots):
        raise ValueError("Every scheduled terminal outcome and frozen choice is required")
    valid_ids = {name for name, outcome in outcomes.items() if outcome["status"] == VALID}
    if any(outcome["status"] not in {VALID, INVALID} for outcome in outcomes.values()) or set(libraries) != valid_ids:
        raise ValueError("HALT/missing/partial libraries cannot be analyzed")
    modes = (*matched.BASELINES, *(matched.semantic.skill_mode(skill)
                                  for bank in libraries.values() for skill in bank))
    matched.validate_rows(rows, modes, seeds, "heldout")
    anchors = {choice["selection"]["target_anchor_mode"] for choice in choices.values()}
    if len(anchors) != 1 or not anchors <= set(matched.BASELINES):
        raise ValueError("A single shared calibration anchor is required")
    anchor = next(iter(anchors))
    for identifier, choice in choices.items():
        if choice["status"] != outcomes[identifier]["status"]:
            raise ValueError("Frozen outcome status changed")
        if choice["status"] == INVALID:
            if choice["primary_mode"] != anchor or choice["selected_mode"] != anchor:
                raise ValueError("Invalid library must use the shared anchor")
        else:
            bank_modes = {matched.semantic.skill_mode(skill) for skill in libraries[identifier]}
            if len(bank_modes) != 5 or choice["primary_mode"] not in bank_modes or choice["selected_mode"] not in bank_modes | {anchor}:
                raise ValueError("A valid library must keep its own five-select-one choice")
    ordered = sorted(seeds)
    index = {(row["mode"], row["seed"]): row for row in rows}
    rng = np.random.default_rng(config["bootstrap_seed"])
    seed_counts = rng.multinomial(len(ordered), np.full(len(ordered), 1 / len(ordered)),
                                  size=config["bootstrap_replicates"])
    pair_counts = rng.multinomial(config["pair_count"],
                                  np.full(config["pair_count"], 1 / config["pair_count"]),
                                  size=config["bootstrap_replicates"])

    def matrix(key, metric="best_so_far_auc", arm=None):
        return np.asarray([
            [float(index[(choices[f"pair_{p:02d}_{arm or 'labeled'}"][key], seed)][metric])
             - float(index[(anchor if arm else choices[f"pair_{p:02d}_anonymous"][key], seed)][metric])
             for seed in ordered]
            for p in range(config["pair_count"])], dtype=float)

    def endpoint(values):
        result = describe(values, ordered, config["heldout_blocks"])
        result["delta_matrix"] = values.tolist()
        result["interval"] = stability.percentile_interval(
            stability.crossed_bootstrap_means(values, pair_counts, seed_counts), 0.95)
        result["interval"]["method"] = "crossed whole-generation-pair/shared-heldout-start percentile bootstrap"
        return result

    primary_matrix = matrix("primary_mode")
    primary = endpoint(primary_matrix)
    both_valid = [p for p in range(config["pair_count"])
                  if all(f"pair_{p:02d}_{arm}" in valid_ids for arm in ARMS)]
    subset = {"available": bool(both_valid), "pair_count": len(both_valid),
              "pair_ids": [f"pair_{p:02d}" for p in both_valid],
              "interpretation": "Post-generation subset is descriptive only; no confirmatory inference or generalization to all generated libraries."}
    if both_valid:
        subset["descriptive_auc"] = describe(primary_matrix[both_valid, :], ordered, config["heldout_blocks"])
    else:
        subset["reason"] = "No pair is valid in both arms; this result is unavailable, not zero."
    all_invalid = not valid_ids
    return {
        "primary_estimand": config["analysis"]["primary_estimand"],
        "primary_interpretation": config["analysis"]["primary_interpretation"],
        "direction": "labeled minus anonymous, retaining all scheduled pairs",
        "pair_ids": [f"pair_{p:02d}" for p in range(config["pair_count"])], "heldout_seeds": ordered,
        "primary_all_pairs_auc": primary,
        "primary_mean_advantage": bool(primary["interval"]["lower"] > 0),
        "stable_advantage": bool(primary["interval"]["lower"] > 0 and primary["all_block_means_positive"]),
        "stability_interpretation": "The five consecutive seed blocks are a descriptive stability convention, not extra confirmatory tests. Pair means do not enter this flag.",
        "secondary_gated_auc": endpoint(matrix("selected_mode")),
        "secondary_anchor_auc": {f"{arm}_{label}": endpoint(matrix(key, arm=arm))
                                 for arm in ARMS for label, key in (("primary", "primary_mode"), ("gated", "selected_mode"))},
        "descriptive_other_metrics": {label: {metric: describe(matrix(key, metric), ordered, config["heldout_blocks"])
                                             for metric in ("final_best", "top10_hit")}
                                      for label, key in (("primary", "primary_mode"), ("gated", "selected_mode"))},
        "both_valid_pairs_descriptive": subset,
        "valid_library_counts_by_arm": {arm: sum(name.endswith("_" + arm) for name in valid_ids) for arm in ARMS},
        "pair_validity_counts": dict(Counter(
            ("both_valid" if all(f"pair_{p:02d}_{arm}" in valid_ids for arm in ARMS)
             else "both_invalid" if all(f"pair_{p:02d}_{arm}" not in valid_ids for arm in ARMS)
             else "labeled_invalid_only" if f"pair_{p:02d}_labeled" not in valid_ids
             else "anonymous_invalid_only")
            for p in range(config["pair_count"]))),
        "gate_accepted_library_counts_by_arm": {
            arm: sum(choices[name]["selection"].get("selected_rule") is True
                     for name in valid_ids if name.endswith("_" + arm)) for arm in ARMS},
        "invalid_reason_categories_by_arm": {arm: dict(Counter(
            outcome["reason_codes"][0] for outcome in outcomes.values()
            if outcome["arm"] == arm and outcome["status"] == INVALID)) for arm in ARMS},
        "all_invalid_constructed_zero": all_invalid,
        "degenerate_interpretation": ("No generated valid libraries. Zero differences and the [0,0] interval follow from assigning both arms the identical anchor; this is not equivalence or precise estimation of knowledge."
                                      if all_invalid else None),
        "target_anchor_mode": anchor,
        "bootstrap": {"replicates": config["bootstrap_replicates"], "seed": config["bootstrap_seed"],
                      "confidence_level": 0.95, "primary_hypotheses": 1,
                      "unit": "Whole paired-generation rows and shared heldout-seed columns",
                      "no_cell_independence": True, "calibration_bootstrapped": False},
        "auc_units": "Original yield percentage points averaged over reveal-round best-so-far values",
        "evidence_boundary": config["analysis"]["scope"], "numpy_version": np.__version__,
    }


def evaluate_phases(config, libraries, outcomes, calibration, heldout, output, protocol_hash, verify):
    skills = tuple(skill for bank in libraries.values() for skill in bank)
    calibration_rows, calibration_seconds = matched.run_phase(
        "calibration", calibration, config, skills, 4, output)
    choices, anchor, anchor_summary = freeze_choices(calibration_rows, libraries, outcomes, calibration, config)
    lock = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "heldout_started": False,
            "manifest_sha256": common.file_hash(output / "manifest.json"),
            "protocol_lock_sha256": protocol_hash,
            "calibration_metrics_sha256": common.file_hash(output / "calibration_metrics.csv"),
            "heldout_seeds": sorted(heldout), "choices": choices,
            "target_anchor_mode": anchor, "target_anchor_summary": anchor_summary}
    common.save(output / "selection_lock.json", lock, exclusive=True)
    selection_hash = common.file_hash(output / "selection_lock.json")
    verify()
    common.save(output / "heldout_start.json", {"selection_lock_sha256": selection_hash,
                "started_at_utc": datetime.now(timezone.utc).isoformat()}, exclusive=True)
    heldout_rows, heldout_seconds = matched.run_phase("heldout", heldout, config, skills, 4, output)
    if (common.file_hash(output / "selection_lock.json") != selection_hash
            or common.file_hash(output / "manifest.json") != lock["manifest_sha256"]
            or common.file_hash(output / "calibration_metrics.csv") != lock["calibration_metrics_sha256"]):
        raise RuntimeError("Locked selection/calibration evidence changed")
    result = analyze(heldout_rows, libraries, outcomes, choices, heldout, config)
    return result, {"selection_lock_sha256": selection_hash,
                    "completed_logical_trajectories": len(calibration_rows) + len(heldout_rows),
                    "summed_seed_worker_seconds": calibration_seconds + heldout_seconds}


def run(config_path, lock_path, freeze_dir, generation_output, output, workers=4):
    config_path, lock_path, freeze_dir, generation_output, output = map(
        Path, (config_path, lock_path, freeze_dir, generation_output, output))
    started = time.perf_counter()
    if output.exists():
        raise ValueError("Evaluation output already exists")
    with matched.offline_guard():
        config, lock = common.verify_freeze(config_path, lock_path, freeze_dir)
        calibration, heldout = validate_config(config, workers)
        if output.resolve() != (ROOT / config["evaluation_output_directory"]).resolve():
            raise ValueError("Use the frozen evaluation output directory")
        if generation_output.resolve() != (ROOT / config["generation_output_directory"]).resolve():
            raise ValueError("Use the frozen generation output directory")
        outcomes, payloads, generation_hashes = verify_generation(
            config_path, lock_path, generation_output, config, lock)
        # Nothing is loaded into the replay executor until all twenty terminal outcomes pass.
        adapter = matched.replay.DATASET_BUILDERS[config["target_dataset"]]()
        if len(adapter.candidates) != config["expected_candidate_count"]:
            raise ValueError("Candidate pool size changed")
        catalog = matched.semantic.semantic_field_catalog(adapter)
        libraries, normalization = {}, {}
        for identifier in names(config):
            if identifier in payloads:
                outcome = outcomes[identifier]
                libraries[identifier], normalization[identifier] = v1.normalize_without_loss(
                    payloads[identifier], catalog, outcome["pair_index"], outcome["arm"])
        skills = tuple(skill for bank in libraries.values() for skill in bank)
        representatives, aliases = matched.execution_plan(skills, config["exact_execution_reuse"])
    protocol_hash = common.file_hash(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    try:
        def copy_verified(source, destination, digest):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            checked(destination, digest)

        for group, destination in (("source_sha256", "source_snapshot"),
                                   ("data_sha256", "input_snapshot"), ("input_sha256", "input_snapshot")):
            for relative, digest in lock[group].items():
                copy_verified(ROOT / relative, output / destination / relative, digest)
        for relative, digest in generation_hashes.items():
            copy_verified(generation_output / relative, output / "generation_snapshot" / relative, digest)
        copy_verified(config_path, output / "config.json", lock["config_sha256"])
        copy_verified(lock_path, output / "protocol_lock.json", protocol_hash)
        manifest = {
            "config": config, "config_sha256": lock["config_sha256"], "protocol_lock_sha256": protocol_hash,
            "source_sha256": lock["source_sha256"], "data_sha256": lock["data_sha256"],
            "input_sha256": lock["input_sha256"], "generation_sha256": generation_hashes,
            "outcomes": outcomes, "libraries": {name: [asdict(skill) for skill in bank] for name, bank in libraries.items()},
            "normalization_and_id_provenance": normalization, "execution_aliases": aliases,
            "scheduled_library_slots": 20, "valid_library_count": len(libraries),
            "invalid_library_count": 20 - len(libraries), "logical_skill_count": len(skills),
            "unique_execution_skills": len(representatives),
            "planned_logical_trajectories": (len(matched.BASELINES) + len(skills)) * (len(calibration) + len(heldout)),
            "planned_physical_trajectories": (len(matched.BASELINES) + len(representatives)) * (len(calibration) + len(heldout)),
            "model_api_calls_in_evaluation": 0,
        }
        common.save(output / "manifest.json", manifest, exclusive=True)

        def verify():
            current_config, current_lock = common.verify_freeze(config_path, lock_path, freeze_dir)
            if current_config != config or current_lock != lock or common.file_hash(lock_path) != protocol_hash:
                raise RuntimeError("Frozen protocol changed")
            _, _, current_hashes = verify_generation(config_path, lock_path, generation_output, config, lock)
            if current_hashes != generation_hashes:
                raise RuntimeError("Generation evidence changed during evaluation")

        with matched.offline_guard():
            analysis, execution = evaluate_phases(
                config, libraries, outcomes, calibration, heldout, output, protocol_hash, verify)
            verify()
        common.save(output / "analysis.json", analysis, exclusive=True)
        result = {"status": "complete", "model_api_calls": 0, "generation_pair_count": 10,
                  "valid_library_count": len(libraries), "invalid_library_count": 20 - len(libraries),
                  "wall_seconds": time.perf_counter() - started, **execution,
                  "primary_mean_auc_delta": analysis["primary_all_pairs_auc"]["mean_delta"],
                  "primary_95_interval": analysis["primary_all_pairs_auc"]["interval"],
                  "primary_mean_advantage": analysis["primary_mean_advantage"],
                  "stable_advantage": analysis["stable_advantage"],
                  "all_invalid_constructed_zero": analysis["all_invalid_constructed_zero"],
                  "interpretation": analysis["primary_interpretation"]}
        common.save(output / "summary.json", result, exclusive=True)
        common.save(output / "integrity.json", {
            path.relative_to(output).as_posix(): common.file_hash(path)
            for path in sorted(output.rglob("*")) if path.is_file()}, exclusive=True)
        return result
    except BaseException as error:
        common.save(output / "failure.json", {"status": "failed", "error": repr(error), "model_api_calls": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--generation-output", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.protocol_lock, args.freeze_dir,
                         args.generation_output, args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
