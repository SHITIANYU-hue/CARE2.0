"""Shared integrity and authorization checks for the fresh V2 protocol.

No credential access or network operations. The original 116 scripts are
read-only dependencies and must still match the V1 lock before every phase.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL_REL = "experiments/care_replay/protocols/bh_semantic_anonymization_v2"
CONFIG_REL = "experiments/care_replay/configs/bh_semantic_anonymization_v2.json"
LEGACY_COUNT = 116
ARMS = ("labeled", "anonymous")
PRIOR_COST = Decimal("0.0125145")
EXPECTED = {'target_dataset': 'real_buchwald_hartwig',
 'expected_candidate_count': 3955,
 'llm_record': 'experiments/care_replay/skill_banks/frozen_records/bh_semantic_headline_20260721.json',
 'llm_record_sha256': 'a3b4c29d4a55ec24c9b4305f8309ed401e9248ef66c92a4268b5a4a6d049fbdc',
 'data_files': ['experiments/care_replay/data/raw/dreher_doyle_buchwald_hartwig.xlsx',
                'experiments/care_replay/data/descriptors/reaction_component_descriptors.csv'],
 'initial_observations': 5,
 'reveal_rounds': 10,
 'gp_beta': 1.5,
 'gp_xi': 0.01,
 'numeric_length_scale': 0.35,
 'categorical_length_scale': 3.0,
 'gp_noise': 0.05,
 'min_risk_adjusted_gain': 0.25,
 'min_positive_fold_rate': 0.8,
 'exact_execution_reuse': True,
 'protocol_id': 'bh_semantic_anonymization_v2',
 'pair_count': 10,
 'calibration_seed_start': 202610205000,
 'calibration_seed_count': 50,
 'heldout_seed_start': 202610206000,
 'heldout_seed_count': 100,
 'bootstrap_seed': 2026102004,
 'bootstrap_replicates': 10000,
 'heldout_blocks': 5,
 'require_protocol_lock': True,
 'status': 'approved_for_execution_pending_freeze',
 'generation_output_directory': 'experiments/care_replay/results/2026-10-20-bh-semantic-anonymization-v2-generation',
 'model.id': 'openai/gpt-4o-mini',
 'model.temperature': 0.45,
 'model.max_tokens': 5000,
 'model.response_format': 'json_object',
 'model.sampling_seed': None,
 'api.base_url': 'https://openrouter.ai/api/v1',
 'api.max_cost_usd': 1.0,
 'api.reservation_per_attempt_usd': 0.045,
 'api.context_token_bound': 128000,
 'api.timeout_seconds': 120,
 'api.transport_retries': 0,
 'api.provider': {'only': ['openai'],
                  'order': ['openai'],
                  'allow_fallbacks': False,
                  'require_parameters': True,
                  'max_price': {'prompt': 0.15, 'completion': 0.6, 'request': 0}},
 'api.cumulative_prior_spend_usd': '0.0125145',
 'paired_design.ordered_rule_counts': [2, 1, 1, 1, 1],
 'paired_design.private_map_sent_to_model': False,
 'paired_design.knowledge_base_injection': False,
 'paired_design.target_outcomes_in_generation': False,
 'paired_design.same_neutral_system_and_contract': True,
 'paired_design.alias_seed_by_pair': {'pair_00': 202610207000,
                                      'pair_01': 202610207001,
                                      'pair_02': 202610207002,
                                      'pair_03': 202610207003,
                                      'pair_04': 202610207004,
                                      'pair_05': 202610207005,
                                      'pair_06': 202610207006,
                                      'pair_07': 202610207007,
                                      'pair_08': 202610207008,
                                      'pair_09': 202610207009},
 'paired_design.order_seed': 202610208000,
 'paired_design.v1_generated_libraries_reused': False,
 'selection.strategies_per_library': 5,
 'selection.expand_variants': False,
 'selection.across_library_selection': False,
 'selection.gate_inherited_conditions': {'mean_delta_final_min': 0.0,
                                         'mean_delta_auc_min': 0.0,
                                         'final_nonloss_rate_min': 0.55,
                                         'positive_fold_rate_min': 0.8,
                                         'risk_adjusted_composite_gain_min': 0.25,
                                         'risk_adjustment': 'mean(delta_final+delta_auc) - 0.5*SE'},
 'selection.lock_selection_before_any_heldout': True,
 'selection.target_anchor.modes': ['gp_ucb', 'mixed_kernel_gp_ei', 'target_acquisition_portfolio'],
 'selection.target_anchor.calibration_only': True,
 'selection.target_anchor.same_for_every_pair_and_arm': True,
 'selection.target_anchor.lock_before_heldout': True,
 'analysis.primary_metric': 'best_so_far_auc',
 'analysis.primary_policy': 'ungated_calibration_winner_if_valid_else_shared_target_anchor',
 'analysis.primary_contrast': 'labeled_minus_anonymous',
 'analysis.confidence_level': 0.95,
 'analysis.primary_hypotheses': 1,
 'analysis.no_heldout_reselection': True,
 'generation_completion_contract': {'required_slots': 20,
                                    'allowed_terminal_states': ['VALID', 'INVALID_ANCHOR_FALLBACK'],
                                    'valid_library_file_only_when_valid': True,
                                    'failure_records_required_for_invalid': True,
                                    'mixed_valid_invalid_run_is_complete': True,
                                    'all_invalid_run_is_complete_for_policy_comparison': True,
                                    'halt_never_counts_as_invalid': True,
                                    'same_policy_for_both_arms': True},
 'response_classification.manual_subjective_validity_labels': False,
 'implementation_boundary.workers': 4,
 'implementation_boundary.per_process_blas_threads': 1,
 'implementation_boundary.no_heldout_evaluation_until_all_twenty_terminal_slots_and_selection_lock': True,
 'implementation_boundary.no_calibration_until_all_twenty_terminal_slots': True}


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(path, value, exclusive=False):
    """Fsync before exclusive creation/atomic replace; never partially update JSON."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        if exclusive:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        else:
            fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary is not None:
            temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Nonfinite JSON number")
        return parsed
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_float=finite, parse_constant=reject)


def repo_path(relative, *, must_exist=True):
    """Resolve one repository-relative path without traversal or symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Repository-relative path required")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError("Unsafe repository-relative path")
    root = ROOT.resolve()
    path = root / rel
    walk = root
    for part in rel.parts:
        walk /= part
        if walk.is_symlink():
            raise ValueError("Symlink is not a frozen input/output path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Path escapes repository")
    if must_exist and not path.is_file():
        raise ValueError("Frozen input is missing or not an ordinary file: " + relative)
    return path


def relative_path(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Path traversal is not permitted")
    if not path.is_absolute():
        return str(path.as_posix())
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Path is outside repository") from exc


def source_inventory():
    scripts = ROOT / "experiments/care_replay/scripts"
    protocol = ROOT / PROTOCOL_REL
    paths = list(scripts.glob("*.py")) + list(protocol.rglob("*.py"))
    result = {p.relative_to(ROOT).as_posix() for p in paths}
    if len(result) != len(paths) or len(list(scripts.glob("*.py"))) != LEGACY_COUNT:
        raise ValueError("Original script inventory must contain exactly 116 files")
    required = {PROTOCOL_REL + "/" + n for n in ("common.py", "freeze.py", "generation.py", "evaluation.py")}
    if not required <= result or not any(n.startswith(PROTOCOL_REL + "/tests/") for n in result):
        raise ValueError("V2 production source/test inventory is incomplete")
    for name in result:
        repo_path(name)
    return sorted(result)


def expected_slots(config):
    order = config.get("generation_order")
    if type(config.get("pair_count")) is not int or config["pair_count"] != 10:
        raise ValueError("Exactly ten fresh pairs are required")
    if not isinstance(order, list) or len(order) != 20:
        raise ValueError("Exactly twenty new slots are required")
    for slot in order:
        if (not isinstance(slot, dict) or set(slot) != {"pair_index", "arm"}
                or type(slot["pair_index"]) is not int or slot["arm"] not in ARMS):
            raise ValueError("Invalid generation slot")
    flags = [True] * 5 + [False] * 5
    random.Random(202610208000).shuffle(flags)
    expected_order = [dict(pair_index=pair, arm=arm) for pair, first in enumerate(flags)
                      for arm in (ARMS if first else ARMS[::-1])]
    if order != expected_order:
        raise ValueError("Generation order differs from new balanced seeded order")
    return {f"pair_{slot['pair_index']:02d}_{slot['arm']}" for slot in order}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    for dotted, expected in EXPECTED.items():
        actual = config
        try:
            for part in dotted.split("."):
                actual = actual[part]
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing frozen parameter: " + dotted) from exc
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("Frozen V2 parameter changed: " + dotted)
    auth = config.get("resource_authorization", {})
    for key in ("generation_enabled", "new_model_calls_approved", "new_invalid_output_policy_approved"):
        if auth.get(key) is not True:
            raise ValueError("Explicit production authorization is required: " + key)
    for key, expected in (("approved_pair_count", 10), ("approved_new_logical_requests", 20),
                          ("maximum_new_post_attempts", 20), ("prior_experiment_actual_post_attempts", 11)):
        if type(auth.get(key)) is not int or auth[key] != expected:
            raise ValueError("Authorized request count changed: " + key)
    if not isinstance(auth.get("authorization_reference"), str) or not auth["authorization_reference"].strip():
        raise ValueError("Authorization reference is required")
    money = {"prior_known_cost_usd": PRIOR_COST, "proposed_cumulative_max_cost_usd": Decimal(1),
             "remaining_new_spend_allowance_usd": Decimal("0.9874855"),
             "per_attempt_reservation_usd": Decimal("0.045"),
             "prior_cost_plus_twenty_reservations_usd": Decimal("0.9125145"),
             "approved_max_cost_usd": Decimal(1)}
    for key, expected in money.items():
        value = auth.get(key)
        if isinstance(value, bool) or Decimal(str(value)) != expected:
            raise ValueError("Authorized cumulative budget changed: " + key)
    expected_slots(config)
    calibration = set(range(config["calibration_seed_start"], config["calibration_seed_start"] + 50))
    heldout = set(range(config["heldout_seed_start"], config["heldout_seed_start"] + 100))
    aliases = set(config["paired_design"]["alias_seed_by_pair"].values())
    auxiliary = {config["bootstrap_seed"], config["paired_design"]["order_seed"]}
    if len(calibration | heldout | aliases | auxiliary) != 162:
        raise ValueError("V2 calibration, heldout and generation seeds overlap")
    excluded = config.get("excluded_prior_seed_ranges")
    if not isinstance(excluded, list) or not excluded:
        raise ValueError("Prior seed exclusion ranges are required")
    for bounds in excluded:
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) is not int for v in bounds) or bounds[0] > bounds[1]):
            raise ValueError("Invalid prior seed exclusion range")
        if any(bounds[0] <= seed <= bounds[1] for seed in calibration | heldout | aliases | auxiliary):
            raise ValueError("V2 seeds intersect a prior range")
    for name in (*config["data_files"], config["llm_record"], config["parent_v1_protocol_lock"], config["prior_spend_ledger"]):
        repo_path(name)
    repo_path(config["generation_output_directory"], must_exist=False)
    return calibration, heldout


def prior_context(config):
    parent_path = repo_path(config["parent_v1_protocol_lock"])
    parent = load(parent_path)
    if (parent.get("protocol_id") != "bh_semantic_anonymization_v1"
            or parent.get("status") != "frozen_before_generation"
            or parent.get("model_api_calls_at_freeze") != 0):
        raise ValueError("Original V1 freeze is required")
    scripts = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments/care_replay/scripts").glob("*.py")}
    if len(scripts) != LEGACY_COUNT or set(parent.get("source_sha256", {})) != scripts:
        raise ValueError("Original V1 script inventory changed")
    if set(parent.get("data_sha256", {})) != set(config["data_files"]):
        raise ValueError("Data dependencies differ from V1")
    original_inputs = {config["llm_record"], "uv.lock", "pyproject.toml"}
    if set(parent.get("input_sha256", {})) != original_inputs:
        raise ValueError("Original frozen dependency inputs changed")
    for group in ("source_sha256", "data_sha256", "input_sha256"):
        for relative, digest in parent[group].items():
            if file_hash(repo_path(relative)) != digest:
                raise ValueError("Original V1 bytes changed: " + relative)
    if file_hash(repo_path(config["llm_record"])) != config["llm_record_sha256"]:
        raise ValueError("Historical record hash changed")
    ledger_path = repo_path(config["prior_spend_ledger"])
    ledger = load(ledger_path)
    attempts = ledger.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 11:
        raise ValueError("Expected exactly eleven accounted V1 attempts")
    values = []
    for attempt in attempts:
        raw = attempt.get("cost_usd")
        if isinstance(raw, bool) or raw is None:
            raise ValueError("Prior attempt cost is unresolved")
        value = Decimal(str(raw))
        if not value.is_finite() or value < 0:
            raise ValueError("Invalid prior cost")
        values.append(value)
    if sum(values) != PRIOR_COST or Decimal(str(ledger.get("known_cost_usd"))) != PRIOR_COST:
        raise ValueError("Prior cumulative spend changed; no duplicated initial ledger allowed")
    return parent, ledger


def input_inventory(config):
    return sorted({config["llm_record"], "uv.lock", "pyproject.toml",
                   config["parent_v1_protocol_lock"], config["prior_spend_ledger"]})


def verify_freeze(config_path, lock_path, freeze_dir):
    config_name = relative_path(config_path)
    if config_name != CONFIG_REL:
        raise ValueError("Only the formal V2 production config may execute")
    config_path = repo_path(config_name)
    lock_path = repo_path(relative_path(lock_path))
    freeze_dir = repo_path(relative_path(freeze_dir), must_exist=False)
    if not freeze_dir.is_dir():
        raise ValueError("Freeze directory is missing")
    config, lock = load(config_path), load(lock_path)
    validate_config(config)
    prior_context(config)
    if (lock.get("protocol_id") != config["protocol_id"]
            or lock.get("status") != "frozen_before_generation"
            or type(lock.get("model_api_calls_at_freeze")) is not int
            or lock["model_api_calls_at_freeze"] != 0):
        raise ValueError("V2 pre-generation freeze is required")
    if lock.get("config_sha256") != file_hash(config_path):
        raise ValueError("Production config changed after freeze")
    if file_hash(repo_path(relative_path(freeze_dir / "config.json"))) != lock["config_sha256"]:
        raise ValueError("Frozen config snapshot changed")
    if lock.get("freeze_directory") != relative_path(freeze_dir):
        raise ValueError("Wrong frozen directory")
    if file_hash(repo_path(relative_path(freeze_dir / "protocol_lock.json"))) != file_hash(lock_path):
        raise ValueError("Frozen protocol-lock snapshot changed")
    expected_maps = {"source_sha256": set(source_inventory()), "data_sha256": set(config["data_files"]),
                     "input_sha256": set(input_inventory(config))}
    snapshots = {"source_snapshot": set(), "input_snapshot": set()}
    for group, names in expected_maps.items():
        mapping = lock.get(group)
        if not isinstance(mapping, dict) or set(mapping) != names:
            raise ValueError("Frozen inventory mismatch: " + group)
        for relative, digest in mapping.items():
            if file_hash(repo_path(relative)) != digest:
                raise ValueError("Frozen file hash mismatch: " + relative)
            folder = "source_snapshot" if group == "source_sha256" else "input_snapshot"
            snapshot = repo_path(relative_path(freeze_dir / folder / relative))
            if file_hash(snapshot) != digest:
                raise ValueError("Frozen input/source snapshot mismatch: " + relative)
            snapshots[folder].add(relative)
    for folder, expected_names in snapshots.items():
        directory = freeze_dir / folder
        if {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()} != expected_names:
            raise ValueError("Frozen snapshot file inventory differs: " + folder)
    checks = {"authorization_reference": config["resource_authorization"]["authorization_reference"],
              "parent_v1_protocol_lock": config["parent_v1_protocol_lock"],
              "prior_spend_ledger": config["prior_spend_ledger"],
              "new_authorized_model_calls": 20, "cumulative_max_cost_usd": "1",
              "parent_v1_protocol_lock_sha256": file_hash(repo_path(config["parent_v1_protocol_lock"])),
              "prior_spend_ledger_sha256": file_hash(repo_path(config["prior_spend_ledger"])),
              "prior_known_cost_usd": str(PRIOR_COST), "prior_model_api_calls": 11}
    if any(lock.get(key) != value for key, value in checks.items()):
        raise ValueError("Authorization/parent/billing provenance changed")
    for group, folder, names in (("messages_sha256", "messages", expected_slots(config)),
                                 ("pair_material_sha256", "private_pairs", {f"pair_{p:02d}" for p in range(10)})):
        mapping = lock.get(group)
        if not isinstance(mapping, dict) or set(mapping) != names:
            raise ValueError("Incomplete frozen paired material inventory")
        directory = freeze_dir / folder
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("Frozen material directory is missing or a symlink")
        if {p.name for p in directory.iterdir()} != {name + ".json" for name in names}:
            raise ValueError("Unexpected frozen paired material file inventory")
        for name, digest in mapping.items():
            path = repo_path(relative_path(directory / (name + ".json")))
            if file_hash(path) != digest:
                raise ValueError("Frozen paired material changed: " + name)
    log = repo_path(relative_path(freeze_dir / "offline_validation.log"))
    if file_hash(log) != lock.get("offline_validation_log_sha256"):
        raise ValueError("Validation log changed")
    return config, lock
