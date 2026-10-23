#!/usr/bin/env python3
"""Predeclared crossed-bank/start analysis of a completed offline BH study.

This module never selects policies. It reads the calibration winners already
saved in selection_lock.json, validates complete held-out cells, and evaluates
all random libraries with equal weight. It does not import an experiment or API
client. The two families' primary intervals control familywise error through
Bonferroni; deployment intervals are separately labelled exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BASELINES = ("gp_ucb", "mixed_kernel_gp_ei", "target_acquisition_portfolio")
FAMILIES = ("r1", "r2")
METRICS = ("best_so_far_auc", "final_best", "top10_hit")
PROTOCOL = {
    "analysis_id": "bh_matched_rule_stability_v1",
    "run_protocol_id": "bh_random_rule_controls_scale_v1",
    "calibration_seed_start": 202610183000,
    "heldout_seed_start": 202610184000,
    "random_seed": 2026101802,
    "banks_per_family": 100,
    "skills_per_bank": 5,
    "calibration_seed_count": 50,
    "heldout_seed_count": 100,
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 2026101803,
    "block_count": 5,
    "primary_confidence_level": 0.975,
    "secondary_confidence_level": 0.95,
    "primary_selection_key": "ungated_calibration_winner",
    "secondary_selection_key": "selected_mode",
    "stability_rule": "Both families have a strictly positive adjusted primary CI lower bound and all five primary block means are strictly positive.",
}

COVERAGE_PROTOCOL = {
    **PROTOCOL,
    "analysis_id": "bh_coverage_matched_rule_stability_v1",
    "run_protocol_id": "bh_coverage_matched_rule_controls_v1",
    "calibration_seed_start": 202610185000,
    "heldout_seed_start": 202610186000,
    "random_seed": 2026101804,
    "bootstrap_seed": 2026101805,
}
PROTOCOLS = {protocol["run_protocol_id"]: protocol for protocol in (PROTOCOL, COVERAGE_PROTOCOL)}
COVERAGE_MATCHING = {
    "absolute_fraction_tolerance": 0.005,
    "match_ordered_rule_support": True,
    "match_joint_activation_histogram": True,
    "sampling": "original_null_conditioned_on_public_coverage",
    "identity_and_duplicates": "allowed",
}

ANALYSIS_SOURCE = "experiments/care_replay/scripts/analyze_matched_rule_stability.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def expected_analysis_plan(protocol):
    return {
        "bootstrap_seed": protocol["bootstrap_seed"],
        "bootstrap_replicates": protocol["bootstrap_replicates"],
        "resampling": "independent_library_and_shared_heldout_seed_axes",
        "primary_metric": "best_so_far_auc",
        "primary_policy": protocol["primary_selection_key"],
        "primary_family_interval_level": protocol["primary_confidence_level"],
        "familywise_error_control": "Bonferroni for two primary family comparisons; approximate percentile bootstrap",
        "heldout_blocks": protocol["block_count"],
        "positive_block_requirement": "all five consecutive equal-size blocks in both families",
        "stable_advantage": "both primary family adjusted interval lower bounds strictly positive AND all five block means strictly positive in both families",
        "secondary_policy": protocol["secondary_selection_key"],
        "secondary_intervals": "exploratory unadjusted 95 percent crossed bootstrap",
        "scope": "Conditional on frozen calibration selections, historical LLM library and known BH pool; not new tasks or new LLM proposals.",
    }


def protocol_for(config):
    protocol_id = config.get("protocol_id")
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unsupported predeclared run protocol: {protocol_id!r}")
    return dict(PROTOCOLS[protocol_id])


def validate_coverage_config(config):
    """Keep conditioned and unconditioned null protocols unambiguous."""
    if config.get("protocol_id") == COVERAGE_PROTOCOL["run_protocol_id"]:
        actual = config.get("coverage_matching")
        if (not isinstance(actual, dict) or set(actual) != set(COVERAGE_MATCHING)
                or any(type(actual[key]) is not type(expected) or actual[key] != expected
                       for key, expected in COVERAGE_MATCHING.items())):
            raise ValueError("Coverage-matching configuration differs from the predeclared protocol")
    elif config.get("coverage_matching") is not None:
        raise ValueError("The original random-rule protocol cannot include coverage matching")


def validate_coverage_provenance(manifest, frozen):
    validate_coverage_config(manifest["config"])
    if manifest["config"]["protocol_id"] == COVERAGE_PROTOCOL["run_protocol_id"]:
        diagnostics = manifest.get("coverage_matching_diagnostics")
        if not isinstance(diagnostics, dict) or not diagnostics:
            raise ValueError("Coverage-matched run requires nonempty public coverage diagnostics")
        if frozen.get("coverage_matching_sha256") != digest(diagnostics):
            raise ValueError("Frozen coverage-matching diagnostics hash does not match manifest")


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def skill_mode(skill: dict[str, Any]) -> str:
    return "llm_semantic_" + skill["skill_id"]


def validate_inputs(manifest, lock, rows):
    """Check every held-out cell and frozen choice; do not use outcomes to choose."""
    config = manifest["config"]
    heldout = list(range(config["heldout_seed_start"],
                         config["heldout_seed_start"] + config["heldout_seed_count"]))
    calibration = set(range(config["calibration_seed_start"],
                            config["calibration_seed_start"] + config["calibration_seed_count"]))
    if not heldout or calibration.intersection(heldout):
        raise ValueError("Empty held-out seeds or calibration/held-out overlap")
    libraries = manifest["libraries"]
    banks = config["null_replicates"]
    expected_libraries = {"llm", *(f"{family}_{index:03d}"
                                  for family in FAMILIES for index in range(1, banks + 1))}
    if set(libraries) != expected_libraries or set(lock["selections"]) != expected_libraries:
        raise ValueError("Library/selection IDs do not match the planned complete families")
    bank_modes = {key: tuple(skill_mode(skill) for skill in bank)
                  for key, bank in libraries.items()}
    if not bank_modes["llm"] or any(len(modes) != len(bank_modes["llm"])
                                    for modes in bank_modes.values()):
        raise ValueError("Every library must have the same nonzero selection budget")
    all_modes = [*BASELINES, *(mode for modes in bank_modes.values() for mode in modes)]
    if len(all_modes) != len(set(all_modes)):
        raise ValueError("Policy mode IDs must be globally unique")
    choices = lock["selections"]
    anchor = choices["llm"]["selection"]["target_anchor_mode"]
    if anchor not in BASELINES:
        raise ValueError("Unknown calibration-selected baseline anchor")
    for library_id, choice in choices.items():
        if choice["selection"]["target_anchor_mode"] != anchor:
            raise ValueError("All libraries must use the same calibration-selected anchor")
        if choice["ungated_calibration_winner"] not in bank_modes[library_id]:
            raise ValueError("Ungated winner must belong to its own frozen library")
        if choice["selected_mode"] not in (*bank_modes[library_id], anchor):
            raise ValueError("Deployment must use its own library or the frozen anchor")
    expected = {(mode, seed) for mode in all_modes for seed in heldout}
    index = {}
    for row in rows:
        if row["split"] != "heldout":
            raise ValueError("Analysis accepts held-out rows only")
        raw_seed = row["seed"]
        if isinstance(raw_seed, bool) or isinstance(raw_seed, float):
            raise ValueError("Seeds must be integers")
        seed = int(raw_seed)
        key = (row["mode"], seed)
        if key not in expected or key in index:
            raise ValueError("Unexpected or duplicate held-out cell")
        values = tuple(float(row[metric]) for metric in METRICS)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Nonfinite held-out metric")
        if values[2] not in (0.0, 1.0):
            raise ValueError("top10_hit must be binary")
        index[key] = values
    if set(index) != expected:
        raise ValueError("Incomplete held-out cells; no partial effect estimate is allowed")
    return heldout, bank_modes, index


def delta_matrix(index, selections, bank_ids, seeds, selection_key, metric):
    """Rows are independent library draws; columns are shared replay starts."""
    metric_index = METRICS.index(metric)
    llm_mode = selections["llm"][selection_key]
    llm = np.array([index[(llm_mode, seed)][metric_index] for seed in seeds])
    controls = np.array([[index[(selections[bank][selection_key], seed)][metric_index]
                          for seed in seeds] for bank in bank_ids])
    return llm[np.newaxis, :] - controls


def crossed_bootstrap_means(matrix, bank_counts, seed_counts):
    """Apply product resampling weights, preserving the one shared LLM trace.

    Resampling individual cells or independently resampling LLM per bank would
    spuriously reduce the common-start contribution. The same seed counts apply
    to every bank and both sides of each paired difference.
    """
    matrix = np.asarray(matrix, dtype=float)
    bank_counts = np.asarray(bank_counts, dtype=float)
    seed_counts = np.asarray(seed_counts, dtype=float)
    if matrix.ndim != 2 or bank_counts.ndim != 2 or seed_counts.ndim != 2:
        raise ValueError("Matrix and bootstrap count arrays must be two-dimensional")
    if (bank_counts.shape[1] != matrix.shape[0]
            or seed_counts.shape[1] != matrix.shape[1]
            or bank_counts.shape[0] != seed_counts.shape[0]):
        raise ValueError("Crossed bootstrap count dimensions do not match")
    if (not all(np.isfinite(array).all() for array in (matrix, bank_counts, seed_counts))
            or np.any(bank_counts < 0) or np.any(seed_counts < 0)):
        raise ValueError("Nonfinite matrix or invalid bootstrap counts")
    if (not np.all(bank_counts.sum(axis=1) == matrix.shape[0])
            or not np.all(seed_counts.sum(axis=1) == matrix.shape[1])):
        raise ValueError("Bootstrap must resample each axis at its original size")
    return np.einsum("bi,ij,bj->b", bank_counts, matrix, seed_counts, optimize=True) / matrix.size


def describe_deltas(matrix, seeds, block_count=5):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(seeds) or not matrix.size:
        raise ValueError("Delta matrix shape does not match held-out seeds")
    if len(seeds) % block_count or seeds != sorted(set(seeds)):
        raise ValueError("Use ordered unique seeds and equal prespecified blocks")
    bank_means = matrix.mean(axis=1)
    quantiles = np.quantile(bank_means, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], method="linear")
    size = len(seeds) // block_count
    blocks = []
    for block in range(block_count):
        begin, end = block * size, (block + 1) * size
        blocks.append({"block": block + 1, "seed_start": seeds[begin],
                       "seed_end": seeds[end - 1], "seed_count": size,
                       "mean_delta": float(matrix[:, begin:end].mean())})
    return {
        "mean_delta": float(matrix.mean()),
        "positive_bank_fraction": float(np.mean(bank_means > 0)),
        "zero_bank_fraction": float(np.mean(bank_means == 0)),
        "negative_bank_fraction": float(np.mean(bank_means < 0)),
        "bank_mean_quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                       map(float, quantiles))),
        "bank_mean_deltas": list(map(float, bank_means)),
        "ordered_seed_blocks": blocks,
        "all_block_means_positive": all(block["mean_delta"] > 0 for block in blocks),
    }


def percentile_interval(draws, confidence_level):
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], method="linear")
    return {"confidence_level": confidence_level, "lower": float(lower), "upper": float(upper),
            "method": "crossed bank/start percentile bootstrap; linear sample quantiles"}


def analyze(manifest, lock, rows, *, protocol=None):
    """Pure analysis function; alternate small protocols are for synthetic tests."""
    config = manifest["config"]
    registered = protocol_for(config)
    protocol = dict(registered if protocol is None else protocol)
    validate_coverage_config(config)
    if config.get("target_dataset") != "real_buchwald_hartwig":
        raise ValueError("This predeclared analysis applies only to the BH dataset")
    if config.get("protocol_id") != protocol["run_protocol_id"]:
        raise ValueError("Run protocol ID does not match the predeclared analysis")
    if config.get("stability_analysis") != expected_analysis_plan(protocol):
        raise ValueError("Frozen analysis plan differs from this analysis implementation")
    for key, expected in (("null_replicates", protocol["banks_per_family"]),
                          ("calibration_seed_count", protocol["calibration_seed_count"]),
                          ("heldout_seed_count", protocol["heldout_seed_count"]),
                          ("calibration_seed_start", protocol["calibration_seed_start"]),
                          ("heldout_seed_start", protocol["heldout_seed_start"]),
                          ("random_seed", protocol["random_seed"])):
        if config[key] != expected:
            raise ValueError(f"Predeclared protocol mismatch for {key}: expected {expected}")
    seeds, bank_modes, index = validate_inputs(manifest, lock, rows)
    if len(bank_modes["llm"]) != protocol["skills_per_bank"]:
        raise ValueError("Predeclared per-library skill count changed")
    selections = lock["selections"]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    draws = protocol["bootstrap_replicates"]
    # Shared start weights also preserve dependence across families/endpoints.
    seed_counts = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=draws)
    families = {}
    for family in FAMILIES:
        bank_ids = sorted(key for key in bank_modes if key.startswith(family + "_"))
        bank_counts = rng.multinomial(len(bank_ids), np.full(len(bank_ids), 1 / len(bank_ids)), size=draws)
        result = {"bank_ids": bank_ids,
                  "frozen_choices": {bank: {"ungated": selections[bank]["ungated_calibration_winner"],
                                            "deployed": selections[bank]["selected_mode"]}
                                     for bank in bank_ids}}
        for endpoint, key, confidence in (
            ("primary_ungated_auc", protocol["primary_selection_key"], protocol["primary_confidence_level"]),
            ("secondary_deployed_auc", protocol["secondary_selection_key"], protocol["secondary_confidence_level"]),
        ):
            matrix = delta_matrix(index, selections, bank_ids, seeds, key, "best_so_far_auc")
            stats = describe_deltas(matrix, seeds, protocol["block_count"])
            bootstrapped = crossed_bootstrap_means(matrix, bank_counts, seed_counts)
            stats["interval"] = percentile_interval(bootstrapped, confidence)
            stats["multiplicity"] = ("Bonferroni-adjusted across two primary families; familywise 95%"
                                      if endpoint.startswith("primary") else
                                      "Exploratory, unadjusted 95%; excluded from stable-advantage decision")
            result[endpoint] = stats
        result["descriptive_other_metrics"] = {
            label: {metric: describe_deltas(
                delta_matrix(index, selections, bank_ids, seeds, key, metric),
                seeds, protocol["block_count"])
                for metric in ("final_best", "top10_hit")}
            for label, key in (("ungated", protocol["primary_selection_key"]),
                               ("deployed", protocol["secondary_selection_key"]))
        }
        primary = result["primary_ungated_auc"]
        result["primary_family_stability_pass"] = bool(
            primary["interval"]["lower"] > 0 and primary["all_block_means_positive"])
        families[family] = result
    return {
        "analysis_protocol": protocol,
        "coverage_matching": config.get("coverage_matching"),
        "direction": "Positive differences favour the frozen LLM library: LLM minus random library.",
        "primary_estimand": "Mean held-out AUC difference between the LLM calibration ungated winner and a random bank's calibration ungated winner, averaging all independently drawn banks and shared held-out starts.",
        "secondary_estimand": "The corresponding difference between gate-deployed policies, including baseline fallback; reported separately and excluded from the primary decision.",
        "auc_units": "Original yield percentage points, averaged across reveal-round best-so-far values.",
        "bootstrap_unit": "Crossed independent bank draws and shared held-out starts; not individual trajectories or cells.",
        "llm_frozen_choices": {"ungated": selections["llm"]["ungated_calibration_winner"],
                               "deployed": selections["llm"]["selected_mode"]},
        "families": families,
        "stable_advantage": all(result["primary_family_stability_pass"] for result in families.values()),
        "evidence_boundary": "Conditional on this one historical LLM library, the frozen calibration selection, the null generators and known BH candidate pool. Calibration is not reselected or bootstrapped. These intervals do not cover new LLM generations, model families or new scientific tasks. Consecutive start blocks are a stability check, not five independent replications. No best-of-bank selection is performed.",
        "numpy_version": np.__version__,
    }


def render_text(result):
    lines = ["BH 随机规则对照：预设稳定性分析", "",
             "差值均为 LLM − 随机库；正值有利于 LLM。每个随机库先在校准集选定策略，留出集不再选择。",
             "主终点比较未过 gate 限制的校准赢家；每族 97.5% bootstrap 区间对两族作 Bonferroni 调整。"]
    if result.get("coverage_matching") is not None:
        lines.append("本协议额外约束公开候选池中的逐条规则覆盖比例和完整联合激活直方图；各比例的绝对容差为 0.005（0.5 个百分点），允许恒等映射与重复抽样。")
    for family in FAMILIES:
        item = result["families"][family]
        primary, secondary = item["primary_ungated_auc"], item["secondary_deployed_auc"]
        ci, secondary_ci = primary["interval"], secondary["interval"]
        blocks = ", ".join(f"{block['mean_delta']:+.4f}" for block in primary["ordered_seed_blocks"])
        lines.extend(["", f"{family.upper()} 主终点：AUC 均差 {primary['mean_delta']:+.4f}，调整区间 [{ci['lower']:+.4f}, {ci['upper']:+.4f}]；正差库比例 {primary['positive_bank_fraction']:.1%}。",
                      f"五个预设连续起点块均差：{blocks}。",
                      f"部署次终点：AUC 均差 {secondary['mean_delta']:+.4f}，探索性未调整 95% 区间 [{secondary_ci['lower']:+.4f}, {secondary_ci['upper']:+.4f}]。"])
    lines.extend(["", ("达到" if result["stable_advantage"] else "未达到") +
                  "预设稳定优势标准：两族主终点调整区间下界均大于 0，且两族各自五块均差全部大于 0。",
                  "未达到标准不等于证明两种方法等效。部署次终点不参与这项判定。",
                  "区间同时重抽随机库和共享留出起点，保留同一 LLM 的共享轨迹；只针对本历史规则库、冻结校准选择和 BH 池，不代表新模型生成或新任务的泛化证据。",
                  "final_best 与 top10_hit 仅作描述，详见 JSON；连续起点块不作为独立重复实验。"])
    return "\n".join(lines) + "\n"


def analyze_run(run_dir: Path, output_prefix: Path):
    json_path, text_path = Path(str(output_prefix) + ".json"), Path(str(output_prefix) + ".txt")
    if json_path.exists() or text_path.exists():
        raise ValueError("Analysis outputs already exist; use a new output prefix")
    paths = {name: run_dir / name for name in (
        "manifest.json", "selection_lock.json", "calibration_metrics.csv",
        "heldout_metrics.csv", "summary.json", "integrity.json", "frozen_protocol.json")}
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    lock = json.loads(paths["selection_lock.json"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    integrity = json.loads(paths["integrity.json"].read_text(encoding="utf-8"))
    frozen = json.loads(paths["frozen_protocol.json"].read_text(encoding="utf-8"))
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if summary.get("status") != "complete":
        raise ValueError("Only a completed study can be analyzed")
    if manifest.get("model_api_calls") != 0 or summary.get("model_api_calls") != 0:
        raise ValueError("This analysis requires a recorded offline run with zero model API calls")
    for name, actual in hashes.items():
        if name != "integrity.json" and integrity.get(name) != actual:
            raise ValueError(f"Run integrity mismatch: {name}")
    if (lock["manifest_sha256"] != hashes["manifest.json"]
            or lock["calibration_metrics_sha256"] != hashes["calibration_metrics.csv"]
            or summary["selection_lock_sha256"] != hashes["selection_lock.json"]):
        raise ValueError("Frozen selection provenance hashes do not match")
    if (manifest.get("frozen_protocol_sha256") != hashes["frozen_protocol.json"]
            or lock.get("frozen_protocol_sha256") != hashes["frozen_protocol.json"]):
        raise ValueError("Frozen protocol copy does not match manifest/selection lock")
    validate_coverage_provenance(manifest, frozen)
    expected_frozen = {key: manifest[key] for key in ("config_sha256", "source_sha256", "data_sha256")}
    expected_frozen.update(libraries_sha256=digest(manifest["libraries"]),
                           execution_aliases_sha256=digest(manifest["execution_aliases"]))
    if any(frozen.get(key) != value for key, value in expected_frozen.items()):
        raise ValueError("Frozen protocol source/data/config/library provenance mismatch")
    if manifest["source_sha256"].get(ANALYSIS_SOURCE) != file_hash(Path(__file__)):
        raise ValueError("Analysis source differs from the version frozen before this run")
    with paths["heldout_metrics.csv"].open(newline="", encoding="utf-8") as handle:
        result = analyze(manifest, lock, csv.DictReader(handle))
    result["provenance"] = {"run_dir": str(run_dir.resolve()), "input_sha256": hashes,
                            "analysis_source_sha256": file_hash(Path(__file__)),
                            "calibration_values_read": False, "policy_reselection": False}
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    text_path.write_text(render_text(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_dir, args.output_prefix)
    print(render_text(result))


if __name__ == "__main__":
    main()
