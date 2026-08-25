#!/usr/bin/env python3
"""Falsify source-outcome transfer with matched outcome permutations.

The LLM-compiled source/target role map, kernel patches, target-only anchor,
target budget, and target seeds stay fixed.  The only intervention is whether
measured source outcomes remain attached to their original source candidates
or are permuted across those candidates.  Sequential transfer mass is fixed to
zero so the experiment isolates the source-informed initial design.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import platform
import random
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import llm_semantic_skills as semantic
import run_calibrated_source_outcome_transfer as calibrated
import run_llm_kernel_skill_evolution as evolution
import run_llm_transfer_router as outcome_router
import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("final_best", "best_so_far_auc", "top10_hit")
BASELINE = "target_only_anchor"
TRUE_OUTCOMES = "true_source_outcomes"
CHECKPOINT_SCHEMA = "care.source_outcome_falsification_job/v1"


_RUNTIME_CACHE: dict[str, tuple[Any, ...]] = {}


def resolve_path(value: str) -> Path:
    return (ROOT / value).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def verify_sha256(path: Path) -> bool:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not path.exists() or not sidecar.exists():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == file_sha256(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    write_sha256(path)


def implementation_fingerprint() -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(),
        "calibrated": Path(calibrated.__file__).resolve(),
        "evolution": Path(evolution.__file__).resolve(),
        "outcome_router": Path(outcome_router.__file__).resolve(),
        "replay": Path(replay.__file__).resolve(),
        "semantic": Path(semantic.__file__).resolve(),
        "surrogate": Path(surrogate.__file__).resolve(),
        "transfer": Path(transfer.__file__).resolve(),
        "weighted": Path(weighted.__file__).resolve(),
    }
    file_hashes = {name: file_sha256(path) for name, path in files.items()}
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": (
            None
            if surrogate.np is None
            else str(surrogate.np.__version__)
        ),
        "gp_backend": surrogate.gp_backend_name(),
    }
    combined = hashlib.sha256(
        json.dumps(
            {"files": file_hashes, "runtime_versions": runtime_versions},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "sha256": combined,
        "files": file_hashes,
        "runtime_versions": runtime_versions,
    }


def permuted_source_observations(
    observed: list[replay.Candidate],
    permutation_seed: int,
) -> list[replay.Candidate]:
    """Preserve source features and outcome marginals while breaking their link."""

    if len(observed) < 2:
        return list(observed)
    original = [candidate.objective_value for candidate in observed]
    permuted = list(original)
    random.Random(permutation_seed).shuffle(permuted)
    if permuted == original:
        permuted = permuted[1:] + permuted[:1]
    return [
        replace(candidate, objective_value=outcome)
        for candidate, outcome in zip(observed, permuted)
    ]


def condition_name(permutation_seed: int) -> str:
    return f"permuted_source_outcomes_{permutation_seed}"


def normalized_pair(pair: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    policy = pair.get("frozen_policy", {})
    return {
        **pair,
        "initial_observations": int(
            policy.get("initial_observations", protocol["initial_observations"])
        ),
        "reveal_rounds": int(
            policy.get("reveal_rounds", protocol["reveal_rounds"])
        ),
        "source_initial_strategy": str(
            policy.get("source_initial_strategy", protocol["source_initial_strategy"])
        ),
    }


def runtime_components(
    pair: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[
    replay.DatasetAdapter,
    replay.DatasetAdapter,
    tuple[evolution.KernelSkillPatch, ...],
    semantic.SemanticSkill | None,
    tuple[float, ...],
]:
    source = replay.DATASET_BUILDERS[pair["source_dataset"]]()
    target = replay.DATASET_BUILDERS[pair["target_dataset"]]()
    patch_record = json.loads(resolve_path(pair["llm_record"]).read_text(encoding="utf-8"))
    patches = evolution.normalize_patches(
        {"patches": patch_record.get("normalized_patches", [])},
        target.decision_columns,
        max_patches=100,
    )
    if not patches:
        raise RuntimeError(f"No executable patches for {pair['pair_id']}")
    target_record = json.loads(
        resolve_path(pair["target_llm_record"]).read_text(encoding="utf-8")
    )
    target_skill = calibrated.resolve_target_llm_skill(
        target_record,
        target,
        pair["target_llm_mode"],
    )
    scales = weighted.parse_scales(
        str(protocol.get("fixed_ensemble_scales", "0.5,1,1.5,2,3,4"))
    )
    return source, target, patches, target_skill, scales


def cached_runtime_components(
    pair: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[
    replay.DatasetAdapter,
    replay.DatasetAdapter,
    tuple[evolution.KernelSkillPatch, ...],
    semantic.SemanticSkill | None,
    list[replay.Candidate],
    dict[str, replay.Candidate],
    dict[str, surrogate.FeatureRecord],
    set[str],
]:
    """Cache immutable per-pair state inside each worker process."""

    key = str(pair["pair_id"])
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    source, target, patches, target_skill, _scales = runtime_components(pair, protocol)
    source_observed = transfer.source_observations(
        source,
        int(protocol["source_seed"]),
        int(pair["source_observations"]),
    )
    by_id = {candidate.candidate_id: candidate for candidate in target.candidates}
    features_by_id = {
        candidate.candidate_id: surrogate.candidate_features(target, candidate)
        for candidate in target.candidates
    }
    top10_ids = {
        candidate.candidate_id
        for candidate in sorted(
            target.candidates,
            key=lambda item: item.objective_value,
            reverse=True,
        )[:10]
    }
    value = (
        source,
        target,
        patches,
        target_skill,
        source_observed,
        by_id,
        features_by_id,
        top10_ids,
    )
    _RUNTIME_CACHE[key] = value
    return value


def run_condition(
    *,
    condition: str,
    source: replay.DatasetAdapter,
    target: replay.DatasetAdapter,
    source_observed: list[replay.Candidate],
    pair: dict[str, Any],
    protocol: dict[str, Any],
    patches: tuple[evolution.KernelSkillPatch, ...],
    target_skill: semantic.SemanticSkill | None,
    target_seed: int,
    source_initial_strategy: str,
    by_id: dict[str, replay.Candidate],
    features_by_id: dict[str, surrogate.FeatureRecord],
    top10_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = replay.make_task(
        target,
        int(pair["initial_observations"]),
        int(pair["reveal_rounds"]),
    )
    initial_observed = calibrated.target_llm_initial_observations(
        target,
        task,
        target_seed,
        pair["target_llm_mode"],
        target_skill,
    )
    anchor_scorer = calibrated.target_llm_anchor_scorer(
        target,
        task,
        pair["target_llm_mode"],
        target_skill,
        float(protocol.get("gp_beta", 1.5)),
        float(protocol.get("gp_xi", 0.01)),
        float(protocol.get("numeric_length_scale", 0.35)),
        float(protocol.get("categorical_length_scale", 3.0)),
        float(protocol.get("gp_noise", 0.05)),
    )
    role_map = transfer.descriptor_transfer_role_map_for(
        source.dataset_id,
        target.dataset_id,
    )
    card = transfer.compile_transfer_card(
        source,
        target,
        source_observed,
        role_map,
        float(protocol.get("discount", 0.65)),
        int(protocol.get("min_source_support", 3)),
    )
    source_priors: dict[str, dict[str, float]] = {}
    additive_priors: dict[str, dict[str, float]] = {}
    interaction_priors: dict[str, dict[str, float]] = {}
    if source_initial_strategy != "matched":
        for patch in patches:
            mode = evolution.patch_mode(patch)
            source_priors[mode], _ = outcome_router.aligned_source_prior(
                source_observed,
                target,
                card,
                patch,
            )
            additive_priors[mode], _ = outcome_router.source_additive_outcome_prior(
                source_observed,
                target,
                card,
                patch,
            )
            interaction_priors[mode], _ = outcome_router.source_interaction_prior(
                source_observed,
                target,
                card,
                patch,
            )
    observed, initial_design = outcome_router.source_informed_initial_observations(
        target,
        initial_observed,
        source_priors,
        additive_priors,
        interaction_priors,
        patches,
        task.initial_observations,
        source_initial_strategy,
    )
    observed_ids = {candidate.candidate_id for candidate in observed}
    selected_top10 = any(candidate.candidate_id in top10_ids for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []
    for round_index in range(task.reveal_budget):
        scores, anchor_diagnostics = anchor_scorer(
            observed_ids,
            observed,
            features_by_id,
            round_index,
        )
        selected_id = replay.top_candidate(scores)
        selected = by_id[selected_id]
        observed.append(selected)
        observed_ids.add(selected_id)
        selected_top10 = selected_top10 or selected_id in top10_ids
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append({
            "dataset_id": target.dataset_id,
            "source_dataset_id": source.dataset_id,
            "seed": target_seed,
            "round_index": round_index,
            "condition": condition,
            "selected_candidate": selected_id,
            "revealed_value": selected.objective_value,
            "best_so_far": best_so_far,
            "source_initial_design": initial_design,
            "target_anchor": pair["target_llm_mode"],
            "anchor_diagnostics": anchor_diagnostics,
            "evidence_boundary": (
                "Source outcomes may change only the initial design. All sequential "
                "choices use the matched frozen target-only anchor."
            ),
        })
    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "top10_hit": int(selected_top10),
    }
    row = {
        "pair_id": pair["pair_id"],
        "source_dataset": source.dataset_id,
        "target_dataset": target.dataset_id,
        "target_seed": target_seed,
        "condition": condition,
        **{metric: float(metrics[metric]) for metric in METRICS},
        "source_outcome_active": bool(initial_design.get("source_outcome_active")),
        "source_probe_ids": "|".join(initial_design.get("source_probe_ids", [])),
        "selected_initial_ids": "|".join(
            initial_design.get("selected_initial_ids", [])
        ),
    }
    return row, audit


def run_target_seed(
    pair: dict[str, Any],
    protocol: dict[str, Any],
    target_seed: int,
    permutation_seeds: list[int],
    retain_audit: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pair = normalized_pair(pair, protocol)
    (
        source,
        target,
        patches,
        target_skill,
        source_observed,
        by_id,
        features_by_id,
        top10_ids,
    ) = cached_runtime_components(pair, protocol)
    rows: list[dict[str, Any]] = []
    audits: dict[str, list[dict[str, Any]]] = {}
    baseline, baseline_audit = run_condition(
        condition=BASELINE,
        source=source,
        target=target,
        source_observed=source_observed,
        pair=pair,
        protocol=protocol,
        patches=patches,
        target_skill=target_skill,
        target_seed=target_seed,
        source_initial_strategy="matched",
        by_id=by_id,
        features_by_id=features_by_id,
        top10_ids=top10_ids,
    )
    rows.append(baseline)
    true_row, true_audit = run_condition(
        condition=TRUE_OUTCOMES,
        source=source,
        target=target,
        source_observed=source_observed,
        pair=pair,
        protocol=protocol,
        patches=patches,
        target_skill=target_skill,
        target_seed=target_seed,
        source_initial_strategy=pair["source_initial_strategy"],
        by_id=by_id,
        features_by_id=features_by_id,
        top10_ids=top10_ids,
    )
    rows.append(true_row)
    if retain_audit:
        audits[BASELINE] = baseline_audit
        audits[TRUE_OUTCOMES] = true_audit
    for permutation_seed in permutation_seeds:
        condition = condition_name(permutation_seed)
        permuted = permuted_source_observations(source_observed, permutation_seed)
        row, audit = run_condition(
            condition=condition,
            source=source,
            target=target,
            source_observed=permuted,
            pair=pair,
            protocol=protocol,
            patches=patches,
            target_skill=target_skill,
            target_seed=target_seed,
            source_initial_strategy=pair["source_initial_strategy"],
            by_id=by_id,
            features_by_id=features_by_id,
            top10_ids=top10_ids,
        )
        rows.append(row)
        if retain_audit:
            audits[condition] = audit
    return rows, audits


def execute_job(
    job: tuple[dict[str, Any], dict[str, Any], int, list[int], bool],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pair, protocol, target_seed, permutation_seeds, retain_audit = job
    return run_target_seed(
        pair,
        protocol,
        target_seed,
        permutation_seeds,
        retain_audit,
    )


def delta_summary(values: Iterable[float]) -> dict[str, float]:
    data = list(values)
    if not data:
        raise ValueError("Cannot summarize an empty delta list")
    value_mean = mean(data)
    if len(data) < 2:
        standard_error = 0.0
    else:
        variance = sum((value - value_mean) ** 2 for value in data) / (len(data) - 1)
        standard_error = math.sqrt(variance / len(data))
    return {
        "n": len(data),
        "mean": value_mean,
        "normal_95ci_low": value_mean - 1.96 * standard_error,
        "normal_95ci_high": value_mean + 1.96 * standard_error,
        "win_rate": sum(value > 1e-12 for value in data) / len(data),
        "non_loss_rate": sum(value >= -1e-12 for value in data) / len(data),
    }


def build_pair_summary(
    pair: dict[str, Any],
    rows: list[dict[str, Any]],
    permutation_seeds: list[int],
) -> dict[str, Any]:
    by_condition_seed = {
        (str(row["condition"]), int(row["target_seed"])): row
        for row in rows
    }
    target_seeds = sorted({int(row["target_seed"]) for row in rows})
    true_vs_baseline: dict[str, Any] = {}
    true_vs_mean_permutation: dict[str, Any] = {}
    permutation_null: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for metric in METRICS:
        true_vs_baseline[metric] = delta_summary(
            float(by_condition_seed[(TRUE_OUTCOMES, seed)][metric])
            - float(by_condition_seed[(BASELINE, seed)][metric])
            for seed in target_seeds
        )
        true_vs_mean_permutation[metric] = delta_summary(
            float(by_condition_seed[(TRUE_OUTCOMES, seed)][metric])
            - mean(
                float(by_condition_seed[(condition_name(pseed), seed)][metric])
                for pseed in permutation_seeds
            )
            for seed in target_seeds
        )
        for pseed in permutation_seeds:
            permutation_null[metric].append(mean(
                float(by_condition_seed[(condition_name(pseed), seed)][metric])
                - float(by_condition_seed[(BASELINE, seed)][metric])
                for seed in target_seeds
            ))
    randomization_tests = {}
    for metric in METRICS:
        observed = float(true_vs_baseline[metric]["mean"])
        null = permutation_null[metric]
        randomization_tests[metric] = {
            "observed_true_minus_baseline": observed,
            "permutation_minus_baseline_means": null,
            "one_sided_empirical_p": (
                1 + sum(value >= observed for value in null)
            ) / (len(null) + 1),
        }
    return {
        "pair_id": pair["pair_id"],
        "source_dataset": pair["source_dataset"],
        "target_dataset": pair["target_dataset"],
        "target_seed_count": len(target_seeds),
        "permutation_count": len(permutation_seeds),
        "true_outcomes_minus_target_only": true_vs_baseline,
        "true_outcomes_minus_mean_permutation": true_vs_mean_permutation,
        "randomization_test": randomization_tests,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_results(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Source-outcome permutation falsification",
        "",
        "The LLM-compiled role map, frozen patches, target anchor, target seeds, and ",
        "budget are matched. Only the assignment of measured source outcomes to source ",
        "candidates is permuted. Sequential transfer mass is zero, so this audit isolates ",
        "source-informed initial design.",
        "",
        "| Source -> target | True vs target AUC | True vs permuted AUC | Empirical p |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pair in report["pairs"]:
        true_target = pair["true_outcomes_minus_target_only"]["best_so_far_auc"]
        true_perm = pair["true_outcomes_minus_mean_permutation"]["best_so_far_auc"]
        p_value = pair["randomization_test"]["best_so_far_auc"]["one_sided_empirical_p"]
        lines.append(
            f"| {pair['source_dataset']} -> {pair['target_dataset']} "
            f"| {true_target['mean']:+.3f} "
            f"[{true_target['normal_95ci_low']:+.3f}, {true_target['normal_95ci_high']:+.3f}] "
            f"| {true_perm['mean']:+.3f} "
            f"[{true_perm['normal_95ci_low']:+.3f}, {true_perm['normal_95ci_high']:+.3f}] "
            f"| {p_value:.4f} |"
        )
    lines.extend([
        "",
        "The paired confidence intervals quantify uncertainty across target seeds, conditional ",
        "on the tested outcome assignments. They do not test whether the true source assignment ",
        "is exceptional among matched permutations. That causal question is evaluated by the ",
        "empirical randomization p-value. Under the frozen 19-permutation audit, none of the ",
        "tested routes rejects the assignment-null at p < 0.05. The result therefore does not ",
        "establish source feature-outcome learning beyond the outcome marginal and initialization ",
        "mechanism, and it does not substitute for a fresh-task or prospective experiment.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def job_checkpoint_path(job_dir: Path, pair_id: str, target_seed: int) -> Path:
    safe_pair_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in pair_id
    )
    return job_dir / f"{safe_pair_id}__seed_{target_seed}.json"


def load_job_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    pair_id: str,
    target_seed: int,
    permutation_seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]] | None:
    if not verify_sha256(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_conditions = {
        BASELINE,
        TRUE_OUTCOMES,
        *(condition_name(seed) for seed in permutation_seeds),
    }
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("config_sha256") != config_sha256
        or payload.get("implementation_sha256") != implementation_sha256
        or payload.get("pair_id") != pair_id
        or int(payload.get("target_seed", -1)) != target_seed
        or not isinstance(rows, list)
        or {str(row.get("condition")) for row in rows} != expected_conditions
        or any(
            row.get("pair_id") != pair_id
            or int(row.get("target_seed", -1)) != target_seed
            for row in rows
        )
    ):
        return None
    audits = payload.get("audits", {})
    if not isinstance(audits, dict):
        return None
    return rows, audits


def write_job_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    pair_id: str,
    target_seed: int,
    rows: list[dict[str, Any]],
    audits: dict[str, list[dict[str, Any]]],
) -> None:
    write_json_atomic(path, {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "pair_id": pair_id,
        "target_seed": target_seed,
        "rows": rows,
        "audits": audits,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-target-seeds", type=int)
    parser.add_argument("--max-permutations", type=int)
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse checksum-verified per-seed checkpoints in the output directory.",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    target_seeds = list(range(
        int(protocol["target_seed_start"]),
        int(protocol["target_seed_start"]) + int(protocol["target_seed_count"]),
    ))
    permutation_seeds = [int(value) for value in protocol["permutation_seeds"]]
    if args.max_target_seeds is not None:
        target_seeds = target_seeds[: args.max_target_seeds]
    if args.max_permutations is not None:
        permutation_seeds = permutation_seeds[: args.max_permutations]
    if not target_seeds or not permutation_seeds:
        parser.error("At least one target seed and one permutation are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = args.output_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(args.config)
    implementation = implementation_fingerprint()
    all_rows: list[dict[str, Any]] = []
    retained_audits: dict[str, Any] = {}
    pairs = [
        pair
        for pair in config["pairs"]
        if not args.pair or pair["pair_id"] in set(args.pair)
    ]
    missing_pairs = set(args.pair) - {pair["pair_id"] for pair in pairs}
    if missing_pairs:
        parser.error(f"Unknown pair ids: {sorted(missing_pairs)}")
    jobs = [
        (
            pair,
            protocol,
            target_seed,
            permutation_seeds,
            target_seed == target_seeds[0],
        )
        for pair in pairs
        for target_seed in target_seeds
    ]
    pending_jobs = []
    resumed_jobs = 0

    def collect_job(
        job: tuple[dict[str, Any], dict[str, Any], int, list[int], bool],
        output: tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]],
        *,
        checkpoint: bool,
    ) -> None:
        pair, _protocol, target_seed, _permutations, _retain_audit = job
        rows, audits = output
        if checkpoint:
            write_job_checkpoint(
                job_checkpoint_path(job_dir, pair["pair_id"], target_seed),
                config_sha256=config_sha256,
                implementation_sha256=implementation["sha256"],
                pair_id=pair["pair_id"],
                target_seed=target_seed,
                rows=rows,
                audits=audits,
            )
        all_rows.extend(rows)
        if audits:
            retained_audits[f"{pair['pair_id']}::{target_seed}"] = audits

    for job in jobs:
        pair, _protocol, target_seed, _permutations, _retain_audit = job
        checkpoint_path = job_checkpoint_path(job_dir, pair["pair_id"], target_seed)
        checkpoint = None
        if args.resume:
            checkpoint = load_job_checkpoint(
                checkpoint_path,
                config_sha256=config_sha256,
                implementation_sha256=implementation["sha256"],
                pair_id=pair["pair_id"],
                target_seed=target_seed,
                permutation_seeds=permutation_seeds,
            )
        if checkpoint is None:
            pending_jobs.append(job)
        else:
            collect_job(job, checkpoint, checkpoint=False)
            resumed_jobs += 1

    if not args.resume and any(job_dir.glob("*.json")):
        parser.error(
            f"Checkpoint files already exist in {job_dir}; use --resume or a new output directory"
        )

    completed_new_jobs = 0
    if args.workers == 1:
        for job in pending_jobs:
            collect_job(job, execute_job(job), checkpoint=True)
            completed_new_jobs += 1
            print(
                f"completed {resumed_jobs + completed_new_jobs}/{len(jobs)} jobs",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(execute_job, job): job for job in pending_jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                collect_job(job, future.result(), checkpoint=True)
                completed_new_jobs += 1
                print(
                    f"completed {resumed_jobs + completed_new_jobs}/{len(jobs)} jobs",
                    flush=True,
                )

    all_rows.sort(key=lambda row: (
        str(row["pair_id"]),
        int(row["target_seed"]),
        str(row["condition"]),
    ))
    report = {
        "schema_version": "care.source_outcome_falsification/v1",
        "evidence_class": "retrospective_outcome_assignment_falsification",
        "config": args.config.name,
        "config_sha256": config_sha256,
        "implementation": implementation,
        "protocol": {
            **protocol,
            "executed_target_seed_count": len(target_seeds),
            "executed_permutation_count": len(permutation_seeds),
        },
        "execution": {
            "job_count": len(jobs),
            "resumed_job_count": resumed_jobs,
            "new_job_count": completed_new_jobs,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "gp_backend": surrogate.gp_backend_name(),
        },
        "intervention": (
            "Permute measured source outcomes across fixed source candidates while preserving "
            "the source features, outcome marginal distribution, LLM role map, LLM patches, "
            "target anchor, target seeds, and target budget."
        ),
        "pairs": [
            build_pair_summary(
                pair,
                [row for row in all_rows if row["pair_id"] == pair["pair_id"]],
                permutation_seeds,
            )
            for pair in pairs
        ],
    }
    metrics_path = args.output_dir / "trajectory_metrics.csv"
    report_path = args.output_dir / "falsification_report.json"
    results_path = args.output_dir / "RESULTS.md"
    audit_path = args.output_dir / "canonical_audits.json"
    manifest_path = args.output_dir / "job_manifest.json"
    write_csv(metrics_path, all_rows)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_results(results_path, report)
    audit_path.write_text(json.dumps(retained_audits, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "care.source_outcome_falsification_manifest/v1",
        "config_sha256": config_sha256,
        "implementation": implementation,
        "jobs": [
            {
                "pair_id": pair["pair_id"],
                "target_seed": target_seed,
                "path": str(job_checkpoint_path(job_dir, pair["pair_id"], target_seed).relative_to(args.output_dir)),
                "sha256": file_sha256(job_checkpoint_path(job_dir, pair["pair_id"], target_seed)),
            }
            for pair, _protocol, target_seed, _permutations, _retain_audit in jobs
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in (metrics_path, report_path, results_path, audit_path, manifest_path):
        write_sha256(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
