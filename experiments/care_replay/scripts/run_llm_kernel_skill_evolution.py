#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"


@dataclass(frozen=True)
class KernelSkillPatch:
    patch_id: str
    scales: tuple[float, ...]
    role_multipliers: dict[str, float]
    gp_beta: float
    gp_beta_end: float
    source_prior_strength: float
    source_similarity_temperature: float
    source_neighbor_count: int
    calibration_mode: str
    min_cv_gain: float
    confidence: float
    reason: str
    source_interaction_strength: float = 0.0
    source_interaction_min_support: int = 4
    canonicalize_source_values: bool = True


def bounded_float(raw: Any, default: float, lower: float, upper: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def bounded_int(raw: Any, default: int, lower: int, upper: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(lower, min(upper, value))


def normalize_patch_id(raw: Any, index: int) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(raw).lower()).strip("_")
    return text[:48] or f"llm_patch_{index + 1}"


def normalize_patches(
    parsed: dict[str, Any],
    target_fields: tuple[str, ...],
    max_patches: int,
) -> tuple[KernelSkillPatch, ...]:
    raw_patches = parsed.get("patches", [])
    if not isinstance(raw_patches, list):
        raw_patches = []
    patches: list[KernelSkillPatch] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_patches[:max_patches]):
        if not isinstance(raw, dict):
            continue
        raw_scales = raw.get("scales", [raw.get("scale", 1.0)])
        if not isinstance(raw_scales, list):
            raw_scales = [raw_scales]
        scales = tuple(
            dict.fromkeys(
                bounded_float(value, 1.0, 0.0, 8.0)
                for value in raw_scales[:6]
            )
        )
        if not scales:
            continue
        raw_multipliers = raw.get("role_multipliers", {})
        if not isinstance(raw_multipliers, dict):
            raw_multipliers = {}
        multipliers = {
            field: bounded_float(raw_multipliers.get(field, 1.0), 1.0, 0.20, 3.0)
            for field in target_fields
        }
        patch_id = normalize_patch_id(raw.get("patch_id"), index)
        if patch_id in seen_ids:
            patch_id = f"{patch_id}_{index + 1}"
        seen_ids.add(patch_id)
        patches.append(
            KernelSkillPatch(
                patch_id=patch_id,
                scales=scales,
                role_multipliers=multipliers,
                gp_beta=bounded_float(raw.get("gp_beta", 1.5), 1.5, 0.5, 3.0),
                gp_beta_end=bounded_float(
                    raw.get("gp_beta_end", raw.get("gp_beta", 1.5)),
                    1.0,
                    0.5,
                    3.0,
                ),
                source_prior_strength=bounded_float(
                    raw.get("source_prior_strength", 0.0), 0.0, 0.0, 2.5
                ),
                source_similarity_temperature=bounded_float(
                    raw.get("source_similarity_temperature", 0.35), 0.35, 0.08, 2.0
                ),
                source_neighbor_count=bounded_int(raw.get("source_neighbor_count", 12), 12, 3, 48),
                calibration_mode=(
                    str(raw.get("calibration_mode", "signed")).strip().lower()
                    if str(raw.get("calibration_mode", "signed")).strip().lower()
                    in {"off", "positive_only", "signed"}
                    else "signed"
                ),
                min_cv_gain=bounded_float(raw.get("min_cv_gain", 0.02), 0.02, 0.0, 0.35),
                confidence=bounded_float(raw.get("confidence", 0.5), 0.5, 0.0, 1.0),
                reason=str(raw.get("reason", ""))[:600],
                source_interaction_strength=bounded_float(
                    raw.get("source_interaction_strength", 0.0), 0.0, 0.0, 2.5
                ),
                source_interaction_min_support=bounded_int(
                    raw.get("source_interaction_min_support", 4), 4, 3, 32
                ),
                canonicalize_source_values=bool(
                    raw.get("canonicalize_source_values", True)
                ),
            )
        )
    return tuple(patches)


def target_schema(adapter: replay.DatasetAdapter) -> dict[str, Any]:
    return {
        "dataset_id": adapter.dataset_id,
        "decision_columns": list(adapter.decision_columns),
        "field_cardinality": {
            field: len({str(candidate.metadata.get(field, "")) for candidate in adapter.candidates})
            for field in adapter.decision_columns
        },
        "pool_size": len(adapter.candidates),
        "hidden_target": adapter.hidden_target,
        "boundary": "Only public feature schema is exposed. No target outcomes are included.",
    }


def propose_kernel_skill_patches(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    card: transfer.TransferCard,
    config: replay.LLMConfig,
    max_patches: int,
) -> tuple[tuple[KernelSkillPatch, ...], dict[str, Any]]:
    roles = [asdict(role) for role in card.roles if role.transfer_weight > 0.0]
    value_priors = sorted(
        (asdict(prior) for prior in card.value_priors),
        key=lambda prior: (float(prior["confidence"]), abs(float(prior["source_effect"]))),
        reverse=True,
    )[:80]
    prompt_payload = {
        "task": (
            "Generate a small portfolio of executable cross-domain optimization skills. "
            "A skill may reshape the GP kernel and may add a source-neighbor prior only when "
            "leave-one-out target calibration supports it."
        ),
        "source_dataset": source_adapter.dataset_id,
        "source_objective": source_adapter.hidden_target,
        "target_schema": target_schema(target_adapter),
        "transfer_card": {
            "source_observation_count": card.source_observation_count,
            "discount": card.discount,
            "roles": roles,
            "shared_value_priors": value_priors,
            "evidence_summary": card.evidence_summary,
        },
        "base_policy": {
            "name": "mixed_kernel_gp_ucb",
            "gp_beta": 1.5,
            "categorical_weight_formula": "1 + scale * source_role_confidence * role_multiplier",
            "normalization": "categorical weights are normalized to mean 1",
            "ensemble": "multiple scales are aggregated by mean normalized candidate rank",
            "source_prior": (
                "mapped-role nearest-neighbor prediction from source observations; sign and strength "
                "are recalibrated using only currently revealed target observations"
            ),
        },
        "design_rules": [
            "Return diverse executable patches, not candidate IDs and not review prose.",
            "At least one patch should include scale 0 as a target-only safety anchor in an ensemble.",
            "At least one patch should focus on the two highest-confidence roles.",
            "Include an exploration-first patch with gp_beta above 2.2 and gp_beta_end at or below 1.2.",
            "Include a conservative patch with gp_beta and gp_beta_end both at or below 1.5.",
            "Make at least one patch test a counter-hypothesis rather than only reinforcing the strongest source role.",
            "At least two patches must downweight a weak, uncertain, or failure-associated role below 0.8; do not make every role multiplier at least 1.",
            "Separate exploitation patches from exploration patches instead of giving every patch the same beta schedule.",
            "Include at least two source-prior patches when mapped roles have shared vocabularies.",
            "Include source interaction transfer when two mapped roles have enough joint source support.",
            "Use signed calibration when source and target objectives can be inversely related.",
            "Use positive_only calibration only when the objective semantics support the same direction.",
            "Set source_prior_strength to 0 when role vocabularies are not meaningfully comparable.",
            "Downweight weak or semantically uncertain role mappings instead of treating all roles equally.",
            "Vary confidence according to support and uncertainty; do not reuse a fixed value such as 0.7 across patches.",
            "Use multi-scale ensembles when transfer strength is uncertain.",
            "Do not claim access to target outcomes. Calibration and held-out evaluation happen after generation.",
        ],
        "allowed_space": {
            "patch_count": max_patches,
            "scales": "1 to 6 numbers, each in [0, 8]",
            "role_multipliers": "map every target decision field to [0.20, 3.0]",
            "gp_beta": "number in [0.5, 3.0] used at the first replay round",
            "gp_beta_end": "number in [0.5, 3.0] reached linearly at the final replay round",
            "source_prior_strength": "number in [0, 2.5]",
            "source_similarity_temperature": "number in [0.08, 2.0]",
            "source_neighbor_count": "integer in [3, 48]",
            "source_interaction_strength": "number in [0, 2.5]",
            "source_interaction_min_support": "integer in [3, 32]",
            "canonicalize_source_values": "boolean",
            "calibration_mode": "one of off, positive_only, signed",
            "min_cv_gain": "leave-one-out gain threshold in [0, 0.35]",
        },
        "output_contract": {
            "patches": [
                {
                    "patch_id": "short_unique_name",
                    "scales": [0.0, 1.0, 2.0],
                    "role_multipliers": {field: 1.0 for field in target_adapter.decision_columns},
                    "gp_beta": 1.5,
                    "gp_beta_end": 1.0,
                    "source_prior_strength": 0.8,
                    "source_similarity_temperature": 0.35,
                    "source_neighbor_count": 12,
                    "source_interaction_strength": 0.6,
                    "source_interaction_min_support": 4,
                    "canonicalize_source_values": True,
                    "calibration_mode": "signed",
                    "min_cv_gain": 0.03,
                    "confidence": "evidence-calibrated number in [0, 1]; do not reuse one default",
                    "reason": "brief hypothesis, counter-hypothesis, mechanism, and failure trigger",
                }
            ]
        },
    }
    system = (
        "You are the skill-evolution component of CARE 2.0. "
        "Design reusable source-to-target skill patches for finite-pool scientific optimization. "
        "Use source evidence to decide which feature roles deserve more similarity weight and whether "
        "a mapped source-neighbor prior is scientifically defensible. "
        "Keep exploration through GP-UCB and scale ensembles. "
        "You never see target outcomes and must return only JSON matching the requested schema."
    )
    user = (
        f"Generate exactly {max_patches} diverse kernel skill patches. "
        "The goal is to beat target-only GP-UCB without relying on a single brittle transfer scale. JSON input:\n"
        + json.dumps(prompt_payload, ensure_ascii=False)
    )
    content, response_meta = replay.chat_completion_text(
        config,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    parse_error = ""
    try:
        parsed = replay.extract_json_object(content)
    except ValueError as exc:
        parsed = {}
        parse_error = str(exc)
    patches = normalize_patches(parsed, target_adapter.decision_columns, max_patches)
    if not patches:
        raise RuntimeError(f"LLM returned no usable kernel skill patches: {parse_error or content[:300]}")
    record = {
        "model": response_meta["model"],
        "usage": response_meta["usage"],
        "llm_generation_call_count": 1,
        "api_configuration": {
            "base_url": config.base_url,
            "api_mode": config.api_mode,
            "structured_mode": config.structured_mode,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        "prompt_payload": prompt_payload,
        "raw_response": content,
        "parsed_response": parsed,
        "normalized_patches": [asdict(patch) for patch in patches],
        "parse_error": parse_error,
    }
    return patches, record


def patch_mode(patch: KernelSkillPatch) -> str:
    return f"llm_evolved_kernel_{patch.patch_id}"


def run_patch(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    patch: KernelSkillPatch,
    card: transfer.TransferCard,
    normalize: bool,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scale_weights: list[tuple[float, tuple[float, ...], dict[str, Any]]] = []
    for scale in patch.scales:
        weights, diagnostics = weighted.transfer_categorical_weights(
            adapter,
            card,
            scale,
            normalize,
            patch.role_multipliers,
        )
        scale_weights.append((scale, weights, diagnostics))
    mode = patch_mode(patch)
    if len(scale_weights) == 1:
        return weighted.run_policy(
            adapter,
            task,
            seed,
            mode,
            scale_weights[0][1],
            {
                "llm_patch": asdict(patch),
                "scale_weight": scale_weights[0][2],
            },
            patch.gp_beta,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            patch.gp_beta_end,
        )
    metrics, audit = weighted.run_ensemble_policy(
        adapter,
        task,
        seed,
        mode,
        scale_weights,
        patch.gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
        patch.gp_beta_end,
    )
    for row in audit:
        row["hypothesis_snapshot"]["llm_patch"] = asdict(patch)
    return metrics, audit


def summarize(rows: list[dict[str, Any]], seeds: set[int]) -> dict[str, dict[str, float]]:
    selected_rows = [row for row in rows if int(row["seed"]) in seeds]
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for mode, items in by_mode.items():
        summary[mode] = {}
        for field in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = [float(item[field]) for item in items]
            summary[mode][f"{field}_mean"] = round(mean(values), 4)
            summary[mode][f"{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        summary[mode]["seed_count"] = float(len(items))
    return summary


def selection_score(metrics: dict[str, float]) -> float:
    return metrics["final_best_mean"] + metrics["best_so_far_auc_mean"]


def select_calibration_robust_patch(
    rows: list[dict[str, Any]],
    calibration_seeds: set[int],
    patches: tuple[KernelSkillPatch, ...],
) -> tuple[str, dict[str, Any]]:
    patch_by_mode = {patch_mode(patch): patch for patch in patches}
    scores_by_mode: dict[str, dict[int, float]] = {}
    for row in rows:
        mode = str(row["mode"])
        seed = int(row["seed"])
        if mode not in patch_by_mode or seed not in calibration_seeds:
            continue
        scores_by_mode.setdefault(mode, {})[seed] = (
            float(row["final_best"]) + float(row["best_so_far_auc"])
        )
    if not scores_by_mode:
        raise RuntimeError("No LLM patch rows are available for calibration selection.")

    mean_scores = {
        mode: mean(seed_scores.values())
        for mode, seed_scores in scores_by_mode.items()
    }
    empirical_best = max(mean_scores, key=mean_scores.get)
    best_seed_scores = scores_by_mode[empirical_best]
    eligible: list[str] = []
    diagnostics: dict[str, Any] = {}
    for mode, seed_scores in scores_by_mode.items():
        shared_seeds = sorted(set(seed_scores) & set(best_seed_scores))
        paired_deficits = [
            best_seed_scores[seed] - seed_scores[seed]
            for seed in shared_seeds
        ]
        mean_deficit = mean(paired_deficits) if paired_deficits else float("inf")
        standard_error = (
            pstdev(paired_deficits) / math.sqrt(len(paired_deficits))
            if len(paired_deficits) > 1
            else 0.0
        )
        # Small calibration sets can produce a very wide one-SE band. Cap the
        # equivalence margin at one point on the 0-200 composite scale so a
        # high-confidence patch cannot erase a meaningful empirical deficit.
        tolerance = min(1.0, max(0.25, standard_error))
        is_eligible = mean_deficit <= tolerance + 1e-12
        if is_eligible:
            eligible.append(mode)
        patch = patch_by_mode[mode]
        diagnostics[mode] = {
            "calibration_composite_mean": round(mean_scores[mode], 6),
            "paired_deficit_vs_empirical_best": round(mean_deficit, 6),
            "paired_standard_error": round(standard_error, 6),
            "eligibility_tolerance": round(tolerance, 6),
            "eligible": is_eligible,
            "llm_confidence": patch.confidence,
            "has_target_only_anchor": 0.0 in patch.scales,
        }

    selected = max(
        eligible,
        key=lambda mode: (
            patch_by_mode[mode].confidence,
            int(0.0 in patch_by_mode[mode].scales),
            -abs(patch_by_mode[mode].gp_beta - patch_by_mode[mode].gp_beta_end),
            -len(patch_by_mode[mode].scales),
            mode,
        ),
    )
    return selected, {
        "rule": (
            "Paired practical-equivalence set on calibration final_best + AUC: "
            "the one-standard-error margin is bounded to [0.25, 1.0] composite points; "
            "within that set prefer higher LLM confidence, then a scale-0 target-only anchor."
        ),
        "empirical_best_mode": empirical_best,
        "eligible_modes": sorted(eligible),
        "selected_mode": selected,
        "modes": diagnostics,
    }


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    llm_record: dict[str, Any],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_RUNS / f"{output_id}_llm_record.json").write_text(
        json.dumps(llm_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for (mode, seed), audit_rows in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as file:
            for row in audit_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_seed_evaluation(
    seed: int,
    source_dataset: str,
    target_dataset: str,
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    patches: tuple[KernelSkillPatch, ...],
    fixed_scales: tuple[float, ...],
    normalize: bool,
    initial: int,
    rounds: int,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    source_seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    role_map = transfer.descriptor_transfer_role_map_for(
        source_dataset,
        target_dataset,
    )
    source_observed = transfer.source_observations(
        source_adapter,
        seed if source_seed is None else source_seed,
        source_observation_count,
    )
    card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        source_observed,
        role_map,
        discount,
        min_source_support,
    )
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}

    baseline_weights, baseline_diagnostics = weighted.transfer_categorical_weights(
        target_adapter, card, 0.0, normalize
    )
    metrics, audit = weighted.run_policy(
        target_adapter,
        task,
        seed,
        "gp_ucb",
        baseline_weights,
        baseline_diagnostics,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows.append(metrics)
    audits[("gp_ucb", seed)] = audit

    fixed_weights = []
    for scale in fixed_scales:
        weights, diagnostics = weighted.transfer_categorical_weights(
            target_adapter, card, scale, normalize
        )
        fixed_weights.append((scale, weights, diagnostics))
    fixed_mode = f"fixed_scale_ensemble_{weighted.scales_label(fixed_scales)}"
    metrics, audit = weighted.run_ensemble_policy(
        target_adapter,
        task,
        seed,
        fixed_mode,
        fixed_weights,
        gp_beta,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    rows.append(metrics)
    audits[(fixed_mode, seed)] = audit

    for patch in patches:
        metrics, audit = run_patch(
            target_adapter,
            task,
            seed,
            patch,
            card,
            normalize,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        rows.append(metrics)
        audits[(patch_mode(patch), seed)] = audit
    return rows, audits


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    source_adapter = replay.DATASET_BUILDERS[args.source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[args.target_dataset]()
    role_map = transfer.descriptor_transfer_role_map_for(
        args.source_dataset,
        args.target_dataset,
    )
    proposal_source = transfer.source_observations(source_adapter, 0, args.source_observations)
    proposal_card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        proposal_source,
        role_map,
        args.discount,
        args.min_source_support,
    )
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(f"Set {args.llm_api_key_env}, COMMONSTACK_API_KEY, or CARE_LLM_API_KEY.")
    llm_config = replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_mode=args.llm_api_mode,
        structured_mode=args.llm_structured_mode,
    )
    patches, llm_record = propose_kernel_skill_patches(
        source_adapter,
        target_adapter,
        proposal_card,
        llm_config,
        args.patch_count,
    )

    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    fixed_scales = weighted.parse_scales(args.fixed_ensemble_scales)
    worker_args = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_observation_count": args.source_observations,
        "discount": args.discount,
        "min_source_support": args.min_source_support,
        "patches": patches,
        "fixed_scales": fixed_scales,
        "normalize": not args.no_normalize,
        "initial": args.initial,
        "rounds": args.rounds,
        "gp_beta": args.gp_beta,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
    }
    if args.workers == 1:
        seed_results = [run_seed_evaluation(seed=seed, **worker_args) for seed in range(args.seeds)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_seed_evaluation, seed=seed, **worker_args)
                for seed in range(args.seeds)
            ]
            seed_results = [future.result() for future in futures]
    for seed_rows, seed_audits in seed_results:
        rows.extend(seed_rows)
        audits.update(seed_audits)
    rows.sort(key=lambda row: (int(row["seed"]), str(row["mode"])))

    calibration_seeds = set(range(args.calibration_seeds))
    heldout_seeds = set(range(args.calibration_seeds, args.seeds))
    calibration = summarize(rows, calibration_seeds)
    heldout = summarize(rows, heldout_seeds)
    llm_modes = [patch_mode(patch) for patch in patches]
    naive_selected_mode = max(llm_modes, key=lambda mode: selection_score(calibration[mode]))
    selected_mode, selection_diagnostics = select_calibration_robust_patch(
        rows,
        calibration_seeds,
        patches,
    )
    gp_mode = "gp_ucb"
    fixed_mode = f"fixed_scale_ensemble_{weighted.scales_label(fixed_scales)}"
    selected_eval = heldout[selected_mode]
    summary = {
        "experiment": "care_llm_kernel_skill_evolution",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "llm": {
            "model": llm_record["model"],
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "call_count": 1,
        },
        "evidence_boundary": (
            "The LLM sees source transfer-card evidence and target public schema only. "
            "It does not see target outcomes. Calibration seeds select one generated patch; held-out seeds report performance."
        ),
        "seeds": args.seeds,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_count": args.seeds - args.calibration_seeds,
        "patches": [asdict(patch) for patch in patches],
        "selected_llm_mode": selected_mode,
        "naive_calibration_selected_llm_mode": naive_selected_mode,
        "selection_diagnostics": selection_diagnostics,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_deltas": {
            "vs_gp_ucb_final_best": round(selected_eval["final_best_mean"] - heldout[gp_mode]["final_best_mean"], 4),
            "vs_gp_ucb_auc": round(selected_eval["best_so_far_auc_mean"] - heldout[gp_mode]["best_so_far_auc_mean"], 4),
            "vs_fixed_ensemble_final_best": round(selected_eval["final_best_mean"] - heldout[fixed_mode]["final_best_mean"], 4),
            "vs_fixed_ensemble_auc": round(selected_eval["best_so_far_auc_mean"] - heldout[fixed_mode]["best_so_far_auc_mean"], 4),
        },
    }
    output_id = f"llm_kernel_evolution_{args.source_dataset}_to_{args.target_dataset}"
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    write_outputs(output_id, rows, summary, audits, llm_record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and held-out evaluate LLM-evolved CARE GP-kernel skills.")
    parser.add_argument("--source-dataset", default="real_suzuki_miyaura", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--calibration-seeds", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=96)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--patch-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--fixed-ensemble-scales", default="0.5,1,1.5,2,3,4")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--llm-base-url", default=os.environ.get("CARE_LLM_BASE_URL", "https://api.commonstack.ai/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("CARE_LLM_MODEL", "openai/gpt-5.5"))
    parser.add_argument("--llm-api-key-env", default="CARE_LLM_API_KEY")
    parser.add_argument(
        "--llm-api-mode",
        choices=("chat", "completion"),
        default=os.environ.get("CARE_LLM_API_MODE", "chat"),
    )
    parser.add_argument(
        "--llm-structured-mode",
        choices=("tool", "json"),
        default=os.environ.get("CARE_LLM_STRUCTURED_MODE", "tool"),
    )
    parser.add_argument("--llm-temperature", type=float, default=0.4)
    parser.add_argument("--llm-max-tokens", type=int, default=2200)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    if args.calibration_seeds <= 0 or args.calibration_seeds >= args.seeds:
        parser.error("--calibration-seeds must be between 1 and seeds-1")
    summary = run_experiment(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
