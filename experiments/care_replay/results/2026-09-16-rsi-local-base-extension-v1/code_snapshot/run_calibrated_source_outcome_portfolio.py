#!/usr/bin/env python3
"""Select a frozen source-outcome transfer portfolio on calibration seeds.

The ordinary calibrated runner evaluates one transfer configuration.  This
runner evaluates a pre-registered, small portfolio of transfer configurations
under the same target-only baselines.  Only calibration outcomes may select a
candidate; held-out seeds only execute the selected candidate (or the frozen
matched target-only LLM fallback).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cross_task_router
import llm_semantic_skills as semantic
import run_calibrated_frozen_llm_selector as selector
import run_calibrated_llm_semantic_selector as semantic_selector
import run_calibrated_source_outcome_transfer as calibrated
import run_llm_kernel_skill_evolution as evolution
import run_llm_transfer_router as outcome_router
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer
import run_transfer_weighted_kernel as weighted


SELECTOR_MODE = "care_source_outcome_portfolio"
MATCHED_TARGET_LLM_MODE = calibrated.MATCHED_TARGET_LLM_MODE
ALLOWED_INITIAL_STRATEGIES = {
    "matched",
    "source_extremes",
    "source_positive",
    "source_negative",
    "source_positive_quantile",
    "source_negative_quantile",
}


@dataclass(frozen=True)
class PortfolioCandidate:
    candidate_id: str
    router_max_transfer_mass: float
    source_initial_strategy: str

    @property
    def mode(self) -> str:
        return f"source_outcome_candidate__{self.candidate_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "router_max_transfer_mass": self.router_max_transfer_mass,
            "source_initial_strategy": self.source_initial_strategy,
        }


def parse_portfolio(spec: str) -> tuple[PortfolioCandidate, ...]:
    """Parse ``id:max_transfer_mass:initial_strategy`` entries."""
    candidates: list[PortfolioCandidate] = []
    seen: set[str] = set()
    for raw in (part.strip() for part in spec.split(",")):
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid portfolio entry {raw!r}; expected id:max_transfer_mass:initial_strategy."
            )
        candidate_id, raw_mass, initial_strategy = parts
        if not candidate_id or candidate_id in seen or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in candidate_id
        ):
            raise ValueError(f"Portfolio candidate id must be unique and shell-safe: {candidate_id!r}")
        try:
            mass = float(raw_mass)
        except ValueError as exc:
            raise ValueError(f"Invalid transfer mass in {raw!r}") from exc
        if not 0.0 <= mass <= 0.45:
            raise ValueError("router_max_transfer_mass must be between 0 and 0.45.")
        if initial_strategy not in ALLOWED_INITIAL_STRATEGIES:
            raise ValueError(
                f"Unknown source initial strategy {initial_strategy!r}; "
                f"choose from {sorted(ALLOWED_INITIAL_STRATEGIES)}."
            )
        candidates.append(
            PortfolioCandidate(candidate_id, round(mass, 6), initial_strategy)
        )
        seen.add(candidate_id)
    if not candidates:
        raise ValueError("The transfer portfolio must contain at least one candidate.")
    return tuple(candidates)


def _rename_route(
    metrics: dict[str, Any],
    audit: list[dict[str, Any]],
    candidate: PortfolioCandidate,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    renamed = dict(metrics)
    renamed["mode"] = candidate.mode
    renamed["portfolio_candidate_id"] = candidate.candidate_id
    renamed["router_max_transfer_mass"] = candidate.router_max_transfer_mass
    renamed["source_initial_strategy"] = candidate.source_initial_strategy
    renamed_audit: list[dict[str, Any]] = []
    for event in audit:
        item = deepcopy(event)
        item["mode"] = candidate.mode
        item["portfolio_candidate"] = candidate.as_dict()
        renamed_audit.append(item)
    return renamed, renamed_audit


def _source_context(
    source_dataset: str,
    target_dataset: str,
    patches: tuple[evolution.KernelSkillPatch, ...],
    source_observations: int,
    source_seed: int,
    discount: float,
    min_source_support: int,
) -> tuple[Any, Any, transfer.TransferCard, list[replay.Candidate]]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    source_observed = transfer.source_observations(
        source_adapter,
        source_seed,
        source_observations,
    )
    card = transfer.compile_transfer_card(
        source_adapter,
        target_adapter,
        source_observed,
        transfer.descriptor_transfer_role_map_for(source_dataset, target_dataset),
        discount,
        min_source_support,
    )
    return source_adapter, target_adapter, card, source_observed


def evaluate_seed(
    seed: int,
    split: str,
    source_dataset: str,
    target_dataset: str,
    candidates: tuple[PortfolioCandidate, ...],
    patches: tuple[evolution.KernelSkillPatch, ...],
    target_llm_mode: str,
    target_llm_skill: semantic.SemanticSkill | None,
    fixed_scales: tuple[float, ...],
    source_observations: int,
    source_seed: int,
    discount: float,
    min_source_support: int,
    initial: int,
    rounds: int,
    normalize: bool,
    gp_beta: float,
    gp_xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    router_min_observations: int,
    router_min_quality: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    initial_observed = calibrated.target_llm_initial_observations(
        target_adapter,
        task,
        seed,
        target_llm_mode,
        target_llm_skill,
    )
    anchor_scorer = calibrated.target_llm_anchor_scorer(
        target_adapter,
        task,
        target_llm_mode,
        target_llm_skill,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    first = candidates[0]
    # This call supplies the fixed target-only BO baselines.  Its route row is
    # discarded; route variants below reuse the same public target setup.
    baseline_rows, baseline_audits = outcome_router.run_seed(
        seed,
        source_dataset,
        target_dataset,
        patches,
        fixed_scales,
        source_observations,
        source_seed,
        discount,
        min_source_support,
        initial,
        rounds,
        normalize,
        gp_beta,
        gp_xi,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
        initial_observed,
        anchor_scorer,
        target_llm_mode,
        router_min_observations,
        router_min_quality,
        first.router_max_transfer_mass,
        first.source_initial_strategy,
    )
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        if str(row["mode"]) == "llm_transfer_router":
            continue
        copied = dict(row)
        copied["split"] = split
        copied["selected_source_mode"] = ""
        rows.append(copied)
    for (mode, audit_seed), events in baseline_audits.items():
        if mode == "llm_transfer_router":
            continue
        for event in events:
            event["split"] = split
        audits[(mode, audit_seed)] = events

    source_adapter, target_adapter, card, source_observed = _source_context(
        source_dataset,
        target_dataset,
        patches,
        source_observations,
        source_seed,
        discount,
        min_source_support,
    )
    for candidate in candidates:
        route_metrics, route_audit = outcome_router.run_router_policy(
            source_adapter,
            target_adapter,
            task,
            seed,
            card,
            source_observed,
            patches,
            normalize,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
            initial_observed,
            anchor_scorer,
            target_llm_mode,
            router_min_observations,
            router_min_quality,
            candidate.router_max_transfer_mass,
            candidate.source_initial_strategy,
        )
        metrics, renamed_audit = _rename_route(route_metrics, route_audit, candidate)
        metrics["split"] = split
        metrics["selected_source_mode"] = candidate.mode
        for event in renamed_audit:
            event["split"] = split
        rows.append(metrics)
        audits[(candidate.mode, seed)] = renamed_audit

    target_metrics, target_audit = calibrated.run_matched_target_llm(
        target_adapter,
        task,
        seed,
        target_llm_mode,
        target_llm_skill,
        rows,
        audits,
        numeric_length_scale,
        categorical_length_scale,
        gp_noise,
    )
    target_metrics["split"] = split
    for event in target_audit:
        event["split"] = split
    rows.append(target_metrics)
    audits[(MATCHED_TARGET_LLM_MODE, seed)] = target_audit
    return rows, audits


def _diagnose_candidate(
    rows: list[dict[str, Any]],
    candidate: PortfolioCandidate,
    calibration_seeds: set[int],
    anchor: str,
    min_risk_adjusted_gain: float,
    min_positive_fold_rate: float,
    min_final_non_loss_rate: float,
    min_composite_ci_low: float,
) -> dict[str, Any]:
    comparisons: dict[str, dict[str, Any]] = {}
    eligible_by_anchor: dict[str, bool] = {}
    for comparison_anchor in (MATCHED_TARGET_LLM_MODE, anchor):
        final = selector.paired_deltas(
            rows, candidate.mode, comparison_anchor, calibration_seeds, "final_best"
        )
        auc = selector.paired_deltas(
            rows, candidate.mode, comparison_anchor, calibration_seeds, "best_so_far_auc"
        )
        composite = [left + right for left, right in zip(final, auc)]
        final_stats = selector.delta_summary(final)
        auc_stats = selector.delta_summary(auc)
        composite_stats = selector.delta_summary(composite)
        fold_rate, fold_means = selector.positive_fold_rate(composite, 5)
        risk_adjusted = composite_stats["mean"] - composite_stats["se"]
        eligible_by_anchor[comparison_anchor] = bool(
            final_stats["mean"] >= 0.0
            and auc_stats["mean"] >= 0.0
            and final_stats["non_loss_rate"] >= min_final_non_loss_rate
            and fold_rate >= min_positive_fold_rate
            and risk_adjusted >= min_risk_adjusted_gain
            and composite_stats["normal_95ci_low"] >= min_composite_ci_low
        )
        comparisons[comparison_anchor] = {
            "final_best": final_stats,
            "best_so_far_auc": auc_stats,
            "composite": composite_stats,
            "positive_fold_rate": round(fold_rate, 6),
            "fold_composite_means": fold_means,
            "risk_adjusted_composite_gain": round(risk_adjusted, 6),
            "eligible": eligible_by_anchor[comparison_anchor],
        }
    return {
        "candidate": candidate.as_dict(),
        "eligible": all(eligible_by_anchor.values()),
        "calibration_composite_vs_matched_llm": comparisons[
            MATCHED_TARGET_LLM_MODE
        ]["composite"]["mean"],
        "eligible_by_anchor": eligible_by_anchor,
        "comparisons": comparisons,
    }


def select_portfolio(
    rows: list[dict[str, Any]],
    candidates: tuple[PortfolioCandidate, ...],
    calibration_seeds: set[int],
    min_risk_adjusted_gain: float,
    min_positive_fold_rate: float,
    min_final_non_loss_rate: float,
    min_composite_ci_low: float,
) -> tuple[str, dict[str, Any]]:
    target_summary = selector.mode_means(
        rows,
        selector.TARGET_MODES,
        calibration_seeds,
    )
    anchor = max(
        selector.TARGET_MODES,
        key=lambda mode: (target_summary[mode]["composite"], mode),
    )
    diagnostics = [
        _diagnose_candidate(
            rows,
            candidate,
            calibration_seeds,
            anchor,
            min_risk_adjusted_gain,
            min_positive_fold_rate,
            min_final_non_loss_rate,
            min_composite_ci_low,
        )
        for candidate in candidates
    ]
    eligible = [item for item in diagnostics if item["eligible"]]
    if eligible:
        chosen = max(
            eligible,
            key=lambda item: (
                float(item["calibration_composite_vs_matched_llm"]),
                str(item["candidate"]["candidate_id"]),
            ),
        )
        selected_mode = str(chosen["candidate"]["mode"])
        selected_candidate_id = str(chosen["candidate"]["candidate_id"])
        deployed = True
    else:
        selected_mode = MATCHED_TARGET_LLM_MODE
        selected_candidate_id = None
        deployed = False
    selected_candidate = (
        chosen["candidate"] if eligible else None
    )
    selected_initial_strategy = (
        selected_candidate.get("source_initial_strategy")
        if selected_candidate is not None
        else None
    )
    selected_max_transfer_mass = (
        float(selected_candidate["router_max_transfer_mass"])
        if selected_candidate is not None
        else 0.0
    )
    return selected_mode, {
        "selected_mode": selected_mode,
        "selected_candidate_id": selected_candidate_id,
        "selected_source_outcome_transfer": deployed,
        "selected_source_outcome_continuous": bool(
            deployed and selected_max_transfer_mass > 0.0
        ),
        "selected_source_outcome_initial_design": bool(
            deployed and selected_initial_strategy not in {None, "matched"}
        ),
        "selected_initial_strategy": selected_initial_strategy,
        "selected_router_max_transfer_mass": selected_max_transfer_mass,
        "target_anchor_mode": anchor,
        "matched_target_only_llm_mode": MATCHED_TARGET_LLM_MODE,
        "target_anchor_summary": target_summary,
        "candidate_diagnostics": diagnostics,
        "thresholds": {
            "min_risk_adjusted_gain": min_risk_adjusted_gain,
            "min_positive_fold_rate": min_positive_fold_rate,
            "min_final_non_loss_rate": min_final_non_loss_rate,
            "min_composite_95ci_low": min_composite_ci_low,
        },
        "rule": (
            "Evaluate the pre-registered transfer candidates on calibration seeds only. "
            "A candidate must beat both the frozen matched target-only LLM and the "
            "strongest target-only BO on final value, AUC, fold stability, non-loss rate, "
            "one-standard-error gain, and the composite confidence bound. Select the "
            "best eligible candidate; otherwise deploy the matched target-only LLM."
        ),
    }


def add_portfolio_alias(
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    selected_mode: str,
    heldout_seeds: set[int],
    selection: dict[str, Any],
) -> None:
    """Expose only the calibration-frozen choice as the deployed route."""
    for source_row in list(rows):
        seed = int(source_row["seed"])
        if str(source_row["mode"]) != selected_mode or seed not in heldout_seeds:
            continue
        row = dict(source_row)
        row["mode"] = SELECTOR_MODE
        row["selected_source_mode"] = selected_mode
        rows.append(row)
        alias_events: list[dict[str, Any]] = []
        for source_event in audits[(selected_mode, seed)]:
            event = deepcopy(source_event)
            event["mode"] = SELECTOR_MODE
            event["source_outcome_selector"] = {
                "selected_source_mode": selected_mode,
                "selected_source_outcome_transfer": selection[
                    "selected_source_outcome_transfer"
                ],
                "target_anchor_mode": selection["target_anchor_mode"],
                "selection_rule": selection["rule"],
                "evidence_boundary": (
                    "The portfolio was frozen on calibration seeds. The held-out seed did not "
                    "change the candidate portfolio, source history, target anchor, or fallback."
                ),
            }
            alias_events.append(event)
        audits[(SELECTOR_MODE, seed)] = alias_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a frozen portfolio of source-outcome transfer routes."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--target-llm-record", type=Path, required=True)
    parser.add_argument("--target-llm-mode", required=True)
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--calibration-seed-start", type=int, default=32000)
    parser.add_argument("--calibration-seeds", type=int, default=30)
    parser.add_argument("--heldout-seed-start", type=int, default=32100)
    parser.add_argument("--heldout-seeds", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--source-observations", type=int, default=512)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--fixed-ensemble-scales", default="0.5,1,1.5,2,3,4")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--gp-xi", type=float, default=0.01)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--router-min-observations", type=int, default=5)
    parser.add_argument("--router-min-quality", type=float, default=0.20)
    parser.add_argument(
        "--portfolio",
        default="standard:0.45:matched,conservative:0.25:matched,extremes:0.45:source_extremes,positive:0.45:source_positive",
        help="Comma-separated id:max_transfer_mass:initial_strategy entries.",
    )
    parser.add_argument("--min-risk-adjusted-gain", type=float, default=0.25)
    parser.add_argument("--min-positive-fold-rate", type=float, default=0.8)
    parser.add_argument("--min-final-non-loss-rate", type=float, default=0.6)
    parser.add_argument("--min-composite-ci-low", type=float, default=0.0)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    try:
        candidates = parse_portfolio(args.portfolio)
    except ValueError as exc:
        parser.error(str(exc))
    calibration_seeds = set(range(args.calibration_seed_start, args.calibration_seed_start + args.calibration_seeds))
    heldout_seeds = set(range(args.heldout_seed_start, args.heldout_seed_start + args.heldout_seeds))
    if calibration_seeds & heldout_seeds:
        parser.error("Calibration and held-out seed ranges must not overlap.")

    record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    target = replay.DATASET_BUILDERS[args.target_dataset]()
    route_proposal = cross_task_router.propose_route(
        args.source_dataset,
        args.target_dataset,
        replay.DATASET_BUILDERS[args.source_dataset]().decision_columns,
        target.decision_columns,
    ).as_dict()
    target_llm_record = json.loads(args.target_llm_record.read_text(encoding="utf-8"))
    target_llm_skill = calibrated.resolve_target_llm_skill(
        target_llm_record,
        target,
        args.target_llm_mode,
    )
    patches = evolution.normalize_patches(
        {"patches": record.get("normalized_patches", [])},
        target.decision_columns,
        max_patches=100,
    )
    if not patches:
        raise RuntimeError("The record contains no executable source-outcome patches.")
    fixed_scales = weighted.parse_scales(args.fixed_ensemble_scales)
    jobs = [
        *((seed, "calibration") for seed in sorted(calibration_seeds)),
        *((seed, "heldout") for seed in sorted(heldout_seeds)),
    ]
    worker_args = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "candidates": candidates,
        "patches": patches,
        "target_llm_mode": args.target_llm_mode,
        "target_llm_skill": target_llm_skill,
        "fixed_scales": fixed_scales,
        "source_observations": args.source_observations,
        "source_seed": args.source_seed,
        "discount": args.discount,
        "min_source_support": args.min_source_support,
        "initial": args.initial,
        "rounds": args.rounds,
        "normalize": not args.no_normalize,
        "gp_beta": args.gp_beta,
        "gp_xi": args.gp_xi,
        "numeric_length_scale": args.numeric_length_scale,
        "categorical_length_scale": args.categorical_length_scale,
        "gp_noise": args.gp_noise,
        "router_min_observations": args.router_min_observations,
        "router_min_quality": args.router_min_quality,
    }
    if args.workers == 1:
        results = [evaluate_seed(seed=seed, split=split, **worker_args) for seed, split in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(evaluate_seed, seed=seed, split=split, **worker_args)
                for seed, split in jobs
            ]
            results = [future.result() for future in futures]

    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed_rows, seed_audits in results:
        rows.extend(seed_rows)
        audits.update(seed_audits)
    selected_mode, selection = select_portfolio(
        rows,
        candidates,
        calibration_seeds,
        args.min_risk_adjusted_gain,
        args.min_positive_fold_rate,
        args.min_final_non_loss_rate,
        args.min_composite_ci_low,
    )
    selection["schema_route_proposal"] = route_proposal
    add_portfolio_alias(rows, audits, selected_mode, heldout_seeds, selection)
    all_modes = (
        *selector.TARGET_MODES,
        MATCHED_TARGET_LLM_MODE,
        *(candidate.mode for candidate in candidates),
        SELECTOR_MODE,
    )
    calibration = selector.mode_means(rows, all_modes, calibration_seeds)
    heldout = selector.mode_means(rows, all_modes, heldout_seeds)
    pairwise = {
        baseline: {
            field: selector.delta_summary(
                selector.paired_deltas(rows, SELECTOR_MODE, baseline, heldout_seeds, field)
            )
            for field in ("final_best", "best_so_far_auc", "top10_hit")
        }
        for baseline in (*selector.TARGET_MODES, MATCHED_TARGET_LLM_MODE)
    }
    summary = {
        "experiment": "care_calibrated_source_outcome_portfolio",
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "schema_route_proposal": route_proposal,
        "portfolio_candidates": [candidate.as_dict() for candidate in candidates],
        "source_history": {
            "seed": args.source_seed,
            "observation_count": args.source_observations,
            "fixed_across_target_seeds": True,
        },
        "source_outcome_components": [
            "source-outcome-informed initial target probes",
            "mapped source-neighbor outcome prior",
            "mapped additive source outcome-effect prior",
            "source pairwise interaction residual prior",
            "source-outcome-informed GP categorical geometry",
            "prequential target calibration and exact target-only fallback",
        ],
        "llm_model": record.get("model"),
        "llm_generation_call_count": int(record.get("llm_generation_call_count", 1)),
        "matched_target_only_llm": {
            "model": target_llm_record.get("model"),
            "generation_call_count": int(target_llm_record.get("llm_generation_call_count", 1)),
            "frozen_mode": args.target_llm_mode,
            "record_path": str(args.target_llm_record),
        },
        "patches": [asdict(patch) for patch in patches],
        "calibration_seed_start": args.calibration_seed_start,
        "calibration_seed_count": args.calibration_seeds,
        "heldout_seed_start": args.heldout_seed_start,
        "heldout_seed_count": args.heldout_seeds,
        "initial_observations": args.initial,
        "rounds": args.rounds,
        "router": {
            "min_observations": args.router_min_observations,
            "min_quality": args.router_min_quality,
        },
        "selection": selection,
        "calibration": calibration,
        "heldout": heldout,
        "heldout_pairwise": pairwise,
        "evidence_boundary": (
            "The portfolio and candidate identities are frozen before replay. Source history, "
            "mappings, and patches are fixed before target replay. Target outcomes enter only "
            "after reveal and are used prequentially; held-out outcomes never select the "
            "portfolio candidate."
        ),
    }
    # Keep the legacy CSV writer's rectangular schema while retaining the
    # candidate identity on route rows and in the JSONL audit/summary.
    for row in rows:
        row.setdefault("portfolio_candidate_id", "")
        row.setdefault("router_max_transfer_mass", "")
        row.setdefault("source_initial_strategy", "")
    output_id = (
        f"calibrated_source_outcome_portfolio_{args.source_dataset}_to_{args.target_dataset}"
    )
    if args.output_tag:
        output_id += f"_{args.output_tag}"
    rows.sort(key=lambda row: (str(row.get("split", "")), int(row["seed"]), str(row["mode"])))
    outcome_router.write_outputs(output_id, rows, audits, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
