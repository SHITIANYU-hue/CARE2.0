#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_surrogate_baselines as surrogate
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

HybridMode = str
DEFAULT_MODES: tuple[HybridMode, ...] = (
    "gp_ucb",
    "hybrid_transfer_gp_ucb_gate_v1",
    "hybrid_transfer_gp_ucb_no_gate",
    "hybrid_value_prior_gp_ucb_gate_v1",
    "hybrid_value_prior_gp_ucb_no_gate",
    "hybrid_value_prior_gp_ucb_strict_gate_v1",
    "hybrid_value_prior_gp_ucb_target_calibrated_gate_v1",
    "hybrid_value_prior_gp_ucb_adaptive_gate_v1",
    "hybrid_value_prior_gp_ucb_warm3_gate_v1",
    "hybrid_value_prior_gp_ucb_warm5_gate_v1",
)


def parse_modes(raw: str) -> tuple[HybridMode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(DEFAULT_MODES)}")
    return modes


def is_value_prior_mode(mode: HybridMode) -> bool:
    return mode in {
        "hybrid_value_prior_gp_ucb_gate_v1",
        "hybrid_value_prior_gp_ucb_no_gate",
        "hybrid_value_prior_gp_ucb_strict_gate_v1",
        "hybrid_value_prior_gp_ucb_target_calibrated_gate_v1",
        "hybrid_value_prior_gp_ucb_adaptive_gate_v1",
        "hybrid_value_prior_gp_ucb_warm3_gate_v1",
        "hybrid_value_prior_gp_ucb_warm5_gate_v1",
    }


def is_no_gate_mode(mode: HybridMode) -> bool:
    return mode in {"hybrid_transfer_gp_ucb_no_gate", "hybrid_value_prior_gp_ucb_no_gate"}


def warm_start_round_limit(mode: HybridMode) -> int | None:
    if mode == "hybrid_value_prior_gp_ucb_warm3_gate_v1":
        return 3
    if mode == "hybrid_value_prior_gp_ucb_warm5_gate_v1":
        return 5
    return None


def uses_strict_source_value_prior(mode: HybridMode) -> bool:
    return mode in {
        "hybrid_value_prior_gp_ucb_strict_gate_v1",
        "hybrid_value_prior_gp_ucb_adaptive_gate_v1",
    }


def uses_target_calibrated_value_prior(mode: HybridMode) -> bool:
    return mode in {
        "hybrid_value_prior_gp_ucb_target_calibrated_gate_v1",
        "hybrid_value_prior_gp_ucb_adaptive_gate_v1",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    numeric_fields = [
        "final_best",
        "best_so_far_auc",
        "simple_regret",
        "top10_hit",
        "intervention_count",
        "bad_intervention_count",
        "rejected_good_challenger_count",
        "transfer_active_rounds",
        "transfer_scored_candidates_total",
        "mean_transfer_confidence",
    ]
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in numeric_fields:
            values = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(values), 4),
                "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
            }
    return out


def run_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: HybridMode,
    card: transfer.TransferCard,
    min_target_support: int,
    effect_threshold: float,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[replay.AuditEntry]]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {candidate.candidate_id: surrogate.candidate_features(adapter, candidate) for candidate in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    audit: list[replay.AuditEntry] = []
    best_trace: list[float] = []
    intervention_count = 0
    bad_interventions = 0
    rejected_good_challengers = 0
    transfer_active_rounds = 0
    transfer_scored_candidates_total = 0

    for round_index in range(task.reveal_budget):
        base_scores, gp_diagnostics = surrogate.gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            "mixed_kernel_gp_ucb",
            gp_beta,
            0.005,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
        transfer_cert: dict[str, Any] | None = None
        value_prior_cert: dict[str, Any] | None = None
        target_calibrated_value_prior_cert: dict[str, Any] | None = None
        adjustments = {candidate_id: 0.0 for candidate_id in base_scores}
        target_calibrated_value_prior_adjustments: dict[str, float] | None = None
        active_skill_ids: list[str] = []
        row_order_stable = True

        warm_limit = warm_start_round_limit(mode)
        if mode == "gp_ucb" or (warm_limit is not None and round_index >= warm_limit):
            incumbent = replay.top_candidate(base_scores)
            gate = replay.GateCertificate(
                gate_version="none",
                incumbent_candidate=incumbent,
                challenger_candidate=incumbent,
                selected_candidate=incumbent,
                authorized=False,
                gate_margin=0.0,
                acquisition_loss=0.0,
                row_order_stable=True,
                applied_skill_ids=(),
                reason="baseline_mixed_kernel_gp_ucb" if mode == "gp_ucb" else "warm_start_transfer_expired_gp_ucb",
            )
        else:
            transfer_adjustments, transfer_cert = transfer.transfer_adjustments(
                adapter,
                pool,
                observed_ids,
                observed,
                card,
                min_target_support,
                effect_threshold,
            )
            row_order_stable = transfer.transfer_row_order_stability_check(
                adapter,
                pool,
                observed_ids,
                observed,
                card,
                min_target_support,
                effect_threshold,
                transfer_adjustments,
            )
            transfer_skill = transfer_cert["skills"]["cross_domain_transfer_card"]
            if transfer_skill.get("active"):
                active_skill_ids.append("cross_domain_transfer_card")
                transfer_scored_candidates_total += int(transfer_skill.get("scored_candidates", 0))

            if is_value_prior_mode(mode):
                value_prior_adjustments: dict[str, float] | None = None
                if mode != "hybrid_value_prior_gp_ucb_target_calibrated_gate_v1":
                    strict_source_value_prior = uses_strict_source_value_prior(mode)
                    source_value_prior_skill_id = (
                        "strict_source_value_prior" if strict_source_value_prior else "source_value_prior"
                    )
                    value_prior_adjustments, value_prior_cert = transfer.source_value_prior_adjustments(
                        pool,
                        observed_ids,
                        card,
                        strict=strict_source_value_prior,
                        min_prior_confidence=0.10,
                        min_positive_priors=1,
                        max_negative_priors=0,
                        skill_id=source_value_prior_skill_id,
                        signed_adjustment_cap=0.08,
                        positive_adjustment_cap=0.05,
                    )
                    value_prior_skill = value_prior_cert["skills"][source_value_prior_skill_id]
                    if value_prior_skill.get("active"):
                        active_skill_ids.append(source_value_prior_skill_id)
                        transfer_scored_candidates_total += int(value_prior_skill.get("scored_candidates", 0))

                if uses_target_calibrated_value_prior(mode):
                    target_calibrated_value_prior_adjustments, target_calibrated_value_prior_cert = (
                        transfer.target_calibrated_descriptor_prior_adjustments(
                            adapter,
                            pool,
                            observed_ids,
                            observed,
                            card,
                            min_target_support,
                            max(1.5, effect_threshold * 0.5),
                            strict=False,
                            signed_adjustment_cap=0.05,
                            positive_adjustment_cap=0.04,
                        )
                    )
                    target_calibrated_skill = target_calibrated_value_prior_cert["skills"][
                        "target_calibrated_descriptor_prior"
                    ]
                    if target_calibrated_skill.get("active"):
                        active_skill_ids.append("target_calibrated_descriptor_prior")
                        transfer_scored_candidates_total += int(target_calibrated_skill.get("scored_candidates", 0))

                if mode == "hybrid_value_prior_gp_ucb_target_calibrated_gate_v1":
                    adjustments = transfer.combine_adjustments(
                        transfer_adjustments,
                        target_calibrated_value_prior_adjustments or {},
                    )
                elif mode == "hybrid_value_prior_gp_ucb_adaptive_gate_v1":
                    adjustments = transfer.combine_adjustments(
                        transfer_adjustments,
                        transfer.combine_adjustments(
                            value_prior_adjustments or {},
                            target_calibrated_value_prior_adjustments or {},
                        ),
                    )
                else:
                    adjustments = transfer.combine_adjustments(transfer_adjustments, value_prior_adjustments or {})
            else:
                adjustments = transfer_adjustments

            adjusted_scores = {candidate_id: base_scores[candidate_id] + adjustments.get(candidate_id, 0.0) for candidate_id in base_scores}
            if is_no_gate_mode(mode):
                gate = replay.no_gate_decision(base_scores, adjusted_scores, row_order_stable, tuple(active_skill_ids))
            else:
                gate = replay.gate_decision("gate_v1", base_scores, adjusted_scores, adjustments, row_order_stable, tuple(active_skill_ids))

            transfer_active = False
            if transfer_cert and transfer_cert["skills"]["cross_domain_transfer_card"].get("active"):
                transfer_active = True
            if value_prior_cert and any(skill.get("active") for skill in value_prior_cert["skills"].values()):
                transfer_active = True
            if target_calibrated_value_prior_cert and any(
                skill.get("active") for skill in target_calibrated_value_prior_cert["skills"].values()
            ):
                transfer_active = True
            transfer_active_rounds += int(transfer_active)

            if gate.authorized:
                intervention_count += 1
                if by_id[gate.challenger_candidate].objective_value < by_id[gate.incumbent_candidate].objective_value:
                    bad_interventions += 1
            elif by_id[gate.challenger_candidate].objective_value > by_id[gate.incumbent_candidate].objective_value:
                rejected_good_challengers += 1

        selected = by_id[gate.selected_candidate]
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        hypothesis_snapshot: dict[str, Any] = {
            "hybrid_surrogate": {
                "base_acquisition": "mixed_kernel_gp_ucb",
                "gp_diagnostics": gp_diagnostics,
                "transfer_card": asdict(card),
            }
        }
        if transfer_cert is not None:
            hypothesis_snapshot["transfer_policy"] = transfer_cert
        if value_prior_cert is not None:
            hypothesis_snapshot["source_value_prior_policy"] = value_prior_cert
        if target_calibrated_value_prior_cert is not None:
            hypothesis_snapshot["target_calibrated_value_prior_policy"] = target_calibrated_value_prior_cert
        audit.append(
            replay.AuditEntry(
                dataset_id=adapter.dataset_id,
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by=mode if gate.authorized or mode == "gp_ucb" else "gp_ucb",
                gate=gate,
                revealed_value=selected.objective_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=hypothesis_snapshot,
            )
        )

    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
        "intervention_count": intervention_count,
        "bad_intervention_count": bad_interventions,
        "rejected_good_challenger_count": rejected_good_challengers,
        "transfer_active_rounds": transfer_active_rounds,
        "transfer_scored_candidates_total": transfer_scored_candidates_total,
        "mean_transfer_confidence": round(mean(role.confidence for role in card.roles), 4),
    }
    return metrics, audit


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[replay.AuditEntry]],
    cards: dict[int, transfer.TransferCard],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for seed, card in sorted(cards.items()):
        (OUTPUT_RUNS / f"{output_id}_transfer_card_seed{seed}.json").write_text(
            json.dumps(asdict(card), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for (mode, seed), audit in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for entry in audit:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def run_hybrid_transfer(
    source_dataset: str,
    target_dataset: str,
    seeds: int,
    rounds: int,
    initial: int,
    source_observation_count: int,
    discount: float,
    min_source_support: int,
    min_target_support: int,
    effect_threshold: float,
    gp_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    modes: tuple[HybridMode, ...],
    output_tag: str,
) -> dict[str, Any]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    role_map = transfer.role_map_for(source_dataset, target_dataset)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[replay.AuditEntry]] = {}
    cards: dict[int, transfer.TransferCard] = {}
    for seed in range(seeds):
        source_observed = transfer.source_observations(source_adapter, seed, source_observation_count)
        card = transfer.compile_transfer_card(
            source_adapter,
            target_adapter,
            source_observed,
            role_map,
            discount,
            min_source_support,
        )
        cards[seed] = card
        for mode in modes:
            metrics, audit = run_policy(
                target_adapter,
                task,
                seed,
                mode,
                card,
                min_target_support,
                effect_threshold,
                gp_beta,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit

    output_id = f"hybrid_surrogate_transfer_{source_dataset}_to_{target_dataset}"
    if output_tag:
        output_id = f"{output_id}_{output_tag}"
    summary = {
        "experiment": "care_hybrid_surrogate_transfer",
        "output_id": output_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "role_map": role_map,
        "base_acquisition": "mixed_kernel_gp_ucb",
        "transfer_boundary": (
            "The GP-UCB score is the incumbent acquisition. Transfer-card and "
            "shared-value-prior adjustments can only modify that acquisition "
            "through bounded adjustments and, in gate modes, must pass gate_v1."
        ),
        "source_observation_count": source_observation_count,
        "discount": discount,
        "min_source_support": min_source_support,
        "min_target_support": min_target_support,
        "effect_threshold": effect_threshold,
        "gp_beta": gp_beta,
        "numeric_length_scale": numeric_length_scale,
        "categorical_length_scale": categorical_length_scale,
        "gp_noise": gp_noise,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, summary, audits, cards)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE transfer over a mixed-kernel GP-UCB incumbent.")
    parser.add_argument("--source-dataset", default="real_moleculenet_freesolv", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_moleculenet_lipophilicity", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=96)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--min-target-support", type=int, default=2)
    parser.add_argument("--effect-threshold", type=float, default=4.0)
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(DEFAULT_MODES)}")
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()
    summary = run_hybrid_transfer(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        seeds=args.seeds,
        rounds=args.rounds,
        initial=args.initial,
        source_observation_count=args.source_observations,
        discount=args.discount,
        min_source_support=args.min_source_support,
        min_target_support=args.min_target_support,
        effect_threshold=args.effect_threshold,
        gp_beta=args.gp_beta,
        numeric_length_scale=args.numeric_length_scale,
        categorical_length_scale=args.categorical_length_scale,
        gp_noise=args.gp_noise,
        modes=parse_modes(args.modes),
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
