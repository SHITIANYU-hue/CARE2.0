#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

TransferMode = str
DEFAULT_MODES: tuple[TransferMode, ...] = (
    "no_care_random",
    "incumbent",
    "target_local_no_gate",
    "target_local_gate_v1",
    "transfer_no_gate",
    "transfer_gate_v1",
    "transfer_plus_local_gate_v1",
    "transfer_strict_no_gate",
    "transfer_strict_gate_v1",
    "transfer_strict_plus_local_gate_v1",
)


@dataclass(frozen=True)
class TransferRole:
    source_field: str
    target_field: str
    source_support_count: int
    source_mean_abs_effect: float
    source_max_abs_effect: float
    confidence: float
    transfer_weight: float


@dataclass(frozen=True)
class TransferCard:
    card_id: str
    source_dataset: str
    target_dataset: str
    source_observation_count: int
    discount: float
    min_source_support: int
    role_map: dict[str, str]
    roles: tuple[TransferRole, ...]
    evidence_summary: str


def parse_modes(raw: str) -> tuple[TransferMode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(DEFAULT_MODES)}")
    return modes


def role_map_for(source_dataset: str, target_dataset: str) -> dict[str, str]:
    if source_dataset == "real_buchwald_hartwig" and target_dataset == "real_suzuki_miyaura":
        return {
            "ligand": "ligand",
            "base": "reagent",
            "aryl_halide": "reactant_1",
            "additive": "solvent",
        }
    raise ValueError(f"No role map defined for {source_dataset} -> {target_dataset}")


def source_observations(adapter: replay.DatasetAdapter, seed: int, count: int) -> list[replay.Candidate]:
    shuffled = list(adapter.candidates)
    random.Random(10_000 + seed).shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def compile_transfer_card(
    source_adapter: replay.DatasetAdapter,
    target_adapter: replay.DatasetAdapter,
    observed: list[replay.Candidate],
    role_map: dict[str, str],
    discount: float,
    min_source_support: int,
) -> TransferCard:
    factor_summary = replay.factor_stats(observed, source_adapter.decision_columns)
    global_mean = replay.observed_mean(observed)
    effects_by_field: dict[str, list[float]] = {field: [] for field in role_map}
    for (field_name, _value), (count, value_mean) in factor_summary.items():
        if field_name not in role_map or count < min_source_support:
            continue
        effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
        effects_by_field[field_name].append(effect)

    roles: list[TransferRole] = []
    for source_field, target_field in role_map.items():
        effects = effects_by_field[source_field]
        source_support_count = len(effects)
        mean_abs = mean(abs(effect) for effect in effects) if effects else 0.0
        max_abs = max((abs(effect) for effect in effects), default=0.0)
        support_confidence = min(1.0, math.log1p(source_support_count) / math.log1p(12.0))
        effect_confidence = min(1.0, max_abs / 18.0)
        confidence = round(discount * support_confidence * effect_confidence, 4)
        transfer_weight = round(min(1.20, 0.35 + confidence), 4) if source_support_count else 0.0
        roles.append(
            TransferRole(
                source_field=source_field,
                target_field=target_field,
                source_support_count=source_support_count,
                source_mean_abs_effect=round(mean_abs, 4),
                source_max_abs_effect=round(max_abs, 4),
                confidence=confidence,
                transfer_weight=transfer_weight,
            )
        )

    active = [role for role in roles if role.transfer_weight > 0.0]
    summary = (
        f"Compiled {len(active)} active role transfers from {source_adapter.dataset_id} "
        f"using {len(observed)} observed source rows. Direction is not transferred as a "
        "raw chemical rule; the card transfers role-level evidence strength and lets "
        "target observations determine candidate-level direction."
    )
    return TransferCard(
        card_id=f"transfer_{source_adapter.dataset_id}_to_{target_adapter.dataset_id}",
        source_dataset=source_adapter.dataset_id,
        target_dataset=target_adapter.dataset_id,
        source_observation_count=len(observed),
        discount=discount,
        min_source_support=min_source_support,
        role_map=dict(role_map),
        roles=tuple(roles),
        evidence_summary=summary,
    )


def transfer_adjustments(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    strict: bool = False,
    min_role_confidence: float = 0.18,
    min_positive_roles: int = 2,
    max_negative_roles: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    if len(observed) < 8:
        return adjustments, {
            "skills": {"cross_domain_transfer_card": {"active": False, "reason": "observed_count_below_8"}},
            "max_abs_adjustment": 0.0,
        }

    factor_summary = replay.factor_stats(observed, adapter.decision_columns)
    global_mean = replay.observed_mean(observed)
    role_by_target = {
        role.target_field: role
        for role in card.roles
        if role.transfer_weight > 0.0 and (not strict or role.confidence >= min_role_confidence)
    }
    applied_specs: list[dict[str, Any]] = []
    positive = 0
    negative = 0
    scored = 0
    strict_rejected = 0

    for c in pool:
        if c.candidate_id not in adjustments:
            continue
        signals: list[float] = []
        for target_field, role in role_by_target.items():
            value = str(c.metadata.get(target_field, ""))
            if not value:
                continue
            count, value_mean = factor_summary.get((target_field, value), (0, global_mean))
            if count < min_target_support:
                continue
            effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
            if abs(effect) < effect_threshold:
                continue
            signals.append((effect / 100.0) * role.transfer_weight)
        if not signals:
            continue
        if strict:
            positive_signals = [signal for signal in signals if signal > 0]
            negative_signals = [signal for signal in signals if signal < 0]
            if len(positive_signals) < min_positive_roles or len(negative_signals) > max_negative_roles:
                strict_rejected += 1
                continue
            bounded = max(0.0, min(0.055, mean(positive_signals)))
        else:
            bounded = max(-0.10, min(0.10, mean(signals)))
        adjustments[c.candidate_id] += bounded
        scored += 1
        positive += int(bounded > 0)
        negative += int(bounded < 0)

    for target_field, role in role_by_target.items():
        active_values = []
        for (field_name, value), (count, value_mean) in factor_summary.items():
            if field_name != target_field or count < min_target_support:
                continue
            effect = replay.smoothed_mean(count, value_mean, global_mean, prior_weight=2.0) - global_mean
            if abs(effect) >= effect_threshold:
                active_values.append(
                    {
                        "value": value,
                        "count": count,
                        "effect": round(effect, 4),
                    }
                )
        applied_specs.append(
            {
                "source_field": role.source_field,
                "target_field": target_field,
                "transfer_weight": role.transfer_weight,
                "confidence": role.confidence,
                "strict_enabled": strict,
                "active_target_values": sorted(active_values, key=lambda item: abs(item["effect"]), reverse=True)[:8],
            }
        )

    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert = {
        "skills": {
            "cross_domain_transfer_card": {
                "active": scored > 0,
                "card_id": card.card_id,
                "scored_candidates": scored,
                "positive_adjustments": positive,
                "negative_adjustments": negative,
                "strict_rejected_candidates": strict_rejected,
                "strict_min_role_confidence": min_role_confidence if strict else 0.0,
                "strict_min_positive_roles": min_positive_roles if strict else 0,
                "strict_max_negative_roles": max_negative_roles if strict else 0,
                "applied_specs": applied_specs,
            }
        },
        "max_abs_adjustment": round(max_abs, 6),
    }
    return adjustments, cert


def transfer_row_order_stability_check(
    adapter: replay.DatasetAdapter,
    pool: tuple[replay.Candidate, ...],
    observed_ids: set[str],
    observed: list[replay.Candidate],
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    reference_adjustments: dict[str, float],
    strict: bool = False,
    min_role_confidence: float = 0.18,
    min_positive_roles: int = 2,
    max_negative_roles: int = 0,
) -> bool:
    shuffled = list(pool)
    random.Random(20_000 + len(observed)).shuffle(shuffled)
    shuffled_adjustments, _ = transfer_adjustments(
        adapter,
        tuple(shuffled),
        observed_ids,
        observed,
        card,
        min_target_support,
        effect_threshold,
        strict,
        min_role_confidence,
        min_positive_roles,
        max_negative_roles,
    )
    return all(abs(reference_adjustments[k] - shuffled_adjustments[k]) < 1e-12 for k in reference_adjustments)


def combine_adjustments(*items: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for adjustments in items:
        for key, value in adjustments.items():
            out[key] = out.get(key, 0.0) + value
    for key, value in list(out.items()):
        out[key] = max(-0.14, min(0.14, value))
    return out


def run_target_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: TransferMode,
    card: TransferCard,
    min_target_support: int,
    effect_threshold: float,
    strict_min_role_confidence: float,
    strict_min_positive_roles: int,
    strict_max_negative_roles: int,
) -> tuple[dict[str, Any], list[replay.AuditEntry], replay.HypothesisEntry]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {c.candidate_id: c for c in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {c.candidate_id for c in observed}
    skills = replay.make_skills(adapter)
    hypothesis = replay.make_hypothesis(adapter)
    if mode == "no_care_random":
        hypothesis.status = "inactive"
        hypothesis.evidence_summary = "No CARE hypothesis or transfer card is used in this random-search baseline."

    audit: list[replay.AuditEntry] = []
    top10 = {c.candidate_id for c in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    best_trace: list[float] = []
    intervention_count = 0
    bad_interventions = 0
    rejected_good_challengers = 0
    selected_top10 = any(c.candidate_id in top10 for c in observed)
    transfer_active_rounds = 0
    transfer_scored_candidates_total = 0

    for round_index in range(task.reveal_budget):
        transfer_cert: dict[str, Any] | None = None
        if mode == "no_care_random":
            selected_candidate = rng.choice([c for c in pool if c.candidate_id not in observed_ids])
            gate = replay.GateCertificate(
                gate_version="no_care",
                incumbent_candidate=selected_candidate.candidate_id,
                challenger_candidate=selected_candidate.candidate_id,
                selected_candidate=selected_candidate.candidate_id,
                authorized=False,
                gate_margin=0.0,
                acquisition_loss=0.0,
                row_order_stable=True,
                applied_skill_ids=(),
                reason="baseline_random_search_no_care",
            )
        else:
            base_scores = replay.public_incumbent_scores(adapter, observed_ids, observed)
            if mode == "incumbent":
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
                    reason="baseline_incumbent_only",
                )
            else:
                local_adjustments: dict[str, float] | None = None
                local_cert: dict[str, Any] | None = None
                transfer_adjustment_values: dict[str, float] | None = None
                row_order_stable = True
                active_skill_ids: list[str] = []

                if mode in {"target_local_no_gate", "target_local_gate_v1", "transfer_plus_local_gate_v1", "transfer_strict_plus_local_gate_v1"}:
                    local_adjustments, local_cert = replay.skill_adjustments(
                        adapter, pool, observed_ids, observed, skills, round_index
                    )
                    row_order_stable = row_order_stable and replay.row_order_stability_check(
                        adapter, pool, observed_ids, observed, skills, round_index, local_adjustments
                    )
                    active_skill_ids.extend(k for k, v in local_cert["skills"].items() if v.get("active"))

                if mode in {
                    "transfer_no_gate",
                    "transfer_gate_v1",
                    "transfer_plus_local_gate_v1",
                    "transfer_strict_no_gate",
                    "transfer_strict_gate_v1",
                    "transfer_strict_plus_local_gate_v1",
                }:
                    strict_transfer = mode in {
                        "transfer_strict_no_gate",
                        "transfer_strict_gate_v1",
                        "transfer_strict_plus_local_gate_v1",
                    }
                    transfer_adjustment_values, transfer_cert = transfer_adjustments(
                        adapter,
                        pool,
                        observed_ids,
                        observed,
                        card,
                        min_target_support,
                        effect_threshold,
                        strict_transfer,
                        strict_min_role_confidence,
                        strict_min_positive_roles,
                        strict_max_negative_roles,
                    )
                    row_order_stable = row_order_stable and transfer_row_order_stability_check(
                        adapter,
                        pool,
                        observed_ids,
                        observed,
                        card,
                        min_target_support,
                        effect_threshold,
                        transfer_adjustment_values,
                        strict_transfer,
                        strict_min_role_confidence,
                        strict_min_positive_roles,
                        strict_max_negative_roles,
                    )
                    transfer_skill = transfer_cert["skills"]["cross_domain_transfer_card"]
                    if transfer_skill.get("active"):
                        transfer_active_rounds += 1
                        transfer_scored_candidates_total += int(transfer_skill.get("scored_candidates", 0))
                        active_skill_ids.append("cross_domain_transfer_card")

                if mode in {"transfer_plus_local_gate_v1", "transfer_strict_plus_local_gate_v1"}:
                    adjustments = combine_adjustments(local_adjustments or {}, transfer_adjustment_values or {})
                else:
                    adjustments = local_adjustments or transfer_adjustment_values or {
                        c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids
                    }
                adjusted_scores = {cid: base_scores[cid] + adjustments.get(cid, 0.0) for cid in base_scores}
                if mode in {"target_local_no_gate", "transfer_no_gate", "transfer_strict_no_gate"}:
                    gate = replay.no_gate_decision(base_scores, adjusted_scores, row_order_stable, tuple(active_skill_ids))
                else:
                    gate = replay.gate_decision("gate_v1", base_scores, adjusted_scores, adjustments, row_order_stable, tuple(active_skill_ids))

                if gate.authorized:
                    intervention_count += 1
                    if by_id[gate.challenger_candidate].objective_value < by_id[gate.incumbent_candidate].objective_value:
                        bad_interventions += 1
                elif by_id[gate.challenger_candidate].objective_value > by_id[gate.incumbent_candidate].objective_value:
                    rejected_good_challengers += 1

        selected = by_id[gate.selected_candidate]
        if mode != "no_care_random":
            replay.update_hypothesis_from_reveal(hypothesis, selected, observed, round_index, adapter.preferred_groups)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(c.objective_value for c in observed)
        best_trace.append(best_so_far)
        hypothesis_snapshot = asdict(hypothesis)
        hypothesis_snapshot["transfer_card"] = asdict(card)
        if transfer_cert is not None:
            hypothesis_snapshot["transfer_policy"] = transfer_cert
        audit.append(
            replay.AuditEntry(
                dataset_id=adapter.dataset_id,
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by=mode if gate.authorized or mode in {"no_care_random", "incumbent"} else "incumbent",
                gate=gate,
                revealed_value=selected.objective_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=hypothesis_snapshot,
            )
        )

    final_best = max(c.objective_value for c in observed)
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
        "source_observation_count": card.source_observation_count,
        "mean_transfer_confidence": round(mean(role.confidence for role in card.roles), 4),
        "hypothesis_confidence": round(hypothesis.confidence, 4),
        "hypothesis_support_count": hypothesis.support_count,
    }
    return metrics, audit, hypothesis


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
        "source_observation_count",
        "mean_transfer_confidence",
        "hypothesis_confidence",
        "hypothesis_support_count",
    ]
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in numeric_fields:
            vals = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(vals), 4),
                "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audits: dict[tuple[str, int], list[replay.AuditEntry]],
    hypotheses: dict[tuple[str, int], replay.HypothesisEntry],
    cards: dict[int, TransferCard],
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
    for (mode, seed), hypothesis in sorted(hypotheses.items()):
        (OUTPUT_RUNS / f"{output_id}_knowledge_{mode}_seed{seed}.json").write_text(
            json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def run_transfer_ablation(
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
    strict_min_role_confidence: float,
    strict_min_positive_roles: int,
    strict_max_negative_roles: int,
    modes: tuple[TransferMode, ...],
    output_tag: str,
) -> dict[str, Any]:
    source_adapter = replay.DATASET_BUILDERS[source_dataset]()
    target_adapter = replay.DATASET_BUILDERS[target_dataset]()
    task = replay.make_task(target_adapter, initial, rounds)
    role_map = role_map_for(source_dataset, target_dataset)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[replay.AuditEntry]] = {}
    hypotheses: dict[tuple[str, int], replay.HypothesisEntry] = {}
    cards: dict[int, TransferCard] = {}
    for seed in range(seeds):
        source_observed = source_observations(source_adapter, seed, source_observation_count)
        card = compile_transfer_card(
            source_adapter,
            target_adapter,
            source_observed,
            role_map,
            discount,
            min_source_support,
        )
        cards[seed] = card
        for mode in modes:
            metrics, audit, hypothesis = run_target_policy(
                target_adapter,
                task,
                seed,
                mode,
                card,
                min_target_support,
                effect_threshold,
                strict_min_role_confidence,
                strict_min_positive_roles,
                strict_max_negative_roles,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit
            hypotheses[(mode, seed)] = hypothesis

    output_id = f"transfer_{source_dataset}_to_{target_dataset}" if not output_tag else f"transfer_{source_dataset}_to_{target_dataset}_{output_tag}"
    summary = {
        "experiment": "care_cross_domain_transfer_ablation",
        "output_id": output_id,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "role_map": role_map,
        "transfer_boundary": (
            "Source outcomes are used only to estimate role-level evidence strength. "
            "No target hidden outcomes or direct source factor values are transferred."
        ),
        "source_observation_count": source_observation_count,
        "discount": discount,
        "min_source_support": min_source_support,
        "min_target_support": min_target_support,
        "effect_threshold": effect_threshold,
        "strict_min_role_confidence": strict_min_role_confidence,
        "strict_min_positive_roles": strict_min_positive_roles,
        "strict_max_negative_roles": strict_max_negative_roles,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, summary, audits, hypotheses, cards)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE 2.0 cross-domain transfer ablations.")
    parser.add_argument("--source-dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", default="real_suzuki_miyaura", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--source-observations", type=int, default=48)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--min-source-support", type=int, default=3)
    parser.add_argument("--min-target-support", type=int, default=2)
    parser.add_argument("--effect-threshold", type=float, default=4.0)
    parser.add_argument("--strict-min-role-confidence", type=float, default=0.18)
    parser.add_argument("--strict-min-positive-roles", type=int, default=2)
    parser.add_argument("--strict-max-negative-roles", type=int, default=0)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(DEFAULT_MODES)}")
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()

    summary = run_transfer_ablation(
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
        strict_min_role_confidence=args.strict_min_role_confidence,
        strict_min_positive_roles=args.strict_min_positive_roles,
        strict_max_negative_roles=args.strict_max_negative_roles,
        modes=parse_modes(args.modes),
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
