#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

SkillFamily = Literal["ranker", "constraint", "exploration", "data_analysis", "fallback"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    ligand_identity: str
    residence_time: float
    temperature: float
    catalyst_loading: float
    yield_value: float


@dataclass(frozen=True)
class TaskSpec:
    dataset_id: str
    objective: str
    decision_columns: tuple[str, ...]
    hidden_target: str
    initial_observations: int
    reveal_budget: int
    oracle_value: float


@dataclass(frozen=True)
class SkillCard:
    skill_id: str
    version: str
    family: SkillFamily
    scope: str
    trigger_rules: tuple[dict[str, Any], ...]
    bounded_parameters: dict[str, Any]
    certificate_schema: dict[str, Any]
    required_checks: tuple[str, ...]
    prohibited_behaviors: tuple[str, ...]
    provenance: dict[str, Any]
    rationale: str


@dataclass
class HypothesisEntry:
    hypothesis_id: str
    status: Literal["active", "inactive", "falsified", "pending"]
    scope: Literal["group_preference", "similarity_region", "mechanism"]
    trigger: dict[str, Any]
    target_spec: dict[str, Any]
    claim: str
    confidence: float
    support_count: int
    alpha: float
    beta: float
    evidence_summary: str
    known_failure_modes: list[str]
    created_round: int
    last_updated_round: int

    def update(self, supports: bool, round_index: int, evidence_summary: str) -> None:
        if supports:
            self.alpha += 1.0
            self.support_count += 1
        else:
            self.beta += 1.0
        self.confidence = self.alpha / (self.alpha + self.beta)
        self.last_updated_round = round_index
        self.evidence_summary = evidence_summary


@dataclass(frozen=True)
class GateCertificate:
    gate_version: str
    incumbent_candidate: str
    challenger_candidate: str
    selected_candidate: str
    authorized: bool
    gate_margin: float
    acquisition_loss: float
    row_order_stable: bool
    applied_skill_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class AuditEntry:
    seed: int
    round_index: int
    public_observed_count: int
    incumbent_candidate: str
    challenger_candidate: str
    selected_candidate: str
    selected_by: str
    gate: GateCertificate
    revealed_yield: float
    best_so_far: float
    hypothesis_snapshot: dict[str, Any]


def stable_noise(*parts: object, scale: float = 2.0) -> float:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 2.0 * scale


def synthetic_suzuki_pool() -> list[Candidate]:
    ligands = [f"L{i}" for i in range(7)]
    residence_times = [30.0, 60.0, 90.0, 120.0]
    temperatures = [50.0, 70.0, 90.0, 110.0]
    loadings = [0.5, 1.0, 2.0, 4.0]
    ligand_base = {
        "L0": 42.0,
        "L1": 48.0,
        "L2": 76.0,
        "L3": 62.0,
        "L4": 58.0,
        "L5": 72.0,
        "L6": 55.0,
    }
    ligand_temp_opt = {"L0": 70.0, "L1": 70.0, "L2": 90.0, "L3": 90.0, "L4": 70.0, "L5": 110.0, "L6": 90.0}
    pool: list[Candidate] = []
    for ligand in ligands:
        for time in residence_times:
            for temp in temperatures:
                for loading in loadings:
                    temp_effect = -0.018 * (temp - ligand_temp_opt[ligand]) ** 2
                    time_effect = -0.0018 * (time - 90.0) ** 2 + 5.0
                    loading_effect = 4.0 * math.log1p(loading) - 1.1 * loading
                    interaction = 4.0 if ligand in {"L2", "L5"} and temp >= 90.0 and time >= 90.0 else 0.0
                    low_temp_low_loading_penalty = -10.0 if temp <= 50.0 and loading <= 0.5 else 0.0
                    y = (
                        ligand_base[ligand]
                        + temp_effect
                        + time_effect
                        + loading_effect
                        + interaction
                        + low_temp_low_loading_penalty
                        + stable_noise(ligand, time, temp, loading, scale=2.3)
                    )
                    y = max(0.0, min(100.0, y))
                    cid = f"{ligand}_T{int(temp)}_R{int(time)}_C{str(loading).replace('.', 'p')}"
                    pool.append(Candidate(cid, ligand, time, temp, loading, round(y, 4)))
    return pool


def make_task(pool: list[Candidate], initial_observations: int, reveal_budget: int) -> TaskSpec:
    return TaskSpec(
        dataset_id="synthetic_suzuki_i",
        objective="maximize_yield",
        decision_columns=("ligand_identity", "residence_time", "temperature", "catalyst_loading"),
        hidden_target="yield_value",
        initial_observations=initial_observations,
        reveal_budget=reveal_budget,
        oracle_value=max(c.yield_value for c in pool),
    )


def make_skills() -> list[SkillCard]:
    return [
        SkillCard(
            skill_id="suzuki_ligand_prior",
            version="1.0.0",
            family="ranker",
            scope="Add bounded prior bonus to literature-preferred Suzuki ligands.",
            trigger_rules=({"field": "round_index", "operator": ">=", "value": 0},),
            bounded_parameters={"preferred_ligands": {"value": ["L2", "L5"]}, "prior_bonus_cap": {"value": 0.10}},
            certificate_schema={"required": ["applied_bonuses", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id", "modify_observed_data"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": "synthetic_suzuki_i"},
            rationale="Convert an unstable natural-language ligand prior into bounded full-pool score adjustments.",
        ),
        SkillCard(
            skill_id="suzuki_ligand_risk_penalty",
            version="1.0.0",
            family="constraint",
            scope="Penalize ligand groups that repeatedly underperform in public observations.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
            bounded_parameters={"penalty_cap": {"value": -0.12}, "min_support": {"value": 2}},
            certificate_schema={"required": ["penalized_ligands", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("block_candidate_permanently", "access_hidden_outcomes"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": "synthetic_suzuki_i"},
            rationale="Use public repeated failures as bounded risk evidence without permanently blocking candidates.",
        ),
        SkillCard(
            skill_id="suzuki_ligand_diversity_explorer",
            version="1.0.0",
            family="exploration",
            scope="Add a small bounded bonus to unseen ligand groups.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
            bounded_parameters={"unseen_ligand_bonus_cap": {"value": 0.05}},
            certificate_schema={"required": ["unseen_ligands", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": "synthetic_suzuki_i"},
            rationale="Encourage controlled exploration of public feature groups not yet covered by observations.",
        ),
    ]


def make_hypothesis() -> HypothesisEntry:
    return HypothesisEntry(
        hypothesis_id="suzuki_L2_L5_high_yield_preference",
        status="active",
        scope="group_preference",
        trigger={"dataset": "synthetic_suzuki_i", "condition": "observed_count >= 4"},
        target_spec={"group_column": "ligand_identity", "group_values": ["L2", "L5"]},
        claim="Ligands L2 and L5 tend to produce higher Suzuki yields under suitable temperature and residence time.",
        confidence=0.5,
        support_count=0,
        alpha=1.0,
        beta=1.0,
        evidence_summary="Initialized from prior; no synthetic reveal evidence yet.",
        known_failure_modes=["Low temperature and low catalyst loading can erase the ligand advantage."],
        created_round=0,
        last_updated_round=0,
    )


def observed_mean(observed: list[Candidate]) -> float:
    return mean(c.yield_value for c in observed) if observed else 50.0


def ligand_stats(observed: list[Candidate]) -> dict[str, tuple[int, float]]:
    by_lig: dict[str, list[float]] = {}
    for c in observed:
        by_lig.setdefault(c.ligand_identity, []).append(c.yield_value)
    return {lig: (len(vals), mean(vals)) for lig, vals in by_lig.items()}


def public_incumbent_scores(pool: list[Candidate], observed_ids: set[str], observed: list[Candidate]) -> dict[str, float]:
    stats = ligand_stats(observed)
    global_mean = observed_mean(observed)
    scores: dict[str, float] = {}
    for c in pool:
        if c.candidate_id in observed_ids:
            continue
        count, lig_mean = stats.get(c.ligand_identity, (0, global_mean))
        uncertainty = 12.0 / math.sqrt(count + 1.0)
        public_condition_prior = (
            0.035 * (c.temperature / 110.0)
            + 0.020 * (c.residence_time / 120.0)
            + 0.015 * math.log1p(c.catalyst_loading)
        )
        estimated = (lig_mean + uncertainty) / 100.0 + public_condition_prior
        scores[c.candidate_id] = estimated
    return scores


def trigger_satisfied(rule: dict[str, Any], round_index: int, observed: list[Candidate]) -> bool:
    field = rule["field"]
    value = rule["value"]
    actual = round_index if field == "round_index" else len(observed)
    op = rule["operator"]
    if op == ">=":
        return actual >= value
    if op == "==":
        return actual == value
    raise ValueError(f"Unsupported operator: {op}")


def skill_adjustments(
    pool: list[Candidate],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    cert: dict[str, Any] = {"skills": {}, "max_abs_adjustment": 0.0}
    stats = ligand_stats(observed)
    global_mean = observed_mean(observed)
    seen_ligands = {c.ligand_identity for c in observed}
    for skill in skills:
        if not all(trigger_satisfied(rule, round_index, observed) for rule in skill.trigger_rules):
            cert["skills"][skill.skill_id] = {"active": False, "reason": "trigger_not_satisfied"}
            continue
        if skill.skill_id == "suzuki_ligand_prior":
            preferred = set(skill.bounded_parameters["preferred_ligands"]["value"])
            cap = float(skill.bounded_parameters["prior_bonus_cap"]["value"])
            applied = {}
            for c in pool:
                if c.candidate_id not in adjustments or c.ligand_identity not in preferred:
                    continue
                bonus = cap
                adjustments[c.candidate_id] += bonus
                applied[c.candidate_id] = bonus
            cert["skills"][skill.skill_id] = {"active": True, "applied_count": len(applied), "cap": cap}
        elif skill.skill_id == "suzuki_ligand_risk_penalty":
            cap = float(skill.bounded_parameters["penalty_cap"]["value"])
            min_support = int(skill.bounded_parameters["min_support"]["value"])
            risky = {lig for lig, (count, lig_mean) in stats.items() if count >= min_support and lig_mean < global_mean - 8.0}
            for c in pool:
                if c.candidate_id in adjustments and c.ligand_identity in risky:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "penalized_ligands": sorted(risky), "cap": cap}
        elif skill.skill_id == "suzuki_ligand_diversity_explorer":
            cap = float(skill.bounded_parameters["unseen_ligand_bonus_cap"]["value"])
            unseen = sorted({c.ligand_identity for c in pool} - seen_ligands)
            for c in pool:
                if c.candidate_id in adjustments and c.ligand_identity in unseen:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "unseen_ligands": unseen, "cap": cap}
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert["max_abs_adjustment"] = round(max_abs, 6)
    return adjustments, cert


def top_candidate(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]


def row_order_stability_check(
    pool: list[Candidate],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
    reference_adjustments: dict[str, float],
) -> bool:
    shuffled = list(pool)
    random.Random(1000 + round_index + len(observed)).shuffle(shuffled)
    shuffled_adjustments, _ = skill_adjustments(shuffled, observed_ids, observed, skills, round_index)
    return all(abs(reference_adjustments[k] - shuffled_adjustments[k]) < 1e-12 for k in reference_adjustments)


def gate_decision(
    gate_version: str,
    base_scores: dict[str, float],
    adjusted_scores: dict[str, float],
    adjustments: dict[str, float],
    row_order_stable: bool,
    active_skill_ids: tuple[str, ...],
) -> GateCertificate:
    incumbent = top_candidate(base_scores)
    challenger = top_candidate(adjusted_scores)
    if challenger == incumbent:
        return GateCertificate(
            gate_version=gate_version,
            incumbent_candidate=incumbent,
            challenger_candidate=challenger,
            selected_candidate=incumbent,
            authorized=False,
            gate_margin=0.0,
            acquisition_loss=0.0,
            row_order_stable=row_order_stable,
            applied_skill_ids=active_skill_ids,
            reason="challenger_matches_incumbent",
        )
    epsilon = 0.05 if gate_version == "gate_v1" else 0.12
    min_margin = 0.025 if gate_version == "gate_v1" else 0.010
    gate_margin = adjusted_scores[challenger] - base_scores[incumbent]
    acquisition_loss = max(0.0, base_scores[incumbent] - base_scores[challenger])
    max_adjustment = max(abs(v) for v in adjustments.values()) if adjustments else 0.0
    authorized = row_order_stable and max_adjustment <= 0.20 and gate_margin >= min_margin and acquisition_loss <= epsilon
    reason = "authorized_bounded_skill_adjustment" if authorized else "rejected_by_gate_bounds"
    return GateCertificate(
        gate_version=gate_version,
        incumbent_candidate=incumbent,
        challenger_candidate=challenger,
        selected_candidate=challenger if authorized else incumbent,
        authorized=authorized,
        gate_margin=round(gate_margin, 6),
        acquisition_loss=round(acquisition_loss, 6),
        row_order_stable=row_order_stable,
        applied_skill_ids=active_skill_ids,
        reason=reason,
    )


def update_hypothesis_from_reveal(h: HypothesisEntry, selected: Candidate, observed: list[Candidate], round_index: int) -> None:
    if selected.ligand_identity not in {"L2", "L5"}:
        return
    public_mean_before = observed_mean(observed)
    supports = selected.yield_value >= public_mean_before
    h.update(
        supports=supports,
        round_index=round_index,
        evidence_summary=(
            f"Round {round_index}: {selected.candidate_id} yielded {selected.yield_value:.2f}; "
            f"public mean before reveal was {public_mean_before:.2f}; supports={supports}."
        ),
    )


def run_policy(
    pool: list[Candidate],
    task: TaskSpec,
    seed: int,
    mode: Literal["incumbent", "gate_v1", "gate_v2"],
) -> tuple[dict[str, Any], list[AuditEntry], HypothesisEntry]:
    rng = random.Random(seed)
    by_id = {c.candidate_id: c for c in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {c.candidate_id for c in observed}
    skills = make_skills()
    hypothesis = make_hypothesis()
    audit: list[AuditEntry] = []
    top10 = {c.candidate_id for c in sorted(pool, key=lambda x: x.yield_value, reverse=True)[:10]}
    best_trace: list[float] = []
    intervention_count = 0
    bad_interventions = 0
    rejected_good_challengers = 0
    selected_top10 = False

    for round_index in range(task.reveal_budget):
        base_scores = public_incumbent_scores(pool, observed_ids, observed)
        if mode == "incumbent":
            incumbent = top_candidate(base_scores)
            gate = GateCertificate(
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
            adjustments, skill_cert = skill_adjustments(pool, observed_ids, observed, skills, round_index)
            adjusted_scores = {cid: base_scores[cid] + adjustments.get(cid, 0.0) for cid in base_scores}
            row_order_stable = row_order_stability_check(pool, observed_ids, observed, skills, round_index, adjustments)
            active_skill_ids = tuple(k for k, v in skill_cert["skills"].items() if v.get("active"))
            gate = gate_decision(mode, base_scores, adjusted_scores, adjustments, row_order_stable, active_skill_ids)
            if gate.authorized:
                intervention_count += 1
                if by_id[gate.challenger_candidate].yield_value < by_id[gate.incumbent_candidate].yield_value:
                    bad_interventions += 1
            elif by_id[gate.challenger_candidate].yield_value > by_id[gate.incumbent_candidate].yield_value:
                rejected_good_challengers += 1

        selected = by_id[gate.selected_candidate]
        update_hypothesis_from_reveal(hypothesis, selected, observed, round_index)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(c.yield_value for c in observed)
        best_trace.append(best_so_far)
        audit.append(
            AuditEntry(
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by="gate_authorized_challenger" if gate.authorized else "incumbent",
                gate=gate,
                revealed_yield=selected.yield_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=asdict(hypothesis),
            )
        )
    final_best = max(c.yield_value for c in observed)
    metrics = {
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10 or any(c.candidate_id in top10 for c in observed)),
        "intervention_count": intervention_count,
        "bad_intervention_count": bad_interventions,
        "rejected_good_challenger_count": rejected_good_challengers,
        "hypothesis_confidence": round(hypothesis.confidence, 4),
        "hypothesis_support_count": hypothesis.support_count,
    }
    return metrics, audit, hypothesis


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out: dict[str, Any] = {}
    numeric_fields = [
        "final_best",
        "best_so_far_auc",
        "simple_regret",
        "top10_hit",
        "intervention_count",
        "bad_intervention_count",
        "rejected_good_challenger_count",
        "hypothesis_confidence",
        "hypothesis_support_count",
    ]
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in numeric_fields:
            vals = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(vals), 4),
                "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            }
    return out


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], audit_seed0: list[AuditEntry], hypothesis_seed0: HypothesisEntry) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_TABLES / "synthetic_suzuki_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / "synthetic_suzuki_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_RUNS / "synthetic_suzuki_audit_seed0.jsonl").open("w", encoding="utf-8") as f:
        for entry in audit_seed0:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    (OUTPUT_RUNS / "synthetic_suzuki_knowledge_seed0.json").write_text(
        json.dumps(asdict(hypothesis_seed0), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CARE 2.0 synthetic Suzuki replay smoke test.")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    args = parser.parse_args()

    pool = synthetic_suzuki_pool()
    task = make_task(pool, args.initial, args.rounds)
    rows: list[dict[str, Any]] = []
    seed0_audit: list[AuditEntry] = []
    seed0_hypothesis = make_hypothesis()
    for mode in ("incumbent", "gate_v1", "gate_v2"):
        for seed in range(args.seeds):
            metrics, audit, hypothesis = run_policy(pool, task, seed, mode)  # type: ignore[arg-type]
            rows.append(metrics)
            if seed == 0 and mode == "gate_v2":
                seed0_audit = audit
                seed0_hypothesis = hypothesis
    summary = {
        "experiment": "synthetic_suzuki_skill_knowledge_replay",
        "disclaimer": "Synthetic smoke test; not a CARE 1.0 paper reproduction.",
        "task": asdict(task),
        "candidate_count": len(pool),
        "seeds": args.seeds,
        "rounds": args.rounds,
        "initial_observations": args.initial,
        "aggregate": aggregate(rows),
    }
    write_outputs(rows, summary, seed0_audit, seed0_hypothesis)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

