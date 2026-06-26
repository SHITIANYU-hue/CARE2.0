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
from typing import Any, Callable, Literal


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

SkillFamily = Literal["ranker", "constraint", "exploration", "data_analysis", "fallback"]
Mode = Literal["incumbent", "gate_v1", "gate_v2"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: str
    x1: float
    x2: float
    x3: float
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetAdapter:
    dataset_id: str
    title: str
    objective: str
    decision_columns: tuple[str, ...]
    hidden_target: str
    group_column: str
    preferred_groups: tuple[str, ...]
    failure_note: str
    candidates: tuple[Candidate, ...]


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
    dataset_id: str
    seed: int
    round_index: int
    public_observed_count: int
    incumbent_candidate: str
    challenger_candidate: str
    selected_candidate: str
    selected_by: str
    gate: GateCertificate
    revealed_value: float
    best_so_far: float
    hypothesis_snapshot: dict[str, Any]


def stable_noise(*parts: object, scale: float = 2.0) -> float:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * 2.0 * scale


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 4)


def synthetic_suzuki_adapter() -> DatasetAdapter:
    ligands = [f"L{i}" for i in range(7)]
    residence_times = [30.0, 60.0, 90.0, 120.0]
    temperatures = [50.0, 70.0, 90.0, 110.0]
    loadings = [0.5, 1.0, 2.0, 4.0]
    ligand_base = {"L0": 42.0, "L1": 48.0, "L2": 76.0, "L3": 62.0, "L4": 58.0, "L5": 72.0, "L6": 55.0}
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
                    penalty = -10.0 if temp <= 50.0 and loading <= 0.5 else 0.0
                    y = clamp_score(
                        ligand_base[ligand]
                        + temp_effect
                        + time_effect
                        + loading_effect
                        + interaction
                        + penalty
                        + stable_noise(ligand, time, temp, loading, scale=2.3)
                    )
                    cid = f"{ligand}_T{int(temp)}_R{int(time)}_C{str(loading).replace('.', 'p')}"
                    pool.append(
                        Candidate(
                            candidate_id=cid,
                            group=ligand,
                            x1=temp / 110.0,
                            x2=time / 120.0,
                            x3=math.log1p(loading) / math.log1p(4.0),
                            objective_value=y,
                            metadata={
                                "ligand_identity": ligand,
                                "temperature": temp,
                                "residence_time": time,
                                "catalyst_loading": loading,
                                "yield_value": y,
                            },
                        )
                    )
    return DatasetAdapter(
        dataset_id="synthetic_suzuki_i",
        title="Synthetic Suzuki finite-pool replay",
        objective="maximize_yield",
        decision_columns=("ligand_identity", "residence_time", "temperature", "catalyst_loading"),
        hidden_target="yield_value",
        group_column="ligand_identity",
        preferred_groups=("L2", "L5"),
        failure_note="Low temperature and low catalyst loading can erase the ligand advantage.",
        candidates=tuple(pool),
    )


def synthetic_chemlex_adapter() -> DatasetAdapter:
    acids = [f"A{i}" for i in range(6)]
    amines = [f"N{i}" for i in range(6)]
    solvent_polarities = [0.2, 0.5, 0.8]
    base_equivalents = [0.5, 1.0, 1.5, 2.0]
    temperatures = [25.0, 50.0, 75.0, 100.0]
    acid_base = {"A0": 40.0, "A1": 47.0, "A2": 68.0, "A3": 55.0, "A4": 72.0, "A5": 63.0}
    amine_base = {"N0": 0.0, "N1": 8.0, "N2": -4.0, "N3": 11.0, "N4": 5.0, "N5": -2.0}
    pair_bonus = {("A2", "N3"): 10.0, ("A4", "N1"): 8.0, ("A5", "N4"): 6.0}
    pool: list[Candidate] = []
    for acid in acids:
        for amine in amines:
            group = f"{acid}-{amine}"
            for polarity in solvent_polarities:
                for base_eq in base_equivalents:
                    for temp in temperatures:
                        polarity_opt = 0.8 if acid in {"A2", "A4"} else 0.5
                        temp_opt = 75.0 if amine in {"N1", "N3", "N4"} else 50.0
                        solvent_effect = -22.0 * (polarity - polarity_opt) ** 2 + 4.0
                        base_effect = -4.0 * (base_eq - 1.5) ** 2 + 5.0
                        temp_effect = -0.006 * (temp - temp_opt) ** 2 + 4.0
                        y = clamp_score(
                            acid_base[acid]
                            + amine_base[amine]
                            + pair_bonus.get((acid, amine), 0.0)
                            + solvent_effect
                            + base_effect
                            + temp_effect
                            + stable_noise(acid, amine, polarity, base_eq, temp, scale=2.8)
                        )
                        cid = f"{acid}_{amine}_S{int(polarity * 10)}_B{str(base_eq).replace('.', 'p')}_T{int(temp)}"
                        pool.append(
                            Candidate(
                                candidate_id=cid,
                                group=group,
                                x1=polarity,
                                x2=base_eq / 2.0,
                                x3=temp / 100.0,
                                objective_value=y,
                                metadata={
                                    "acid": acid,
                                    "amine": amine,
                                    "solvent_polarity": polarity,
                                    "base_equivalents": base_eq,
                                    "temperature": temp,
                                    "yield_value": y,
                                },
                            )
                        )
    return DatasetAdapter(
        dataset_id="synthetic_chemlex_i",
        title="Synthetic ChemLex-style acid-amine replay",
        objective="maximize_yield",
        decision_columns=("acid", "amine", "solvent_polarity", "base_equivalents", "temperature"),
        hidden_target="yield_value",
        group_column="acid_amine_pair",
        preferred_groups=("A2-N3", "A4-N1", "A5-N4"),
        failure_note="Matched acid-amine pairs still depend on solvent polarity and base equivalents.",
        candidates=tuple(pool),
    )


def synthetic_materials_adapter() -> DatasetAdapter:
    dopants = [f"D{i}" for i in range(7)]
    ratios = [0.10, 0.20, 0.35, 0.50]
    anneal_temps = [300.0, 450.0, 600.0, 750.0]
    dwell_times = [10.0, 30.0, 60.0]
    dopant_base = {"D0": 38.0, "D1": 54.0, "D2": 76.0, "D3": 58.0, "D4": 73.0, "D5": 49.0, "D6": 61.0}
    temp_opt = {"D0": 450.0, "D1": 450.0, "D2": 600.0, "D3": 600.0, "D4": 750.0, "D5": 450.0, "D6": 600.0}
    pool: list[Candidate] = []
    for dopant in dopants:
        for ratio in ratios:
            for temp in anneal_temps:
                for dwell in dwell_times:
                    ratio_effect = -90.0 * (ratio - 0.35) ** 2 + 5.0
                    temp_effect = -0.00016 * (temp - temp_opt[dopant]) ** 2 + 6.0
                    dwell_effect = 4.5 * math.log1p(dwell / 10.0) - 0.045 * dwell
                    interaction = 5.0 if dopant in {"D2", "D4"} and ratio >= 0.20 and temp >= 600.0 else 0.0
                    overcook_penalty = -8.0 if temp >= 750.0 and dwell >= 60.0 else 0.0
                    score = clamp_score(
                        dopant_base[dopant]
                        + ratio_effect
                        + temp_effect
                        + dwell_effect
                        + interaction
                        + overcook_penalty
                        + stable_noise(dopant, ratio, temp, dwell, scale=2.5)
                    )
                    cid = f"{dopant}_R{int(ratio * 100)}_T{int(temp)}_D{int(dwell)}"
                    pool.append(
                        Candidate(
                            candidate_id=cid,
                            group=dopant,
                            x1=ratio / 0.50,
                            x2=temp / 750.0,
                            x3=dwell / 60.0,
                            objective_value=score,
                            metadata={
                                "dopant": dopant,
                                "dopant_ratio": ratio,
                                "anneal_temperature": temp,
                                "dwell_time": dwell,
                                "stability_score": score,
                            },
                        )
                    )
    return DatasetAdapter(
        dataset_id="synthetic_materials_i",
        title="Synthetic materials formulation replay",
        objective="maximize_stability_score",
        decision_columns=("dopant", "dopant_ratio", "anneal_temperature", "dwell_time"),
        hidden_target="stability_score",
        group_column="dopant",
        preferred_groups=("D2", "D4"),
        failure_note="Preferred dopants are sensitive to annealing temperature and long dwell overcooking.",
        candidates=tuple(pool),
    )


DATASET_BUILDERS: dict[str, Callable[[], DatasetAdapter]] = {
    "synthetic_suzuki_i": synthetic_suzuki_adapter,
    "synthetic_chemlex_i": synthetic_chemlex_adapter,
    "synthetic_materials_i": synthetic_materials_adapter,
}


def make_task(adapter: DatasetAdapter, initial_observations: int, reveal_budget: int) -> TaskSpec:
    return TaskSpec(
        dataset_id=adapter.dataset_id,
        objective=adapter.objective,
        decision_columns=adapter.decision_columns,
        hidden_target=adapter.hidden_target,
        initial_observations=initial_observations,
        reveal_budget=reveal_budget,
        oracle_value=max(c.objective_value for c in adapter.candidates),
    )


def make_skills(adapter: DatasetAdapter) -> list[SkillCard]:
    return [
        SkillCard(
            skill_id="group_prior",
            version="1.0.0",
            family="ranker",
            scope=f"Add bounded prior bonus to preferred {adapter.group_column} groups.",
            trigger_rules=({"field": "round_index", "operator": ">=", "value": 0},),
            bounded_parameters={"preferred_groups": {"value": list(adapter.preferred_groups)}, "prior_bonus_cap": {"value": 0.10}},
            certificate_schema={"required": ["applied_bonuses", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id", "modify_observed_data"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
            rationale="Convert a dataset-level scientific prior into bounded full-pool score adjustments.",
        ),
        SkillCard(
            skill_id="group_risk_penalty",
            version="1.0.0",
            family="constraint",
            scope=f"Penalize {adapter.group_column} groups that repeatedly underperform in public observations.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
            bounded_parameters={"penalty_cap": {"value": -0.12}, "min_support": {"value": 2}},
            certificate_schema={"required": ["penalized_groups", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("block_candidate_permanently", "access_hidden_outcomes"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
            rationale="Use public repeated failures as bounded risk evidence without permanently blocking candidates.",
        ),
        SkillCard(
            skill_id="group_diversity_explorer",
            version="1.0.0",
            family="exploration",
            scope=f"Add a small bounded bonus to unseen {adapter.group_column} groups.",
            trigger_rules=({"field": "observed_count", "operator": ">=", "value": 5},),
            bounded_parameters={"unseen_group_bonus_cap": {"value": 0.05}},
            certificate_schema={"required": ["unseen_groups", "max_abs_adjustment"]},
            required_checks=("static", "sandbox", "full_pool", "row_order"),
            prohibited_behaviors=("access_hidden_outcomes", "directly_select_candidate_id"),
            provenance={"source": "CARE 2.0 skill specification", "dataset": adapter.dataset_id},
            rationale="Encourage controlled exploration of public feature groups not yet covered by observations.",
        ),
    ]


def make_hypothesis(adapter: DatasetAdapter) -> HypothesisEntry:
    preferred = ", ".join(adapter.preferred_groups)
    return HypothesisEntry(
        hypothesis_id=f"{adapter.dataset_id}_preferred_group_hypothesis",
        status="active",
        scope="group_preference",
        trigger={"dataset": adapter.dataset_id, "condition": "observed_count >= 4"},
        target_spec={"group_column": adapter.group_column, "group_values": list(adapter.preferred_groups)},
        claim=f"Groups {preferred} tend to produce higher objective values under suitable conditions.",
        confidence=0.5,
        support_count=0,
        alpha=1.0,
        beta=1.0,
        evidence_summary="Initialized from dataset prior; no reveal evidence yet.",
        known_failure_modes=[adapter.failure_note],
        created_round=0,
        last_updated_round=0,
    )


def observed_mean(observed: list[Candidate]) -> float:
    return mean(c.objective_value for c in observed) if observed else 50.0


def group_stats(observed: list[Candidate]) -> dict[str, tuple[int, float]]:
    by_group: dict[str, list[float]] = {}
    for c in observed:
        by_group.setdefault(c.group, []).append(c.objective_value)
    return {group: (len(vals), mean(vals)) for group, vals in by_group.items()}


def public_incumbent_scores(pool: tuple[Candidate, ...], observed_ids: set[str], observed: list[Candidate]) -> dict[str, float]:
    stats = group_stats(observed)
    global_mean = observed_mean(observed)
    scores: dict[str, float] = {}
    for c in pool:
        if c.candidate_id in observed_ids:
            continue
        count, group_mean = stats.get(c.group, (0, global_mean))
        uncertainty = 12.0 / math.sqrt(count + 1.0)
        public_condition_prior = 0.035 * c.x1 + 0.020 * c.x2 + 0.015 * c.x3
        estimated = (group_mean + uncertainty) / 100.0 + public_condition_prior
        scores[c.candidate_id] = estimated
    return scores


def trigger_satisfied(rule: dict[str, Any], round_index: int, observed: list[Candidate]) -> bool:
    field_name = rule["field"]
    value = rule["value"]
    actual = round_index if field_name == "round_index" else len(observed)
    op = rule["operator"]
    if op == ">=":
        return actual >= value
    if op == "==":
        return actual == value
    raise ValueError(f"Unsupported operator: {op}")


def skill_adjustments(
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    adjustments = {c.candidate_id: 0.0 for c in pool if c.candidate_id not in observed_ids}
    cert: dict[str, Any] = {"skills": {}, "max_abs_adjustment": 0.0}
    stats = group_stats(observed)
    global_mean = observed_mean(observed)
    seen_groups = {c.group for c in observed}
    for skill in skills:
        if not all(trigger_satisfied(rule, round_index, observed) for rule in skill.trigger_rules):
            cert["skills"][skill.skill_id] = {"active": False, "reason": "trigger_not_satisfied"}
            continue
        if skill.skill_id == "group_prior":
            preferred = set(skill.bounded_parameters["preferred_groups"]["value"])
            cap = float(skill.bounded_parameters["prior_bonus_cap"]["value"])
            applied = {}
            for c in pool:
                if c.candidate_id not in adjustments or c.group not in preferred:
                    continue
                adjustments[c.candidate_id] += cap
                applied[c.candidate_id] = cap
            cert["skills"][skill.skill_id] = {"active": True, "applied_count": len(applied), "cap": cap}
        elif skill.skill_id == "group_risk_penalty":
            cap = float(skill.bounded_parameters["penalty_cap"]["value"])
            min_support = int(skill.bounded_parameters["min_support"]["value"])
            risky = {group for group, (count, group_mean) in stats.items() if count >= min_support and group_mean < global_mean - 8.0}
            for c in pool:
                if c.candidate_id in adjustments and c.group in risky:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "penalized_groups": sorted(risky), "cap": cap}
        elif skill.skill_id == "group_diversity_explorer":
            cap = float(skill.bounded_parameters["unseen_group_bonus_cap"]["value"])
            unseen = sorted({c.group for c in pool} - seen_groups)
            for c in pool:
                if c.candidate_id in adjustments and c.group in unseen:
                    adjustments[c.candidate_id] += cap
            cert["skills"][skill.skill_id] = {"active": True, "unseen_groups": unseen, "cap": cap}
    max_abs = max((abs(v) for v in adjustments.values()), default=0.0)
    cert["max_abs_adjustment"] = round(max_abs, 6)
    return adjustments, cert


def top_candidate(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]


def row_order_stability_check(
    pool: tuple[Candidate, ...],
    observed_ids: set[str],
    observed: list[Candidate],
    skills: list[SkillCard],
    round_index: int,
    reference_adjustments: dict[str, float],
) -> bool:
    shuffled = list(pool)
    random.Random(1000 + round_index + len(observed)).shuffle(shuffled)
    shuffled_adjustments, _ = skill_adjustments(tuple(shuffled), observed_ids, observed, skills, round_index)
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


def update_hypothesis_from_reveal(
    h: HypothesisEntry,
    selected: Candidate,
    observed: list[Candidate],
    round_index: int,
    preferred_groups: tuple[str, ...],
) -> None:
    if selected.group not in preferred_groups:
        return
    public_mean_before = observed_mean(observed)
    supports = selected.objective_value >= public_mean_before
    h.update(
        supports=supports,
        round_index=round_index,
        evidence_summary=(
            f"Round {round_index}: {selected.candidate_id} revealed {selected.objective_value:.2f}; "
            f"public mean before reveal was {public_mean_before:.2f}; supports={supports}."
        ),
    )


def run_policy(adapter: DatasetAdapter, task: TaskSpec, seed: int, mode: Mode) -> tuple[dict[str, Any], list[AuditEntry], HypothesisEntry]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {c.candidate_id: c for c in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {c.candidate_id for c in observed}
    skills = make_skills(adapter)
    hypothesis = make_hypothesis(adapter)
    audit: list[AuditEntry] = []
    top10 = {c.candidate_id for c in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
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
                if by_id[gate.challenger_candidate].objective_value < by_id[gate.incumbent_candidate].objective_value:
                    bad_interventions += 1
            elif by_id[gate.challenger_candidate].objective_value > by_id[gate.incumbent_candidate].objective_value:
                rejected_good_challengers += 1

        selected = by_id[gate.selected_candidate]
        update_hypothesis_from_reveal(hypothesis, selected, observed, round_index, adapter.preferred_groups)
        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(c.objective_value for c in observed)
        best_trace.append(best_so_far)
        audit.append(
            AuditEntry(
                dataset_id=adapter.dataset_id,
                seed=seed,
                round_index=round_index,
                public_observed_count=len(observed) - 1,
                incumbent_candidate=gate.incumbent_candidate,
                challenger_candidate=gate.challenger_candidate,
                selected_candidate=selected.candidate_id,
                selected_by="gate_authorized_challenger" if gate.authorized else "incumbent",
                gate=gate,
                revealed_value=selected.objective_value,
                best_so_far=best_so_far,
                hypothesis_snapshot=asdict(hypothesis),
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


def write_outputs(
    dataset_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    audit_seed0: list[AuditEntry],
    hypothesis_seed0: HypothesisEntry,
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_TABLES / f"{dataset_id}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{dataset_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_RUNS / f"{dataset_id}_audit_seed0.jsonl").open("w", encoding="utf-8") as f:
        for entry in audit_seed0:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    (OUTPUT_RUNS / f"{dataset_id}_knowledge_seed0.json").write_text(
        json.dumps(asdict(hypothesis_seed0), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_dataset(adapter: DatasetAdapter, seeds: int, rounds: int, initial: int) -> dict[str, Any]:
    task = make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    seed0_audit: list[AuditEntry] = []
    seed0_hypothesis = make_hypothesis(adapter)
    for mode in ("incumbent", "gate_v1", "gate_v2"):
        for seed in range(seeds):
            metrics, audit, hypothesis = run_policy(adapter, task, seed, mode)  # type: ignore[arg-type]
            rows.append(metrics)
            if seed == 0 and mode == "gate_v2":
                seed0_audit = audit
                seed0_hypothesis = hypothesis
    summary = {
        "experiment": "care_multi_dataset_skill_knowledge_replay",
        "disclaimer": "Synthetic smoke test; not a CARE 1.0 paper reproduction.",
        "dataset": {
            "dataset_id": adapter.dataset_id,
            "title": adapter.title,
            "group_column": adapter.group_column,
            "preferred_groups": list(adapter.preferred_groups),
        },
        "task": asdict(task),
        "candidate_count": len(adapter.candidates),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "aggregate": aggregate(rows),
    }
    write_outputs(adapter.dataset_id, rows, summary, seed0_audit, seed0_hypothesis)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CARE 2.0 synthetic finite-pool replay smoke tests.")
    parser.add_argument("--dataset", default="synthetic_suzuki_i", choices=[*DATASET_BUILDERS.keys(), "all"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    args = parser.parse_args()

    dataset_ids = list(DATASET_BUILDERS) if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(DATASET_BUILDERS[dataset_id](), args.seeds, args.rounds, args.initial) for dataset_id in dataset_ids]
    if len(summaries) == 1:
        print(json.dumps(summaries[0], ensure_ascii=False, indent=2))
    else:
        combined = {
            "experiment": "care_multi_dataset_skill_knowledge_replay",
            "disclaimer": "Synthetic smoke test; not a CARE 1.0 paper reproduction.",
            "datasets": [summary["dataset"]["dataset_id"] for summary in summaries],
            "summaries": summaries,
        }
        (OUTPUT_RUNS / "all_datasets_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
