"""Experiment data and reveal budget, independent of any selection method.

Only the controller process holds ReplayTask.outcomes. Agents receive a JSON
view containing candidate features, completed source experiments and observations.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Any

from .legacy import import_module


@dataclass
class ReplayTask:
    task: dict[str, Any]
    candidates: list[dict[str, Any]]
    outcomes: dict[str, float]
    source: dict[str, Any] = field(default_factory=lambda: {"task": {}, "observations": []})

    @property
    def task_id(self) -> str:
        return self.task["dataset_id"]


def _task_info(adapter) -> dict:
    return {"direction": "maximize", **{key: getattr(adapter, key) for key in (
        "dataset_id", "title", "objective", "decision_columns", "group_column", "preferred_groups"
    )}}


def _public_candidate(candidate, allowed_fields: set[str]) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "group": candidate.group,
        "x1": candidate.x1, "x2": candidate.x2, "x3": candidate.x3,
        "numeric_features": list(candidate.numeric_features or (
            candidate.x1, candidate.x2, candidate.x3)),
        "metadata": {key: value for key, value in candidate.metadata.items() if key in allowed_fields},
    }


def load_task(config: dict | str) -> ReplayTask:
    """Load an existing dataset adapter or an explicit finite-pool JSON fixture.

JSON fixtures contain task, candidates, outcomes, and optionally source. Candidate
records are already public; outcomes are a separate candidate-id -> value mapping.
Legacy metadata is exported by a feature allowlist (never all raw row columns).
"""
    if isinstance(config, str):
        config = {"target_dataset": config}
    if "task_file" in config:
        data = json.loads(Path(config["task_file"]).read_text())
        return ReplayTask(data["task"], data["candidates"], data["outcomes"],
                          data.get("source", {"task": {}, "observations": []}))
    replay = import_module("run_synthetic_suzuki")
    transfer = import_module("run_transfer_ablation")
    semantic = import_module("llm_semantic_skills")
    target = replay.DATASET_BUILDERS[config["target_dataset"]]()
    source = replay.DATASET_BUILDERS[config["source_dataset"]]() if config.get("source_dataset") else None
    role_map = transfer.descriptor_transfer_role_map_for(source.dataset_id, target.dataset_id) if source else {}
    public_identity = {
        "smiles", "composition", "sequence", "mutation_tokens", "mutation_class_tokens",
        "mutation_count", "wild_type", "ligand_raw_name", "base_raw_name", "additive_raw_name",
        "aryl_halide_raw_name", "catalyst_raw_name", "reagent_raw_name", "solvent_raw_name",
        "residence_time_seconds", "temperature_celsius", "catalyst_loading_mol_percent",
    }

    def fields(adapter, roles):
        return set(adapter.decision_columns) | set(roles) | public_identity | set(
            semantic.SEMANTIC_FIELDS.get(adapter.dataset_id, ()))

    target_fields = fields(target, role_map.values())
    candidates = [_public_candidate(c, target_fields) for c in target.candidates]
    source_view = {"task": {}, "observations": []}
    if source:
        observed = transfer.source_observations(source, config.get("source_seed", 0),
                                               config.get("source_observations", 512))
        source_fields = fields(source, role_map)
        source_view = {
            "task": _task_info(source),
            "observations": [dict(_public_candidate(c, source_fields),
                                  outcome=c.objective_value) for c in observed],
        }
    return ReplayTask(_task_info(target), candidates,
                      {c.candidate_id: c.objective_value for c in target.candidates}, source_view)


class ReplayEnvironment:
    """Single campaign; only reveal() accesses hidden target outcomes."""

    def __init__(self, task: ReplayTask, seed: int, initial_observations: int,
                 reveal_rounds: int, initial_ids: list[str] | None = None):
        self.task = task
        if task.task.get("direction", "maximize") != "maximize":
            raise ValueError("Convert outcomes to a larger-is-better utility before loading the task")
        self.seed = seed
        self.initial_observations = initial_observations
        self.reveal_rounds = reveal_rounds
        self._by_id = {c["candidate_id"]: c for c in task.candidates}
        if len(self._by_id) != len(task.candidates):
            raise ValueError("Candidate IDs must be unique")
        if set(task.outcomes) != set(self._by_id):
            raise ValueError("Candidate table and outcome table must contain the same IDs")
        if initial_observations < 1 or reveal_rounds < 1 or initial_observations + reveal_rounds > len(self._by_id):
            raise ValueError("Budget requires at least one initial and one adaptive experiment within pool size")
        if initial_ids is None:
            ids = list(self._by_id)
            random.Random(seed).shuffle(ids)
            initial_ids = ids[:initial_observations]
        if len(initial_ids) != initial_observations or len(set(initial_ids)) != len(initial_ids):
            raise ValueError("Initial IDs must be distinct and match the initial budget")
        self.observations = [self._observation(candidate_id) for candidate_id in initial_ids]
        self.initial_ids = list(initial_ids)
        self.events: list[dict] = []

    def _observation(self, candidate_id: str) -> dict:
        if candidate_id not in self._by_id:
            raise ValueError(f"Unknown candidate: {candidate_id}")
        value = float(self.task.outcomes[candidate_id])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite measured outcome for {candidate_id}")
        return {"candidate_id": candidate_id, "outcome": value}

    @property
    def remaining_budget(self) -> int:
        return self.reveal_rounds - len(self.events)

    def view(self) -> dict:
        return deepcopy({
            "task": self.task.task,
            "candidates": self.task.candidates,
            "observations": self.observations,
            "source": self.task.source,
            "round_index": len(self.events),
            "remaining_budget": self.remaining_budget,
            "seed": self.seed,
            "initial_observations": self.initial_observations,
            "reveal_rounds": self.reveal_rounds,
        })

    def reveal(self, candidate_id: str) -> dict:
        if not self.remaining_budget:
            raise ValueError("Experiment budget exhausted")
        if candidate_id in {o["candidate_id"] for o in self.observations}:
            raise ValueError(f"Candidate already observed: {candidate_id}")
        observation = self._observation(candidate_id)
        self.observations.append(observation)
        event = {"round_index": len(self.events), **observation,
                 "best_so_far": max(o["outcome"] for o in self.observations)}
        self.events.append(event)
        return deepcopy(event)

    def metrics(self) -> dict:
        """Evaluator-only pool extrema/ranks are used AFTER decisions, never in view()."""
        values = list(self.task.outcomes.values())
        low, high = min(values), max(values)
        scale = high - low
        best = max(o["outcome"] for o in self.observations)
        auc = sum(e["best_so_far"] for e in self.events) / len(self.events) if self.events else None
        observed = {o["candidate_id"] for o in self.observations}
        ranked = sorted(self.task.outcomes, key=lambda x: (-self.task.outcomes[x], x))
        top_one_percent = max(1, math.ceil(len(ranked) * 0.01))
        return {
            "final_best": best,
            "best_so_far_auc": auc,
            "simple_regret": high - best,
            "normalized_regret": (high - best) / scale if scale else 0.0,
            "normalized_best_so_far_auc": ((auc - low) / scale if scale else 1.0) if auc is not None else None,
            "top10_hit": bool(observed.intersection(ranked[:10])),
            "top1_percent_hit": bool(observed.intersection(ranked[:top_one_percent])),
            "initial_count": self.initial_observations,
            "adaptive_count": len(self.events),
        }
