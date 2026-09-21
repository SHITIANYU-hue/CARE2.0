"""Incremental adapters for the frozen CARE source-outcome router.

Only public views enter this module. Legacy Candidate objects have an unknown
outcome until that candidate appears in the view's observation history.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from care_harness.legacy import REPLAY_ROOT, import_module


DEFAULTS = {
    "discount": 0.65,
    "min_source_support": 3,
    "normalize_kernel_weights": True,
    "gp_beta": 1.5,
    "gp_xi": 0.01,
    "numeric_length_scale": 0.35,
    "categorical_length_scale": 3.0,
    "gp_noise": 0.05,
    "router_min_observations": 5,
    "router_min_quality": 0.20,
    "router_max_transfer_mass": 0.45,
    "source_initial_strategy": "matched",
    "target_llm_mode": "target_acquisition_portfolio",
    "deployment_mode": "llm_transfer_router",
    "calibration_status": "not_run",
}


def load_pair_config(
    pair_id: str,
    config_path: str | Path | None = None,
    strategy: str = "full_source_outcome",
) -> dict[str, Any]:
    """Resolve the same pair/protocol precedence as the canonical suite runner."""
    path = Path(config_path) if config_path else REPLAY_ROOT / "configs/source_outcome_benchmark.json"
    suite = json.loads(path.read_text())
    pair = next(pair for pair in suite["pairs"] if pair["pair_id"] == pair_id)
    config = {
        **DEFAULTS,
        **suite["protocol"],
        **suite["strategy_candidates"][strategy],
        **pair,
        **pair.get("frozen_policy", {}),
        "strategy": strategy,
        "suite_config": str(path.resolve()),
    }
    for key in ("llm_record", "target_llm_record"):
        record = Path(config[key])
        config[key] = str(record if record.is_absolute() else REPLAY_ROOT / record)
    return config


def _candidate(public: dict[str, Any], outcome: float = float("nan")) -> Any:
    replay = import_module("run_synthetic_suzuki")
    return replay.Candidate(
        candidate_id=str(public["candidate_id"]),
        group=str(public["group"]),
        x1=float(public.get("x1", 0.0)),
        x2=float(public.get("x2", 0.0)),
        x3=float(public.get("x3", 0.0)),
        objective_value=outcome,
        metadata=dict(public["metadata"]),
        numeric_features=tuple(public["numeric_features"]),
    )


def _adapter(task: dict[str, Any], candidates: list[Any]) -> Any:
    replay = import_module("run_synthetic_suzuki")
    return replay.DatasetAdapter(
        dataset_id=task["dataset_id"],
        title=task["title"],
        objective=task["objective"],
        decision_columns=tuple(task["decision_columns"]),
        hidden_target="outcome",
        group_column=task["group_column"],
        preferred_groups=tuple(task["preferred_groups"]),
        failure_note="",
        candidates=tuple(candidates),
    )


def _task(adapter: Any, view: dict[str, Any]) -> Any:
    replay = import_module("run_synthetic_suzuki")
    return replay.TaskSpec(
        dataset_id=adapter.dataset_id,
        objective=adapter.objective,
        decision_columns=adapter.decision_columns,
        hidden_target="outcome",
        initial_observations=view["initial_observations"],
        reveal_budget=view["reveal_rounds"],
        oracle_value=float("nan"),
    )


class TargetGPAgent:
    """Canonical UCB/EI/portfolio or a frozen target-only semantic policy."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**DEFAULTS, **(config or {})}

    def _skill(self, adapter: Any) -> tuple[str, Any]:
        calibrated = import_module("run_calibrated_source_outcome_transfer")
        mode = self.config.get("mode", "target_acquisition_portfolio")
        record = (json.loads(Path(self.config["target_llm_record"]).read_text())
                  if mode not in {"gp_ucb", "mixed_kernel_gp_ei", "target_acquisition_portfolio"}
                  else {})
        return mode, calibrated.resolve_target_llm_skill(record, adapter, mode)

    def initial_candidate_ids(self, view: dict[str, Any]) -> list[str]:
        calibrated = import_module("run_calibrated_source_outcome_transfer")
        adapter = _adapter(view["task"], [_candidate(row) for row in view["candidates"]])
        mode, skill = self._skill(adapter)
        return [candidate.candidate_id for candidate in calibrated.target_llm_initial_observations(
            adapter, _task(adapter, view), view["seed"], mode, skill,
        )]

    def select(self, view: dict[str, Any]) -> dict[str, Any]:
        replay = import_module("run_synthetic_suzuki")
        surrogate = import_module("run_surrogate_baselines")
        calibrated = import_module("run_calibrated_source_outcome_transfer")
        candidates = [_candidate(row) for row in view["candidates"]]
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        adapter = _adapter(view["task"], candidates)
        observed = [replace(by_id[row["candidate_id"]], objective_value=float(row["outcome"]))
                    for row in view["observations"]]
        mode, skill = self._skill(adapter)
        scorer = calibrated.target_llm_anchor_scorer(
            adapter, _task(adapter, view), mode, skill,
            self.config["gp_beta"], self.config["gp_xi"],
            self.config["numeric_length_scale"], self.config["categorical_length_scale"],
            self.config["gp_noise"],
        )
        scores, diagnostics = scorer(
            {candidate.candidate_id for candidate in observed}, observed,
            {candidate.candidate_id: surrogate.candidate_features(adapter, candidate)
             for candidate in candidates}, view["round_index"],
        )
        return {"candidate_id": replay.top_candidate(scores), "diagnostics": diagnostics}


class CareAgent:
    """Frozen LLM skills + the unchanged numerical source-outcome router.

    `deployment_mode` identifies the route already chosen by offline calibration.
    Its default is the raw transfer route, explicitly marked as uncalibrated.
    This adapter never generates skills or calls a model endpoint.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = {**DEFAULTS, **config}
        if self.config["deployment_mode"] not in {
            "llm_transfer_router", "matched_target_only_llm", "gp_ucb",
            "mixed_kernel_gp_ei", "target_acquisition_portfolio",
        }:
            raise ValueError(f"Unknown CARE deployment mode: {self.config['deployment_mode']}")
        self._policy = None
        self._pending = None
        self._by_id: dict[str, Any] = {}
        deployment = self.config["deployment_mode"]
        self._target_agent = None
        if deployment != "llm_transfer_router":
            mode = self.config["target_llm_mode"] if deployment == "matched_target_only_llm" else deployment
            self._target_agent = TargetGPAgent({**self.config, "mode": mode})

    def _build(self, view: dict[str, Any], *, preserve_initial: bool) -> Any:
        transfer = import_module("run_transfer_ablation")
        evolution = import_module("run_llm_kernel_skill_evolution")
        calibrated = import_module("run_calibrated_source_outcome_transfer")
        router = import_module("run_llm_transfer_router")
        config = self.config
        candidates = [_candidate(row) for row in view["candidates"]]
        self._by_id = {candidate.candidate_id: candidate for candidate in candidates}
        target = _adapter(view["task"], candidates)
        source_observed = [_candidate(row, float(row["outcome"]))
                           for row in view["source"]["observations"]]
        source = _adapter(view["source"]["task"], source_observed)
        task = _task(target, view)
        source_record = json.loads(Path(config["llm_record"]).read_text())
        target_record = json.loads(Path(config["target_llm_record"]).read_text())
        patches = evolution.normalize_patches(
            {"patches": source_record["normalized_patches"]}, target.decision_columns, 100,
        )
        mode = config["target_llm_mode"]
        skill = calibrated.resolve_target_llm_skill(target_record, target, mode)
        scorer = calibrated.target_llm_anchor_scorer(
            target, task, mode, skill, config["gp_beta"], config["gp_xi"],
            config["numeric_length_scale"], config["categorical_length_scale"], config["gp_noise"],
        )
        initial_strategy = config["source_initial_strategy"]
        if preserve_initial:
            initial = [self._by_id[row["candidate_id"]]
                       for row in view["observations"][:view["initial_observations"]]]
            initial_strategy = "matched"
        else:
            initial = calibrated.target_llm_initial_observations(target, task, view["seed"], mode, skill)
        card = transfer.compile_transfer_card(
            source, target, source_observed,
            transfer.descriptor_transfer_role_map_for(source.dataset_id, target.dataset_id),
            config["discount"], config["min_source_support"],
        )
        return router.router_decisions(
            source, target, task, view["seed"], card, source_observed, patches,
            config["normalize_kernel_weights"], config["gp_beta"], config["gp_xi"],
            config["numeric_length_scale"], config["categorical_length_scale"], config["gp_noise"],
            initial, scorer, mode, config["router_min_observations"],
            config["router_min_quality"], config["router_max_transfer_mass"], initial_strategy,
        )

    def initial_candidate_ids(self, view: dict[str, Any]) -> list[str]:
        """Compute the canonical design before its outcomes are revealed."""
        if self._target_agent is not None:
            return self._target_agent.initial_candidate_ids(view)
        policy = self._build(view, preserve_initial=False)
        initial = next(policy)
        policy.close()
        return initial["initial_candidate_ids"]

    def select(self, view: dict[str, Any]) -> dict[str, Any]:
        if self._target_agent is not None:
            decision = self._target_agent.select(view)
            decision["diagnostics"].update(
                deployment_mode=self.config["deployment_mode"],
                calibration_status=self.config["calibration_status"],
                method="frozen_source_outcome_router",
            )
            decision["usage"] = {"model_calls": 0}
            return decision
        if self._policy is None:
            self._policy = self._build(view, preserve_initial=False)
            design = next(self._policy)
            observed_ids = [row["candidate_id"] for row in view["observations"]]
            if design["initial_candidate_ids"] != observed_ids:
                # A benchmark may deliberately prescribe a shared random start.
                self._policy.close()
                self._policy = self._build(view, preserve_initial=True)
                next(self._policy)
            response = [replace(self._by_id[row["candidate_id"]], objective_value=float(row["outcome"]))
                        for row in view["observations"]]
        else:
            observation = next(row for row in view["observations"]
                               if row["candidate_id"] == self._pending)
            response = replace(self._by_id[self._pending], objective_value=float(observation["outcome"]))
        event = self._policy.send(response)
        self._pending = event["selected_candidate"]
        return {
            "candidate_id": self._pending,
            "diagnostics": {
                **event["hypothesis_snapshot"],
                "deployment_mode": self.config["deployment_mode"],
                "calibration_status": self.config["calibration_status"],
                "method": "frozen_source_outcome_router",
            },
            "usage": {"model_calls": 0},
        }
