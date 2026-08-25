#!/usr/bin/env python3
"""Run an evidence-bounded LLM scientist at every target reveal round."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import run_classical_transfer_baselines as classical
import run_llm_initial_design_hypothesis as initial_design
import run_multisource_warmstart as warmstart
import run_synthetic_suzuki as replay

SCHEMA_VERSION = "care.online_llm_scientist/v1"
INITIAL_SCHEMA_VERSION = "care.online_llm_scientist_initial/v1"
MODE = "online_llm_scientist"
SAME_INITIAL_GP_MODE = "llm_initial_target_gp"


class LLMValidationError(ValueError):
    def __init__(self, message: str, attempts: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = [dict(item) for item in attempts]


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_fingerprint(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def numeric_feature_names(adapter: replay.DatasetAdapter) -> list[str]:
    if adapter.dataset_id.startswith("real_flip2_hydro_"):
        return [
            "aromatic_core_fraction",
            "beta_branched_core_fraction",
            "leucine_core_fraction",
            "methionine_core_fraction",
            "core_residue_diversity",
        ]
    if adapter.dataset_id.startswith("real_baumgartner_cn_"):
        return [
            "base_equivalents",
            "temperature_celsius",
            "residence_time_minutes",
            "precatalyst_loading_mol_percent",
        ]
    if adapter.dataset_id.startswith("real_baumgartner_suzuki_"):
        return [
            "temperature_celsius",
            "residence_time_seconds",
            "precatalyst_fraction",
        ]
    dimension = len(adapter.candidates[0].numeric_features)
    return [f"normalized_feature_{index}" for index in range(dimension)]


def selected_source_evidence(source_ids: Sequence[str]) -> dict[str, Any]:
    correlation_rows = []
    observations = []
    for source_id in source_ids:
        adapter = replay.DATASET_BUILDERS[source_id]()
        summary = initial_design.source_summary(adapter)
        correlation_rows.append(
            [
                float(item["outcome_correlation"])
                for item in summary["numeric_feature_outcome_correlations"]
            ]
        )
        observations.extend(
            {
                "source": source_id,
                "outcome": float(item["source_outcome"]),
                "conditions": item["conditions"],
            }
            for item in summary["top_source_observations"][:2]
        )
    names = numeric_feature_names(replay.DATASET_BUILDERS[source_ids[0]]())
    means = {
        name: round(
            float(
                np.mean(
                    [row[index] for row in correlation_rows if index < len(row)]
                )
            ),
            4,
        )
        for index, name in enumerate(names)
    }
    observations.sort(key=lambda item: item["outcome"], reverse=True)
    return {
        "source_ids": list(source_ids),
        "feature_outcome_correlations": means,
        "top_observations": observations[:3],
    }


def compact_shortlist(
    shortlist: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    condition_fields = sorted(
        {
            field
            for row in shortlist
            for field in row.get("conditions", {})
        }
    )
    view_names = sorted(
        {
            name
            for row in shortlist
            for name in row.get("source_evidence", {})
        }
    )
    columns = [
        "candidate_id",
        "group",
        *condition_fields,
        *(f"{name}_rank" for name in view_names),
    ]
    rows = []
    for row in shortlist:
        evidence = row.get("source_evidence", {})
        rows.append(
            [
                row["candidate_id"],
                row.get("group"),
                *(row.get("conditions", {}).get(field) for field in condition_fields),
                *(evidence.get(name, {}).get("rank") for name in view_names),
            ]
        )
    return {"columns": columns, "rows": rows}


def compact_source_tables(
    source_ids: Sequence[str],
    views: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    summaries = {
        source_id: initial_design.source_summary(
            replay.DATASET_BUILDERS[source_id]()
        )
        for source_id in source_ids
    }
    campaign_rows = []
    for source_id in source_ids:
        summary = summaries[source_id]
        descriptor = summary["descriptor"]
        campaign_rows.append(
            [
                source_id,
                descriptor.get("domain"),
                descriptor.get("representation"),
                descriptor.get("substrate"),
                descriptor.get("precatalyst"),
                summary["outcome_summary"]["q75"],
                summary["outcome_summary"]["maximum"],
            ]
        )
    view_rows = []
    for view_name, view_ids in views.items():
        observations = []
        correlations = []
        for source_id in view_ids:
            summary = summaries[source_id]
            observations.extend(
                [
                    [
                        source_id,
                        row["source_outcome"],
                        row["conditions"],
                    ]
                    for row in summary["top_source_observations"][:1]
                ]
            )
            correlations.append(
                [
                    float(item["outcome_correlation"])
                    for item in summary[
                        "numeric_feature_outcome_correlations"
                    ]
                ]
            )
        observations.sort(key=lambda item: float(item[1]), reverse=True)
        dimension = max((len(row) for row in correlations), default=0)
        means = [
            round(
                float(np.mean([row[index] for row in correlations if index < len(row)])),
                4,
            )
            for index in range(dimension)
        ]
        feature_names = numeric_feature_names(
            replay.DATASET_BUILDERS[view_ids[0]]()
        )
        view_rows.append(
            [
                view_name,
                {
                    name: means[index]
                    for index, name in enumerate(feature_names)
                    if index < len(means)
                },
                observations[:2],
            ]
        )
    return {
        "campaign_columns": [
            "dataset_id",
            "domain",
            "representation",
            "substrate",
            "precatalyst",
            "q75",
            "maximum",
        ],
        "campaign_rows": campaign_rows,
        "view_columns": [
            "view_name",
            "mean_feature_outcome_correlations",
            "top_observations[source,outcome,conditions]",
        ],
        "view_rows": view_rows,
    }


def build_initial_prompt(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    source_tables = compact_source_tables(source_ids, views)
    return {
        "task": (
            "Design the first three experiments for a new target before any target "
            "outcome is known. Optimize early best-so-far AUC first and final-best "
            "outcome second across the fixed sequential budget."
        ),
        "target": {
            "dataset_id": target.dataset_id,
            "objective": target.objective,
            "decision_columns": list(target.decision_columns),
            "descriptor": warmstart.task_descriptor(target),
            "candidate_count": len(target.candidates),
        },
        "completed_source_campaigns": {
            "columns": source_tables["campaign_columns"],
            "rows": source_tables["campaign_rows"],
        },
        "source_view_evidence": {
            "columns": source_tables["view_columns"],
            "rows": source_tables["view_rows"],
        },
        "source_view_names": list(views),
        "candidate_shortlist": compact_shortlist(shortlist),
        "execution_protocol": {
            "initial_observations": int(protocol["initial_observations"]),
            "sequential_reveal_rounds": int(protocol["reveal_rounds"]),
            "later_rounds": (
                "The LLM will choose one experiment per round from a menu containing "
                "GP-UCB, source-prior, and geometric candidates."
            ),
        },
        "required_output": {
            "hypothesis": "one falsifiable source-to-target scientific claim",
            "mechanism": "why the pattern should transfer",
            "selected_source_view": "one name from source_view_names",
            "selected_candidate_ids": ["quality_anchor", "mechanism_probe", "geometry_probe"],
            "candidate_assessments": [
                {
                    "candidate_id": "candidate ID",
                    "role": "quality_anchor, mechanism_probe, or geometry_probe",
                    "expected_outcome": 0.0,
                    "confidence": 0.0,
                    "reason": "short scientific reason",
                }
            ],
            "failure_conditions": ["observable falsification condition"],
            "confidence": 0.0,
        },
        "decision_rules": [
            "Return one JSON object only.",
            "Select exactly three unique IDs from candidate_shortlist.",
            "Keep hypothesis and mechanism under 55 words each.",
            "Use exactly three candidate assessments and keep each reason under 25 words.",
            "Use at most three failure conditions and keep each under 25 words.",
            "Select at least one candidate ranked in the top five by source evidence.",
            "Use one plausible quality anchor, one mechanism discriminator, and one geometry probe.",
            "The three candidates must include at least two public categorical condition signatures when possible.",
            "At least one pair must differ materially in normalized numeric conditions.",
            "Do not cluster all three candidates in the same local operating region.",
            "Candidate assessments must cover all three selected candidates.",
            "Source scores are priors, not target measurements. Never invent target outcomes.",
        ],
        "evidence_boundary": "No target outcomes are present in this prompt.",
    }


def assert_no_unrevealed_outcomes(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).lower()
    prohibited = (
        '"objective_value"',
        '"yield_value"',
        '"target_outcome"',
        '"unrevealed_outcome"',
    )
    leaked = [token for token in prohibited if token in encoded]
    if leaked:
        raise ValueError(f"Prompt contains prohibited outcome fields: {leaked}")


def normalize_initial_response(
    response: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    allowed = {str(row["candidate_id"]) for row in shortlist}
    selected = [str(item) for item in response.get("selected_candidate_ids", [])]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("The LLM must select exactly three unique candidates.")
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unknown initial candidate IDs: {unknown}")
    view = str(response.get("selected_source_view", ""))
    if view not in views:
        raise ValueError(f"Unknown source view: {view}")
    confidence = float(response.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Initial confidence must lie in [0, 1].")
    assessments_by_id: dict[str, dict[str, Any]] = {}
    for item in response.get("candidate_assessments", []):
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in allowed:
            continue
        expected = float(item.get("expected_outcome", 0.0))
        item_confidence = float(item.get("confidence", 0.0))
        assessments_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "role": str(item.get("role", "")).strip(),
            "expected_outcome": max(0.0, min(100.0, expected)),
            "confidence": max(0.0, min(1.0, item_confidence)),
            "reason": str(item.get("reason", "")).strip(),
        }
    missing = [candidate_id for candidate_id in selected if candidate_id not in assessments_by_id]
    if missing:
        raise ValueError(f"Missing assessments for selected candidates: {missing}")
    return {
        "hypothesis": str(response.get("hypothesis", "")).strip(),
        "mechanism": str(response.get("mechanism", "")).strip(),
        "selected_source_view": view,
        "selected_candidate_ids": selected,
        "candidate_assessments": [assessments_by_id[item] for item in selected],
        "failure_conditions": [
            str(item).strip() for item in response.get("failure_conditions", [])
        ],
        "confidence": confidence,
    }


def llm_config(args: argparse.Namespace, model: str | None = None) -> replay.LLMConfig:
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Set {args.llm_api_key_env}, COMMONSTACK_API_KEY, or CARE_LLM_API_KEY."
        )
    return replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=model or args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_mode=args.llm_api_mode,
        structured_mode="json",
    )


def validated_llm_json(
    config: replay.LLMConfig,
    messages: Sequence[Mapping[str, str]],
    normalizer: Callable[[Mapping[str, Any]], dict[str, Any]],
    repair_context: Mapping[str, Any],
    repair_attempts: int,
) -> tuple[dict[str, Any], str, Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]]:
    if repair_attempts < 0:
        raise ValueError("repair_attempts must be non-negative.")
    current_messages = [dict(message) for message in messages]
    attempts = []
    for attempt_index in range(repair_attempts + 1):
        content, metadata = replay.chat_completion_text(config, current_messages)
        parsed: Mapping[str, Any] = {}
        try:
            parsed = replay.extract_json_object(content)
            normalized = normalizer(parsed)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "valid": True,
                    "usage": metadata.get("usage", {}),
                    "raw_response": content,
                }
            )
            return normalized, content, metadata, parsed, attempts
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "valid": False,
                    "usage": metadata.get("usage", {}),
                    "raw_response": content,
                    "validation_error": str(exc),
                }
            )
            if attempt_index >= repair_attempts:
                raise LLMValidationError(str(exc), attempts) from exc
            repair_prompt = {
                "task": "Repair the previous response into one valid JSON object.",
                "validation_error": str(exc),
                "previous_response": content,
                "constraints": dict(repair_context),
            }
            current_messages = [
                {
                    "role": "system",
                    "content": (
                        "Repair JSON only. Preserve the scientific decision when valid, "
                        "change only what is required by the validation error, and return "
                        "one JSON object with no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(repair_prompt, ensure_ascii=False),
                },
            ]
    raise AssertionError("Unreachable validated LLM loop.")


def aggregate_attempt_usage(attempts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in attempts:
        usage = attempt.get("usage", {})
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def generate_initial(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    target = replay.DATASET_BUILDERS[args.target_task]()
    source_ids = [str(item) for item in args.source_tasks]
    shortlist, views, _priors = initial_design.candidate_shortlist(
        target,
        source_ids,
        config["protocol"],
        args.per_view_limit,
    )
    prompt = build_initial_prompt(
        target,
        source_ids,
        config["protocol"],
        shortlist,
        views,
    )
    assert_no_unrevealed_outcomes(prompt)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the lead scientist for sequential experimental design. "
                "Optimize the stated metric, distinguish evidence from speculation, "
                "and return one valid JSON object only."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    normalized, content, metadata, parsed, attempts = validated_llm_json(
        llm_config(args),
        [
            *messages,
        ],
        lambda response: normalize_initial_response(response, shortlist, views),
        {
            "allowed_candidate_ids": [row["candidate_id"] for row in shortlist],
            "allowed_source_views": list(views),
            "selected_candidate_count": 3,
            "candidate_assessment_required_for_each_selected_id": True,
            "required_output_schema": prompt["required_output"],
        },
        args.llm_repair_attempts,
    )
    record = {
        "schema_version": INITIAL_SCHEMA_VERSION,
        "status": "frozen_before_target_replay",
        "model": metadata.get("model", args.llm_model),
        "usage": aggregate_attempt_usage(attempts),
        "api_configuration": {
            "base_url": args.llm_base_url,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "structured_mode": "json",
        },
        "config_fingerprint": warmstart.config_fingerprint(config),
        "target_task": args.target_task,
        "source_tasks": source_ids,
        "target_outcomes_available_to_llm": False,
        "prompt": prompt,
        "prompt_sha256": payload_hash(prompt),
        "raw_response": content,
        "parsed_response": parsed,
        "api_attempts": attempts,
        "frozen_initial_policy": normalized,
    }
    write_json(args.output, record)
    write_fingerprint(args.output)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def posterior_state(
    target: replay.DatasetAdapter,
    observed_indices: Sequence[int],
    kernel: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = classical.feature_arrays(target, target.candidates)
    observed_y, center, scale = classical.normalized_outcomes(
        [target.candidates[index] for index in observed_indices]
    )
    mean_normalized, variance, _diagnostics = classical.target_gp_posterior(
        features,
        list(observed_indices),
        observed_y,
        float(kernel["numeric_length_scale"]),
        float(kernel["categorical_length_scale"]),
        float(kernel["gp_noise"]),
    )
    posterior_mean = 100.0 * (center + scale * mean_normalized)
    posterior_std = 100.0 * scale * np.sqrt(variance)
    ucb = posterior_mean + float(kernel["gp_beta"]) * posterior_std
    return posterior_mean, posterior_std, ucb


def candidate_distance_to_observed(
    target: replay.DatasetAdapter,
    observed_indices: Sequence[int],
) -> np.ndarray:
    features = classical.feature_arrays(target, target.candidates)
    numeric = features.numeric
    categorical = features.categorical
    distances = np.full(len(target.candidates), np.inf, dtype=np.float64)
    for observed_index in observed_indices:
        numeric_distance = np.sqrt(
            np.sum((numeric - numeric[observed_index]) ** 2, axis=1)
        )
        categorical_distance = np.mean(
            categorical != categorical[observed_index], axis=1
        )
        distances = np.minimum(distances, numeric_distance + categorical_distance)
    return distances


def rank_map(values: np.ndarray, observed_set: set[int]) -> dict[int, int]:
    # Match classical.top_unobserved: np.argmax resolves exact ties to the
    # lowest pool index. An unstable argsort made the online "GP rank one"
    # disagree with the matched target-only GP baseline on tied candidates.
    ordered = sorted(
        (index for index in range(len(values)) if index not in observed_set),
        key=lambda index: (-float(values[index]), index),
    )
    return {index: rank + 1 for rank, index in enumerate(ordered)}


def build_candidate_menu(
    target: replay.DatasetAdapter,
    observed_indices: Sequence[int],
    source_prior: np.ndarray,
    kernel: Mapping[str, Any],
    gp_count: int,
    source_count: int,
    consensus_count: int,
    decision_consensus_limit: int,
    diversity_count: int,
    eligibility_mode: str = "transfer_consensus",
    max_transfer_gp_rank: int | None = None,
    safety_fallback_gp_count: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if eligibility_mode not in {"transfer_consensus", "target_gp", "full_menu"}:
        raise ValueError(f"Unknown eligibility mode: {eligibility_mode}")
    if safety_fallback_gp_count < 1:
        raise ValueError("safety_fallback_gp_count must be positive")
    observed_set = set(observed_indices)
    posterior_mean, posterior_std, ucb = posterior_state(
        target, observed_indices, kernel
    )
    current_best = max(
        float(target.candidates[index].objective_value)
        for index in observed_indices
    )
    distances = candidate_distance_to_observed(target, observed_indices)
    gp_ranks = rank_map(ucb, observed_set)
    source_ranks = rank_map(source_prior, observed_set)
    diversity_ranks = rank_map(distances, observed_set)
    consensus_order = sorted(
        gp_ranks,
        key=lambda index: (
            gp_ranks[index] + source_ranks[index],
            max(gp_ranks[index], source_ranks[index]),
            gp_ranks[index],
            index,
        ),
    )
    consensus_ranks = {
        index: rank + 1 for rank, index in enumerate(consensus_order)
    }
    selected_indices = {
        index for index, rank in gp_ranks.items() if rank <= gp_count
    }
    selected_indices.update(
        index for index, rank in source_ranks.items() if rank <= source_count
    )
    selected_indices.update(
        index for index, rank in consensus_ranks.items() if rank <= consensus_count
    )
    selected_indices.update(
        index for index, rank in diversity_ranks.items() if rank <= diversity_count
    )
    rows = []
    for index in sorted(selected_indices, key=lambda item: (gp_ranks[item], item)):
        candidate = target.candidates[index]
        sigma = max(float(posterior_std[index]), 1e-9)
        improvement_z = (float(posterior_mean[index]) - current_best) / sigma
        probability_improvement = 0.5 * (
            1.0 + math.erf(improvement_z / math.sqrt(2.0))
        )
        expected_improvement = (
            (float(posterior_mean[index]) - current_best)
            * probability_improvement
            + sigma
            * math.exp(-0.5 * improvement_z * improvement_z)
            / math.sqrt(2.0 * math.pi)
        )
        rows.append(
            {
                **initial_design.public_candidate(candidate, target),
                "model_evidence": {
                    "gp_rank": gp_ranks[index],
                    "gp_posterior_mean": round(float(posterior_mean[index]), 4),
                    "gp_posterior_std": round(float(posterior_std[index]), 4),
                    "gp_ucb": round(float(ucb[index]), 4),
                    "gp_probability_improvement": round(
                        probability_improvement, 6
                    ),
                    "gp_expected_improvement": round(
                        max(0.0, expected_improvement), 6
                    ),
                    "source_prior_rank": source_ranks[index],
                    "source_prior_score": round(float(source_prior[index]), 6),
                    "consensus_rank": consensus_ranks[index],
                    "rank_sum": gp_ranks[index] + source_ranks[index],
                    "decision_eligible": (
                        (
                            max_transfer_gp_rank is None
                            or gp_ranks[index] <= max_transfer_gp_rank
                        )
                        if eligibility_mode == "full_menu"
                        else (
                            consensus_ranks[index] <= decision_consensus_limit
                            and (
                                max_transfer_gp_rank is None
                                or gp_ranks[index] <= max_transfer_gp_rank
                            )
                            if eligibility_mode == "transfer_consensus"
                            else gp_ranks[index] <= decision_consensus_limit
                        )
                    ),
                    "eligibility_mode": eligibility_mode,
                    "distance_to_observed": round(float(distances[index]), 6),
                    "menu_reasons": [
                        reason
                        for reason, active in (
                            ("gp_top", gp_ranks[index] <= gp_count),
                            ("source_top", source_ranks[index] <= source_count),
                            (
                                "consensus_top",
                                consensus_ranks[index] <= consensus_count,
                            ),
                            ("diversity_top", diversity_ranks[index] <= diversity_count),
                        )
                        if active
                    ],
                },
            }
        )
    effective_eligibility_mode = eligibility_mode
    if not any(row["model_evidence"]["decision_eligible"] for row in rows):
        effective_eligibility_mode = "target_gp_safety_fallback"
        for row in rows:
            row["model_evidence"]["decision_eligible"] = (
                row["model_evidence"]["gp_rank"] <= safety_fallback_gp_count
            )
            row["model_evidence"]["eligibility_mode"] = (
                effective_eligibility_mode
            )
    diagnostics = {
        "gp_incumbent_candidate": target.candidates[
            min(gp_ranks, key=gp_ranks.get)
        ].candidate_id,
        "menu_size": len(rows),
        "gp_count": gp_count,
        "source_count": source_count,
        "consensus_count": consensus_count,
        "decision_consensus_limit": decision_consensus_limit,
        "eligibility_mode": effective_eligibility_mode,
        "requested_eligibility_mode": eligibility_mode,
        "max_transfer_gp_rank": max_transfer_gp_rank,
        "safety_fallback_gp_count": safety_fallback_gp_count,
        "diversity_count": diversity_count,
        "consensus_candidate": target.candidates[consensus_order[0]].candidate_id,
    }
    return rows, diagnostics


def build_round_prompt(
    target: replay.DatasetAdapter,
    initial_policy: Mapping[str, Any],
    observed_indices: Sequence[int],
    menu: Sequence[Mapping[str, Any]],
    round_index: int,
    total_rounds: int,
    previous_decision: Mapping[str, Any] | None,
    menu_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    public_candidates = {
        index: initial_design.public_candidate(target.candidates[index], target)
        for index in range(len(target.candidates))
    }
    condition_fields = list(target.decision_columns)
    for item in public_candidates.values():
        for field in item["conditions"]:
            if field not in condition_fields:
                condition_fields.append(field)
    observed = {
        "columns": [
            "candidate_id",
            "group",
            *condition_fields,
            "revealed_target_outcome",
        ],
        "rows": [
            [
                target.candidates[index].candidate_id,
                target.candidates[index].group,
                *(
                    public_candidates[index]["conditions"].get(field)
                    for field in condition_fields
                ),
                round(float(target.candidates[index].objective_value), 6),
            ]
            for index in observed_indices
        ],
    }
    menu_rows = []
    for item in menu:
        evidence = item["model_evidence"]
        menu_rows.append(
            [
                item["candidate_id"],
                item["group"],
                *(item["conditions"].get(field) for field in condition_fields),
                evidence["gp_rank"],
                evidence["gp_posterior_mean"],
                evidence["gp_posterior_std"],
                evidence["gp_ucb"],
                evidence["gp_probability_improvement"],
                evidence["gp_expected_improvement"],
                evidence["source_prior_rank"],
                evidence["source_prior_score"],
                evidence["consensus_rank"],
                evidence["rank_sum"],
                evidence["decision_eligible"],
                evidence["distance_to_observed"],
                "+".join(evidence["menu_reasons"]),
            ]
        )
    compact_initial = {
        "hypothesis": initial_policy.get("hypothesis", ""),
        "mechanism": initial_policy.get("mechanism", ""),
        "selected_source_view": initial_policy.get("selected_source_view", ""),
        "selected_candidate_ids": initial_policy.get("selected_candidate_ids", []),
        "failure_conditions": initial_policy.get("failure_conditions", []),
        "confidence": initial_policy.get("confidence", 0.0),
        "selected_source_evidence": initial_policy.get(
            "selected_source_evidence", {}
        ),
    }
    observed_condition_performance: dict[str, list[dict[str, Any]]] = {}
    for field in condition_fields:
        grouped: dict[str, list[float]] = {}
        for index in observed_indices:
            value = public_candidates[index]["conditions"].get(field)
            if value is None:
                continue
            grouped.setdefault(str(value), []).append(
                float(target.candidates[index].objective_value)
            )
        ranked = sorted(
            (
                {
                    "value": value,
                    "count": len(values),
                    "mean_revealed": round(float(np.mean(values)), 4),
                    "best_revealed": round(float(max(values)), 4),
                }
                for value, values in grouped.items()
            ),
            key=lambda item: (-item["best_revealed"], -item["count"], item["value"]),
        )
        if ranked:
            observed_condition_performance[field] = ranked[:4]

    diagnostics = dict(menu_diagnostics or {})
    return {
        "task": (
            "Choose exactly one next experiment. Optimize best-so-far AUC, so an "
            "early improvement is more valuable than a late one. Use scientific "
            "transfer evidence to choose among statistically credible candidates."
        ),
        "round": {
            "index": round_index,
            "total": total_rounds,
            "remaining_including_this_round": total_rounds - round_index,
        },
        "target": initial_design.target_public_spec(target),
        "frozen_initial_policy": compact_initial,
        "observed_target_history": observed,
        "current_best": max(row[-1] for row in observed["rows"]),
        "observed_condition_performance": observed_condition_performance,
        "decision_context": {
            "gp_default_candidate": diagnostics.get("gp_incumbent_candidate"),
            "consensus_default_candidate": diagnostics.get("consensus_candidate"),
            "eligibility_mode": diagnostics.get("eligibility_mode"),
            "eligible_candidate_count": sum(
                bool(item["model_evidence"]["decision_eligible"])
                for item in menu
            ),
            "policy_note": (
                "You have final authority over every eligible candidate. GP rank one "
                "is a strong target-only default, not a mandatory choice. Deviate only "
                "when observed target evidence or a source-grounded mechanism predicts "
                "a better immediate best-so-far outcome."
            ),
        },
        "candidate_menu": {
            "columns": [
                "candidate_id",
                "group",
                *condition_fields,
                "gp_rank",
                "gp_posterior_mean",
                "gp_posterior_std",
                "gp_ucb",
                "gp_probability_improvement",
                "gp_expected_improvement",
                "source_prior_rank",
                "source_prior_score",
                "consensus_rank",
                "rank_sum",
                "decision_eligible",
                "distance_to_observed",
                "menu_reasons",
            ],
            "rows": menu_rows,
        },
        "previous_llm_decision": dict(previous_decision or {}),
        "required_output": {
            "hypothesis_status": "supported, mixed, falsified, or insufficient",
            "updated_hypothesis": "current falsifiable scientific claim",
            "selected_candidate_id": "one ID from candidate_menu",
            "decision_type": "exploit, mechanism_test, boundary_probe, or recovery",
            "expected_outcome": 0.0,
            "probability_of_improving_current_best": 0.0,
            "confidence": 0.0,
            "evidence_for": ["short evidence item"],
            "evidence_against": ["short contradiction or uncertainty"],
            "reasoning_summary": "concise reason this candidate best serves the reward",
            "continue_source_transfer": True,
            "decision_verdict": "accept_proposal, revise_to_gp, or revise_to_alternative",
            "gp_default_comparison": "why the final choice should beat or defer to GP rank one",
        },
        "decision_rules": [
            "Return one JSON object only.",
            "Select exactly one candidate ID from candidate_menu rows.",
            "The selected row must have decision_eligible=true.",
            "Keep updated_hypothesis and reasoning_summary under 45 words each.",
            "Return at most two evidence_for items and two evidence_against items; keep each under 25 words.",
            "GP quantities are target-only predictions, not observed outcomes.",
            "Prefer a high probability of improving current_best when such a candidate exists.",
            "Use gp_expected_improvement as the default reward-aligned comparator; it accounts for both predicted value and uncertainty.",
            "Do not select a low-mean source candidate merely because its variance is small; that protects prediction error, not best-so-far AUC.",
            "Choose an informative probe only when its expected outcome is competitive enough for the AUC objective.",
            "Do not spend consecutive rounds confirming the same mechanism unless the prior reveal improved current_best.",
            "Treat consensus_rank=1 as the risk-calibrated default because it combines independent GP and source evidence.",
            "Override consensus rank 1 only when another candidate has a materially stronger expected-improvement argument grounded in observed target evidence.",
            "Use selected_source_evidence correlations before general scientific intuition; do not assert a dominant variable contradicted by source data.",
            "You may override GP rank 1 when source evidence or mechanism provides a concrete reason.",
            "If source evidence is falsified, set continue_source_transfer=false and prefer GP-supported recovery.",
            "Never invent outcomes for unexecuted candidates.",
        ],
        "evidence_boundary": (
            "Only observed_target_history contains target outcomes. Candidate-menu "
            "numbers are predictions derived from those observations or source data."
        ),
    }


def build_critic_prompt(
    round_prompt: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": (
            "Act as the final scientific decision critic. Audit the proposer against "
            "the target-only GP default, observed target evidence, source evidence, and "
            "the best-so-far AUC reward. Return the final executable decision."
        ),
        "round": round_prompt["round"],
        "target": round_prompt["target"],
        "current_best": round_prompt["current_best"],
        "frozen_initial_policy": round_prompt["frozen_initial_policy"],
        "observed_target_history": round_prompt["observed_target_history"],
        "observed_condition_performance": round_prompt[
            "observed_condition_performance"
        ],
        "candidate_menu": round_prompt["candidate_menu"],
        "decision_context": round_prompt["decision_context"],
        "proposer_decision": dict(proposal),
        "required_output": round_prompt["required_output"],
        "critic_rules": [
            "Return one JSON object only and select one decision_eligible candidate.",
            "You may accept the proposal, revise to GP rank one, or choose another eligible candidate.",
            "Set decision_verdict to accept_proposal, revise_to_gp, or revise_to_alternative.",
            "Prefer the candidate most likely to improve current_best in this round; early gains dominate the AUC reward.",
            "Do not reward novelty by itself. A mechanism probe must retain competitive expected outcome.",
            "Treat GP rank one as the default under weak, contradictory, or non-target evidence.",
            "A non-GP choice should have competitive gp_expected_improvement and a target-observed or source-grounded reason to outperform the default.",
            "A source-prior override needs both a plausible mechanism and support from revealed target observations.",
            "If the source hypothesis is falsified, set continue_source_transfer=false.",
            "Never use or infer unobserved target outcomes.",
        ],
        "evidence_boundary": round_prompt["evidence_boundary"],
    }


def normalize_round_response(
    response: Mapping[str, Any], menu: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    allowed = {str(item["candidate_id"]) for item in menu}
    eligible = {
        str(item["candidate_id"])
        for item in menu
        if item.get("model_evidence", {}).get("decision_eligible", True)
    }
    selected = str(response.get("selected_candidate_id", ""))
    if selected not in allowed:
        raise ValueError(f"LLM selected a candidate outside the menu: {selected}")
    if selected not in eligible:
        raise ValueError(
            "LLM selected a candidate outside the calibrated decision set: "
            f"{selected}. Eligible IDs: {sorted(eligible)}"
        )
    status = str(response.get("hypothesis_status", "")).strip().lower()
    if status not in {"supported", "mixed", "falsified", "insufficient"}:
        raise ValueError(f"Unknown hypothesis status: {status}")
    confidence = max(0.0, min(1.0, float(response.get("confidence", 0.0))))
    probability = max(
        0.0,
        min(1.0, float(response.get("probability_of_improving_current_best", 0.0))),
    )
    continue_source_transfer = response.get("continue_source_transfer", True)
    if not isinstance(continue_source_transfer, bool):
        raise ValueError("continue_source_transfer must be a JSON boolean.")
    decision_verdict = str(
        response.get("decision_verdict", "accept_proposal")
    ).strip()
    if decision_verdict not in {
        "accept_proposal",
        "revise_to_gp",
        "revise_to_alternative",
    }:
        raise ValueError(f"Unknown decision verdict: {decision_verdict}")
    return {
        "hypothesis_status": status,
        "updated_hypothesis": str(response.get("updated_hypothesis", "")).strip(),
        "selected_candidate_id": selected,
        "decision_type": str(response.get("decision_type", "")).strip(),
        "expected_outcome": max(
            0.0, min(100.0, float(response.get("expected_outcome", 0.0)))
        ),
        "probability_of_improving_current_best": probability,
        "confidence": confidence,
        "evidence_for": [str(item).strip() for item in response.get("evidence_for", [])],
        "evidence_against": [
            str(item).strip() for item in response.get("evidence_against", [])
        ],
        "reasoning_summary": str(response.get("reasoning_summary", "")).strip(),
        "continue_source_transfer": continue_source_transfer,
        "decision_verdict": decision_verdict,
        "gp_default_comparison": str(
            response.get("gp_default_comparison", "")
        ).strip(),
    }


def append_trace(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def calibration_prediction_error(
    decision: Mapping[str, Any], revealed_value: float
) -> float | None:
    """Score only predictions the LLM chose to assert rather than defer to GP."""
    if decision.get("decision_verdict") == "revise_to_gp":
        return None
    expected = decision.get("expected_outcome")
    if expected is None:
        return None
    return abs(float(expected) - float(revealed_value))


def calibration_gate_snapshot(
    prediction_errors: Sequence[float],
    *,
    hard_abstention: bool,
    threshold: float | None,
    force_fallback: bool = False,
) -> dict[str, Any]:
    mean_error = (
        float(np.mean(np.asarray(prediction_errors, dtype=np.float64)))
        if prediction_errors
        else None
    )
    reasons = []
    if force_fallback:
        reasons.append("bounded_authority_round_limit")
    if hard_abstention:
        reasons.append("hard_abstention")
    if threshold is not None and mean_error is not None and mean_error > threshold:
        reasons.append("prediction_mae_above_threshold")
    return {
        "scored_prediction_count": len(prediction_errors),
        "mean_absolute_prediction_error": (
            round(mean_error, 6) if mean_error is not None else None
        ),
        "hard_abstention": hard_abstention,
        "threshold": float(threshold) if threshold is not None else None,
        "force_fallback": force_fallback,
        "switch_to_target_gp": bool(reasons),
        "trigger_reasons": reasons,
    }


def initial_policy_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(record.get("schema_version", ""))
    if schema == INITIAL_SCHEMA_VERSION:
        policy = dict(record["frozen_initial_policy"])
        if record.get("source_tasks") and record.get("target_task"):
            source_ids = [str(item) for item in record["source_tasks"]]
            target = replay.DATASET_BUILDERS[str(record["target_task"])]()
            views = initial_design.source_views(target, source_ids)
            selected_view = str(policy["selected_source_view"])
            policy["selected_source_evidence"] = selected_source_evidence(
                views[selected_view]
            )
        return policy
    if schema == "care.llm_initial_design_hypothesis/v1":
        hypothesis = record["frozen_hypothesis"]
        role_by_id = {
            str(item.get("candidate_id")): str(item.get("role", ""))
            for item in hypothesis.get("selection_roles", [])
        }
        policy = {
            "hypothesis": str(hypothesis.get("hypothesis", "")),
            "mechanism": str(hypothesis.get("mechanism", "")),
            "selected_source_view": str(
                hypothesis.get("selected_source_view", "all_sources")
            ),
            "selected_candidate_ids": [
                str(item) for item in hypothesis["selected_candidate_ids"]
            ],
            "candidate_assessments": [
                {
                    "candidate_id": str(candidate_id),
                    "role": role_by_id.get(str(candidate_id), "initial_probe"),
                    "expected_outcome": None,
                    "confidence": float(hypothesis.get("confidence", 0.0)),
                    "reason": "Imported from the frozen outcome-blind LLM record.",
                }
                for candidate_id in hypothesis["selected_candidate_ids"]
            ],
            "failure_conditions": [
                str(item) for item in hypothesis.get("failure_conditions", [])
            ],
            "confidence": float(hypothesis.get("confidence", 0.0)),
        }
        if record.get("source_tasks") and record.get("target_task"):
            source_ids = [str(item) for item in record["source_tasks"]]
            target = replay.DATASET_BUILDERS[str(record["target_task"])]()
            views = initial_design.source_views(target, source_ids)
            selected_view = str(policy["selected_source_view"])
            policy["selected_source_evidence"] = selected_source_evidence(
                views[selected_view]
            )
        return policy
    raise ValueError(f"Unsupported online-scientist initial record schema: {schema}")


def resolve_initial_design(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    initial_policy: Mapping[str, Any],
    requested_mode: str,
) -> tuple[list[int], str, dict[str, Any]]:
    if requested_mode not in {"auto", "direct", "compiled"}:
        raise ValueError(f"Unknown initial design mode: {requested_mode}")
    distinct_groups = {candidate.group for candidate in target.candidates}
    resolved_mode = requested_mode
    if resolved_mode == "auto":
        resolved_mode = "compiled" if len(distinct_groups) > 1 else "direct"
    if resolved_mode == "direct":
        indices = initial_design.indices_for_ids(
            target, initial_policy["selected_candidate_ids"]
        )
        return indices, resolved_mode, {
            "compiler": "direct_llm_selection/v1",
            "candidate_ids": [target.candidates[index].candidate_id for index in indices],
        }
    compiler_hypothesis = dict(initial_policy)
    compiler_hypothesis["selection_roles"] = [
        {
            "candidate_id": item["candidate_id"],
            "role": item.get("role", ""),
        }
        for item in initial_policy.get("candidate_assessments", [])
    ]
    indices, _selected_sources, compiler_record = initial_design.compiled_llm_initial(
        target,
        source_ids,
        protocol,
        compiler_hypothesis,
    )
    return indices, resolved_mode, compiler_record


def run_online(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    record = json.loads(args.initial_record.read_text(encoding="utf-8"))
    if record["config_fingerprint"] != warmstart.config_fingerprint(config):
        raise ValueError("Initial record does not match the experiment config.")
    expected_hash_path = args.initial_record.with_suffix(
        args.initial_record.suffix + ".sha256"
    )
    if expected_hash_path.exists():
        expected = expected_hash_path.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(args.initial_record.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("Initial-record fingerprint mismatch.")

    protocol = config["protocol"]
    target = replay.DATASET_BUILDERS[record["target_task"]]()
    source_ids = [str(item) for item in record["source_tasks"]]
    initial_policy = initial_policy_from_record(record)
    initial_indices, resolved_initial_mode, initial_compiler_record = (
        resolve_initial_design(
            target,
            source_ids,
            protocol,
            initial_policy,
            args.initial_design_mode,
        )
    )
    executed_initial_ids = [
        target.candidates[index].candidate_id for index in initial_indices
    ]
    views = initial_design.source_views(target, source_ids)
    selected_view = str(initial_policy["selected_source_view"])
    source_prior, selected_sources = warmstart.build_source_consensus(
        target, views[selected_view], protocol
    )
    observed_indices = list(initial_indices)
    observed_set = set(observed_indices)
    requested_rounds = int(args.rounds or protocol["reveal_rounds"])
    rounds = min(requested_rounds, len(target.candidates) - len(observed_indices))
    if rounds <= 0:
        raise ValueError("The initial design exhausts the target candidate pool.")
    calibration_gate_round = getattr(args, "calibration_gate_round", None)
    calibration_gate_threshold = getattr(
        args, "calibration_gate_mae_threshold", None
    )
    calibration_gate_hard_abstention = bool(
        getattr(args, "calibration_gate_hard_abstention", True)
    )
    calibration_gate_force_fallback = bool(
        getattr(args, "calibration_gate_force_fallback", False)
    )
    calibration_gate_enabled = calibration_gate_round is not None
    if calibration_gate_force_fallback and not calibration_gate_enabled:
        raise ValueError(
            "Bounded-authority fallback requires --calibration-gate-round."
        )
    if calibration_gate_enabled:
        calibration_gate_round = int(calibration_gate_round)
        if calibration_gate_threshold is None and not calibration_gate_force_fallback:
            raise ValueError(
                "Calibration gate requires --calibration-gate-mae-threshold."
            )
        if calibration_gate_threshold is not None:
            calibration_gate_threshold = float(calibration_gate_threshold)
        if not 1 <= calibration_gate_round <= rounds:
            raise ValueError(
                "Calibration gate round must be within the executed reveal budget."
            )
        if (
            calibration_gate_threshold is not None
            and calibration_gate_threshold < 0.0
        ):
            raise ValueError("Calibration gate threshold must be non-negative.")
    by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    trace_path = args.output_dir / "llm_trace.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing trace: {trace_path}")
    best_trace: list[float] = []
    previous_decision: dict[str, Any] | None = None
    llm_selected_rounds = 0
    llm_choice_rounds = 0
    eligible_candidate_total = 0
    critic_revision_rounds = 0
    gp_override_rounds = 0
    gp_rank_one_choices = 0
    consensus_rank_one_choices = 0
    source_active_rounds = 0
    calibration_prediction_errors: list[float] = []
    calibration_hard_abstention_seen = False
    calibration_gate_triggered = False
    calibration_gate_trigger_reason: list[str] = []
    calibration_gate_snapshot_record: dict[str, Any] | None = None
    calibration_gate_fallback_rounds = 0
    logical_llm_calls_per_round = (
        2 if getattr(args, "deliberation_mode", "single") == "proposal_critic" else 1
    )
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    append_trace(
        trace_path,
        {
            "event": "initial_design_revealed",
            "candidate_ids": executed_initial_ids,
            "public_conditions": [
                initial_design.public_candidate(
                    target.candidates[index],
                    target,
                )["conditions"]
                for index in initial_indices
            ],
            "revealed_values": [
                target.candidates[index].objective_value for index in initial_indices
            ],
            "initial_model": record["model"],
            "initial_policy": initial_policy,
            "initial_design_mode": resolved_initial_mode,
            "initial_compiler_record": initial_compiler_record,
        },
    )

    for round_index in range(rounds):
        if calibration_gate_triggered:
            menu, menu_diagnostics = build_candidate_menu(
                target,
                observed_indices,
                source_prior,
                protocol["kernel"],
                args.menu_gp_count,
                args.menu_source_count,
                args.menu_consensus_count,
                1,
                args.menu_diversity_count,
                "target_gp",
                getattr(args, "max_transfer_gp_rank", None),
                getattr(args, "safety_fallback_gp_count", 1),
            )
            selected_id = str(menu_diagnostics["gp_incumbent_candidate"])
            selected_by = "calibration_gate_target_gp"
            decision = {
                "hypothesis_status": "insufficient",
                "updated_hypothesis": str(
                    (previous_decision or initial_policy).get(
                        "updated_hypothesis",
                        initial_policy.get("hypothesis", ""),
                    )
                ),
                "selected_candidate_id": selected_id,
                "decision_type": "calibration_gate_fallback",
                "expected_outcome": None,
                "probability_of_improving_current_best": None,
                "confidence": None,
                "evidence_for": [],
                "evidence_against": list(calibration_gate_trigger_reason),
                "reasoning_summary": (
                    "The frozen calibration gate transferred control to target-only "
                    "GP-UCB; no LLM request was made this round."
                ),
                "continue_source_transfer": False,
                "decision_verdict": "revise_to_gp",
                "gp_default_comparison": "Deterministic target-only GP-UCB rank one.",
            }
            append_trace(
                trace_path,
                {
                    "event": "calibration_gate_target_gp_decision",
                    "round_index": round_index,
                    "selected_candidate": selected_id,
                    "selected_by": selected_by,
                    "gate_trigger_reasons": calibration_gate_trigger_reason,
                    "menu_diagnostics": menu_diagnostics,
                    "normalized_decision": decision,
                },
            )

            selected_index = by_id[selected_id]
            if selected_index in observed_set:
                raise RuntimeError("The calibration gate selected an observed candidate.")
            selected_menu_row = next(
                row for row in menu if row["candidate_id"] == selected_id
            )
            gp_rank = int(selected_menu_row["model_evidence"]["gp_rank"])
            consensus_rank = int(
                selected_menu_row["model_evidence"]["consensus_rank"]
            )
            if gp_rank == 1:
                gp_rank_one_choices += 1
            if consensus_rank == 1:
                consensus_rank_one_choices += 1
            observed_indices.append(selected_index)
            observed_set.add(selected_index)
            revealed = float(target.candidates[selected_index].objective_value)
            best_so_far = max(
                float(target.candidates[index].objective_value)
                for index in observed_indices
            )
            best_trace.append(best_so_far)
            calibration_gate_fallback_rounds += 1
            append_trace(
                trace_path,
                {
                    "event": "target_reveal",
                    "round_index": round_index,
                    "selected_candidate": selected_id,
                    "selected_by": selected_by,
                    "gp_rank_at_selection": gp_rank,
                    "source_prior_rank_at_selection": int(
                        selected_menu_row["model_evidence"]["source_prior_rank"]
                    ),
                    "consensus_rank_at_selection": consensus_rank,
                    "public_conditions": initial_design.public_candidate(
                        target.candidates[selected_index],
                        target,
                    )["conditions"],
                    "revealed_value": revealed,
                    "best_so_far": best_so_far,
                    "prediction_error": None,
                    "calibration_prediction_scored": False,
                    "calibration_gate_active": True,
                },
            )
            previous_decision = decision
            continue

        source_transfer_stopped = previous_decision is not None and not bool(
            previous_decision.get("continue_source_transfer", True)
        )
        high_authority = (
            getattr(args, "decision_policy", "calibrated") == "high_authority"
        )
        if source_transfer_stopped:
            eligibility_mode = "target_gp"
        elif high_authority:
            eligibility_mode = "full_menu"
        else:
            eligibility_mode = "transfer_consensus"
        if high_authority and eligibility_mode == "target_gp":
            decision_consensus_limit = args.menu_gp_count
        else:
            decision_consensus_limit = (
                1
                if eligibility_mode == "target_gp"
                or (round_index == 0 and args.force_first_consensus)
                else args.menu_consensus_count
            )
        menu, menu_diagnostics = build_candidate_menu(
            target,
            observed_indices,
            source_prior,
            protocol["kernel"],
            args.menu_gp_count,
            args.menu_source_count,
            args.menu_consensus_count,
            decision_consensus_limit,
            args.menu_diversity_count,
            eligibility_mode,
            getattr(args, "max_transfer_gp_rank", None),
            getattr(args, "safety_fallback_gp_count", 1),
        )
        eligible_candidate_count = sum(
            bool(row["model_evidence"]["decision_eligible"])
            for row in menu
        )
        eligible_candidate_total += eligible_candidate_count
        if eligible_candidate_count > 1:
            llm_choice_rounds += 1
        prompt = build_round_prompt(
            target,
            initial_policy,
            observed_indices,
            menu,
            round_index,
            rounds,
            previous_decision,
            menu_diagnostics,
        )
        assert_no_unrevealed_outcomes(prompt)
        request_event: dict[str, Any] = {
            "event": "llm_round_request",
            "round_index": round_index,
            "model": args.llm_model,
            "prompt_sha256": payload_hash(prompt),
            "prompt": prompt,
            "menu_diagnostics": menu_diagnostics,
        }
        append_trace(trace_path, request_event)
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an online AI scientist controlling a sequential "
                        "experiment. Use observed evidence, calibrated model predictions, "
                        "and scientific mechanism together. Optimize the stated reward "
                        "and return one valid JSON object only."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
            proposal, content, metadata, parsed, attempts = validated_llm_json(
                llm_config(args),
                [
                    *messages,
                ],
                lambda response, current_menu=menu: normalize_round_response(
                    response,
                    current_menu,
                ),
                {
                    "allowed_candidate_ids": [
                        row["candidate_id"]
                        for row in menu
                        if row["model_evidence"]["decision_eligible"]
                    ],
                    "selected_candidate_count": 1,
                    "allowed_hypothesis_status": [
                        "supported",
                        "mixed",
                        "falsified",
                        "insufficient",
                    ],
                    "required_output_schema": prompt["required_output"],
                },
                args.llm_repair_attempts,
            )
            usage = aggregate_attempt_usage(attempts)
            for key in usage_totals:
                usage_totals[key] += int(usage.get(key, 0) or 0)
            append_trace(
                trace_path,
                {
                    "event": "llm_round_proposal_response",
                    "round_index": round_index,
                    "model": metadata.get("model", args.llm_model),
                    "usage": usage,
                    "raw_response": content,
                    "parsed_response": parsed,
                    "normalized_decision": proposal,
                    "api_attempts": attempts,
                },
            )

            decision = proposal
            selected_by = "llm_proposer"
            if getattr(args, "deliberation_mode", "single") == "proposal_critic":
                critic_prompt = build_critic_prompt(prompt, proposal)
                assert_no_unrevealed_outcomes(critic_prompt)
                append_trace(
                    trace_path,
                    {
                        "event": "llm_round_critic_request",
                        "round_index": round_index,
                        "model": args.llm_model,
                        "prompt_sha256": payload_hash(critic_prompt),
                        "prompt": critic_prompt,
                    },
                )
                critic_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are the final decision critic for an online scientific "
                            "experiment. Independently audit the proposal, protect the "
                            "best-so-far AUC reward, and return one valid JSON object only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(critic_prompt, ensure_ascii=False),
                    },
                ]
                critic, critic_content, critic_metadata, critic_parsed, critic_attempts = (
                    validated_llm_json(
                        llm_config(args),
                        critic_messages,
                        lambda response, current_menu=menu: normalize_round_response(
                            response,
                            current_menu,
                        ),
                        {
                            "allowed_candidate_ids": [
                                row["candidate_id"]
                                for row in menu
                                if row["model_evidence"]["decision_eligible"]
                            ],
                            "selected_candidate_count": 1,
                            "required_output_schema": critic_prompt["required_output"],
                        },
                        args.llm_repair_attempts,
                    )
                )
                critic_usage = aggregate_attempt_usage(critic_attempts)
                for key in usage_totals:
                    usage_totals[key] += int(critic_usage.get(key, 0) or 0)
                if critic["selected_candidate_id"] != proposal["selected_candidate_id"]:
                    critic_revision_rounds += 1
                decision = critic
                selected_by = "llm_critic"
                append_trace(
                    trace_path,
                    {
                        "event": "llm_round_critic_response",
                        "round_index": round_index,
                        "model": critic_metadata.get("model", args.llm_model),
                        "usage": critic_usage,
                        "raw_response": critic_content,
                        "parsed_response": critic_parsed,
                        "normalized_decision": critic,
                        "proposal_candidate": proposal["selected_candidate_id"],
                        "api_attempts": critic_attempts,
                    },
                )
            selected_id = decision["selected_candidate_id"]
            llm_selected_rounds += 1
            response_event = {
                "event": "llm_round_response",
                "round_index": round_index,
                "model": args.llm_model,
                "selected_by": selected_by,
                "proposal_candidate": proposal["selected_candidate_id"],
                "normalized_decision": decision,
            }
            append_trace(trace_path, response_event)
        except Exception as exc:
            eligible_ids = {
                str(row["candidate_id"])
                for row in menu
                if row["model_evidence"]["decision_eligible"]
            }
            consensus_id = str(menu_diagnostics["consensus_candidate"])
            gp_id = str(menu_diagnostics["gp_incumbent_candidate"])
            if consensus_id in eligible_ids:
                selected_id = consensus_id
                selected_by = "consensus_fallback_after_llm_error"
            else:
                selected_id = gp_id
                selected_by = "gp_fallback_after_llm_error"
            decision = {
                "hypothesis_status": "insufficient",
                "updated_hypothesis": str(
                    (previous_decision or initial_policy).get("updated_hypothesis")
                    or initial_policy.get("hypothesis", "")
                ),
                "selected_candidate_id": selected_id,
                "decision_type": "recovery",
                "expected_outcome": 0.0,
                "probability_of_improving_current_best": 0.0,
                "confidence": 0.0,
                "evidence_for": [],
                "evidence_against": [f"LLM call failed: {type(exc).__name__}"],
                "reasoning_summary": "Calibrated rank-fusion fallback.",
                "continue_source_transfer": False,
                "decision_verdict": "revise_to_gp",
                "gp_default_comparison": "LLM call failed; use the eligible fallback.",
            }
            append_trace(
                trace_path,
                {
                    "event": "llm_round_error",
                    "round_index": round_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "api_attempts": getattr(exc, "attempts", []),
                    "fallback_candidate": selected_id,
                },
            )
            if args.fail_on_llm_error:
                raise

        selected_index = by_id[selected_id]
        if selected_index in observed_set:
            raise RuntimeError("The online policy selected an observed candidate.")
        selected_menu_row = next(
            row for row in menu if row["candidate_id"] == selected_id
        )
        gp_rank = int(selected_menu_row["model_evidence"]["gp_rank"])
        if selected_id != str(menu_diagnostics["gp_incumbent_candidate"]):
            gp_override_rounds += 1
        if gp_rank == 1:
            gp_rank_one_choices += 1
        consensus_rank = int(selected_menu_row["model_evidence"]["consensus_rank"])
        if consensus_rank == 1:
            consensus_rank_one_choices += 1
        if decision["continue_source_transfer"]:
            source_active_rounds += 1
        observed_indices.append(selected_index)
        observed_set.add(selected_index)
        revealed = float(target.candidates[selected_index].objective_value)
        best_so_far = max(
            float(target.candidates[index].objective_value)
            for index in observed_indices
        )
        best_trace.append(best_so_far)
        round_calibration_error = calibration_prediction_error(decision, revealed)
        append_trace(
            trace_path,
            {
                "event": "target_reveal",
                "round_index": round_index,
                "selected_candidate": selected_id,
                "selected_by": selected_by,
                "gp_rank_at_selection": gp_rank,
                "source_prior_rank_at_selection": int(
                    selected_menu_row["model_evidence"]["source_prior_rank"]
                ),
                "consensus_rank_at_selection": consensus_rank,
                "public_conditions": initial_design.public_candidate(
                    target.candidates[selected_index],
                    target,
                )["conditions"],
                "revealed_value": revealed,
                "best_so_far": best_so_far,
                "prediction_error": round(
                    abs(float(decision["expected_outcome"]) - revealed), 6
                ),
                "calibration_prediction_error": (
                    round(round_calibration_error, 6)
                    if round_calibration_error is not None
                    else None
                ),
                "calibration_prediction_scored": (
                    round_calibration_error is not None
                ),
                "calibration_gate_active": False,
            },
        )
        if round_calibration_error is not None:
            calibration_prediction_errors.append(round_calibration_error)
        if calibration_gate_hard_abstention and (
            decision.get("hypothesis_status") == "falsified"
            or decision.get("continue_source_transfer") is False
        ):
            calibration_hard_abstention_seen = True
        if (
            calibration_gate_enabled
            and round_index + 1 == calibration_gate_round
        ):
            calibration_gate_snapshot_record = calibration_gate_snapshot(
                calibration_prediction_errors,
                hard_abstention=calibration_hard_abstention_seen,
                threshold=calibration_gate_threshold,
                force_fallback=calibration_gate_force_fallback,
            )
            calibration_gate_triggered = bool(
                calibration_gate_snapshot_record["switch_to_target_gp"]
            )
            calibration_gate_trigger_reason = list(
                calibration_gate_snapshot_record["trigger_reasons"]
            )
            append_trace(
                trace_path,
                {
                    "event": "calibration_gate_evaluation",
                    "round_index": round_index,
                    "gate_round": calibration_gate_round,
                    **calibration_gate_snapshot_record,
                    "next_round_policy": (
                        "target_only_gp_ucb"
                        if calibration_gate_triggered
                        else "continue_online_llm"
                    ),
                },
            )
        previous_decision = decision

    oracle = max(candidate.objective_value for candidate in target.candidates)
    online_metrics = {
        "target_dataset": target.dataset_id,
        "mode": MODE,
        "initial_observations": len(initial_indices),
        "reveal_rounds": rounds,
        "initial_candidate_ids": ";".join(initial_policy["selected_candidate_ids"]),
        "executed_initial_candidate_ids": ";".join(executed_initial_ids),
        "initial_design_mode": resolved_initial_mode,
        "source_datasets": ";".join(selected_sources),
        "final_best": round(max(best_trace), 6),
        "best_so_far_auc": round(float(np.mean(best_trace)), 6),
        "simple_regret": round(oracle - max(best_trace), 6),
        "llm_selected_rounds": llm_selected_rounds,
        "llm_participation_rate": round(llm_selected_rounds / rounds, 6),
        "llm_choice_rounds": llm_choice_rounds,
        "llm_decision_authority_rate": round(llm_choice_rounds / rounds, 6),
        "llm_mean_eligible_candidate_count": round(
            eligible_candidate_total / rounds, 6
        ),
        "llm_mean_eligible_candidate_count_when_called": round(
            eligible_candidate_total / max(1, llm_selected_rounds), 6
        ),
        "llm_gp_override_rate": round(gp_override_rounds / rounds, 6),
        "llm_critic_revision_rate": round(critic_revision_rounds / rounds, 6),
        "gp_rank_one_choice_rate": round(gp_rank_one_choices / rounds, 6),
        "consensus_rank_one_choice_rate": round(
            consensus_rank_one_choices / rounds, 6
        ),
        "source_transfer_active_rate": round(source_active_rounds / rounds, 6),
        "calibration_gate_enabled": calibration_gate_enabled,
        "calibration_gate_round": calibration_gate_round,
        "calibration_gate_mae_threshold": calibration_gate_threshold,
        "calibration_gate_hard_abstention_enabled": (
            calibration_gate_hard_abstention
        ),
        "calibration_gate_force_fallback_enabled": (
            calibration_gate_force_fallback
        ),
        "calibration_gate_triggered": calibration_gate_triggered,
        "calibration_gate_trigger_reasons": calibration_gate_trigger_reason,
        "calibration_gate_scored_prediction_count": (
            calibration_gate_snapshot_record["scored_prediction_count"]
            if calibration_gate_snapshot_record is not None
            else None
        ),
        "calibration_gate_prediction_mae": (
            calibration_gate_snapshot_record[
                "mean_absolute_prediction_error"
            ]
            if calibration_gate_snapshot_record is not None
            else None
        ),
        "calibration_gate_fallback_rounds": calibration_gate_fallback_rounds,
        "calibration_gate_llm_rounds_saved": calibration_gate_fallback_rounds,
        "calibration_gate_nominal_llm_calls_avoided": (
            calibration_gate_fallback_rounds * logical_llm_calls_per_round
        ),
    }
    same_initial_metrics, _same_initial_audit = warmstart.run_target_gp(
        target,
        initial_indices,
        rounds,
        SAME_INITIAL_GP_MODE,
        -1,
        protocol["kernel"],
        selected_sources,
        {"initial_policy": "llm"},
    )
    fixed_indices, fixed_sources, fixed_policy = initial_design.fixed_v2_initial(
        target, source_ids, protocol
    )
    fixed_metrics, _fixed_audit = warmstart.run_target_gp(
        target,
        fixed_indices,
        rounds,
        initial_design.FIXED_V2_MODE,
        -1,
        protocol["kernel"],
        fixed_sources,
        fixed_policy,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "care_online_llm_scientist",
        "target_task": target.dataset_id,
        "source_tasks": source_ids,
        "initial_model": record["model"],
        "online_model": args.llm_model,
        "initial_design_mode": resolved_initial_mode,
        "initial_compiler_record": initial_compiler_record,
        "evidence_boundary": (
            "Each LLM decision sees completed source evidence, public target conditions, "
            "and only target outcomes revealed before that round."
        ),
        "matched_target_budget": True,
        "requested_reveal_rounds": requested_rounds,
        "actual_reveal_rounds": rounds,
        "calibration_gate_policy": {
            "enabled": calibration_gate_enabled,
            "evaluation_after_llm_guided_reveals": calibration_gate_round,
            "mean_absolute_prediction_error_threshold": (
                calibration_gate_threshold
            ),
            "hard_abstention_enabled": calibration_gate_hard_abstention,
            "force_fallback_after_evaluation": calibration_gate_force_fallback,
            "hard_abstention_conditions": [
                "hypothesis_status=falsified",
                "continue_source_transfer=false",
            ],
            "prediction_scoring_rule": (
                "Score absolute error only when the LLM asserts an expected outcome; "
                "exclude decision_verdict=revise_to_gp."
            ),
            "fallback": (
                "Stop all later LLM requests and execute target-only GP-UCB rank one "
                "from every observation accumulated before the gate."
            ),
            "triggered": calibration_gate_triggered,
            "trigger_reasons": calibration_gate_trigger_reason,
        },
        "menu_policy": {
            "gp_count": args.menu_gp_count,
            "source_count": args.menu_source_count,
            "consensus_count": args.menu_consensus_count,
            "force_first_consensus": args.force_first_consensus,
            "diversity_count": args.menu_diversity_count,
            "adaptive_routing_from_llm_continue_source_transfer": True,
            "decision_policy": getattr(args, "decision_policy", "calibrated"),
            "deliberation_mode": getattr(args, "deliberation_mode", "single"),
            "target_gp_fallback_eligible_count": (
                args.menu_gp_count
                if getattr(args, "decision_policy", "calibrated")
                == "high_authority"
                else 1
            ),
            "max_transfer_gp_rank": getattr(
                args,
                "max_transfer_gp_rank",
                None,
            ),
            "safety_fallback_gp_count": getattr(
                args,
                "safety_fallback_gp_count",
                1,
            ),
        },
        "metrics": {
            MODE: online_metrics,
            SAME_INITIAL_GP_MODE: same_initial_metrics,
            initial_design.FIXED_V2_MODE: fixed_metrics,
        },
        "deltas": {
            "online_llm_increment_over_same_initial_gp": {
                "best_so_far_auc": round(
                    online_metrics["best_so_far_auc"]
                    - same_initial_metrics["best_so_far_auc"],
                    6,
                ),
                "final_best": round(
                    online_metrics["final_best"] - same_initial_metrics["final_best"],
                    6,
                ),
            },
            "full_llm_scientist_vs_fixed_v2": {
                "best_so_far_auc": round(
                    online_metrics["best_so_far_auc"] - fixed_metrics["best_so_far_auc"],
                    6,
                ),
                "final_best": round(
                    online_metrics["final_best"] - fixed_metrics["final_best"],
                    6,
                ),
            },
        },
        "usage": usage_totals,
        "trace_file": str(trace_path),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_fingerprint(trace_path)
    write_fingerprint(args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_llm_arguments(parser: argparse.ArgumentParser, max_tokens: int) -> None:
    parser.add_argument(
        "--llm-base-url", default="https://api.commonstack.ai/v1"
    )
    parser.add_argument("--llm-model", default="anthropic/claude-fable-5")
    parser.add_argument("--llm-api-key-env", default="COMMONSTACK_API_KEY")
    parser.add_argument(
        "--llm-api-mode",
        choices=("chat", "completion", "anthropic"),
        default="chat",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-max-tokens", type=int, default=max_tokens)
    parser.add_argument("--llm-repair-attempts", type=int, default=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-initial")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--target-task", required=True)
    generate_parser.add_argument("--source-tasks", nargs="+", required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--per-view-limit", type=int, default=10)
    add_llm_arguments(generate_parser, 2200)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--initial-record", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--initial-design-mode",
        choices=("auto", "direct", "compiled"),
        default="auto",
    )
    run_parser.add_argument("--rounds", type=int)
    run_parser.add_argument("--menu-gp-count", type=int, default=5)
    run_parser.add_argument("--menu-source-count", type=int, default=2)
    run_parser.add_argument("--menu-consensus-count", type=int, default=2)
    run_parser.add_argument("--max-transfer-gp-rank", type=int, default=5)
    run_parser.add_argument("--safety-fallback-gp-count", type=int, default=1)
    run_parser.add_argument(
        "--force-first-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_parser.add_argument("--menu-diversity-count", type=int, default=1)
    run_parser.add_argument(
        "--decision-policy",
        choices=("calibrated", "high_authority"),
        default="calibrated",
    )
    run_parser.add_argument(
        "--deliberation-mode",
        choices=("single", "proposal_critic"),
        default="single",
    )
    run_parser.add_argument(
        "--calibration-gate-round",
        type=int,
        help=(
            "Evaluate the frozen prediction-error gate after this many "
            "LLM-guided reveals. Omit to disable the gate."
        ),
    )
    run_parser.add_argument(
        "--calibration-gate-mae-threshold",
        type=float,
        help="Switch to target-only GP-UCB when prefix prediction MAE exceeds this value.",
    )
    run_parser.add_argument(
        "--calibration-gate-hard-abstention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also switch when the prefix contains a falsified hypothesis or an "
            "explicit request to stop source transfer."
        ),
    )
    run_parser.add_argument(
        "--calibration-gate-force-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Always transfer control to target-only GP-UCB after the configured "
            "gate round. This implements a bounded LLM-authority controller."
        ),
    )
    run_parser.add_argument("--fail-on-llm-error", action="store_true")
    add_llm_arguments(run_parser, 1400)

    args = parser.parse_args()
    if args.command == "generate-initial":
        generate_initial(args)
    else:
        run_online(args)


if __name__ == "__main__":
    main()
