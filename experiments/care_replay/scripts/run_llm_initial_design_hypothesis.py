#!/usr/bin/env python3
"""Generate and evaluate an outcome-blind LLM initial-design hypothesis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import run_classical_transfer_baselines as classical
import run_multisource_warmstart as warmstart
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
LLM_MODE = "llm_hypothesis_initial_design"
LLM_COMPILED_MODE = "llm_hypothesis_compiled_initial_design"
LLM_REFLECTIVE_MODE = "llm_hypothesis_reflective_design"
FIXED_V2_MODE = "fixed_v2_initial_design"
SHORTLIST_RANDOM_MODE = "matched_shortlist_random_initial"

PUBLIC_FIELDS = (
    "base",
    "precatalyst",
    "base_equivalents",
    "temperature_celsius",
    "residence_time_minutes",
    "residence_time_seconds",
    "precatalyst_loading_mol_percent",
    "precatalyst_fraction",
    "substrate",
    "campaign_name",
)

PUBLIC_IDENTITY_FIELDS = (
    "compound_id",
    "iupac",
    "smiles",
    "composition",
)


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def public_candidate(
    candidate: replay.Candidate,
    adapter: replay.DatasetAdapter | None = None,
) -> dict[str, Any]:
    condition_fields = list(PUBLIC_FIELDS)
    if adapter is not None:
        condition_fields.extend(adapter.decision_columns)
    condition_fields.extend(PUBLIC_IDENTITY_FIELDS)
    condition_fields = list(dict.fromkeys(condition_fields))
    return {
        "candidate_id": candidate.candidate_id,
        "group": candidate.group,
        "conditions": {
            key: candidate.metadata.get(key)
            for key in condition_fields
            if candidate.metadata.get(key) is not None
        },
        "normalized_numeric_features": [
            round(float(value), 6) for value in candidate.numeric_features
        ],
    }


def source_summary(adapter: replay.DatasetAdapter) -> dict[str, Any]:
    values = np.asarray(
        [candidate.objective_value for candidate in adapter.candidates],
        dtype=np.float64,
    )
    top = sorted(
        adapter.candidates,
        key=lambda candidate: candidate.objective_value,
        reverse=True,
    )[:5]
    group_values: dict[str, list[float]] = {}
    for candidate in adapter.candidates:
        group_values.setdefault(str(candidate.group), []).append(
            float(candidate.objective_value)
        )
    group_means = sorted(
        (
            {
                "group": group,
                "count": len(group_outcomes),
                "mean_outcome": round(float(np.mean(group_outcomes)), 6),
                "max_outcome": round(float(np.max(group_outcomes)), 6),
            }
            for group, group_outcomes in group_values.items()
        ),
        key=lambda item: item["mean_outcome"],
        reverse=True,
    )[:8]
    correlations = []
    feature_matrix = np.asarray(
        [candidate.numeric_features for candidate in adapter.candidates],
        dtype=np.float64,
    )
    for index in range(feature_matrix.shape[1]):
        feature = feature_matrix[:, index]
        if float(np.std(feature)) <= 1e-12 or float(np.std(values)) <= 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(feature, values)[0, 1])
        correlations.append(
            {"feature_index": index, "outcome_correlation": round(correlation, 6)}
        )
    return {
        "dataset_id": adapter.dataset_id,
        "title": adapter.title,
        "objective": adapter.objective,
        "descriptor": warmstart.task_descriptor(adapter),
        "candidate_count": len(adapter.candidates),
        "outcome_summary": {
            "minimum": round(float(np.min(values)), 6),
            "mean": round(float(np.mean(values)), 6),
            "q75": round(float(np.quantile(values, 0.75)), 6),
            "maximum": round(float(np.max(values)), 6),
        },
        "numeric_feature_outcome_correlations": correlations,
        "group_outcome_summary": group_means,
        "top_source_observations": [
            {
                **public_candidate(candidate, adapter),
                "source_outcome": round(float(candidate.objective_value), 6),
            }
            for candidate in top
        ],
    }


def source_views(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
) -> dict[str, list[str]]:
    target_descriptor = warmstart.task_descriptor(target)
    descriptors = {
        source_id: warmstart.task_descriptor(replay.DATASET_BUILDERS[source_id]())
        for source_id in source_ids
    }
    views = {"all_sources": list(source_ids)}
    same_substrate = [
        source_id
        for source_id in source_ids
        if descriptors[source_id]["substrate"] == target_descriptor["substrate"]
    ]
    same_precatalyst = [
        source_id
        for source_id in source_ids
        if descriptors[source_id]["precatalyst"]
        == target_descriptor["precatalyst"]
    ]
    if same_substrate:
        views["same_substrate"] = same_substrate
    if same_precatalyst:
        views["same_precatalyst"] = same_precatalyst
    return views


def candidate_shortlist(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    per_view_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, np.ndarray]]:
    views = source_views(target, source_ids)
    priors: dict[str, np.ndarray] = {}
    shortlist_indices: set[int] = set()
    for name, view_source_ids in views.items():
        prior, _ = warmstart.build_source_consensus(
            target,
            view_source_ids,
            protocol,
        )
        priors[name] = prior
        shortlist_indices.update(
            int(index) for index in np.argsort(-prior)[:per_view_limit]
        )
    shortlist_indices.update(
        warmstart.space_filling_initial(
            target,
            int(protocol["initial_observations"]),
        )
    )
    rows = []
    for index in sorted(shortlist_indices):
        item = public_candidate(target.candidates[index], target)
        item["source_evidence"] = {
            name: {
                "score": round(float(prior[index]), 6),
                "rank": int(np.where(np.argsort(-prior) == index)[0][0]) + 1,
            }
            for name, prior in priors.items()
        }
        rows.append(item)
    rows.sort(
        key=lambda item: max(
            evidence["score"] for evidence in item["source_evidence"].values()
        ),
        reverse=True,
    )
    return rows, views, priors


def target_public_spec(adapter: replay.DatasetAdapter) -> dict[str, Any]:
    numeric = np.asarray(
        [candidate.numeric_features for candidate in adapter.candidates],
        dtype=np.float64,
    )
    return {
        "dataset_id": adapter.dataset_id,
        "title": adapter.title,
        "objective": adapter.objective,
        "decision_columns": list(adapter.decision_columns),
        "descriptor": warmstart.task_descriptor(adapter),
        "candidate_count": len(adapter.candidates),
        "numeric_feature_ranges": [
            {
                "feature_index": index,
                "minimum": round(float(np.min(numeric[:, index])), 6),
                "maximum": round(float(np.max(numeric[:, index])), 6),
            }
            for index in range(numeric.shape[1])
        ],
        "hidden_field_redacted": True,
        "target_outcomes_available_to_llm": False,
    }


def build_prompt(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    return {
        "task": (
            "Freeze one source-grounded initial-design hypothesis for a new target. "
            "Select exactly three candidate IDs before any target outcome is revealed."
        ),
        "scientific_context": {
            "target": target_public_spec(target),
            "completed_source_experiments": [
                source_summary(replay.DATASET_BUILDERS[source_id]())
                for source_id in source_ids
            ],
            "source_views": {name: list(ids) for name, ids in views.items()},
            "candidate_shortlist": list(shortlist),
            "archived_failure_lessons": [
                (
                    "Unconstrained maximin diversity can leave the source-supported "
                    "quality region and cause negative transfer."
                ),
                (
                    "Prefer a quality anchor plus complementary candidates that remain "
                    "plausible under source evidence."
                ),
                (
                    "Do not infer or invent target outcomes. Candidate source scores are "
                    "priors, not target measurements."
                ),
                (
                    "A quality-only design concentrated in one catalyst or condition family "
                    "can give a good first observation but leave the subsequent target GP "
                    "poorly conditioned. Preserve source-supported geometric coverage."
                ),
            ],
        },
        "fixed_execution_after_initial_design": {
            "initial_observations": int(protocol["initial_observations"]),
            "target_only_gp_ucb_reveal_rounds": int(protocol["reveal_rounds"]),
            "kernel": dict(protocol["kernel"]),
            "source_information_exits_after_initial_design": True,
        },
        "required_output_schema": {
            "hypothesis": "one falsifiable source-to-target claim",
            "selected_candidate_ids": ["candidate_id_1", "candidate_id_2", "candidate_id_3"],
            "selected_source_view": "one key from source_views",
            "selection_roles": [
                {"candidate_id": "candidate_id_1", "role": "quality_anchor or complementary_probe"}
            ],
            "mechanism": "why the source evidence should transfer",
            "confidence": 0.0,
            "failure_conditions": ["conditions that would falsify or limit this skill"],
            "abstain": False,
            "abstain_reason": "empty unless abstain is true",
        },
        "constraints": [
            "Return one JSON object only.",
            "Use exactly three unique candidate IDs from candidate_shortlist unless abstain=true.",
            "Do not request or speculate about hidden target outcomes.",
            "Do not change the target GP-UCB kernel, reveal budget, or candidate pool.",
            "Use source evidence and scientific plausibility, not candidate ID order.",
            (
                "Select one source-quality anchor and two complementary probes. The three "
                "candidates must span at least two distinct group values when the shortlist "
                "contains multiple groups; selecting all three from one group is invalid."
            ),
            (
                "Complementary probes must remain strongly source-supported while adding "
                "coverage across catalyst identity and numeric operating conditions."
            ),
        ],
    }


def assert_outcome_blind_prompt(prompt: Mapping[str, Any]) -> None:
    encoded = json.dumps(prompt, ensure_ascii=False).lower()
    prohibited = (
        '"objective_value"',
        '"yield_value"',
        '"revealed_value"',
        '"target_outcome"',
    )
    leaked = [token for token in prohibited if token in encoded]
    if leaked:
        raise ValueError(f"Prompt contains prohibited target-outcome fields: {leaked}")


def normalize_hypothesis(
    response: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    abstain = bool(response.get("abstain", False))
    allowed_ids = {str(item["candidate_id"]) for item in shortlist}
    selected = [str(item) for item in response.get("selected_candidate_ids", [])]
    if abstain:
        if not str(response.get("abstain_reason", "")).strip():
            raise ValueError("An abstaining LLM response must include abstain_reason.")
        selected = []
    else:
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError("LLM must select exactly three unique candidate IDs.")
        unknown = sorted(set(selected) - allowed_ids)
        if unknown:
            raise ValueError(f"LLM selected candidates outside the shortlist: {unknown}")
        group_by_id = {
            str(item["candidate_id"]): str(item.get("group", ""))
            for item in shortlist
        }
        available_groups = {group for group in group_by_id.values() if group}
        selected_groups = {group_by_id[candidate_id] for candidate_id in selected}
        if len(available_groups) > 1 and len(selected_groups) < 2:
            raise ValueError(
                "LLM initial design must cover at least two candidate groups."
            )
    view = str(response.get("selected_source_view", "all_sources"))
    if view not in views:
        raise ValueError(f"Unknown selected_source_view: {view}")
    confidence = float(response.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1].")
    return {
        "hypothesis": str(response.get("hypothesis", "")).strip(),
        "selected_candidate_ids": selected,
        "selected_source_view": view,
        "selection_roles": list(response.get("selection_roles", [])),
        "mechanism": str(response.get("mechanism", "")).strip(),
        "confidence": confidence,
        "failure_conditions": [
            str(item) for item in response.get("failure_conditions", [])
        ],
        "abstain": abstain,
        "abstain_reason": str(response.get("abstain_reason", "")).strip(),
    }


def compact_source_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": summary.get("dataset_id"),
        "descriptor": summary.get("descriptor", {}),
        "outcome_summary": summary.get("outcome_summary", {}),
        "numeric_feature_outcome_correlations": summary.get(
            "numeric_feature_outcome_correlations", []
        ),
        "group_outcome_summary": list(summary.get("group_outcome_summary", []))[:3],
        "top_source_observations": list(
            summary.get("top_source_observations", [])
        )[:2],
    }


def build_reflection_prompt(
    target: replay.DatasetAdapter,
    protocol: Mapping[str, Any],
    hypothesis_record: Mapping[str, Any],
    followup_count: int,
    initial_candidate_ids: Sequence[str] | None = None,
    initial_design_mode: str = "raw_llm",
) -> dict[str, Any]:
    if followup_count < 1:
        raise ValueError("followup_count must be positive.")
    if followup_count > int(protocol["reveal_rounds"]):
        raise ValueError("followup_count cannot exceed the reveal budget.")
    hypothesis = hypothesis_record["frozen_hypothesis"]
    initial_ids = [
        str(item)
        for item in (
            initial_candidate_ids
            if initial_candidate_ids is not None
            else hypothesis["selected_candidate_ids"]
        )
    ]
    initial_id_set = set(initial_ids)
    index_by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    initial_results = []
    for candidate_id in initial_ids:
        candidate = target.candidates[index_by_id[candidate_id]]
        initial_results.append(
            {
                **public_candidate(candidate, target),
                "revealed_target_outcome": round(
                    float(candidate.objective_value), 6
                ),
            }
        )
    original_context = hypothesis_record["prompt"]["scientific_context"]
    remaining = [
        dict(item)
        for item in original_context["candidate_shortlist"]
        if str(item["candidate_id"]) not in initial_id_set
    ]
    if len(remaining) < followup_count:
        raise ValueError("Not enough remaining shortlist candidates for reflection.")
    return {
        "task": (
            "Act as the second-stage scientist after the first target experiments. "
            "Test the frozen transfer hypothesis against only the newly revealed evidence, "
            "revise it when needed, and select the next experiments within the fixed budget."
        ),
        "scientific_context": {
            "target": target_public_spec(target),
            "completed_source_experiments": [
                compact_source_summary(summary)
                for summary in original_context["completed_source_experiments"]
            ],
            "original_hypothesis": hypothesis,
            "executed_initial_design_mode": initial_design_mode,
            "revealed_initial_target_results": initial_results,
            "remaining_candidate_shortlist": remaining,
            "evidence_boundary": (
                "Only the listed initial target outcomes are revealed. Outcomes for all "
                "remaining candidates are hidden."
            ),
        },
        "fixed_execution": {
            "total_reveal_rounds_including_llm_followups": int(
                protocol["reveal_rounds"]
            ),
            "llm_followup_count": followup_count,
            "remaining_rounds_use_target_only_gp_ucb": True,
            "kernel": dict(protocol["kernel"]),
        },
        "required_output_schema": {
            "hypothesis_status": "supported, mixed, falsified, or insufficient",
            "evidence_interpretation": "what the revealed results support and contradict",
            "revised_hypothesis": "a falsifiable updated claim",
            "selected_candidate_ids": [
                f"next_candidate_{index + 1}" for index in range(followup_count)
            ],
            "selection_roles": [
                {
                    "candidate_id": "next_candidate_1",
                    "role": "confirmation, boundary_probe, or counter_hypothesis",
                }
            ],
            "next_experiment_rationale": "why these experiments are maximally informative",
            "confidence": 0.0,
            "stop_transfer": False,
            "stop_reason": "required when stop_transfer is true",
        },
        "constraints": [
            "Return one JSON object only.",
            (
                f"Select exactly {followup_count} unique IDs from "
                "remaining_candidate_shortlist unless stop_transfer=true."
            ),
            "Use only revealed_initial_target_results as target evidence.",
            "Do not infer or invent outcomes for remaining candidates.",
            "Explicitly identify evidence that contradicts the original hypothesis.",
            "Prefer experiments that discriminate between the revised hypothesis and a counter-hypothesis.",
            "Do not change the total experiment budget, GP kernel, or candidate pool.",
            "Stop transfer when the initial evidence falsifies the mechanism or makes source guidance unsafe.",
        ],
    }


def assert_reflection_prompt_boundary(prompt: Mapping[str, Any]) -> None:
    remaining = prompt["scientific_context"]["remaining_candidate_shortlist"]
    encoded = json.dumps(remaining, ensure_ascii=False).lower()
    prohibited = (
        '"objective_value"',
        '"yield_value"',
        '"revealed_value"',
        '"revealed_target_outcome"',
        '"target_outcome"',
    )
    leaked = [token for token in prohibited if token in encoded]
    if leaked:
        raise ValueError(
            f"Reflection prompt leaks unrevealed target outcomes: {leaked}"
        )


def normalize_reflection(
    response: Mapping[str, Any],
    remaining_shortlist: Sequence[Mapping[str, Any]],
    followup_count: int,
) -> dict[str, Any]:
    status = str(response.get("hypothesis_status", "")).strip().lower()
    if status not in {"supported", "mixed", "falsified", "insufficient"}:
        raise ValueError(f"Unknown hypothesis_status: {status}")
    stop_transfer = bool(response.get("stop_transfer", False))
    selected = [str(item) for item in response.get("selected_candidate_ids", [])]
    if stop_transfer:
        if not str(response.get("stop_reason", "")).strip():
            raise ValueError("A stopped reflection must include stop_reason.")
        selected = []
    else:
        if len(selected) != followup_count or len(set(selected)) != followup_count:
            raise ValueError(
                f"LLM reflection must select exactly {followup_count} unique candidates."
            )
        allowed = {str(item["candidate_id"]) for item in remaining_shortlist}
        unknown = sorted(set(selected) - allowed)
        if unknown:
            raise ValueError(
                f"LLM reflection selected candidates outside the shortlist: {unknown}"
            )
    confidence = float(response.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1].")
    revised = str(response.get("revised_hypothesis", "")).strip()
    if not stop_transfer and not revised:
        raise ValueError("A continuing reflection must include revised_hypothesis.")
    return {
        "hypothesis_status": status,
        "evidence_interpretation": str(
            response.get("evidence_interpretation", "")
        ).strip(),
        "revised_hypothesis": revised,
        "selected_candidate_ids": selected,
        "selection_roles": list(response.get("selection_roles", [])),
        "next_experiment_rationale": str(
            response.get("next_experiment_rationale", "")
        ).strip(),
        "confidence": confidence,
        "stop_transfer": stop_transfer,
        "stop_reason": str(response.get("stop_reason", "")).strip(),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    target = replay.DATASET_BUILDERS[args.target_task]()
    source_ids = list(args.source_tasks)
    shortlist, views, _priors = candidate_shortlist(
        target,
        source_ids,
        protocol,
        args.per_view_limit,
    )
    prompt = build_prompt(target, source_ids, protocol, shortlist, views)
    assert_outcome_blind_prompt(prompt)
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Set {args.llm_api_key_env}, COMMONSTACK_API_KEY, or CARE_LLM_API_KEY."
        )
    llm_config = replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        structured_mode="json",
    )
    content, metadata = replay.chat_completion_text(
        llm_config,
        [
            {
                "role": "system",
                "content": (
                    "You are a scientific experimental-design expert. Return one valid JSON "
                    "object only. The target outcomes are hidden. Make a falsifiable decision "
                    "from completed source evidence and public target conditions."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    parsed = replay.extract_json_object(content)
    normalized = normalize_hypothesis(parsed, shortlist, views)
    record = {
        "schema_version": "care.llm_initial_design_hypothesis/v1",
        "status": "frozen_before_target_replay",
        "model": metadata.get("model", args.llm_model),
        "usage": metadata.get("usage", {}),
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
        "prompt_sha256": sha256_payload(prompt),
        "raw_response": content,
        "parsed_response": parsed,
        "frozen_hypothesis": normalized,
    }
    write_json(args.output, record)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


def reflect(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hypothesis_record = json.loads(
        args.hypothesis_record.read_text(encoding="utf-8")
    )
    if hypothesis_record["config_fingerprint"] != warmstart.config_fingerprint(
        config
    ):
        raise ValueError("Hypothesis record does not match the reflection config.")
    hypothesis = hypothesis_record["frozen_hypothesis"]
    if hypothesis["abstain"]:
        raise ValueError("An abstaining initial hypothesis cannot be reflected.")
    target_id = str(hypothesis_record["target_task"])
    target = replay.DATASET_BUILDERS[target_id]()
    source_ids = [str(item) for item in hypothesis_record["source_tasks"]]
    initial_design_record: dict[str, Any] = {
        "mode": LLM_MODE,
        "candidate_ids": list(hypothesis["selected_candidate_ids"]),
        "source_tasks": source_ids,
    }
    if args.initial_design_mode == "compiled":
        compiled_indices, compiled_sources, compiler_record = compiled_llm_initial(
            target,
            source_ids,
            config["protocol"],
            hypothesis,
        )
        initial_design_record = {
            "mode": LLM_COMPILED_MODE,
            "candidate_ids": [
                target.candidates[index].candidate_id for index in compiled_indices
            ],
            "source_tasks": compiled_sources,
            "compiler_record": compiler_record,
        }
    prompt = build_reflection_prompt(
        target,
        config["protocol"],
        hypothesis_record,
        args.followup_count,
        initial_design_record["candidate_ids"],
        initial_design_record["mode"],
    )
    assert_reflection_prompt_boundary(prompt)
    api_key = (
        os.environ.get(args.llm_api_key_env)
        or os.environ.get("COMMONSTACK_API_KEY")
        or os.environ.get("CARE_LLM_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            f"Set {args.llm_api_key_env}, COMMONSTACK_API_KEY, or CARE_LLM_API_KEY."
        )
    llm_config = replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        structured_mode="json",
    )
    content, metadata = replay.chat_completion_text(
        llm_config,
        [
            {
                "role": "system",
                "content": (
                    "You are the reflection and experiment-planning stage of an AI "
                    "scientist. Update a falsifiable hypothesis from observed evidence, "
                    "actively look for contradictions, and choose the most informative "
                    "next experiments. Never invent hidden outcomes. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    parsed = replay.extract_json_object(content)
    remaining = prompt["scientific_context"]["remaining_candidate_shortlist"]
    normalized = normalize_reflection(parsed, remaining, args.followup_count)
    parent_sha256 = hashlib.sha256(args.hypothesis_record.read_bytes()).hexdigest()
    record = {
        "schema_version": "care.llm_hypothesis_reflection/v1",
        "status": "frozen_after_initial_observations_before_followup_replay",
        "model": metadata.get("model", args.llm_model),
        "usage": metadata.get("usage", {}),
        "api_configuration": {
            "base_url": args.llm_base_url,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
            "structured_mode": "json",
        },
        "config_fingerprint": warmstart.config_fingerprint(config),
        "target_task": target_id,
        "source_tasks": source_ids,
        "initial_design": initial_design_record,
        "parent_hypothesis_record": str(args.hypothesis_record),
        "parent_hypothesis_sha256": parent_sha256,
        "target_evidence_boundary": "initial_observations_only",
        "prompt": prompt,
        "prompt_sha256": sha256_payload(prompt),
        "raw_response": content,
        "parsed_response": parsed,
        "frozen_reflection": normalized,
    }
    write_json(args.output, record)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


def indices_for_ids(
    adapter: replay.DatasetAdapter,
    candidate_ids: Sequence[str],
) -> list[int]:
    index_by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(adapter.candidates)
    }
    return [index_by_id[candidate_id] for candidate_id in candidate_ids]


def fixed_v2_initial(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[int], list[str], dict[str, Any]]:
    policy = {
        "source_scope": "same_substrate_then_precatalyst",
        "diversity_weight": 1.0,
        "source_quantile": 0.5,
    }
    selected_sources = warmstart.source_ids_for_scope(
        target,
        source_ids,
        policy["source_scope"],
    )
    prior, selected_sources = warmstart.build_source_consensus(
        target,
        selected_sources,
        protocol,
    )
    return (
        warmstart.source_diverse_initial(
            target,
            prior,
            int(protocol["initial_observations"]),
            policy["diversity_weight"],
            policy["source_quantile"],
        ),
        selected_sources,
        policy,
    )


def compiled_llm_initial(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    source_quantile: float = 0.5,
) -> tuple[list[int], list[str], dict[str, Any]]:
    """Compile one LLM semantic anchor into a quality-bounded design."""
    views = source_views(target, source_ids)
    view_name = str(hypothesis["selected_source_view"])
    selected_sources = views[view_name]
    prior, selected_sources = warmstart.build_source_consensus(
        target,
        selected_sources,
        protocol,
    )
    index_by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    source_anchor = int(np.argmax(prior))
    role_by_id = {
        str(item.get("candidate_id")): str(item.get("role", ""))
        for item in hypothesis.get("selection_roles", [])
    }
    llm_ids = [str(item) for item in hypothesis["selected_candidate_ids"]]
    semantic_candidates = [
        index_by_id[candidate_id]
        for candidate_id in llm_ids
        if role_by_id.get(candidate_id) == "quality_anchor"
        and index_by_id[candidate_id] != source_anchor
    ]
    if not semantic_candidates:
        semantic_candidates = [
            index_by_id[candidate_id]
            for candidate_id in llm_ids
            if index_by_id[candidate_id] != source_anchor
        ]
    eligible = prior >= np.quantile(prior, source_quantile)
    semantic_candidates = [
        index for index in semantic_candidates if bool(eligible[index])
    ]
    if not semantic_candidates:
        semantic_candidates = [
            int(index)
            for index in np.argsort(-prior)
            if int(index) != source_anchor and bool(eligible[int(index)])
        ]
    semantic_anchor = max(semantic_candidates, key=lambda index: float(prior[index]))

    features = classical.feature_arrays(target, target.candidates)
    selected = [source_anchor, semantic_anchor]
    min_distance = np.full(len(target.candidates), np.inf, dtype=np.float64)
    for index in selected:
        min_distance = np.minimum(
            min_distance,
            classical.mixed_distance_to_point(features, index),
        )
    min_distance[~eligible] = -1.0
    min_distance[np.asarray(selected, dtype=int)] = -1.0
    geometry_probe = int(np.argmax(min_distance))
    selected.append(geometry_probe)
    return selected, list(selected_sources), {
        "compiler": "llm_semantic_anchor_plus_quality_bounded_maximin/v1",
        "selected_source_view": view_name,
        "source_quantile": source_quantile,
        "source_anchor": target.candidates[source_anchor].candidate_id,
        "llm_semantic_anchor": target.candidates[semantic_anchor].candidate_id,
        "geometry_probe": target.candidates[geometry_probe].candidate_id,
        "llm_hypothesis": str(hypothesis.get("hypothesis", "")),
        "failure_conditions": list(hypothesis.get("failure_conditions", [])),
    }


def run_reflective_target_gp(
    adapter: replay.DatasetAdapter,
    initial_indices: Sequence[int],
    followup_indices: Sequence[int],
    rounds: int,
    kernel: Mapping[str, Any],
    source_ids: Sequence[str],
    policy: Mapping[str, Any],
    gate_max_acquisition_loss: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(followup_indices) > rounds:
        raise ValueError("LLM follow-up count cannot exceed the reveal budget.")
    if gate_max_acquisition_loss < 0.0:
        raise ValueError("gate_max_acquisition_loss must be non-negative.")
    pool = adapter.candidates
    features = classical.feature_arrays(adapter, pool)
    observed = [int(index) for index in initial_indices]
    followups = [int(index) for index in followup_indices]
    if len(set([*observed, *followups])) != len(observed) + len(followups):
        raise ValueError("Initial and reflective candidate indices must be unique.")
    observed_set = set(observed)
    audit: list[dict[str, Any]] = [
        {
            "event": "initial_design_revealed",
            "target_dataset": adapter.dataset_id,
            "mode": LLM_REFLECTIVE_MODE,
            "seed": -1,
            "source_datasets": list(source_ids),
            "policy": dict(policy),
            "candidate_ids": [pool[index].candidate_id for index in observed],
            "public_conditions": [
                warmstart.public_conditions(pool[index]) for index in observed
            ],
            "revealed_values": [pool[index].objective_value for index in observed],
        }
    ]
    best_trace: list[float] = []
    for round_index, proposed_index in enumerate(followups):
        observed_y, _center, _scale = classical.normalized_outcomes(
            [pool[index] for index in observed]
        )
        posterior_mean, posterior_variance, _diagnostics = (
            classical.target_gp_posterior(
                features,
                observed,
                observed_y,
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
        scores = posterior_mean + float(kernel["gp_beta"]) * np.sqrt(
            posterior_variance
        )
        incumbent_index = classical.top_unobserved(scores, observed_set)
        proposed_available = proposed_index not in observed_set
        acquisition_loss = (
            max(0.0, float(scores[incumbent_index] - scores[proposed_index]))
            if proposed_available
            else float("inf")
        )
        authorized = (
            proposed_available
            and acquisition_loss <= gate_max_acquisition_loss + 1e-12
        )
        selected_index = proposed_index if authorized else incumbent_index
        selected = pool[selected_index]
        observed.append(selected_index)
        observed_set.add(selected_index)
        best_so_far = max(pool[index].objective_value for index in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "event": "llm_reflection_reveal",
                "target_dataset": adapter.dataset_id,
                "mode": LLM_REFLECTIVE_MODE,
                "seed": -1,
                "round_index": round_index,
                "selected_candidate": selected.candidate_id,
                "llm_proposed_candidate": pool[proposed_index].candidate_id,
                "gp_incumbent_candidate": pool[incumbent_index].candidate_id,
                "public_conditions": warmstart.public_conditions(selected),
                "selected_by": (
                    "llm_reflection"
                    if authorized
                    else "gp_incumbent_after_reflection_gate"
                ),
                "reflection_gate": {
                    "authorized": authorized,
                    "acquisition_loss": (
                        round(acquisition_loss, 6)
                        if np.isfinite(acquisition_loss)
                        else None
                    ),
                    "max_acquisition_loss": gate_max_acquisition_loss,
                    "reason": (
                        "LLM proposal is acquisition-compatible."
                        if authorized
                        else "LLM proposal exceeded the frozen acquisition-risk bound."
                    ),
                },
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "reflection": dict(policy),
            }
        )
    for round_index in range(len(followups), rounds):
        observed_y, _center, _scale = classical.normalized_outcomes(
            [pool[index] for index in observed]
        )
        posterior_mean, posterior_variance, _diagnostics = (
            classical.target_gp_posterior(
                features,
                observed,
                observed_y,
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
        scores = posterior_mean + float(kernel["gp_beta"]) * np.sqrt(
            posterior_variance
        )
        selected_index = classical.top_unobserved(scores, observed_set)
        selected = pool[selected_index]
        observed.append(selected_index)
        observed_set.add(selected_index)
        best_so_far = max(pool[index].objective_value for index in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "event": "target_gp_reveal",
                "target_dataset": adapter.dataset_id,
                "mode": LLM_REFLECTIVE_MODE,
                "seed": -1,
                "round_index": round_index,
                "selected_candidate": selected.candidate_id,
                "public_conditions": warmstart.public_conditions(selected),
                "selected_score": round(float(scores[selected_index]), 6),
                "selected_by": "target_only_gp_ucb_after_llm_reflection",
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
            }
        )
    oracle = max(candidate.objective_value for candidate in pool)
    final_best = max(pool[index].objective_value for index in observed)
    top10 = {
        index
        for index, _candidate in sorted(
            enumerate(pool),
            key=lambda item: item[1].objective_value,
            reverse=True,
        )[: min(10, len(pool))]
    }
    return {
        "target_dataset": adapter.dataset_id,
        "mode": LLM_REFLECTIVE_MODE,
        "seed": -1,
        "initial_observations": len(initial_indices),
        "reveal_rounds": rounds,
        "initial_candidate_ids": ";".join(
            pool[index].candidate_id for index in initial_indices
        ),
        "source_datasets": ";".join(source_ids),
        "final_best": round(final_best, 6),
        "best_so_far_auc": round(float(np.mean(best_trace)), 6),
        "simple_regret": round(oracle - final_best, 6),
        "top10_hit": int(bool(set(observed) & top10)),
    }, audit


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "run_count": len(selected),
            **{
                field: round(
                    float(np.mean([float(row[field]) for row in selected])),
                    6,
                )
                for field in (
                    "final_best",
                    "best_so_far_auc",
                    "simple_regret",
                    "top10_hit",
                )
            },
        }
    return summary


def evaluate(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    record = json.loads(args.hypothesis_record.read_text(encoding="utf-8"))
    if record["config_fingerprint"] != warmstart.config_fingerprint(config):
        raise ValueError("Hypothesis record does not match the evaluation config.")
    expected_hash_path = args.hypothesis_record.with_suffix(
        args.hypothesis_record.suffix + ".sha256"
    )
    if expected_hash_path.exists():
        expected = expected_hash_path.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(args.hypothesis_record.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("Frozen LLM hypothesis fingerprint does not match.")
    target_id = str(record["target_task"])
    target = replay.DATASET_BUILDERS[target_id]()
    source_ids = [str(item) for item in record["source_tasks"]]
    hypothesis = dict(record["frozen_hypothesis"])
    if hypothesis["abstain"]:
        raise ValueError("The frozen LLM hypothesis abstained; no transfer run is allowed.")
    initial_count = int(protocol["initial_observations"])
    rounds = int(protocol["reveal_rounds"])
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    llm_indices = indices_for_ids(target, hypothesis["selected_candidate_ids"])
    llm_metrics, llm_audit = warmstart.run_target_gp(
        target,
        llm_indices,
        rounds,
        LLM_MODE,
        -1,
        protocol["kernel"],
        source_ids,
        {
            "hypothesis_sha256": hashlib.sha256(
                args.hypothesis_record.read_bytes()
            ).hexdigest(),
            "model": record["model"],
            "hypothesis": hypothesis["hypothesis"],
            "selected_source_view": hypothesis["selected_source_view"],
        },
    )
    rows.append(llm_metrics)
    events.extend(llm_audit)

    reflection_record: dict[str, Any] | None = None
    reflection_sha256 = ""
    if args.reflection_record is not None:
        reflection_record = json.loads(
            args.reflection_record.read_text(encoding="utf-8")
        )
        if (
            reflection_record.get("schema_version")
            != "care.llm_hypothesis_reflection/v1"
        ):
            raise ValueError("Unsupported LLM reflection record schema.")
        reflection_sha256 = hashlib.sha256(
            args.reflection_record.read_bytes()
        ).hexdigest()
        expected_reflection_hash_path = args.reflection_record.with_suffix(
            args.reflection_record.suffix + ".sha256"
        )
        if expected_reflection_hash_path.exists():
            expected = expected_reflection_hash_path.read_text(
                encoding="utf-8"
            ).split()[0]
            if expected != reflection_sha256:
                raise ValueError("Frozen LLM reflection fingerprint does not match.")
        if reflection_record["config_fingerprint"] != warmstart.config_fingerprint(
            config
        ):
            raise ValueError("Reflection record does not match the evaluation config.")
        if str(reflection_record["target_task"]) != target_id:
            raise ValueError("Reflection target does not match the hypothesis target.")
        parent_sha256 = hashlib.sha256(
            args.hypothesis_record.read_bytes()
        ).hexdigest()
        if reflection_record["parent_hypothesis_sha256"] != parent_sha256:
            raise ValueError("Reflection record does not match the frozen hypothesis.")
        reflection = dict(reflection_record["frozen_reflection"])
        reflective_initial_ids = [
            str(item)
            for item in reflection_record.get("initial_design", {}).get(
                "candidate_ids", hypothesis["selected_candidate_ids"]
            )
        ]
        reflective_initial_indices = indices_for_ids(
            target, reflective_initial_ids
        )
        followup_indices = indices_for_ids(
            target, reflection["selected_candidate_ids"]
        )
        reflective_metrics, reflective_audit = run_reflective_target_gp(
            target,
            reflective_initial_indices,
            followup_indices,
            rounds,
            protocol["kernel"],
            source_ids,
            {
                "hypothesis_sha256": parent_sha256,
                "reflection_sha256": reflection_sha256,
                "initial_model": record["model"],
                "reflection_model": reflection_record["model"],
                "initial_design_mode": reflection_record.get(
                    "initial_design", {}
                ).get("mode", LLM_MODE),
                "hypothesis_status": reflection["hypothesis_status"],
                "revised_hypothesis": reflection["revised_hypothesis"],
                "confidence": reflection["confidence"],
                "stop_transfer": reflection["stop_transfer"],
            },
            args.reflection_gate_max_acquisition_loss,
        )
        rows.append(reflective_metrics)
        events.extend(reflective_audit)

    compiled_indices, compiled_sources, compiler_record = compiled_llm_initial(
        target,
        source_ids,
        protocol,
        hypothesis,
    )
    compiled_metrics, compiled_audit = warmstart.run_target_gp(
        target,
        compiled_indices,
        rounds,
        LLM_COMPILED_MODE,
        -1,
        protocol["kernel"],
        compiled_sources,
        {
            **compiler_record,
            "hypothesis_sha256": hashlib.sha256(
                args.hypothesis_record.read_bytes()
            ).hexdigest(),
            "model": record["model"],
        },
    )
    rows.append(compiled_metrics)
    events.extend(compiled_audit)

    fixed_indices, fixed_sources, fixed_policy = fixed_v2_initial(
        target,
        source_ids,
        protocol,
    )
    fixed_metrics, fixed_audit = warmstart.run_target_gp(
        target,
        fixed_indices,
        rounds,
        FIXED_V2_MODE,
        -1,
        protocol["kernel"],
        fixed_sources,
        fixed_policy,
    )
    rows.append(fixed_metrics)
    events.extend(fixed_audit)

    space_metrics, space_audit = warmstart.run_target_gp(
        target,
        warmstart.space_filling_initial(target, initial_count),
        rounds,
        warmstart.SPACE_FILLING_MODE,
        -1,
        protocol["kernel"],
    )
    rows.append(space_metrics)
    events.extend(space_audit)

    shortlist_ids = [
        str(item["candidate_id"])
        for item in record["prompt"]["scientific_context"]["candidate_shortlist"]
    ]
    shortlist_indices = indices_for_ids(target, shortlist_ids)
    for seed in range(args.random_seed_start, args.random_seed_start + args.random_seed_count):
        random_ids = list(shortlist_indices)
        np.random.default_rng(seed).shuffle(random_ids)
        metrics, audit = warmstart.run_target_gp(
            target,
            random_ids[:initial_count],
            rounds,
            SHORTLIST_RANDOM_MODE,
            seed,
            protocol["kernel"],
        )
        rows.append(metrics)
        events.extend(audit)
        metrics, audit = warmstart.run_target_gp(
            target,
            warmstart.random_initial(target, initial_count, seed),
            rounds,
            warmstart.RANDOM_MODE,
            seed,
            protocol["kernel"],
        )
        rows.append(metrics)
        events.extend(audit)

    summary_by_mode = summarize_rows(rows)
    comparisons = {}
    challenger_modes = [LLM_MODE, LLM_COMPILED_MODE]
    if reflection_record is not None:
        challenger_modes.append(LLM_REFLECTIVE_MODE)
    for challenger_mode in challenger_modes:
        challenger = summary_by_mode[challenger_mode]
        comparisons[challenger_mode] = {}
        for mode, reference in summary_by_mode.items():
            if mode in challenger_modes:
                continue
            comparisons[challenger_mode][mode] = {
                "final_best_delta": round(
                    challenger["final_best"] - reference["final_best"], 6
                ),
                "best_so_far_auc_delta": round(
                    challenger["best_so_far_auc"]
                    - reference["best_so_far_auc"],
                    6,
                ),
            }
    output = {
        "experiment": (
            "care2_llm_hypothesis_reflective_design"
            if reflection_record is not None
            else "care2_llm_hypothesis_only_initial_design"
        ),
        "evidence_class": "retrospective_component_ablation",
        "target_outcomes_available_to_initial_llm": False,
        "target_outcomes_available_to_reflection_llm": (
            "initial_observations_only" if reflection_record is not None else False
        ),
        "target_task": target_id,
        "source_tasks": source_ids,
        "frozen_hypothesis_record": str(args.hypothesis_record),
        "frozen_hypothesis_sha256": hashlib.sha256(
            args.hypothesis_record.read_bytes()
        ).hexdigest(),
        "llm_model": record["model"],
        "frozen_hypothesis": hypothesis,
        "frozen_reflection_record": (
            str(args.reflection_record) if reflection_record is not None else None
        ),
        "frozen_reflection_sha256": reflection_sha256 or None,
        "frozen_reflection": (
            reflection_record["frozen_reflection"]
            if reflection_record is not None
            else None
        ),
        "reflection_initial_design": (
            reflection_record.get("initial_design")
            if reflection_record is not None
            else None
        ),
        "reflection_gate_max_acquisition_loss": (
            args.reflection_gate_max_acquisition_loss
            if reflection_record is not None
            else None
        ),
        "compiler_record": compiler_record,
        "summary_by_mode": summary_by_mode,
        "challenger_deltas": comparisons,
        "interpretation": (
            "This run measures the incremental value of a frozen LLM initial-design "
            "decision and, when supplied, one evidence-bounded LLM reflection step. "
            "The target has been used previously by the project, so this is a component "
            "ablation rather than fresh external confirmation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "metrics.csv", rows)
    warmstart.write_audits(args.output_dir / "audits.jsonl", events)
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--target-task", required=True)
    generate_parser.add_argument("--source-tasks", nargs="+", required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--per-view-limit", type=int, default=20)
    generate_parser.add_argument(
        "--llm-base-url", default="https://api.commonstack.ai/v1"
    )
    generate_parser.add_argument(
        "--llm-model", default="openai/gpt-5.4-2026-03-05"
    )
    generate_parser.add_argument("--llm-api-key-env", default="COMMONSTACK_API_KEY")
    generate_parser.add_argument("--llm-temperature", type=float, default=0.0)
    generate_parser.add_argument("--llm-max-tokens", type=int, default=3500)

    reflect_parser = subparsers.add_parser("reflect")
    reflect_parser.add_argument("--config", type=Path, required=True)
    reflect_parser.add_argument("--hypothesis-record", type=Path, required=True)
    reflect_parser.add_argument("--output", type=Path, required=True)
    reflect_parser.add_argument("--followup-count", type=int, default=2)
    reflect_parser.add_argument(
        "--initial-design-mode",
        choices=("raw", "compiled"),
        default="raw",
    )
    reflect_parser.add_argument(
        "--llm-base-url", default="https://api.commonstack.ai/v1"
    )
    reflect_parser.add_argument(
        "--llm-model", default="openai/gpt-5.4-2026-03-05"
    )
    reflect_parser.add_argument("--llm-api-key-env", default="COMMONSTACK_API_KEY")
    reflect_parser.add_argument("--llm-temperature", type=float, default=0.0)
    reflect_parser.add_argument("--llm-max-tokens", type=int, default=3000)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--hypothesis-record", type=Path, required=True)
    evaluate_parser.add_argument("--reflection-record", type=Path)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--random-seed-start", type=int, default=96000)
    evaluate_parser.add_argument("--random-seed-count", type=int, default=100)
    evaluate_parser.add_argument(
        "--reflection-gate-max-acquisition-loss",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "reflect":
        reflect(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
