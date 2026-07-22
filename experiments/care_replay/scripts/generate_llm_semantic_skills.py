#!/usr/bin/env python3
"""Compile source evidence and scientific schema into executable LLM skills."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import llm_semantic_skills as semantic
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


EVIDENCE_MODES = ("full", "source_schema_only", "target_only")


def transfer_card_payload(
    source_dataset: str,
    target_dataset: str,
    source_observations: int,
    discount: float,
) -> dict[str, Any]:
    source = replay.DATASET_BUILDERS[source_dataset]()
    target = replay.DATASET_BUILDERS[target_dataset]()
    observed = transfer.source_observations(source, 0, source_observations)
    role_map = transfer.descriptor_transfer_role_map_for(source_dataset, target_dataset)
    card = transfer.compile_transfer_card(
        source,
        target,
        observed,
        role_map,
        discount,
        3,
    )
    return {
        "source_dataset": source_dataset,
        "source_objective": source.objective,
        "source_observation_count": len(observed),
        "roles": [asdict(role) for role in card.roles],
        "shared_value_priors": [asdict(prior) for prior in card.value_priors],
        "boundary": (
            "Only source outcomes and public source/target descriptors are summarized. "
            "No target outcomes are present."
        ),
    }


def source_evidence_payload(
    source_dataset: str,
    target_dataset: str,
    source_observations: int,
    discount: float,
    evidence_mode: str,
) -> dict[str, Any]:
    if evidence_mode == "full":
        payload = transfer_card_payload(
            source_dataset,
            target_dataset,
            source_observations,
            discount,
        )
        payload["evidence_mode"] = evidence_mode
        return payload
    if evidence_mode == "source_schema_only":
        source = replay.DATASET_BUILDERS[source_dataset]()
        role_map = transfer.descriptor_transfer_role_map_for(source_dataset, target_dataset)
        return {
            "evidence_mode": evidence_mode,
            "source_dataset": source_dataset,
            "source_objective": source.objective,
            "mapped_roles": [
                {"source_field": source_field, "target_field": target_field}
                for source_field, target_field in sorted(role_map.items())
            ],
            "source_outcome_statistics": None,
            "shared_value_priors": [],
            "boundary": (
                "Only source identity, objective, and public source-target field alignment are "
                "provided. Source outcomes, effects, support counts, and value priors are withheld."
            ),
        }
    if evidence_mode == "target_only":
        return {
            "evidence_mode": evidence_mode,
            "source_dataset": None,
            "source_objective": None,
            "mapped_roles": [],
            "source_outcome_statistics": None,
            "shared_value_priors": [],
            "boundary": (
                "No source identity, source schema, source outcomes, source effects, or source value "
                "priors are provided. The LLM must use the public target schema and its pretrained "
                "domain knowledge only."
            ),
        }
    raise ValueError(f"Unsupported evidence mode: {evidence_mode}")


def build_prompt_payload(
    source_dataset: str,
    target_dataset: str,
    source_observations: int,
    discount: float,
    skill_count: int,
    evidence_mode: str = "full",
) -> dict[str, Any]:
    target = replay.DATASET_BUILDERS[target_dataset]()
    catalog = semantic.semantic_field_catalog(target)
    evidence_requirement = {
        "full": "Include at least one skill explicitly grounded in the supplied source outcome evidence.",
        "source_schema_only": (
            "Include at least one skill grounded only in the supplied source-target field alignment; "
            "do not imply that source outcome effects were observed."
        ),
        "target_only": (
            "Use only target-schema-compatible domain knowledge; do not imply that a source task or "
            "source outcomes were observed."
        ),
    }[evidence_mode]
    return {
        "task": (
            "Design a diverse library of executable semantic optimization skills for finite-pool "
            "scientific search. Each skill defines interpretable rule features plus a conservative "
            "Bayesian linear surrogate blended with strong target-only GP-UCB and GP-EI anchors."
        ),
        "target": {
            "dataset": target.dataset_id,
            "title": target.title,
            "objective": target.objective,
            "candidate_count": len(target.candidates),
            "public_semantic_fields": catalog,
            "hidden_target": target.hidden_target,
            "boundary": "Field values and counts are public; target outcomes are not included.",
        },
        "source_transfer_evidence": source_evidence_payload(
            source_dataset,
            target_dataset,
            source_observations,
            discount,
            evidence_mode,
        ),
        "executor": {
            "semantic_model": (
                "Rules become binary features. Revealed target outcomes fit a ridge Bayesian linear "
                "surrogate whose coefficient prior follows rule weights."
            ),
            "anchor": "A rank blend of target-only GP-UCB and GP-EI.",
            "fusion": (
                "semantic_mass_start/end controls the rank mass assigned to the semantic surrogate; "
                "the remaining mass stays on the target-only anchor."
            ),
            "selection": (
                "Independent calibration seeds select one frozen skill. Independent held-out seeds "
                "are evaluated only after selection."
            ),
        },
        "design_requirements": [
            f"Return exactly {skill_count} diverse skills.",
            "Use only exact field/value pairs in public_semantic_fields.",
            (
                "Each conditions object must map a real catalog field name directly to one exact "
                "catalog value, for example {\"dominant_family\": \"oxide\"}. Never use literal "
                "placeholder keys such as exact_field, field_name, field, or value."
            ),
            (
                "Give every skill a distinct, descriptive scientific skill_id. Never copy schema "
                "placeholders such as short_unique_name or add numeric suffixes to a placeholder."
            ),
            "Give every skill 3-10 rules; use two-condition interactions only when scientifically meaningful.",
            "Include at least one conservative low-semantic-mass skill.",
            evidence_requirement,
            "Include at least one domain-knowledge skill.",
            "Include a counter-hypothesis skill that reverses or downweights an uncertain source relation.",
            "Prefer mechanistic, chemically or physically interpretable rules over arbitrary coverage rules.",
            "Use positive weights for conditions expected to improve the target objective and negative weights for risks.",
            "Do not claim access to target outcomes and do not output candidate identifiers.",
            *(
                [
                    "For ChemLex, prioritize reagent_coupling_family and acid/amine motif fields over coarse length bins.",
                    "For ChemLex, include chemically plausible two-condition acid/amine or substrate/reagent interactions.",
                ]
                if target_dataset == "real_chemlex_acidamine"
                else []
            ),
        ],
        "allowed_ranges": {
            "rule_weight": "[-1, 1]",
            "ridge": "[0.05, 20]",
            "prior_scale": "[0, 1.5]",
            "semantic_mass_start_end": "[0, 0.85]",
            "ucb_weight": "[0, 1]; EI weight is 1 - ucb_weight",
            "gp_beta_start_end": "[0.2, 4]",
            "gp_xi": "[0, 0.2]",
            "confidence": "[0.05, 1]",
        },
        "output_contract": {
            "skills": [{
                "skill_id": "descriptive_scientific_skill_id",
                "rules": [{
                    "rule_id": "mechanistic_rule",
                    "conditions": {"field_name_from_public_semantic_fields": "exact_catalog_value"},
                    "weight": 0.5,
                    "rationale": "short scientific rationale",
                }],
                "ridge": 1.0,
                "prior_scale": 0.25,
                "semantic_mass_start": 0.30,
                "semantic_mass_end": 0.10,
                "ucb_weight": 0.60,
                "gp_beta_start": 2.0,
                "gp_beta_end": 1.0,
                "gp_xi": 0.01,
                "confidence": 0.70,
                "hypothesis": "what this skill transfers and when it may fail",
            }]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate executable semantic skills with a real LLM call.")
    parser.add_argument("--source-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-observations", type=int, default=512)
    parser.add_argument("--discount", type=float, default=0.65)
    parser.add_argument("--skill-count", type=int, default=8)
    parser.add_argument("--evidence-mode", choices=EVIDENCE_MODES, default="full")
    parser.add_argument("--llm-base-url", default="https://api.commonstack.ai/v1")
    parser.add_argument("--llm-model", default="openai/gpt-4o-mini")
    parser.add_argument("--llm-api-key-env", default="COMMONSTACK_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.45)
    parser.add_argument("--llm-max-tokens", type=int, default=5000)
    args = parser.parse_args()
    api_key = os.environ.get(args.llm_api_key_env) or os.environ.get("CARE_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(f"Set {args.llm_api_key_env} or CARE_LLM_API_KEY.")
    payload = build_prompt_payload(
        args.source_dataset,
        args.target_dataset,
        args.source_observations,
        args.discount,
        args.skill_count,
        args.evidence_mode,
    )
    config = replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        structured_mode="json",
    )
    content, metadata = replay.chat_completion_text(
        config,
        [
            {
                "role": "system",
                "content": (
                    "You are a scientific optimization expert. Return one valid JSON object only. "
                    "Never invent field names or values outside the supplied catalog."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    parsed = replay.extract_json_object(content)
    target = replay.DATASET_BUILDERS[args.target_dataset]()
    catalog = semantic.semantic_field_catalog(target)
    skills = semantic.normalize_skills(parsed, catalog, max_skills=args.skill_count)
    record = {
        "model": metadata.get("model", args.llm_model),
        "usage": metadata.get("usage", {}),
        "llm_generation_call_count": 1,
        "api_configuration": {
            "base_url": args.llm_base_url,
            "api_mode": "chat",
            "structured_mode": "json",
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
        },
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "evidence_mode": args.evidence_mode,
        "prompt_payload": payload,
        "raw_response": content,
        "parsed_response": parsed,
        "normalized_skills": [asdict(skill) for skill in skills],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not skills:
        raise RuntimeError(
            f"The LLM response contained no executable semantic skills; raw response saved to {args.output}."
        )
    print(json.dumps({
        "output": str(args.output),
        "model": record["model"],
        "usage": record["usage"],
        "skill_ids": [skill.skill_id for skill in skills],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
