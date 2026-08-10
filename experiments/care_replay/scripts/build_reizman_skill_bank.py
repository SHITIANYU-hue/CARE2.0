#!/usr/bin/env python3
"""Build the first task-disjoint CARE 2.0 wetlab SKILL.md artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import run_synthetic_suzuki as replay
from skill_bank import ReusableSkill, SkillEvidence, compile_skill_bank


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TASKS = tuple(f"real_reizman_suzuki_case_{index}" for index in (1, 2, 3))
TARGET_TASKS = ("real_reizman_suzuki_case_4",)


def task_digest(adapter: replay.DatasetAdapter) -> str:
    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "conditions": {
                "group": candidate.group,
                "numeric_features": candidate.numeric_features,
            },
            "outcome": candidate.objective_value,
        }
        for candidate in adapter.candidates
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_summary(adapter: replay.DatasetAdapter) -> dict[str, Any]:
    values = [candidate.objective_value for candidate in adapter.candidates]
    return {
        "dataset_id": adapter.dataset_id,
        "candidate_count": len(adapter.candidates),
        "catalyst_count": len({candidate.group for candidate in adapter.candidates}),
        "yield_min": min(values),
        "yield_max": max(values),
        "task_sha256": task_digest(adapter),
    }


def build_bank(selection_record: dict[str, Any] | None = None) -> Any:
    adapters = [replay.DATASET_BUILDERS[dataset_id]() for dataset_id in SOURCE_TASKS]
    summaries = [source_summary(adapter) for adapter in adapters]
    evidence_items = [
        SkillEvidence(
            evidence_id="reizman_mixed_space_contract",
            source_tasks=SOURCE_TASKS,
            task_family="suzuki_reaction_optimization",
            status="validated",
            lesson=(
                "Represent residence time, temperature, and catalyst loading as "
                "continuous dimensions while keeping catalyst identity categorical."
            ),
            applicability=("same declared mixed search space",),
            failure_modes=("incompatible bounds or catalyst vocabulary",),
            trace_references=tuple(item["task_sha256"] for item in summaries),
            observation_count=sum(item["candidate_count"] for item in summaries),
            provenance={"source_commit": "99cd89b378cd8d7a624f22ae138fb6f0a85ef440"},
        ),
        SkillEvidence(
            evidence_id="separate_source_experts",
            source_tasks=SOURCE_TASKS,
            task_family="suzuki_reaction_optimization",
            status="candidate",
            lesson=(
                "Keep each completed substrate task as a separate predictive expert; "
                "combine experts only through online rank evidence or model evidence."
            ),
            applicability=("multiple completed source tasks",),
            failure_modes=(
                "pooling tasks can hide opposing substrate-specific effects",
                "few target observations make expert weights uncertain",
            ),
            trace_references=tuple(item["task_sha256"] for item in summaries),
            observation_count=sum(item["candidate_count"] for item in summaries),
            provenance={"source_summaries": summaries},
        ),
        SkillEvidence(
            evidence_id="exact_target_fallback",
            source_tasks=SOURCE_TASKS,
            task_family="sequential_optimization",
            status="validated",
            lesson=(
                "Run a matched target-only acquisition policy beside transfer and "
                "retain an exact fallback when source evidence is contradictory."
            ),
            applicability=("all sequential target tasks",),
            failure_modes=("claiming fallback performance as source-transfer gain",),
            trace_references=("care2_classical_transfer_confirmation_v1",),
            observation_count=sum(item["candidate_count"] for item in summaries),
            provenance={
                "review_followup": "2026-08-08 classical transfer confirmation"
            },
        ),
    ]
    if selection_record is not None:
        selected_route = dict(selection_record["selected_route"])
        comparison = selection_record["candidate_comparisons"]
        evidence_items.append(
            SkillEvidence(
                evidence_id="reizman_development_transfer_gate_v1",
                source_tasks=SOURCE_TASKS,
                task_family="suzuki_reaction_optimization",
                status="validated",
                lesson=(
                    "On task-disjoint development folds, admit a source-informed "
                    "route only when its paired best-so-far AUC confidence-interval "
                    "lower bound is above zero; otherwise use the exact target-only "
                    f"fallback. The frozen v1 route was {selected_route['mode']}."
                ),
                applicability=("new tasks sharing the declared Reizman search space",),
                failure_modes=(
                    "choosing a source route from evaluation-task outcomes",
                    "forcing transfer when every development confidence interval crosses or falls below zero",
                ),
                trace_references=(
                    selection_record["config_fingerprint"],
                    hashlib.sha256(
                        json.dumps(
                            comparison,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                ),
                observation_count=int(
                    comparison["target_gp_ucb"]["fold_seed_count"]
                ),
                provenance={
                    "selection_scope": selection_record["selection_scope"],
                    "selection_metric": selection_record["selection_metric"],
                    "selection_required_ci_low": selection_record[
                        "selection_required_ci_low"
                    ],
                    "selected_route": selected_route,
                    "evaluation_task_ids_not_loaded": selection_record[
                        "evaluation_task_ids_not_loaded"
                    ],
                },
            )
        )
    evidence = tuple(evidence_items)
    instructions = [
        "Validate that every source and the target share named variables, units, bounds, and categorical semantics.",
        "Normalize continuous variables from the declared design bounds; never infer scaling from hidden target outcomes.",
        "Fit one source expert per completed task instead of concatenating all source rows into one anonymous table.",
        "Update source-versus-target expert weights only from target observations already returned by the wet lab.",
        "Propose one unrevealed experiment and wait for tell(outcome) before the next proposal.",
    ]
    if selection_record is not None:
        instructions.append(
            "Deploy a source-informed route only when task-disjoint development evidence clears the frozen AUC confidence gate; otherwise execute the target-only fallback exactly."
        )
    skills = (
        ReusableSkill(
            skill_id="mixed_variable_multisource_transfer",
            title="Mixed-variable multi-source transfer",
            task_families=("suzuki_reaction_optimization",),
            instructions=tuple(instructions),
            abstain_when=(
                "source and target variables do not have a declared mapping",
                "all source experts receive negligible online weight",
                "the proposed condition violates target bounds or laboratory constraints",
            ),
            evidence_ids=(
                "reizman_mixed_space_contract",
                "separate_source_experts",
                "exact_target_fallback",
                *(
                    ("reizman_development_transfer_gate_v1",)
                    if selection_record is not None
                    else ()
                ),
            ),
        ),
    )
    return compile_skill_bank(
        bank_id="care2-wetlab-transfer",
        development_task_ids=SOURCE_TASKS,
        evaluation_task_ids=TARGET_TASKS,
        evidence=evidence,
        skills=skills,
        provenance={
            "builder": "build_reizman_skill_bank.py",
            "source_repository": "https://github.com/sustainable-processes/multitask",
            "method_references": [
                "https://doi.org/10.1021/acscentsci.3c00050",
                "https://arxiv.org/abs/2603.25158",
                "https://arxiv.org/abs/2602.08234",
            ],
            "target_outcomes_used_for_skill_building": False,
            "development_selection_included": selection_record is not None,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "skill_banks" / "care2-wetlab-transfer",
    )
    parser.add_argument("--selection-record", type=Path)
    args = parser.parse_args()
    selection_record = (
        json.loads(args.selection_record.read_text(encoding="utf-8"))
        if args.selection_record
        else None
    )
    bank = build_bank(selection_record)
    bank.write(args.output_dir)
    print(json.dumps(bank.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
