#!/usr/bin/env python3
"""Compile the frozen Baumgartner initial-design transfer skill."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import run_synthetic_suzuki as replay
from skill_bank import ReusableSkill, SkillEvidence, compile_skill_bank


ROOT = Path(__file__).resolve().parents[1]


def task_digest(adapter: replay.DatasetAdapter) -> str:
    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "public_conditions": {
                "base": candidate.metadata["base"],
                "numeric_features": candidate.numeric_features,
            },
            "outcome": candidate.objective_value,
        }
        for candidate in adapter.candidates
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_bank(
    config: dict[str, Any],
    selection: dict[str, Any],
    confirmation: dict[str, Any] | None = None,
) -> Any:
    protocol = config["protocol"]
    calibration_tasks = tuple(protocol["development_task_ids"])
    external_source_tasks = tuple(
        source_id
        for case in protocol.get("evaluation_cases", [])
        for source_id in case["source_task_ids"]
    )
    development = tuple(dict.fromkeys((*calibration_tasks, *external_source_tasks)))
    evaluation = tuple(protocol["evaluation_task_ids"])
    selected_route = dict(selection["selected_route"])
    diversity_weight = float(selected_route.get("diversity_weight", 0.0))
    source_quantile = float(selected_route.get("source_quantile", 0.0))
    selected_mode = str(selection["selected_mode"])
    comparison = selection["candidate_comparisons"][selected_mode][
        protocol["selection_metric"]
    ]
    adapters = [replay.DATASET_BUILDERS[task_id]() for task_id in calibration_tasks]
    observation_count = sum(len(adapter.candidates) for adapter in adapters)
    task_hashes = tuple(task_digest(adapter) for adapter in adapters)
    evidence = (
        SkillEvidence(
            evidence_id="baumgartner_campaign_space_contract",
            source_tasks=calibration_tasks,
            task_family="cn_reaction_optimization",
            status="validated",
            lesson=(
                "The 13 Baumgartner campaigns expose the same base, equivalents, "
                "temperature, residence-time, and precatalyst-loading decision space."
            ),
            applicability=("declared compatible C-N mixed-variable campaign",),
            failure_modes=("changed units, bounds, or base vocabulary",),
            trace_references=task_hashes,
            observation_count=observation_count,
            provenance={
                "source_commit": "99cd89b378cd8d7a624f22ae138fb6f0a85ef440"
            },
        ),
        SkillEvidence(
            evidence_id="baumgartner_diverse_warmstart_development_v1",
            source_tasks=calibration_tasks,
            task_family="cn_reaction_optimization",
            status="validated",
            lesson=(
                "Task-disjoint development selected a source-rank plus diversity "
                "initial design with an explicit source-quality floor before an "
                "unchanged target-only GP-UCB continuation."
            ),
            applicability=(
                "three initial experiments followed by target-only sequential optimization",
            ),
            failure_modes=(
                "selecting the route from evaluation outcomes",
                "claiming the warm-start result as continuous acquisition transfer",
            ),
            trace_references=(
                selection["config_fingerprint"],
                hashlib.sha256(
                    json.dumps(
                        selection["candidate_comparisons"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            observation_count=observation_count,
            provenance={
                "selection_scope": selection["selection_scope"],
                "selected_route": selected_route,
                "primary_metric": protocol["selection_metric"],
                "development_task_mean_delta": comparison["task_mean_delta"],
                "development_task_95ci": [
                    comparison["task_95ci_low"],
                    comparison["task_95ci_high"],
                ],
                "development_task_nonloss_rate": comparison[
                    "task_nonloss_rate"
                ],
                "evaluation_task_ids_not_loaded": selection[
                    "evaluation_task_ids_not_loaded"
                ],
            },
        ),
    )
    evidence_items = list(evidence)
    if confirmation is not None:
        comparison = confirmation["comparisons"]["development_selected_route"]
        primary = comparison["best_so_far_auc"]
        external_adapters = [
            replay.DATASET_BUILDERS[task_id]() for task_id in external_source_tasks
        ]
        evidence_items.append(
            SkillEvidence(
                evidence_id="baumgartner_suzuki_external_confirmation_v2",
                source_tasks=external_source_tasks,
                task_family="mixed_variable_reaction_optimization",
                status="candidate",
                lesson=(
                    "The frozen source-quality-bounded initial design improved "
                    f"best-so-far AUC by {primary['task_mean_delta']:+.6f} on the "
                    "external Baumgartner Suzuki MINLP2 campaign."
                ),
                applicability=(
                    "source and target campaigns share declared mixed-variable semantics",
                ),
                failure_modes=(
                    "one external target does not establish task-level generalization",
                    "using the target outcome to retune the frozen source quantile",
                ),
                trace_references=(
                    hashlib.sha256(
                        json.dumps(
                            confirmation,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                ),
                observation_count=sum(
                    len(adapter.candidates) for adapter in external_adapters
                ),
                provenance={
                    "external_target_task_ids": list(evaluation),
                    "best_so_far_auc_delta": primary["task_mean_delta"],
                    "final_best_delta": comparison["final_best"]["task_mean_delta"],
                    "route_frozen_before_external_execution": True,
                },
            )
        )
    skill = ReusableSkill(
        skill_id="source_guided_diverse_initial_design",
        title="Source-guided diverse initial design",
        task_families=(
            "cn_reaction_optimization",
            "mixed_variable_reaction_optimization",
        ),
        instructions=(
            "Validate the target variables, units, bounds, and categorical semantics against the source campaign contract.",
            "Choose completed sources with the same substrate; if none exist, use sources with the same precatalyst; otherwise use all compatible sources.",
            "Fit one GP expert per source and aggregate candidate predictions through median normalized ranks.",
            (
                "Select the first experiment by source rank, then select two more "
                f"from the top {100.0 * (1.0 - source_quantile):.0f}% source-ranked "
                f"region with {diversity_weight:.2f} mixed-space diversity weight."
            ),
            "After the three initial outcomes are revealed, discard the source prior and run the frozen target-only GP-UCB for subsequent proposals.",
            "Compare deployment against both random initial design and pure space filling; retain all negative-transfer evidence.",
        ),
        abstain_when=(
            "source and target decision spaces lack a declared mapping",
            "target constraints make any proposed condition invalid",
            "task-disjoint development evidence fails the frozen confidence and non-loss gate",
        ),
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
    )
    return compile_skill_bank(
        bank_id="care2-baumgartner-initial-design",
        development_task_ids=development,
        evaluation_task_ids=evaluation,
        evidence=evidence_items,
        skills=(skill,),
        provenance={
            "builder": "build_baumgartner_warmstart_skill_bank.py",
            "source_repository": "https://github.com/sustainable-processes/multitask",
            "protocol": protocol["version"],
            "target_outcomes_used_for_skill_selection": False,
            "external_confirmation_included": confirmation is not None,
            "claim_boundary": "initial-design transfer only",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "baumgartner_multisource_warmstart_v1.json",
    )
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--confirmation-summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "skill_banks" / "care2-baumgartner-initial-design",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
    confirmation = (
        json.loads(args.confirmation_summary.read_text(encoding="utf-8"))
        if args.confirmation_summary
        else None
    )
    if selection["config_fingerprint"] != hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise ValueError("Selection record does not match the frozen config.")
    bank = build_bank(config, selection, confirmation)
    bank.write(args.output_dir)
    print(json.dumps(bank.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
