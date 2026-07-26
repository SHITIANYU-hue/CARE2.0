#!/usr/bin/env python3
"""Turn frozen LLM hypotheses and zero-shot results into auditable KB cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]


def make_card(
    card_id: str,
    card_type: str,
    title: str,
    summary: str,
    content: str,
    tags: list[str],
    source_ids: list[str],
    related_ids: list[str],
    status: str,
    confidence: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "id": card_id,
        "type": card_type,
        "title": title,
        "summary": summary,
        "content": content,
        "tags": sorted(set(tags)),
        "source_ids": source_ids,
        "related_ids": related_ids,
        "status": status,
        "priority": "P1",
        "confidence": confidence,
        "evidence_boundary": "public",
        "updated_at": updated_at,
    }


def result_lookup(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (run["source_dataset"], run["target_dataset"]): run
        for run in report.get("runs", [])
    }


def hypothesis_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parsed = record.get("parsed_response", {})
    raw = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []
    return {
        str(item.get("hypothesis_id")): item
        for item in raw
        if isinstance(item, dict) and item.get("hypothesis_id")
    }


def skill_result(run: dict[str, Any], skill_id: str) -> dict[str, Any] | None:
    for item in run.get("skills", []):
        if item.get("skill_id") == skill_id:
            return item
    return None


def metric_text(comparison: dict[str, Any]) -> str:
    final = comparison["final_best"]
    auc = comparison["best_so_far_auc"]
    return (
        f"final-best delta {final['mean']:+.4f} "
        f"(95% CI [{final['normal_95ci_low']:+.4f}, {final['normal_95ci_high']:+.4f}]); "
        f"AUC delta {auc['mean']:+.4f} "
        f"(95% CI [{auc['normal_95ci_low']:+.4f}, {auc['normal_95ci_high']:+.4f}])."
    )


def build_cards(
    record_paths: list[Path],
    report: dict[str, Any],
    updated_at: str,
) -> list[dict[str, Any]]:
    runs = result_lookup(report)
    cards: list[dict[str, Any]] = [make_card(
        "mechanism.hypothesis-only-compiler",
        "mechanism",
        "Hypothesis-only LLM compiler",
        "The LLM proposes a falsifiable mechanism claim; the executor fixes rule magnitude and acquisition parameters.",
        (
            "The LLM output is limited to claim, mechanism, public schema conditions, expected direction, "
            "failure conditions, and confidence. The compiler validates public field/value pairs and fixes "
            "the executable rule weight, ridge, prior scale, semantic mass, UCB/EI mixture, and GP schedule. "
            "This separates scientific proposal quality from LLM-chosen hyperparameters."
        ),
        ["LLM", "hypothesis", "compiler", "audit", "zero-shot"],
        [],
        ["mechanism.audit-log", "mechanism.embedding-retrieval"],
        "active",
        "high",
        updated_at,
    ), make_card(
        "mechanism.zero-shot-transfer-protocol",
        "mechanism",
        "Zero-shot transfer protocol",
        "Frozen source and LLM proposals are evaluated on target seeds without target calibration or pre-decision target outcomes.",
        (
            "The target replay consumes outcomes only after each policy selects a candidate. Every frozen "
            "hypothesis, target-only baseline, and matched random null is reported on the same seeds. "
            "This protocol is distinct from the historical target-calibration protocol and cannot be used "
            "to claim that a selected target route was zero-shot."
        ),
        ["transfer", "zero-shot", "target-calibration", "matched-null", "audit"],
        [],
        ["mechanism.hypothesis-only-compiler", "mechanism.audit-log"],
        "active",
        "high",
        updated_at,
    )]

    for record_path in record_paths:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source = str(record.get("source_dataset", "unknown"))
        target = str(record.get("target_dataset", "unknown"))
        pair = f"{source}-to-{target}"
        record_hash = hashlib.sha256(record_path.read_bytes()).hexdigest()
        run = runs.get((source, target))
        run_id = f"run-log.zero-shot-{slug(pair)}-{updated_at}"
        record_id = f"source.llm-hypotheses-{slug(pair)}-{updated_at}"
        dataset_tags = [source, target, "LLM", "hypothesis-only", "zero-shot"]
        cards.append(make_card(
            record_id,
            "source",
            f"Frozen LLM hypothesis record: {source} -> {target}",
            (
                f"A real {record.get('model', 'LLM')} call generated frozen hypotheses for "
                f"{source} -> {target}; no target outcomes were included in the prompt."
            ),
            (
                f"Record path: {record_path.as_posix()}. SHA-256: {record_hash}. "
                f"Evidence mode: {record.get('evidence_mode')}; proposal mode: {record.get('proposal_mode')}; "
                f"compiler diagnostics: {json.dumps(record.get('hypothesis_compilation', {}), ensure_ascii=False)}."
            ),
            dataset_tags + ["provenance"],
            ["mechanism.hypothesis-only-compiler"],
            [run_id],
            "active",
            "high",
            updated_at,
        ))
        if run:
            cards.append(make_card(
                run_id,
                "run_log",
                f"Zero-shot hypothesis replay: {source} -> {target}",
                (
                    f"{run['seed_count']} paired target seeds; {len(run['skills'])} frozen hypotheses; "
                    f"{run['random_null_count']} matched random-null policies."
                ),
                (
                    f"Result directory: {run['result_dir']}. Target calibration seeds: "
                    f"{run['target_calibration_seed_count']}; pre-decision target outcomes: "
                    f"{run['target_predecision_outcome_count']}; target budget per seed: "
                    f"{run['target_observation_budget_per_seed']}. Record SHA-256: {run.get('record_sha256')}."
                ),
                dataset_tags + ["replay", "trace", "matched-null"],
                [record_id, "mechanism.zero-shot-transfer-protocol"],
                [],
                "done",
                "high",
                updated_at,
            ))
        for hypothesis_id, hypothesis in hypothesis_map(record).items():
            result = skill_result(run, hypothesis_id) if run else None
            accepted = result is not None
            conditions = json.dumps(hypothesis.get("conditions", {}), ensure_ascii=False, sort_keys=True)
            failures = hypothesis.get("failure_conditions", [])
            if not isinstance(failures, list):
                failures = [str(failures)]
            card_id = f"hypothesis.{slug(pair)}-{slug(hypothesis_id)}-{updated_at}"
            result_id = f"experiment-result.{slug(pair)}-{slug(hypothesis_id)}-{updated_at}"
            cards.append(make_card(
                card_id,
                "hypothesis",
                f"{hypothesis_id}: {source} -> {target}",
                str(hypothesis.get("claim", "Frozen LLM mechanism hypothesis.")),
                (
                    f"Mechanism: {hypothesis.get('mechanism', '')} Conditions: {conditions}. "
                    f"Expected direction: {hypothesis.get('expected_direction', 'unknown')}. "
                    f"Failure conditions: {'; '.join(str(item) for item in failures)}. "
                    f"LLM confidence: {hypothesis.get('confidence', 'unknown')}. "
                    f"Executable after public-schema validation: {accepted}. "
                    "This card is a proposal, not established scientific knowledge."
                ),
                dataset_tags + ["mechanism", str(hypothesis.get("expected_direction", "unknown"))],
                [record_id, "mechanism.hypothesis-only-compiler"],
                [result_id] if result else [run_id] if run else [],
                "candidate" if accepted else "needs_verification",
                "medium",
                updated_at,
            ))
            if result:
                vs_ei = result["vs_mixed_kernel_gp_ei"]
                vs_random = result.get("vs_random_null")
                status = "done"
                if result["stable_gain_vs_mixed_kernel_gp_ei"]:
                    conclusion = "stable positive versus mixed-kernel GP-EI"
                    confidence = "medium"
                elif result["stable_harm_vs_mixed_kernel_gp_ei"]:
                    conclusion = "stable negative versus mixed-kernel GP-EI"
                    confidence = "medium"
                else:
                    conclusion = "inconclusive at the reported seed count"
                    confidence = "low"
                random_text = (
                    metric_text(vs_random) + ""
                    if vs_random else "matched random-null comparison unavailable."
                )
                cards.append(make_card(
                    result_id,
                    "experiment_result",
                    f"Zero-shot result: {hypothesis_id} on {target}",
                    f"{conclusion}; {metric_text(vs_ei)}",
                    (
                        f"Compared with mixed-kernel GP-EI on {vs_ei['seed_count']} paired seeds. "
                        f"Against the matched random null: {random_text} "
                        f"The target replay used no calibration outcomes and selected no hypothesis "
                        "from target outcomes. The result is evidence about this source-target pair, "
                        "not a universal transfer claim."
                    ),
                    dataset_tags + ["experiment", conclusion.replace(" ", "-")],
                    [card_id, run_id, "mechanism.zero-shot-transfer-protocol"],
                    [],
                    status,
                    confidence,
                    updated_at,
                ))
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updated-at", default="2026-07-25")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cards = build_cards(args.record, report, args.updated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cards": len(cards), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
