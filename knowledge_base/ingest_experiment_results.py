from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_OUT_DIR = ROOT / "generated_cards"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def metric_label(row: dict[str, str]) -> str:
    final_low = float(row.get("final_ci_low", 0.0) or 0.0)
    auc_low = float(row.get("auc_ci_low", 0.0) or 0.0)
    if final_low > 0.0 and auc_low > 0.0:
        return "confirmed_positive"
    if final_low > 0.0 or auc_low > 0.0:
        return "partially_confirmed_positive"
    return "inconclusive"


def find_semantic_skill_definition(
    result_dir: Path,
    skill_id: str,
) -> tuple[dict[str, Any] | None, str]:
    """Recover the frozen executable rule set behind a reported skill."""
    for path in sorted(result_dir.rglob("*.json")):
        if path.name in {"summary.json", "run_manifest.json", "knowledge_feedback.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        skills = value.get("normalized_skills")
        if not isinstance(skills, list):
            continue
        for skill in skills:
            if isinstance(skill, dict) and str(skill.get("skill_id", "")) == skill_id:
                return skill, display_path(path)
    return None, ""


def executable_skill_evidence(
    result_dir: Path,
    skill_id: str,
) -> tuple[str, list[str]]:
    skill, artifact = find_semantic_skill_definition(result_dir, skill_id)
    if skill is None:
        return "", []
    executable = {
        "skill_id": skill_id,
        "hypothesis": skill.get("hypothesis", ""),
        "rules": skill.get("rules", []),
        "confidence": skill.get("confidence"),
        "execution_parameters": {
            key: skill[key]
            for key in (
                "ridge",
                "prior_scale",
                "semantic_mass_start",
                "semantic_mass_end",
                "ucb_weight",
                "gp_beta_start",
                "gp_beta_end",
                "gp_xi",
            )
            if key in skill
        },
    }
    return (
        " Frozen executable definition: "
        + json.dumps(executable, ensure_ascii=False, sort_keys=True)
        + f" Artifact: {artifact}.",
        ["executable-skill", "structured-rules"],
    )


def make_card(
    card_id: str,
    card_type: str,
    title: str,
    summary: str,
    content: str,
    tags: list[str],
    related_ids: list[str],
    date: str,
    confidence: str,
    status: str = "done",
) -> dict[str, Any]:
    return {
        "id": card_id,
        "type": card_type,
        "title": title,
        "summary": summary,
        "content": content,
        "tags": tags,
        "source_ids": [],
        "related_ids": related_ids,
        "status": status,
        "priority": "P1",
        "confidence": confidence,
        "evidence_boundary": "public",
        "updated_at": date,
    }


def llm_initial_design_suite_cards(
    result_dir: Path,
    run_id: str,
    study: str,
    date: str,
) -> list[dict[str, Any]]:
    summary_path = result_dir / "aggregate" / "suite_summary.json"
    if not summary_path.exists():
        return []
    suite = json.loads(summary_path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != "care.llm_initial_design_suite/v1":
        return []

    auc = suite.get("auc_delta_vs_fixed_v2", {})
    final = suite.get("final_best_delta_vs_fixed_v2", {})
    route = suite.get("route", {})
    result_id = f"experiment-result.{slug(study)}-llm-initial-design-{date}"
    task_rows = [row for row in suite.get("per_task", []) if isinstance(row, dict)]
    hypothesis_ids = [
        f"hypothesis.{slug(study)}-{slug(str(row.get('target_task', 'unknown')))}-{date}"
        for row in task_rows
    ]
    ci_low = float(auc.get("task_95ci_low", 0.0) or 0.0)
    ci_high = float(auc.get("task_95ci_high", 0.0) or 0.0)
    significance = (
        "task-level CI excludes zero"
        if ci_low > 0.0
        else "task-level CI crosses zero"
    )
    cards: list[dict[str, Any]] = [make_card(
        result_id,
        "experiment_result",
        f"{study}: LLM hypothesis initial-design suite",
        (
            f"Across {suite.get('task_count', 0)} targets, routed LLM initial design changed "
            f"search AUC by {float(auc.get('task_mean', 0.0) or 0.0):+.4f} versus fixed v2; "
            f"{significance}."
        ),
        (
            f"Evidence class: {suite.get('evidence_class', 'unknown')}. "
            f"AUC 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], task win rate "
            f"{float(auc.get('task_win_rate', 0.0) or 0.0):.3f}, non-loss rate "
            f"{float(auc.get('task_nonloss_rate', 0.0) or 0.0):.3f}. Final-best delta "
            f"{float(final.get('task_mean', 0.0) or 0.0):+.4f}. Route "
            f"{route.get('name', 'unknown')} does not use target outcomes during selection, "
            f"but remains {route.get('status', 'unconfirmed')}."
        ),
        [
            "llm-hypothesis",
            "initial-design",
            "component-ablation",
            "retrospective",
            "needs-external-confirmation",
        ],
        [run_id, *hypothesis_ids],
        date,
        "medium",
    )]

    for row, hypothesis_id in zip(task_rows, hypothesis_ids):
        target = str(row.get("target_task", "unknown"))
        auc_delta = float(row.get("routed_auc_delta_vs_fixed", 0.0) or 0.0)
        final_delta = float(row.get("routed_final_delta_vs_fixed", 0.0) or 0.0)
        cards.append(make_card(
            hypothesis_id,
            "hypothesis",
            f"LLM transfer hypothesis for {target}",
            str(row.get("hypothesis", "No hypothesis text was recorded.")),
            (
                f"Route: {row.get('route_mode', 'unknown')} ({row.get('route_reason', 'no reason recorded')}). "
                f"Against fixed v2, routed AUC delta was {auc_delta:+.4f} and final-best delta "
                f"was {final_delta:+.4f}. LLM confidence was "
                f"{float(row.get('llm_confidence', 0.0) or 0.0):.3f}. Trace: "
                f"{row.get('hypothesis_record', 'not recorded')}. This retrospective card is a "
                "candidate claim, not a validated transferable mechanism."
            ),
            [
                target,
                str(row.get("route_mode", "unknown")),
                "llm-reasoning-trace",
                "outcome-blind-selection",
                "candidate-hypothesis",
            ],
            [run_id, result_id],
            date,
            "medium",
            "candidate",
        ))
    return cards


def llm_reflective_suite_cards(
    result_dir: Path,
    run_id: str,
    study: str,
    date: str,
) -> list[dict[str, Any]]:
    summary_path = result_dir / "aggregate" / "suite_summary.json"
    if not summary_path.exists():
        return []
    suite = json.loads(summary_path.read_text(encoding="utf-8"))
    if suite.get("schema_version") != "care.llm_reflective_scientist_suite/v1":
        return []

    auc = suite.get("auc_delta_vs_fixed_v2", {})
    gate = suite.get("gate", {})
    result_id = f"experiment-result.{slug(study)}-llm-reflective-scientist-{date}"
    task_rows = [row for row in suite.get("per_task", []) if isinstance(row, dict)]
    hypothesis_ids = [
        f"hypothesis.{slug(study)}-{slug(str(row.get('target_task', 'unknown')))}-reflection-{date}"
        for row in task_rows
    ]
    ci_low = float(auc.get("task_95ci_low", 0.0) or 0.0)
    ci_high = float(auc.get("task_95ci_high", 0.0) or 0.0)
    status_counts = suite.get("reflection_hypothesis_status_counts", {})
    cards: list[dict[str, Any]] = [make_card(
        result_id,
        "experiment_result",
        f"{study}: evidence-bounded LLM scientist loop",
        (
            f"Across {suite.get('task_count', 0)} targets, the reflective LLM route changed "
            f"search AUC by {float(auc.get('task_mean', 0.0) or 0.0):+.4f} versus fixed v2; "
            f"95% CI [{ci_low:+.4f}, {ci_high:+.4f}]."
        ),
        (
            f"The LLM classified hypotheses as {status_counts}, stopped transfer on "
            f"{float(suite.get('reflection_stop_rate', 0.0) or 0.0):.3f} of tasks, and the "
            f"zero-loss acquisition gate authorized {gate.get('acceptance_count', 0)} of "
            f"{gate.get('proposal_count', 0)} follow-up proposals. This retrospective suite "
            "demonstrates an auditable scientist loop, not independent external confirmation."
        ),
        [
            "llm-as-scientist",
            "hypothesis-revision",
            "counterexample-planning",
            "reflection-gate",
            "retrospective",
        ],
        [run_id, *hypothesis_ids],
        date,
        "medium",
    )]
    for row, hypothesis_id in zip(task_rows, hypothesis_ids):
        status = str(row.get("hypothesis_status", "unknown"))
        cards.append(make_card(
            hypothesis_id,
            "hypothesis",
            f"Reflected transfer hypothesis for {row.get('target_task', 'unknown')}",
            str(row.get("revised_hypothesis", "No revised hypothesis recorded.")),
            (
                f"Reflection status: {status}. Evidence interpretation: "
                f"{row.get('evidence_interpretation', 'not recorded')} Against fixed v2, "
                f"AUC delta was {float(row.get('auc_delta_vs_fixed', 0.0) or 0.0):+.4f}; "
                f"stop_transfer={bool(row.get('stop_transfer', False))}; gate accepted "
                f"{row.get('gate_acceptance_count', 0)} of {row.get('gate_proposal_count', 0)} "
                f"proposals. Trace: {row.get('reflection_record', 'not recorded')}."
            ),
            [
                str(row.get("target_task", "unknown")),
                f"hypothesis-{status}",
                "llm-reasoning-trace",
                "hypothesis-revision",
                "candidate-hypothesis",
            ],
            [run_id, result_id],
            date,
            "medium",
            "candidate",
        ))
    return cards


def cards_from_result_dir(result_dir: Path) -> list[dict[str, Any]]:
    manifest_path = result_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    date = str(manifest.get("date") or result_dir.name[:10])
    study = str(manifest.get("study") or result_dir.name)
    run_id = f"run-log.{slug(study)}-{date}"
    cards: list[dict[str, Any]] = [make_card(
        run_id,
        "run_log",
        f"{study} reproducibility archive",
        f"Reproducibility archive for {study}.",
        (
            f"Result directory: {display_path(result_dir)}. The archive contains the manifest, "
            "per-seed metrics, summaries, model-call records, audit logs, and figures when present."
        ),
        ["replay", "audit", "trace", date],
        [],
        date,
        "high",
    )]
    cards.extend(llm_initial_design_suite_cards(result_dir, run_id, study, date))
    cards.extend(llm_reflective_suite_cards(result_dir, run_id, study, date))
    for row in load_rows(result_dir / "headline_results.csv"):
        source = row.get("source", "")
        target = row["target"]
        skill = row["selected_skill"]
        label = metric_label(row)
        pair_slug = f"{slug(source)}-to-{slug(target)}" if source else slug(target)
        result_id = f"experiment-result.{slug(study)}-{pair_slug}-{date}"
        skill_id = f"skill.semantic-{pair_slug}-{slug(skill)}-{date}"
        evidence_mode = row.get("evidence_mode", "unknown")
        final_delta = float(row.get("delta_final_best", 0.0) or 0.0)
        auc_delta = float(row.get("delta_auc", 0.0) or 0.0)
        rounds_saved = float(row.get("rounds_saved_top10", 0.0) or 0.0)
        execution = row.get("rule_prior", "unknown")
        executable_evidence, executable_tags = executable_skill_evidence(
            result_dir, skill
        )
        if execution == "warmstart":
            skill_summary = (
                f"For {target}, use the LLM-defined {skill} partition to choose "
                "the initial batch, then continue with the target-only optimizer."
            )
            reusable_lesson = (
                "Reusable lesson: calibration should compare LLM warm-starting "
                "with online semantic fitting and direct-prior execution."
            )
        elif execution == "direct_prior":
            skill_summary = (
                f"For {target}, keep the LLM-defined {skill} rule as a bounded "
                "fixed prior blended with the target-only optimizer."
            )
            reusable_lesson = (
                "Reusable lesson: target calibration can decide that a fixed "
                "semantic prior is more reliable than re-fitting its direction."
            )
        elif execution == "source_outcome":
            skill_summary = (
                f"For {target}, use the LLM-defined source-target role map to "
                "compile measured source outcomes into bounded neighbor, additive, "
                "interaction, initial-design, and kernel priors."
            )
            reusable_lesson = (
                "Reusable lesson: freeze the role map and source history before "
                "target replay, calibrate source priors only on revealed target "
                "observations, and retain an exact target-only fallback."
            )
        elif execution == "exact_fallback":
            skill_summary = (
                f"For {target}, reject the unsupported source-outcome route and "
                "reproduce the matched target-only LLM policy exactly."
            )
            reusable_lesson = (
                "Reusable lesson: a safe transfer platform must preserve negative "
                "source evidence and decline transfer when calibration is unstable."
            )
        else:
            skill_summary = (
                f"For {target}, use the LLM to define the {skill} feature "
                "partition, then fit its direction and magnitude from revealed "
                "target observations."
            )
            reusable_lesson = (
                "Reusable lesson: preserve the LLM-proposed feature partition, "
                "but calibrate or remove unsupported coefficient signs before "
                "held-out evaluation."
            )
        summary = (
            f"{skill} on {target}: final-best delta {final_delta:+.4f}, "
            f"AUC delta {auc_delta:+.4f}, top-10 rounds saved {rounds_saved:+.3f}; {label}."
        )
        cards.append(make_card(
            result_id,
            "experiment_result",
            f"{source + ' → ' if source else ''}{target}: {skill} ({evidence_mode})",
            summary,
            (
                f"Compared with {row.get('baseline')} on {row.get('seeds')} paired seeds. "
                f"Final-best 95% CI [{row.get('final_ci_low')}, {row.get('final_ci_high')}]; "
                f"AUC 95% CI [{row.get('auc_ci_low')}, {row.get('auc_ci_high')}]."
            ),
            [
                value
                for value in (
                    source,
                    target,
                    skill,
                    evidence_mode,
                    label,
                    "llm-semantic-skill",
                )
                if value
            ],
            [run_id, skill_id],
            date,
            "high" if label == "confirmed_positive" else "medium",
        ))
        cards.append(make_card(
            skill_id,
            "skill",
            f"Reusable semantic feature pattern: {skill}",
            skill_summary,
            (
                f"Evidence mode: {evidence_mode}. Execution: {execution}. "
                f"{reusable_lesson}{executable_evidence}"
            ),
            [
                value
                for value in (
                    source,
                    target,
                    skill,
                    evidence_mode,
                    "schema-to-skill",
                    "target-calibration",
                    *executable_tags,
                )
                if value
            ],
            [run_id, result_id],
            date,
            "high" if label == "confirmed_positive" else "medium",
            "candidate",
        ))
    negatives = manifest.get("negative_results")
    if not isinstance(negatives, list):
        legacy = manifest.get("negative_result")
        negatives = [legacy] if isinstance(legacy, dict) else []
    for negative in negatives:
        if not isinstance(negative, dict):
            continue
        target = str(negative.get("target", "unknown"))
        source = str(negative.get("source", "unknown"))
        case = slug(str(negative.get("case", "gate-reject")))
        cards.append(make_card(
            f"transfer.{slug(source)}-to-{slug(target)}-{case}-{date}",
            "transfer",
            f"Gate-rejected transfer: {source} to {target}",
            str(negative.get("result", "Transfer was rejected by calibration.")),
            (
                "Reusable lesson: do not force a source-derived semantic policy when every evidence "
                "mode fails calibration. Preserve the target-only fallback and improve the reaction "
                "representation before retrying."
            ),
            [source, target, "negative-transfer", "gate-reject"],
            [run_id],
            date,
            "high",
        ))
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CARE replay results into auditable KB cards.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    cards = cards_from_result_dir(args.result_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.result_dir.name}.json"
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cards={len(cards)}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
