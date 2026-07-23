from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "generated_cards"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_label(row: dict[str, str]) -> str:
    final_low = float(row.get("final_ci_low", 0.0) or 0.0)
    auc_low = float(row.get("auc_ci_low", 0.0) or 0.0)
    if final_low > 0.0 and auc_low > 0.0:
        return "confirmed_positive"
    if final_low > 0.0 or auc_low > 0.0:
        return "partially_confirmed_positive"
    return "inconclusive"


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
        "status": "done",
        "priority": "P1",
        "confidence": confidence,
        "evidence_boundary": "public",
        "updated_at": date,
    }


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
            f"Result directory: {result_dir.as_posix()}. The archive contains the manifest, "
            "per-seed metrics, summaries, model-call records, audit logs, and figures when present."
        ),
        ["replay", "audit", "trace", date],
        [],
        date,
        "high",
    )]
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
                f"{reusable_lesson}"
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
                )
                if value
            ],
            [run_id, result_id],
            date,
            "high" if label == "confirmed_positive" else "medium",
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
