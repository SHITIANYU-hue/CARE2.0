#!/usr/bin/env python3
"""Close the CARE experiment -> skill -> retrieval loop.

This is the single finalization entry point for a completed experiment. It
ingests the result, applies the frozen promotion policy, rebuilds the search
indexes, and writes a receipt showing what the next CARE run can retrieve.
Scientific validation remains deterministic: the LLM may propose a skill, but
only task-disjoint evidence declared in the run manifest can activate it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import retrieval
import self_update


SCHEMA_VERSION = "care.kb.closed_loop_receipt/v1"
DEFAULT_RECEIPT = "knowledge_feedback.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(self_update.REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = " ".join(str(raw).split())
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def feedback_queries(result_dir: Path) -> list[str]:
    """Build outcome-free queries that mirror a future task prompt."""
    queries: list[str] = []
    headline = result_dir / "headline_results.csv"
    if headline.exists():
        with headline.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                queries.append(
                    " ".join(
                        value
                        for value in (
                            row.get("source", ""),
                            row.get("target", ""),
                            row.get("selected_skill", ""),
                            "transfer skill calibration negative transfer gate",
                        )
                        if value
                    )
                )

    for path in result_dir.glob("**/suite_summary.json"):
        value = _read_json(path, {})
        for row in value.get("per_task", []) if isinstance(value, dict) else []:
            if isinstance(row, dict):
                queries.append(
                    f"{row.get('target_task', '')} transfer skill hypothesis gate"
                )

    manifest = _read_json(result_dir / "run_manifest.json", {})
    if not queries and isinstance(manifest, dict):
        queries.append(
            " ".join(
                str(manifest.get(key, ""))
                for key in ("study", "experiment", "source_dataset", "target_dataset")
                if manifest.get(key)
            )
            + " transfer skill gate"
        )
    if not queries:
        queries.append(f"{result_dir.name} transfer skill gate")
    return _unique(queries)


def _cards_for_result(
    result_dir: Path,
    state_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    state = self_update.load_state(state_path)
    key = result_dir.resolve().as_posix()
    entry = state["processed"].get(key, {})
    generated = entry.get("generated_card_file", "")
    if not generated:
        return [], ""
    path = Path(generated)
    cards = _read_json(path, [])
    if not isinstance(cards, list):
        raise ValueError(f"Generated card file must contain a list: {path}")
    return [card for card in cards if isinstance(card, dict)], _display_path(path)


def close_loop(
    result_dirs: list[Path],
    *,
    seed: Path = self_update.build_kb.DEFAULT_SEED,
    generated_dir: Path = self_update.DEFAULT_GENERATED_DIR,
    state_path: Path = self_update.DEFAULT_STATE,
    audit_path: Path = self_update.DEFAULT_AUDIT,
    db: Path = self_update.DEFAULT_DB,
    export: Path = self_update.DEFAULT_EXPORT,
    embeddings: Path = self_update.DEFAULT_EMBEDDINGS,
    embedding_provider: str = "hashed",
    embedding_model: str = "text-embedding-3-small",
    embedding_dims: int = 256,
    embedding_batch_size: int = 32,
    stage_only: bool = False,
    force: bool = False,
    dry_run: bool = False,
    retrieval_limit: int = 8,
    receipt_name: str = DEFAULT_RECEIPT,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Finalize results and prove that accepted experience returns to runtime."""
    update_report = self_update.self_update(
        result_dirs,
        seed=seed,
        generated_dir=generated_dir,
        state_path=state_path,
        audit_path=audit_path,
        db=db,
        export=export,
        embeddings=embeddings,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        embedding_batch_size=embedding_batch_size,
        force=force,
        allow_promotion=not stage_only,
        dry_run=dry_run,
        timestamp=timestamp,
    )
    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "dry_run",
            "knowledge_update": update_report,
            "receipts": [],
        }

    receipts: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        cards, generated_card_file = _cards_for_result(result_dir, state_path)
        feedback = []
        for query in feedback_queries(result_dir):
            hits = retrieval.retrieve_runtime_cards(db, query, retrieval_limit)
            feedback.append(
                {
                    "query": query,
                    "retrieved_card_ids": [item["id"] for item in hits],
                    "retrieved_cards": hits,
                }
            )

        skill_states = {
            "active": sorted(
                str(card["id"])
                for card in cards
                if card.get("type") == "skill" and card.get("status") == "active"
            ),
            "candidate": sorted(
                str(card["id"])
                for card in cards
                if card.get("type") == "skill" and card.get("status") == "candidate"
            ),
            "negative_transfer": sorted(
                str(card["id"])
                for card in cards
                if card.get("type") == "transfer"
                and "negative-transfer" in card.get("tags", [])
            ),
        }
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "result_dir": _display_path(result_dir),
            "result_fingerprint": self_update.result_fingerprint(result_dir),
            "generated_card_file": generated_card_file,
            "skill_states": skill_states,
            "return_channel": {
                "knowledge_db": _display_path(db),
                "embedding_index": _display_path(embeddings),
                "runtime_policy": (
                    "Only active skill cards and public transfer/mechanism evidence are returned. "
                    "Candidate skills remain archived but cannot steer the next LLM run."
                ),
                "feedback": feedback,
            },
            "closed_loop": {
                "experiment_to_evidence": True,
                "evidence_to_candidate_skill": bool(skill_states["candidate"] or skill_states["active"]),
                "qualified_skill_to_runtime": bool(skill_states["active"]),
                "negative_evidence_to_runtime": bool(skill_states["negative_transfer"]),
                "next_llm_generation_reads_runtime_by_default": True,
            },
        }
        _write_json_atomic(result_dir / receipt_name, receipt)
        receipts.append(receipt)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "closed",
        "knowledge_update": update_report,
        "receipts": receipts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="+")
    parser.add_argument("--seed", type=Path, default=self_update.build_kb.DEFAULT_SEED)
    parser.add_argument("--generated-dir", type=Path, default=self_update.DEFAULT_GENERATED_DIR)
    parser.add_argument("--state", type=Path, default=self_update.DEFAULT_STATE)
    parser.add_argument("--audit", type=Path, default=self_update.DEFAULT_AUDIT)
    parser.add_argument("--db", type=Path, default=self_update.DEFAULT_DB)
    parser.add_argument("--export", type=Path, default=self_update.DEFAULT_EXPORT)
    parser.add_argument("--embeddings", type=Path, default=self_update.DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--embedding-provider",
        choices=["none", "hashed", "openai", "sentence_transformers"],
        default="hashed",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dims", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Archive all skills as candidates even when the manifest qualifies promotion.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retrieval-limit", type=int, default=8)
    parser.add_argument("--receipt-name", default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = close_loop(
        args.result_dir,
        seed=args.seed,
        generated_dir=args.generated_dir,
        state_path=args.state,
        audit_path=args.audit,
        db=args.db,
        export=args.export,
        embeddings=args.embeddings,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dims=args.embedding_dims,
        embedding_batch_size=args.embedding_batch_size,
        stage_only=args.stage_only,
        force=args.force,
        dry_run=args.dry_run,
        retrieval_limit=args.retrieval_limit,
        receipt_name=args.receipt_name,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
