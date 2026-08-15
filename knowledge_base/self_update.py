#!/usr/bin/env python3
"""Safely ingest completed CARE experiments into the runtime knowledge base.

The updater is automatic about evidence ingestion and index rebuilding, but
conservative about scientific claims. New reusable skills remain candidates
unless an explicit, task-disjoint confirmation manifest and a human-enabled
promotion flag are both present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import build_embeddings
import build_kb
import ingest_experiment_results


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "care_replay" / "results"
DEFAULT_GENERATED_DIR = ROOT / "generated_cards"
DEFAULT_STATE = ROOT / "self_update" / "state.json"
DEFAULT_AUDIT = ROOT / "self_update" / "audit.jsonl"
DEFAULT_DB = ROOT / "care_kb.sqlite"
DEFAULT_EXPORT = ROOT / "exports" / "CARE-KB-index.md"
DEFAULT_EMBEDDINGS = ROOT / "embeddings" / "card_embeddings.jsonl"

STATE_SCHEMA = "care.kb.self_update/v1"
AUDIT_SCHEMA = "care.kb.self_update.audit/v1"
PROMOTION_KEYS = (
    "allow_skill_promotion",
    "task_disjoint_confirmation",
    "protocol_frozen_before_evaluation",
    "external_outcomes_not_used_during_selection",
)
VALID_CARD_STATUSES = {
    "active",
    "candidate",
    "needs_verification",
    "done",
    "blocked",
}
FINGERPRINT_NAMES = {"run_manifest.json", "headline_results.csv", "summary.json"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_audit(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def result_evidence_files(result_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(result_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in FINGERPRINT_NAMES or path.name.endswith("_summary.json"):
            files.append(path)
    return files


def result_fingerprint(result_dir: Path) -> str:
    files = result_evidence_files(result_dir)
    if not files:
        raise ValueError(
            f"{result_dir} has no run manifest, headline table, or summary file."
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(result_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def discover_result_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    discovered = {path.parent for path in results_root.rglob("run_manifest.json")}
    for path in results_root.rglob("suite_summary.json"):
        if path.parent.name != "aggregate":
            continue
        value = read_json(path, {})
        if isinstance(value, dict) and value.get("schema_version") in {
            "care.llm_initial_design_suite/v1",
            "care.llm_reflective_scientist_suite/v1",
        }:
            discovered.add(path.parent.parent)
    return sorted(discovered)


def existing_ingested_results(generated_dir: Path) -> dict[Path, Path]:
    """Map result directories already represented by a generated run-log card."""
    represented: dict[Path, Path] = {}
    for card_file in sorted(generated_dir.glob("*.json")):
        value = read_json(card_file, [])
        if not isinstance(value, list):
            continue
        for card in value:
            if not isinstance(card, dict) or card.get("type") != "run_log":
                continue
            match = re.search(r"Result directory: ([^.]+)\. The archive", str(card.get("content", "")))
            if not match:
                continue
            path = Path(match.group(1))
            if not path.is_absolute():
                path = REPO_ROOT / path
            represented[path.resolve()] = card_file.resolve()
    return represented


def load_manifest(result_dir: Path) -> dict[str, Any]:
    value = read_json(result_dir / "run_manifest.json", {})
    if not isinstance(value, dict):
        raise ValueError(f"run_manifest.json must contain an object: {result_dir}")
    return value


def promotion_ready(manifest: dict[str, Any], card: dict[str, Any]) -> bool:
    policy = manifest.get("knowledge_update", {})
    return (
        isinstance(policy, dict)
        and all(policy.get(key) is True for key in PROMOTION_KEYS)
        and card.get("confidence") == "high"
    )


def normalize_cards(
    cards: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    allow_promotion: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    promoted: list[str] = []
    for raw in cards:
        card = json.loads(json.dumps(raw, ensure_ascii=False))
        tags = card.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"Card {card.get('id')} tags must be a list.")
        card["tags"] = sorted({str(tag) for tag in [*tags, "self-update"]})
        if card.get("type") == "skill":
            if allow_promotion and promotion_ready(manifest, card):
                card["status"] = "active"
                promoted.append(str(card["id"]))
            else:
                card["status"] = "candidate"
        normalized.append(card)
    return normalized, promoted


def validate_cards(cards: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for card in cards:
        missing = [field for field in build_kb.CARD_COLUMNS if field not in card]
        if missing:
            raise ValueError(f"{card.get('id', '<missing id>')} missing fields: {missing}")
        card_id = str(card["id"])
        if not card_id or card_id in seen:
            raise ValueError(f"Duplicate or empty card id: {card_id}")
        seen.add(card_id)
        if card["status"] not in VALID_CARD_STATUSES:
            raise ValueError(f"Unsupported status for {card_id}: {card['status']}")
        for field in ("tags", "source_ids", "related_ids"):
            if not isinstance(card[field], list) or not all(
                isinstance(value, str) for value in card[field]
            ):
                raise ValueError(f"{card_id} field {field} must be a string list.")


def canonical_card(card: dict[str, Any]) -> str:
    return json.dumps(card, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_evidence(card: dict[str, Any]) -> str:
    """Compare cards while ignoring self-update bookkeeping differences."""
    comparable = json.loads(json.dumps(card, ensure_ascii=False))
    repo_prefix = REPO_ROOT.resolve().as_posix() + "/"
    for field in ("title", "summary", "content"):
        if isinstance(comparable.get(field), str):
            comparable[field] = comparable[field].replace(repo_prefix, "")
    comparable["tags"] = sorted(
        tag for tag in comparable.get("tags", []) if tag != "self-update"
    )
    if comparable.get("type") == "skill" and comparable.get("status") in {
        "candidate",
        "done",
    }:
        comparable["status"] = "candidate"
    return canonical_card(comparable)


def load_state(path: Path) -> dict[str, Any]:
    state = read_json(path, {"schema_version": STATE_SCHEMA, "processed": {}})
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError(f"Unsupported self-update state schema: {state.get('schema_version')}")
    if not isinstance(state.get("processed"), dict):
        raise ValueError("Self-update state 'processed' must be an object.")
    return state


def build_embedding_index(
    db: Path,
    out: Path,
    provider: str,
    model: str,
    dims: int,
    batch_size: int,
) -> dict[str, Any]:
    rows = build_embeddings.load_cards(db)
    texts = [build_embeddings.card_text(row) for row in rows]
    if provider == "hashed":
        vectors = [build_embeddings.hashed_embedding(text, dims) for text in texts]
        resolved_model = f"hashed-{dims}"
    elif provider == "sentence_transformers":
        resolved_model = model
        vectors = build_embeddings.sentence_transformer_embeddings(
            texts, resolved_model, batch_size
        )
    elif provider == "openai":
        resolved_model = model
        vectors = build_embeddings.openai_embeddings(texts, resolved_model, batch_size)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    build_embeddings.write_jsonl(rows, vectors, provider, resolved_model, out)
    return {"provider": provider, "model": resolved_model, "card_count": len(rows)}


def self_update(
    result_dirs: list[Path],
    *,
    seed: Path = build_kb.DEFAULT_SEED,
    generated_dir: Path = DEFAULT_GENERATED_DIR,
    state_path: Path = DEFAULT_STATE,
    audit_path: Path = DEFAULT_AUDIT,
    db: Path = DEFAULT_DB,
    export: Path = DEFAULT_EXPORT,
    embeddings: Path = DEFAULT_EMBEDDINGS,
    embedding_provider: str = "hashed",
    embedding_model: str = "text-embedding-3-small",
    embedding_dims: int = 256,
    embedding_batch_size: int = 32,
    force: bool = False,
    allow_promotion: bool = False,
    dry_run: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    timestamp = timestamp or utc_now()
    state = load_state(state_path)
    represented_results = existing_ingested_results(generated_dir)
    unique_dir_map = {path.resolve(): path for path in result_dirs}
    unique_dirs = [unique_dir_map[key] for key in sorted(unique_dir_map)]
    plans: list[dict[str, Any]] = []
    skipped: list[str] = []
    adopted: list[dict[str, Any]] = []

    for result_dir in unique_dirs:
        if not result_dir.is_dir():
            raise ValueError(f"Result directory does not exist: {result_dir}")
        fingerprint = result_fingerprint(result_dir)
        key = result_dir.resolve().as_posix()
        previous = state["processed"].get(key, {})
        if not force and previous.get("fingerprint") == fingerprint:
            skipped.append(key)
            continue
        represented_by = represented_results.get(result_dir.resolve())
        if not force and not previous and represented_by is not None:
            evidence_mtime = max(
                path.stat().st_mtime for path in result_evidence_files(result_dir)
            )
            if evidence_mtime <= represented_by.stat().st_mtime:
                adopted.append(
                    {
                        "key": key,
                        "fingerprint": fingerprint,
                        "generated_card_file": represented_by.as_posix(),
                    }
                )
                continue
        manifest = load_manifest(result_dir)
        raw_cards = ingest_experiment_results.cards_from_result_dir(result_dir)
        cards, promoted = normalize_cards(raw_cards, manifest, allow_promotion)
        output = represented_by or (
            generated_dir
            / f"auto-{ingest_experiment_results.slug(result_dir.name)}.json"
        )
        plans.append(
            {
                "result_dir": result_dir,
                "key": key,
                "fingerprint": fingerprint,
                "manifest": manifest,
                "cards": cards,
                "promoted": promoted,
                "output": output,
            }
        )

    if not plans:
        report = {
            "schema_version": AUDIT_SCHEMA,
            "status": "dry_run" if dry_run else "state_initialized" if adopted else "noop",
            "updated_at": timestamp,
            "processed": [],
            "skipped": skipped,
            "adopted_existing": [item["key"] for item in adopted],
            "message": (
                "Existing generated evidence was adopted into self-update state."
                if adopted
                else "No new or changed experiment evidence was found."
            ),
        }
        if not dry_run and adopted:
            for item in adopted:
                state["processed"][item["key"]] = {
                    "fingerprint": item["fingerprint"],
                    "generated_card_file": item["generated_card_file"],
                    "updated_at": timestamp,
                    "card_count": "legacy_existing",
                    "promoted_skill_ids": [],
                }
            write_json_atomic(state_path, state)
            append_audit(audit_path, report)
        return report

    replaced_outputs = {plan["output"].resolve() for plan in plans}
    base_generated = [
        path
        for path in sorted(generated_dir.glob("*.json"))
        if path.resolve() not in replaced_outputs
    ]
    base_cards = build_kb.load_cards([seed, *base_generated])
    base_by_id = {card["id"]: card for card in base_cards}
    accepted_by_output: dict[Path, list[dict[str, Any]]] = {}
    accepted_ids: set[str] = set()
    reused_ids: list[str] = []

    for plan in plans:
        accepted: list[dict[str, Any]] = []
        for card in plan["cards"]:
            card_id = card["id"]
            existing = base_by_id.get(card_id)
            if existing is not None:
                if canonical_evidence(existing) != canonical_evidence(card):
                    raise ValueError(
                        f"Self-update card conflicts with existing evidence: {card_id}"
                    )
                reused_ids.append(card_id)
                continue
            if card_id in accepted_ids:
                raise ValueError(f"Two new result directories generated card id: {card_id}")
            accepted_ids.add(card_id)
            accepted.append(card)
        accepted_by_output[plan["output"]] = accepted

    combined_cards = [
        *base_cards,
        *(card for cards in accepted_by_output.values() for card in cards),
    ]
    validate_cards(combined_cards)

    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "dry_run" if dry_run else "updated",
        "updated_at": timestamp,
        "processed": [plan["key"] for plan in plans],
        "skipped": skipped,
        "adopted_existing": [item["key"] for item in adopted],
        "new_card_count": sum(len(cards) for cards in accepted_by_output.values()),
        "reused_card_ids": sorted(set(reused_ids)),
        "promoted_skill_ids": sorted(
            skill_id for plan in plans for skill_id in plan["promoted"]
        ),
        "promotion_policy": {
            "human_enabled": allow_promotion,
            "required_manifest_keys": list(PROMOTION_KEYS),
        },
        "total_card_count": len(combined_cards),
    }
    if dry_run:
        return report

    generated_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".care-kb-update-", dir=ROOT) as raw:
        transaction = Path(raw)
        staged_outputs: list[tuple[Path, Path]] = []
        for output, cards in accepted_by_output.items():
            staged = transaction / "cards" / output.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(
                json.dumps(cards, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staged_outputs.append((staged, output))

        staged_db = transaction / "care_kb.sqlite"
        staged_export = transaction / "CARE-KB-index.md"
        build_kb.build_sqlite(combined_cards, staged_db)
        build_kb.write_markdown(combined_cards, staged_export)

        embedding_report: dict[str, Any] | None = None
        staged_embeddings = transaction / "card_embeddings.jsonl"
        if embedding_provider != "none":
            embedding_report = build_embedding_index(
                staged_db,
                staged_embeddings,
                embedding_provider,
                embedding_model,
                embedding_dims,
                embedding_batch_size,
            )

        for staged, output in staged_outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, output)
        db.parent.mkdir(parents=True, exist_ok=True)
        export.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_db, db)
        os.replace(staged_export, export)
        if embedding_report is not None:
            embeddings.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_embeddings, embeddings)
        report["embedding_index"] = embedding_report

    for plan in plans:
        state["processed"][plan["key"]] = {
            "fingerprint": plan["fingerprint"],
            "generated_card_file": plan["output"].resolve().as_posix(),
            "updated_at": timestamp,
            "card_count": len(accepted_by_output[plan["output"]]),
            "promoted_skill_ids": plan["promoted"],
        }
    for item in adopted:
        state["processed"][item["key"]] = {
            "fingerprint": item["fingerprint"],
            "generated_card_file": item["generated_card_file"],
            "updated_at": timestamp,
            "card_count": "legacy_existing",
            "promoted_skill_ids": [],
        }
    write_json_atomic(state_path, state)
    append_audit(audit_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="*")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Scan --results-root and process every new or changed supported result suite.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seed", type=Path, default=build_kb.DEFAULT_SEED)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--embedding-provider",
        choices=["none", "hashed", "openai", "sentence_transformers"],
        default="hashed",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("CARE_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument("--embedding-dims", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-promotion",
        action="store_true",
        help="Allow only manifest-qualified, high-confidence skills to become active.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = list(args.result_dir)
    if args.discover:
        result_dirs.extend(discover_result_dirs(args.results_root))
    if not result_dirs:
        raise SystemExit("Provide at least one result_dir or use --discover.")
    report = self_update(
        result_dirs,
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
        force=args.force,
        allow_promotion=args.allow_promotion,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
