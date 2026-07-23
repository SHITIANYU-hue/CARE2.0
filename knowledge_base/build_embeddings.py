#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "care_kb.sqlite"
DEFAULT_OUT = ROOT / "embeddings" / "card_embeddings.jsonl"


def card_text(row: sqlite3.Row) -> str:
    tags = " ".join(json.loads(row["tags_json"]))
    sources = " ".join(json.loads(row["source_ids_json"]))
    related = " ".join(json.loads(row["related_ids_json"]))
    return "\n".join([row["title"], row["summary"], row["content"], tags, sources, related])


def load_cards(db: Path) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("select * from cards order by id").fetchall()
    con.close()
    return rows


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def hashed_embedding(text: str, dims: int) -> list[float]:
    vec = [0.0] * dims
    for token in re.findall(r"[A-Za-z0-9_./:-]+", text.lower()):
        h = 2166136261
        for ch in token:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        sign = 1.0 if h & 1 else -1.0
        vec[(h >> 1) % dims] += sign
    return normalize(vec)


def openai_request(base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Embedding endpoint returned HTTP {exc.code}: {body[:500]}") from exc


def openai_embeddings(texts: list[str], model: str, batch_size: int) -> list[list[float]]:
    api_key = os.environ.get("CARE_OPENAI_API_KEY")
    base_url = os.environ.get("CARE_OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("Set CARE_OPENAI_API_KEY before using --provider openai.")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": model, "input": batch}
        data = openai_request(base_url, api_key, payload)
        ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
        vectors.extend([item["embedding"] for item in ordered])
    return vectors


def sentence_transformer_embeddings(
    texts: list[str],
    model: str,
    batch_size: int,
) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers before using --provider sentence_transformers."
        ) from exc
    encoder = SentenceTransformer(model)
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def write_jsonl(rows: list[sqlite3.Row], vectors: list[list[float]], provider: str, model: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if len(rows) != len(vectors):
            raise RuntimeError(f"Card/vector length mismatch: {len(rows)} cards, {len(vectors)} vectors")
        for row, vector in zip(rows, vectors):
            record = {
                "id": row["id"],
                "type": row["type"],
                "title": row["title"],
                "summary": row["summary"],
                "provider": provider,
                "model": model,
                "embedding": vector,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector embeddings for CARE KB cards.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--provider",
        choices=["hashed", "openai", "sentence_transformers"],
        default="hashed",
    )
    parser.add_argument("--model", default=os.environ.get("CARE_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--dims", type=int, default=256, help="Only used by --provider hashed.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = load_cards(args.db)
    texts = [card_text(row) for row in rows]
    if args.provider == "hashed":
        vectors = [hashed_embedding(text, args.dims) for text in texts]
        model = f"hashed-{args.dims}"
    elif args.provider == "sentence_transformers":
        model = args.model
        vectors = sentence_transformer_embeddings(texts, model, args.batch_size)
    else:
        vectors = openai_embeddings(texts, args.model, args.batch_size)
        model = args.model

    write_jsonl(rows, vectors, args.provider, model, args.out)
    print(f"cards={len(rows)}")
    print(f"provider={args.provider}")
    print(f"model={model}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
