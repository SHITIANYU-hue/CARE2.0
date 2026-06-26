#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = ROOT / "embeddings" / "card_embeddings.jsonl"


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


def openai_embedding(text: str, model: str) -> list[float]:
    api_key = os.environ.get("CARE_OPENAI_API_KEY")
    base_url = os.environ.get("CARE_OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("Set CARE_OPENAI_API_KEY before querying an OpenAI embedding index.")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/embeddings",
        data=json.dumps({"model": model, "input": text}).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Embedding endpoint returned HTTP {exc.code}: {body[:500]}") from exc
    return data["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def load_index(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_vector(records: list[dict[str, Any]], query: str) -> list[float]:
    provider = records[0]["provider"]
    model = records[0]["model"]
    if provider == "hashed":
        dims = len(records[0]["embedding"])
        return hashed_embedding(query, dims)
    if provider == "openai":
        return normalize(openai_embedding(query, model))
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector search over CARE KB card embeddings.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    records = load_index(args.index)
    if not records:
        raise RuntimeError(f"Empty embedding index: {args.index}")
    q = query_vector(records, args.query)
    ranked = sorted(records, key=lambda item: cosine(q, item["embedding"]), reverse=True)[: args.limit]
    for item in ranked:
        score = cosine(q, item["embedding"])
        print(f"{score:.4f} [{item['type']}] {item['id']} | {item['title']}")
        print(f"  {item['summary']}")


if __name__ == "__main__":
    main()
