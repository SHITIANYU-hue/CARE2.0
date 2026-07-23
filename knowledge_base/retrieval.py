from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


ALLOWED_RUNTIME_TYPES = ("skill", "transfer", "mechanism")


def _query_tokens(query: str) -> str:
    tokens = [token for token in query.replace('"', " ").split() if token]
    return " OR ".join(f'"{token}"' for token in tokens)


def retrieve_runtime_cards(
    db: Path,
    query: str,
    limit: int = 5,
    allowed_types: tuple[str, ...] = ALLOWED_RUNTIME_TYPES,
    cutoff: str = "",
) -> list[dict[str, Any]]:
    if not db.exists() or not query.strip() or limit <= 0:
        return []
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in allowed_types)
    params: list[Any] = [_query_tokens(query), *allowed_types]
    cutoff_clause = ""
    if cutoff:
        cutoff_clause = " and c.updated_at <= ?"
        params.append(cutoff)
    params.append(limit)
    rows = con.execute(
        f"""
        select c.*, bm25(cards_fts) as score
        from cards_fts
        join cards c on c.id = cards_fts.id
        where cards_fts match ?
          and c.type in ({placeholders})
          and c.evidence_boundary = 'public'
          {cutoff_clause}
        order by score
        limit ?
        """,
        params,
    ).fetchall()
    con.close()
    return [
        {
            "id": row["id"],
            "type": row["type"],
            "title": row["title"],
            "summary": row["summary"],
            "content": row["content"],
            "tags": json.loads(row["tags_json"]),
            "updated_at": row["updated_at"],
            "score": float(row["score"]),
        }
        for row in rows
    ]
