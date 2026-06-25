from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "care_kb.sqlite"


def escape_fts_query(query: str) -> str:
    tokens = [token.strip() for token in query.replace('"', " ").split() if token.strip()]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


def add_like_terms(where: list[str], params: list[str | int], query: str) -> None:
    tokens = [token.strip() for token in query.replace('"', " ").split() if token.strip()]
    if not tokens:
        return
    token_clauses = []
    for token in tokens:
        token_clauses.append(
            "(c.id like ? or c.title like ? or c.summary like ? or c.content like ? or c.tags_json like ?)"
        )
        params.extend([f"%{token}%"] * 5)
    where.append("(" + " or ".join(token_clauses) + ")")


def render_rows(rows: list[sqlite3.Row]) -> None:
    for row in rows:
        tags = ", ".join(json.loads(row["tags_json"]))
        print(f"[{row['type']}] {row['id']} | {row['title']}")
        print(f"  priority={row['priority'] or '-'} status={row['status']} boundary={row['evidence_boundary']} tags={tags}")
        print(f"  {row['summary']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CARE KB")
    parser.add_argument("query", nargs="?", default="", help="FTS query. Empty query lists cards by type/status.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--type", dest="card_type", default="", help="Filter by card type")
    parser.add_argument("--status", default="", help="Filter by status")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    params: list[str | int] = []
    where: list[str] = []
    if args.card_type:
        where.append("c.type = ?")
        params.append(args.card_type)
    if args.status:
        where.append("c.status = ?")
        params.append(args.status)

    query = args.query.strip()
    if query:
        fts = escape_fts_query(args.query)
        where.append("cards_fts match ?")
        params.append(fts)
        sql = """
            select c.*, bm25(cards_fts) as score
            from cards_fts
            join cards c on c.id = cards_fts.id
        """
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by score limit ?"
    else:
        sql = "select c.*, 0.0 as score from cards c"
        if where:
            sql += " where " + " and ".join(where)
        sql += " order by case c.priority when 'P0' then 0 when 'P1' then 1 when 'P2' then 2 else 3 end, c.type, c.id limit ?"
    params.append(args.limit)

    rows = con.execute(sql, params).fetchall()
    if not rows and query:
        fallback_params: list[str | int] = []
        fallback_where: list[str] = []
        if args.card_type:
            fallback_where.append("c.type = ?")
            fallback_params.append(args.card_type)
        if args.status:
            fallback_where.append("c.status = ?")
            fallback_params.append(args.status)
        add_like_terms(fallback_where, fallback_params, query)
        fallback_sql = "select c.*, 0.0 as score from cards c"
        if fallback_where:
            fallback_sql += " where " + " and ".join(fallback_where)
        fallback_sql += " order by case c.priority when 'P0' then 0 when 'P1' then 1 when 'P2' then 2 else 3 end, c.type, c.id limit ?"
        fallback_params.append(args.limit)
        rows = con.execute(fallback_sql, fallback_params).fetchall()
    render_rows(rows)


if __name__ == "__main__":
    main()
