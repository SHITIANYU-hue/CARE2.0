from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = ROOT / "seed_cards.json"
DEFAULT_DB = ROOT / "care_kb.sqlite"
DEFAULT_EXPORT = ROOT / "exports" / "CARE-KB-index.md"


CARD_COLUMNS = [
    "id",
    "type",
    "title",
    "summary",
    "content",
    "tags",
    "source_ids",
    "related_ids",
    "status",
    "priority",
    "confidence",
    "evidence_boundary",
    "updated_at",
]


def load_cards(path: Path) -> list[dict]:
    cards = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for card in cards:
        missing = [key for key in CARD_COLUMNS if key not in card]
        if missing:
            raise ValueError(f"{card.get('id', '<missing id>')} missing fields: {missing}")
        if card["id"] in seen:
            raise ValueError(f"Duplicate card id: {card['id']}")
        seen.add(card["id"])
    return cards


def as_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(value)
    return "" if value is None else str(value)


def card_search_text(card: dict) -> str:
    return "\n".join(
        [
            card["title"],
            card["summary"],
            card["content"],
            " ".join(card["tags"]),
            " ".join(card["source_ids"]),
            " ".join(card["related_ids"]),
        ]
    )


def build_sqlite(cards: list[dict], out: Path) -> None:
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.execute("pragma journal_mode=delete")
    con.execute(
        """
        create table cards (
            id text primary key,
            type text not null,
            title text not null,
            summary text not null,
            content text not null,
            tags_json text not null,
            source_ids_json text not null,
            related_ids_json text not null,
            status text not null,
            priority text not null,
            confidence text not null,
            evidence_boundary text not null,
            updated_at text not null
        )
        """
    )
    con.execute(
        """
        create virtual table cards_fts using fts5(
            id unindexed,
            type,
            title,
            summary,
            content,
            tags,
            search_text,
            tokenize='unicode61'
        )
        """
    )
    for card in cards:
        con.execute(
            """
            insert into cards values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card["id"],
                card["type"],
                card["title"],
                card["summary"],
                card["content"],
                json.dumps(card["tags"], ensure_ascii=False),
                json.dumps(card["source_ids"], ensure_ascii=False),
                json.dumps(card["related_ids"], ensure_ascii=False),
                card["status"],
                card["priority"],
                card["confidence"],
                card["evidence_boundary"],
                card["updated_at"],
            ),
        )
        con.execute(
            "insert into cards_fts values (?, ?, ?, ?, ?, ?, ?)",
            (
                card["id"],
                card["type"],
                card["title"],
                card["summary"],
                card["content"],
                " ".join(card["tags"]),
                card_search_text(card),
            ),
        )
    con.execute("create index idx_cards_type on cards(type)")
    con.execute("create index idx_cards_priority on cards(priority)")
    con.execute("create index idx_cards_status on cards(status)")
    con.commit()
    con.close()


def write_markdown(cards: list[dict], out: Path) -> None:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_type[card["type"]].append(card)

    type_order = [
        "decision",
        "task",
        "dataset",
        "paper",
        "mechanism",
        "skill",
        "open_question",
        "action",
        "source",
    ]
    lines: list[str] = [
        "# CARE 2.0 Knowledge Base Index",
        "",
        "生成文件：`care_kb.sqlite` + 本 Markdown 索引。",
        "",
        "## 概览",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for typ in type_order:
        if typ in by_type:
            lines.append(f"| `{typ}` | {len(by_type[typ])} |")
    lines.append("")

    lines.extend(
        [
            "## 建议先读",
            "",
            "1. `decision.target-is-experiment`：目标是实验闭环，不是自动写论文。",
            "2. `task.care-knowledge-base`：当前知识库的任务中心定位。",
            "3. `task.hte-optimization`：近期主线任务。",
            "4. `mechanism.versioned-evidence-gate`：Gate 的版本化审计边界。",
            "5. `mechanism.public-private-boundary`：晶泰数据公开边界。",
            "6. `action.create-dataset-cards`：下一步最该补的内容。",
            "",
        ]
    )

    for typ in type_order:
        items = by_type.get(typ)
        if not items:
            continue
        lines.extend([f"## {typ}", ""])
        for card in sorted(items, key=lambda x: (x["priority"], x["id"])):
            tags = ", ".join(f"`{tag}`" for tag in card["tags"])
            sources = ", ".join(f"`{sid}`" for sid in card["source_ids"]) or "-"
            related = ", ".join(f"`{rid}`" for rid in card["related_ids"]) or "-"
            lines.extend(
                [
                    f"### {card['id']} - {card['title']}",
                    "",
                    f"**Priority:** {card['priority'] or '-'}  ",
                    f"**Status:** {card['status']}  ",
                    f"**Confidence:** {card['confidence']}  ",
                    f"**Boundary:** {card['evidence_boundary']}",
                    "",
                    card["summary"],
                    "",
                    card["content"],
                    "",
                    f"**Tags:** {tags}",
                    "",
                    f"**Sources:** {sources}",
                    "",
                    f"**Related:** {related}",
                    "",
                ]
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()

    cards = load_cards(args.seed)
    build_sqlite(cards, args.db)
    write_markdown(cards, args.export)
    print(f"cards={len(cards)}")
    print(f"db={args.db}")
    print(f"export={args.export}")


if __name__ == "__main__":
    main()
