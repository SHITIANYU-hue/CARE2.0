# CARE Knowledge Base Prototype

This directory contains a small, auditable knowledge-base prototype for CARE
2.0. The first version uses JSON cards as source data, SQLite as the local
storage layer, and SQLite FTS5 for text search.

The public seed file intentionally contains only public, release-safe cards.
Internal chat logs, private notes, and generated SQLite databases are not
tracked in this repository.

## Build

```bash
python3 knowledge_base/build_kb.py
```

This creates:

- `knowledge_base/care_kb.sqlite`
- `knowledge_base/exports/CARE-KB-index.md`

These generated files are ignored by git.

## Query

```bash
python3 knowledge_base/query_kb.py gate --limit 5
python3 knowledge_base/query_kb.py --type dataset
python3 knowledge_base/query_kb.py "Suzuki ChemLex"
```

## Card Model

See `schema.md` for the card fields and supported card types.
