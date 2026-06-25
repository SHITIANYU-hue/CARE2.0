# CARE Knowledge Base Schema

Each knowledge item is a JSON object. The first public version uses
`seed_cards.json` as the editable source. `build_kb.py` builds a SQLite database
and an FTS5 search index from those cards.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable card ID, preferably `<type>.<slug>` |
| `type` | string | Card type: `source`, `task`, `dataset`, `paper`, `mechanism`, `skill`, `decision`, `open_question`, or `action` |
| `title` | string | Human-readable title |
| `summary` | string | Short 1-3 sentence summary |
| `content` | string | Longer description |
| `tags` | string[] | Search tags |
| `source_ids` | string[] | Provenance card IDs |
| `related_ids` | string[] | Related card IDs |
| `status` | string | `active`, `candidate`, `needs_verification`, `done`, or `blocked` |
| `priority` | string | `P0`, `P1`, `P2`, or empty string |
| `confidence` | string | `high`, `medium`, or `low` |
| `evidence_boundary` | string | `public`, `deidentified`, `internal`, `mixed`, or `unknown` |
| `updated_at` | string | ISO date |

## Card Types

- `source`: original evidence or reference material.
- `task`: an experimental or search task that CARE can support.
- `dataset`: a benchmark, dataset, or candidate data source.
- `paper`: a paper or technical report.
- `mechanism`: a system mechanism, architecture component, data boundary, or
  audit mechanism.
- `skill`: a reusable scientific decision skill.
- `decision`: an accepted design decision or constraint.
- `open_question`: an unresolved scientific, data, or engineering question.
- `action`: a concrete next step.

## RAG Integration

The SQLite database includes a `cards_fts` FTS5 table. For vector retrieval, use
`title + summary + content + tags` as the chunk text and keep `type`,
`priority`, `status`, and `evidence_boundary` as metadata.
