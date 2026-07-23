# CARE Knowledge Base Prototype

This directory contains a small, auditable knowledge-base prototype for CARE
2.0. The first version uses JSON cards as source data, SQLite as the local
storage layer, SQLite FTS5 for text search, and optional vector indexes for
embedding search.

The public seed file intentionally contains only public, release-safe cards.
Internal chat logs, private notes, and generated SQLite databases are not
tracked in this repository.

Replay result snapshots can be converted into public cards automatically. The
ingester creates `experiment_result`, reusable `skill`, negative-transfer, and
`run_log` cards while preserving the result directory as provenance:

```bash
python3 knowledge_base/ingest_experiment_results.py \
  experiments/care_replay/results/2026-07-21-llm-evidence-causality
python3 knowledge_base/build_kb.py
```

Generated card JSON is tracked under `knowledge_base/generated_cards/`.
SQLite, Markdown exports, and embedding indexes remain reproducible generated
artifacts and are ignored by git.

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

`retrieval.py` is the runtime API used by the LLM skill generator. Runtime
retrieval accepts only public `skill`, `transfer`, and `mechanism` cards and
supports an experiment-date cutoff. Model-call records retain the retrieved
card IDs so every prompt can be reconstructed.

## Embeddings

Build a local development vector index without any API call:

```bash
python3 knowledge_base/build_embeddings.py --provider hashed
python3 knowledge_base/query_embeddings.py "reaction optimization dataset" --limit 5
```

Use an OpenAI-compatible embedding endpoint:

```bash
export CARE_OPENAI_API_KEY="..."
export CARE_OPENAI_BASE_URL="https://your-endpoint/v1"
export CARE_EMBEDDING_MODEL="your-embedding-model"
python3 knowledge_base/build_embeddings.py --provider openai
```

The API key is never stored in the repository. A chat-only model endpoint is not
enough for this path; the endpoint must support `/v1/embeddings`.

## Card Model

See `schema.md` for the card fields and supported card types.

## Source Notes

- `source_notes/awesome_resources.md`: public awesome-list review for chemistry,
  molecular discovery, materials-aware LLMs, and EDA-style verification loops.
