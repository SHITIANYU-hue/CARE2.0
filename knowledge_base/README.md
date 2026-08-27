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

## Controlled self-update

Completed result directories can update the card store, SQLite/FTS database,
Markdown index, and embedding index in one transaction:

```bash
python3 knowledge_base/self_update.py \
  experiments/care_replay/results/<completed-run>
```

To discover every new or changed result manifest under the replay results
directory:

```bash
python3 knowledge_base/self_update.py --discover
```

The updater fingerprints each result, skips unchanged runs, records an audit
event, and rebuilds the local hashed embedding index by default. Discovery
supports standard replay manifests and the LLM initial-design suite's
`aggregate/suite_summary.json`; the latter is expanded into an aggregate result
card plus one candidate hypothesis card per target, including its trace path.
The reflective scientist suite is handled the same way, additionally preserving
hypothesis status, revised claims, stop decisions, and gate outcomes.
Use `--embedding-provider openai` only when a supported embedding endpoint and
`CARE_OPENAI_API_KEY` are configured.

New `skill` cards are staged as `candidate` and are excluded from runtime RAG.
Automatic evidence ingestion does not imply automatic scientific validation.
A skill can become `active` only when the run manifest contains all four
`knowledge_update` confirmations below and the operator supplies
`--allow-promotion`:

```json
{
  "knowledge_update": {
    "allow_skill_promotion": true,
    "task_disjoint_confirmation": true,
    "protocol_frozen_before_evaluation": true,
    "external_outcomes_not_used_during_selection": true
  }
}
```

State and append-only audit records are written under
`knowledge_base/self_update/`. The API function `self_update.self_update(...)`
can also be called by an experiment runner after it has successfully finalized
its result directory.

## Closed experiment-to-skill loop

Use the loop finalizer when an experiment is complete:

```bash
python3 knowledge_base/close_loop.py \
  experiments/care_replay/results/<completed-run>
```

The finalizer performs four linked operations in one run:

1. converts the frozen result and audit provenance into evidence and skill cards;
2. activates only high-confidence skills whose manifest declares task-disjoint
   confirmation, a pre-frozen protocol, and outcome-safe selection;
3. rebuilds FTS and embedding indexes, while retaining failed transfer as
   negative evidence;
4. writes `knowledge_feedback.json` into the result directory to show exactly
   what a subsequent task can retrieve.

`generate_llm_semantic_skills.py` reads `knowledge_base/care_kb.sqlite` by
default. Thus an active skill or negative-transfer lesson from run N enters the
prompt context of run N+1. Use `--no-kb-retrieval` only for a controlled
no-memory ablation. Use `close_loop.py --stage-only` to archive all new skills
without activating them.

Hypothesis-only LLM proposals and their zero-shot replay evidence use a separate
ingester so the scientific claim, mechanism, failure conditions, compiler
boundary, matched random-null comparison, and result provenance remain visible:

```bash
python3 knowledge_base/ingest_hypothesis_transfer.py \
  --report experiments/care_replay/results/2026-07-25-zero-shot-hypothesis-transfer-matrix/zero_shot_transfer_matrix.json \
  --record experiments/care_replay/results/2026-07-25-hypothesis-generation/suzuki_to_bh_hypothesis_record_commonstack.json \
  --output knowledge_base/generated_cards/2026-07-25-zero-shot-hypothesis-transfer.json
python3 knowledge_base/build_kb.py
```

These cards deliberately use `candidate` or `needs_verification` for mechanism
claims. A generated hypothesis is not treated as established domain knowledge
just because it improved one replay.

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
supports an experiment-date cutoff. A skill must be explicitly `active`; legacy
`done`, candidate, and needs-verification skills are not returned to runtime
agents. Model-call records retain the retrieved card IDs so every prompt can be
reconstructed.

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

For a local semantic embedding model, install `sentence-transformers` and use:

```bash
python3 knowledge_base/build_embeddings.py \
  --provider sentence_transformers \
  --model sentence-transformers/all-MiniLM-L6-v2
python3 knowledge_base/query_embeddings.py \
  "negative transfer reaction representation" --limit 5
```

The JSONL index records the provider and model for every vector, so the query
path cannot silently mix hashed, API, and local-model embeddings.

## Card Model

See `schema.md` for the card fields and supported card types.

## Source Notes

- `source_notes/awesome_resources.md`: public awesome-list review for chemistry,
  molecular discovery, materials-aware LLMs, and EDA-style verification loops.
