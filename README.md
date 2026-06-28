# CARE

This repository contains the initial public code release for CARE:
**Controlling LLM-Generated Policies through Auditable Review of Evidence in
Scientific Experimentation**.

Current contents:

- `experiments/care_replay/`: a lightweight finite-pool replay harness for
  testing CARE-style incumbent/challenger/gate loops across synthetic adapters
  and public HTE / molecular property datasets.
- `knowledge_base/`: a small SQLite + FTS knowledge-base prototype for CARE 2.0
  task, dataset, mechanism, and skill cards, with optional vector indexing.

The current replay experiments are smoke tests. Synthetic adapters validate the
interface and audit flow; public adapters let the same loop run on real measured
yields and molecular property targets. Reproducing CARE paper numbers still
requires the original benchmark candidate tables and matched evaluation setup.

## Quick Start

Run the replay smoke test:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 30 --rounds 10
```

Build the public seed knowledge base:

```bash
python3 knowledge_base/build_kb.py
python3 knowledge_base/query_kb.py gate --limit 5
python3 knowledge_base/build_embeddings.py --provider hashed
python3 knowledge_base/query_embeddings.py "dataset gate" --limit 5
```

The default embedding path uses a local hashed vectorizer so the repository can
run without external dependencies. To use an OpenAI-compatible embedding
endpoint, set `CARE_OPENAI_API_KEY`, `CARE_OPENAI_BASE_URL`, and
`CARE_EMBEDDING_MODEL`, then run `build_embeddings.py --provider openai`.

## Repository Status

This is an initial code push. The next expected step is to replace the synthetic
replay adapter with real Minerva/Olympus and ChemLex candidate tables when those
tables are available for release.
