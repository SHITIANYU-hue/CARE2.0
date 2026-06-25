# CARE

This repository contains the initial public code release for CARE:
**Controlling LLM-Generated Policies through Auditable Review of Evidence in
Scientific Experimentation**.

Current contents:

- `experiments/care_replay/`: a lightweight finite-pool replay harness for
  testing CARE-style incumbent/challenger/gate loops.
- `knowledge_base/`: a small SQLite + FTS knowledge-base prototype for CARE 2.0
  task, dataset, mechanism, and skill cards.

The current replay experiment is a synthetic Suzuki-like smoke test. It is meant
to validate the software interface and audit flow, not to reproduce the CARE
paper numbers. Reproducing paper results requires the original benchmark
candidate tables.

## Quick Start

Run the replay smoke test:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --seeds 30 --rounds 10
```

Build the public seed knowledge base:

```bash
python3 knowledge_base/build_kb.py
python3 knowledge_base/query_kb.py gate --limit 5
```

Both scripts use only the Python standard library.

## Repository Status

This is an initial code push. The next expected step is to replace the synthetic
replay adapter with real Minerva/Olympus and ChemLex candidate tables when those
tables are available for release.
