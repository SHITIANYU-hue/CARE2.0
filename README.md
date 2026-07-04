# CARE

This repository contains the initial public code release for CARE:
**Controlling LLM-Generated Policies through Auditable Review of Evidence in
Scientific Experimentation**.

Current contents:

- `experiments/care_replay/`: a lightweight finite-pool replay harness for
  testing CARE-style incumbent/challenger/gate loops across synthetic adapters
  and public HTE / molecular-property / materials datasets.
- `knowledge_base/`: a small SQLite + FTS knowledge-base prototype for CARE 2.0
  task, dataset, mechanism, and skill cards, with optional vector indexing.
- `task_tracker/`: current project tasks, owners, blockers, and next actions.
- `overview.md`: a narrative overview of the current replay experiments,
  datasets, results, and next steps.
- `experiments/care_replay/rule_provenance.md`: provenance notes for the
  current incumbent, transfer rules, LLM roles, and incumbent ablation.

The current replay experiments are smoke tests. Synthetic adapters validate the
interface and audit flow; public adapters let the same loop run on real measured
yields and molecular property targets. Reproducing CARE paper numbers still
requires the original benchmark candidate tables and matched evaluation setup.

## Branches

- `carry1.0`: stable snapshot of the current public replay harness and outputs.
- `carry2.0`: active branch for dataset expansion, transfer experiments,
  knowledge-base card extensions, and project task tracking.

## Current Project Priorities

The latest project sync focuses on three workstreams:

1. **Dataset expansion and experiments**
   - Add Kimi-Lex once the download link and schema are available.
   - Add a Materials Project pathway once access/API requirements are confirmed;
     the current public materials proxy is Matbench experimental band gap.
   - Keep public, release-safe replay datasets separate from private/internal
     data.
2. **Code and documentation**
   - Keep README, `overview.md`, and replay outputs aligned with the latest run.
   - Preserve full raw data, metrics, summaries, and per-seed audit logs under
     `experiments/care_replay/`.
3. **Knowledge base and collaboration**
   - Extend card types for experiment results, run logs, transfer hypotheses,
     dataset requests, and reviews.
   - Track dataset blockers, experiment ideas, and review ownership in
     `task_tracker/`.

## Current Experiment Outputs

The latest tracked replay output is in:

- `experiments/care_replay/results/2026-06-28-doc-guided-sweep/`
- `experiments/care_replay/results/2026-06-29-materials-baselines/`
- `experiments/care_replay/results/2026-06-30-generalization-sweep/`
- `experiments/care_replay/results/2026-06-30-llm-commonstack-5seed/`
- `experiments/care_replay/results/2026-06-30-real-chemlex/`
- `experiments/care_replay/results/2026-06-30-bh-to-suzuki-transfer/`
- `experiments/care_replay/results/2026-07-03-multidomain-transfer-feasibility/`
- `experiments/care_replay/results/2026-07-03-transfer-advantage-sweep/`
- `experiments/care_replay/results/2026-07-03-llm-transfer-followup/`
- `experiments/care_replay/results/2026-07-03-incumbent-rule-ablation/`
- `experiments/care_replay/outputs/`
- `experiments/care_replay/data/raw/`

The tracked output set includes the six-adapter 2026-06-28 sweep plus the
2026-06-29 material baseline pass with `no_care_random`, `incumbent`,
`no_gate`, `gate_v1`, and `gate_v2` modes, the nine-dataset 2026-06-30
non-LLM generalization sweep, and the nine-dataset CommonStack LLM replay using
`llm_no_gate` and `llm_gate_v1`. It now also includes a real ChemLex Acid-Amine
wetlab adapter from Zenodo, multi-domain transfer sweeps, the 50-seed transfer
advantage runs, and a 10-seed real-LLM transfer follow-up with LLM proposer and
LLM auditor modes. The incumbent-rule ablation is included to make clear that
the current incumbent is a strong transparent target-only control, not a
CARE 1.0 reproduction or a named literature baseline.
Raw public data files, per-mode metrics, audit logs, and knowledge snapshots
are kept under `experiments/care_replay/`.

## Quick Start

Run the replay smoke test:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 30 --rounds 10
```

Run a real LLM-in-the-loop replay:

```bash
export CARE_LLM_API_KEY="..."
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset all --seeds 5 --rounds 6 --initial 5 --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 --llm-model openai/gpt-4o-mini --output-tag llm_commonstack_5seed
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

This is an initial code push. Real ChemLex wetlab replay is now connected; the
next expected step is to replace the remaining synthetic-only controls with
real Minerva/Olympus candidate tables when those tables are available for
release.
