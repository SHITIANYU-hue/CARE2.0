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
- `docs/NATURE_SUBMISSION_READINESS.md`: the current journal-positioning,
  claim boundary, reviewer-risk, and evidence-completion plan.
- `CARE2.0_Nature_evidence_update_2026-08-23.pptx`: a 13-slide reviewer-facing
  update built from the frozen online-LLM traces and route-level audit.
- `CARE2.0_Nature_evidence_update_2026-08-23_speaker_notes_CN.md`: detailed
  Chinese notes matched page by page to the evidence update.
- `docs/OVERLEAF_CASE_STUDIES_DRAFT.tex`: manuscript-ready analyses of one
  molecular-property success, one materials success, and one negative route.
- `experiments/care_replay/configs/online_llm_repeated_confirmation_v1.json`:
  the frozen 11-route, 30-trajectory-per-route confirmation protocol and its
  predeclared hierarchical analysis rule.
- `experiments/care_replay/rule_provenance.md`: provenance notes for the
  current incumbent, transfer rules, LLM roles, and incumbent ablation.
- `experiments/care_replay/skill_transfer_layers.md`: layered CARE 2.0 skill
  transfer design, covering representation, mechanism, model, acquisition,
  gate/risk, and workflow transfer.

The repository contains several evidence tiers that must not be conflated.
Synthetic adapters validate interfaces and audit flow; public MoleculeNet,
Matbench, ChemLex, and reaction HTE datasets support matched finite-pool
evaluation; frozen multi-seed selector studies test deterministic transfer
components; and the latest online Opus suite tests an LLM that revises a
hypothesis and selects every next experiment. The latest online-LLM result is
descriptive rather than statistically confirmatory. Reproducing CARE paper
numbers still requires the original benchmark candidate tables and matched
evaluation setup.

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

- `experiments/care_replay/results/2026-08-22-submission-evidence-audit/`
- `experiments/care_replay/results/2026-08-16-opus48-generalization-evidence-v1/`
- `experiments/care_replay/results/2026-08-15-opus5-generalization-study/`
- `experiments/care_replay/results/2026-08-14-llm-reflective-scientist/`
- `experiments/care_replay/results/2026-07-28-evaluation-completion/`
- `experiments/care_replay/results/2026-07-27-materials-replication/`
- `experiments/care_replay/results/2026-07-27-bh-replication-smoke/`
- `experiments/care_replay/results/2026-07-24-source-outcome-transfer/`
- `experiments/care_replay/results/2026-07-23-source-evidence-extension/`
- `experiments/care_replay/results/2026-07-22-multidomain-llm-completion/`
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
- `experiments/care_replay/results/2026-07-03-surrogate-baselines/`
- `experiments/care_replay/results/2026-07-03-hybrid-surrogate-transfer/`
- `experiments/care_replay/results/2026-07-04-transfer-weighted-kernel/`
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
CARE 1.0 reproduction or a named literature baseline. The surrogate-baseline
snapshot adds dependency-free GP-UCB, GP-EI, and kNN-UCB comparisons against the
current 50-seed transfer results. The hybrid-surrogate snapshot then uses
GP-UCB as the incumbent acquisition and tests whether CARE transfer cards can
still improve that stronger optimizer. The transfer-weighted-kernel snapshot
goes one step further: CARE transfer-card role confidence reweights the GP-UCB
categorical kernel, producing a small positive acquisition-level result over
GP-UCB on Suzuki-to-Buchwald-Hartwig and FreeSolv-to-Lipophilicity replay.
Raw public data files, per-mode metrics, audit logs, and knowledge snapshots
are kept under `experiments/care_replay/`.

### Latest online-LLM evidence

The 2026-08-16 suite isolates the online controller by comparing it with the
same LLM-generated initial observations followed by target-only GP-UCB under
the same reveal budget. Opus has an executable choice on every online round.
Across 11 real source-target routes, the controller records six wins, two ties,
and three losses with mean best-so-far AUC delta `+1.0135`. The route-level
bootstrap 95% interval is `[-0.4433, +2.7821]`, and the exact two-sided sign-test
`p` value excluding ties is `0.5078`.

Five chemistry routes were added after the high-authority controller was
frozen. This extension records three wins, one tie, and one loss, but the mean
delta is only `+0.0212`, with bootstrap 95% interval
`[-0.5171, +0.4194]` and sign-test `p=0.6250`. The correct current conclusion is
therefore route-specific, majority-positive transfer evidence, not a universal
or statistically confirmatory cross-domain claim. The complete audit, vector
figures, SHA-256 files, and next-evidence requirements are in
`experiments/care_replay/results/2026-08-22-submission-evidence-audit/`.

This online result is separate from the July frozen-selector studies below.
Those studies use many paired seeds to test deterministic compiled transfer
policies; they do not establish the stochastic generalization of the online
LLM controller.

The 2026-07-28 evaluation-completion package consolidates the frozen seven-pair
suite into a strongest-BO comparison, a matched target-only LLM comparison,
paired iteration-efficiency plots, and a baseline/control coverage matrix. It
also provides the executed equations for LLM-generated kernel patches,
source-outcome priors, target calibration, expert routing, and gates. All
figures are rebuilt from archived JSON/CSV rather than manually entered values.

The 2026-07-24 snapshot is the first complete source-outcome transfer
confirmation. LLM-compiled role maps are combined with fixed measured source
histories to build neighbor, additive, interaction, initial-design, and kernel
priors. A calibration-only selector freezes transfer or an exact matched
target-only LLM fallback before 100 independent held-out seeds. Four of seven
real source-target paths are significantly positive on paired final-best plus
AUC; the other three reproduce the target-only policy exactly. The archive
retains the rejected raw routes, per-seed metrics, round-level traces, full
audits, and development screening results.

The 2026-07-23 extension is the preceding source-schema confirmation. It freezes LLM
skill identity before 300 paired held-out seeds and compares source-schema
transfer directly against matched target-only LLM routing. Four of five
source-target paths show a significant gain on final best or best-so-far AUC,
covering molecular properties, materials properties, and reaction HTE. The
negative Lipophilicity-to-FreeSolv result and the Phonons-to-Dielectric top-10
tradeoff are retained in the same report.

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
