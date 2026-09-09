# RSI implementation audit — 2026-09-09

Audited repository: SHITIANYU-hue/CARE2.0, branch exploration-aware-llm,
base commit `4c6f19f8`. The older local checkout was at `7840681a`; this study
uses the fetched remote head. Existing untracked work was left intact.

## Implemented before this experiment

- `knowledge_base/self_update.py`: ingests completed experiments, fingerprints
  results, rebuilds retrieval indexes, stages candidate skills, and gates promotion.
- `knowledge_base/close_loop.py`: finalization receipt links experiment evidence,
  candidate/active cards, negative-transfer evidence, and next-task retrieval.
- `knowledge_base/retrieval.py`: public-only retrieval; skills must be active.
  A date cutoff is supported.
- `generate_llm_semantic_skills.py`: retrieves knowledge when a database exists,
  includes card IDs and context in the model prompt, archives raw model output,
  and compiles executable proposals. `--no-kb-retrieval` is an ablation hook.
- `run_online_llm_scientist.py`: within-trajectory observation and hypothesis
  revision, with GP and safety routing. This is not model-weight training.
- Archived LLM semantic skills and real measured finite pools are executable
  without regenerating a model response.

## Evidence gaps and limitations

1. Closed-loop unit tests verify ingestion and retrieval, not improved downstream
   performance. A retrieval receipt does not establish an RSI causal effect.
2. Promotion trusts manifest declarations plus card confidence; it does not
   independently reconstruct task disjointness. Same-task outcomes must not be
   represented as new-task confirmation merely by setting manifest booleans.
3. Runtime retrieval has a date cutoff but no enforced task-disjoint provenance
   filter. A future RAG comparison needs isolated frozen/updated database snapshots
   and explicit exclusion of evaluation-task evidence.
4. The online scientist and semantic generator are separate paths. The mere
   presence of a knowledge database does not demonstrate that every online
   controller receives and executes the resulting skills.
5. Existing results explicitly limit self-improvement claims. For example, the
   2026-08-14 reflective stage's strict gate rejected all eight executable
   proposals, so it did not demonstrate independent incremental efficacy.
   The 2026-08-15 cross-domain online study reports mean AUC -0.3223 versus
   same-initial GP, with route-specific successes and failures. These historical
   README claims were inspected, not independently rerun in this audit.
6. No safe LLM credential variable was available to this process. This run makes
   zero new model calls. Archived prompts and responses remain traceable by
   source path and SHA-256. No chat-history key is read or reused.

## New experiment boundary

This experiment adds a persistent, versioned skill-selection state and measures
its effect using fresh seed-disjoint finite-pool executions. It does not update
or activate global knowledge-base cards, generate new skills, run the existing
KB-to-LLM feedback path, train weights, or establish full recursive self-improvement.

The updated selector starts from the highest-confidence archived skill and
accumulates complete development feedback for every candidate skill. Three
updates use cumulative mean development AUC, retaining the incumbent on exact
ties. Evaluation seeds never enter selection. Saved states explicitly remain
experimental. All declared tasks, skills, seeds, losses, and warnings are kept.

The primary comparator is the same initial fixed skill (highest archived LLM confidence), not the separately implemented CARE fixed-v2 warmstart controller, with its selection state
frozen. Target GP still updates normally within every trajectory. Additional
controls freeze the first-generation selection, batch-select from the same final
feedback, shuffle skill-feedback assignments, and use a target-only GP. Batch
selection and cumulative updating are expected to coincide here; any gain over
the initial skill alone cannot demonstrate a distinct benefit of recursion.

Development spends extra offline evaluations: it is not free experience. The
same-feedback batch control has the same development budget. All deployment
comparisons share initial observations, target seeds, and 15 total reveals per
trajectory. No claim of end-to-end cost superiority is made.

Held-out seeds still use the same finite target pool, including previously
measured labels and the repository's objective normalization. Intervals quantify
seed variability conditional on these pools and the fixed archived LLM draws;
they do not quantify new-task, model-call, model-version, or wet-lab uncertainty.

## Remaining full-RSI experiment

Freeze task-disjoint development/confirmation routes and snapshot hashes, obtain
an environment-configured endpoint/model, then compare fixed KB, update-enabled
KB, no-memory and shuffled-memory arms with paired model-call replicates. Retain
all actual prompt/response, retrieved-card, proposed-skill, executed-action and
cost traces. Promotion must be supported by disjoint evidence. This is pending,
not silently replaced by the offline component result.
