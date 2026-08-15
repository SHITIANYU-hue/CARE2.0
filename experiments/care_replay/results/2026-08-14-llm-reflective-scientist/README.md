# CARE 2.0 evidence-bounded LLM scientist loop

This retrospective component study extends the frozen LLM initial-design study
with a second real LLM decision after the first three target experiments.

## Scientist loop

1. A strong LLM reads completed source outcomes and public target conditions,
   writes a falsifiable transfer hypothesis, and selects the initial design.
2. The selected experiments are executed and only those target outcomes are
   revealed to a reflection LLM.
3. The reflection LLM classifies the hypothesis as supported, mixed, falsified,
   or insufficient; writes an evidence interpretation; revises the claim; and
   proposes two follow-up experiments or stops transfer.
4. A frozen zero-loss acquisition gate checks each proposed follow-up against
   the target GP-UCB incumbent. Rejected proposals remain in the audit and
   knowledge base as negative evidence.
5. Target-only GP-UCB spends the remaining reveal budget. Every method uses the
   same total number of target experiments.

Neither LLM stage sees outcomes for unexecuted target candidates. Full prompts,
raw model responses, normalized decisions, hashes, API traces, per-candidate
audits, and aggregate metrics are retained in this directory.

## Results

Across five targets, AUC delta versus fixed v2 was `+2.123392` on average with
a task-level 95% interval of `[-0.225904, +4.472688]`. Three tasks improved, one
tied, and one remained negative. Final-best performance was non-inferior on all
five tasks. The reflection LLM marked three hypotheses mixed and two falsified,
and stopped unsafe transfer on one task.

The strict zero-loss gate rejected all eight executable follow-up proposals.
This preserved the earlier routed result but means the new reflection stage has
not yet demonstrated an independent aggregate performance gain. A retrospective
`0.1` gate sensitivity admitted proposals but damaged Suzuki performance, so it
is retained only as a negative sensitivity result and is not the deployed route.

## Evidence boundary

These targets were used by earlier project development. The model calls are
real and the prompts are evidence-bounded, but the study is not fresh external
confirmation. The next confirmation must freeze the scientist prompt, initial
route, gate, target roots, and metrics before new target outcomes are available.

## Key files

- `aggregate/suite_summary.json`: five-target machine-readable summary.
- `aggregate/suite_metrics.csv`: one row per target.
- `<target>/llm_reflection_record.json`: prompt, raw response, revised hypothesis,
  follow-up decision, and fingerprint.
- `<target>/api_trace.jsonl`: real API call trace.
- `<target>/evaluation/`: strict-gate metrics and per-candidate audit trail.
- `<target>/evaluation_gate_0p10/`: retained sensitivity runs where present.
