# CARE 2.0 real-LLM initial-design study

This archive adds a real LLM decision to the CARE 2.0 source-outcome
warm-start pipeline. It is a retrospective component study, not a new external
confirmation.

## What the LLM does

The model receives:

- completed source outcomes and source-level statistics;
- the target schema and public candidate conditions;
- source-model ranks for an outcome-blind candidate shortlist;
- archived failure lessons, including the v1 diversity failure.

The model does **not** receive target outcomes. It returns one frozen JSON
hypothesis with three candidate IDs, their roles, a mechanism, confidence,
failure conditions, and an abstention decision. The record and SHA-256 digest
are written before target replay.

Two execution variants are audited:

1. `llm_hypothesis_initial_design`: execute the three LLM candidates directly.
2. `llm_hypothesis_compiled_initial_design`: retain one LLM semantic anchor,
   add the source-consensus anchor, and add a deterministic maximin probe inside
   the source-supported top-50% region.

After the three initial observations, both variants use the same target-only
GP-UCB implementation, kernel, candidate pool, and 10-reveal budget as fixed
v2 and the target-only controls.

## Real API calls

- Endpoint: CommonStack OpenAI-compatible API
- Model: `openai/gpt-5.4-2026-03-05`
- Primary five-target suite: 5 calls, 66,293 tokens total
- API keys are never stored in this archive
- Request timing and usage are preserved in each `api_trace.jsonl`
- Full prompts, raw responses, normalized hypotheses, and fingerprints are
  preserved in each `llm_hypothesis_record.json`

## Five-target component result

The exploratory task-structure route uses compiled LLM when only one completed
source exists and raw LLM when multiple completed sources exist. This route
does not read target outcomes, but it was formulated after inspecting component
behavior and therefore still requires fresh confirmation.

| Target | Routed LLM variant | AUC delta vs fixed v2 | Final-best delta |
| --- | --- | ---: | ---: |
| Suzuki MINLP2 | compiled | +1.938790 | 0 |
| Morpholine-AlPhos | raw | +5.202660 | 0 |
| Phenethylamine-AlPhos | raw | -0.946610 | 0 |
| Morpholine-tBuBrettPhos | raw | +4.422120 | 0 |
| Morpholine-tBuBrettPhos preliminary | raw | 0 | 0 |

Aggregate AUC delta versus fixed v2:

- task mean: `+2.123392`;
- task-level 95% CI: `[-0.225904, +4.472688]`;
- win rate: `3/5`;
- non-loss rate: `4/5`;
- final-best non-loss rate: `5/5`.

The mean search-efficiency gain is positive, but the confidence interval still
crosses zero. This is evidence that LLM semantic hypotheses can add value to
the fixed transfer rule; it is not yet evidence of general LLM superiority.

## Representative Suzuki trace

The first frozen LLM call proposed the hypothesis that the same-substrate
source supports high-temperature, high-loading `P2L1 XPhos Cl` conditions. It
selected candidate `024` as its semantic quality anchor and assigned 0.73
confidence. Direct execution concentrated too strongly in one catalyst family:

- raw LLM AUC: `91.819000`;
- fixed v2 AUC: `98.061210`.

The deterministic compiler preserved candidate `024`, added source anchor
`039`, and selected geometry probe `013` without using target outcomes. The
compiled design reached:

- compiled LLM AUC: `100.000000`;
- delta versus fixed v2: `+1.938790`;
- final best: `100%` for both methods.

This trace shows why CARE should not ask the LLM to replace the numerical
optimizer. The LLM supplies a semantic, falsifiable hypothesis; the compiler
enforces quality and geometric coverage.

## Negative and neutral cases

- Raw LLM underperformed fixed v2 on Suzuki MINLP2 and
  Phenethylamine-AlPhos.
- The compiled variant underperformed raw LLM on Morpholine-AlPhos.
- A second Suzuki prompt with an explicit diversity instruction still did not
  beat fixed v2 when executed raw; its full trace is retained under
  `suzuki_minlp2_diversity_compiled/`.
- The preliminary Morpholine-tBuBrettPhos task tied fixed v2 under the routed
  policy.

These cases are retained because they define the limits of raw LLM selection
and of a universal compiler.

## Evidence boundary

All five target datasets had been used previously by the project. The prompts
are outcome-blind, and each response is frozen before its replay, but the suite
must be described as a retrospective component ablation. The next confirmation
must freeze the prompt, compiler, route, task roots, and metric before executing
new target campaigns.

## Files

- `aggregate/suite_summary.json`: machine-readable aggregate.
- `aggregate/suite_metrics.csv`: one row per target.
- `<target>/llm_hypothesis_record.json`: prompt, raw response, normalized
  hypothesis, model usage, and frozen decision.
- `<target>/api_trace.jsonl`: real API timing and token trace.
- `<target>/evaluation/` or `evaluation_compiled/`: metrics, audit events, and
  comparisons.
- `../../scripts/run_llm_initial_design_hypothesis.py`: generation and replay.
- `../../scripts/summarize_llm_initial_design_suite.py`: suite aggregation.

## Reproduce

Set the API key in the environment, then generate a frozen hypothesis:

```bash
export COMMONSTACK_API_KEY="..."
export CARE_LLM_TRACE_LOG="experiments/care_replay/results/<run>/api_trace.jsonl"

python3 experiments/care_replay/scripts/run_llm_initial_design_hypothesis.py generate \
  --config experiments/care_replay/configs/baumgartner_multisource_warmstart_v2.json \
  --target-task real_baumgartner_suzuki_minlp2 \
  --source-tasks real_baumgartner_suzuki_minlp1 \
  --llm-model openai/gpt-5.4-2026-03-05 \
  --output experiments/care_replay/results/<run>/llm_hypothesis_record.json
```

Verify and evaluate the frozen record:

```bash
python3 experiments/care_replay/scripts/run_llm_initial_design_hypothesis.py evaluate \
  --config experiments/care_replay/configs/baumgartner_multisource_warmstart_v2.json \
  --hypothesis-record experiments/care_replay/results/<run>/llm_hypothesis_record.json \
  --output-dir experiments/care_replay/results/<run>/evaluation
```
