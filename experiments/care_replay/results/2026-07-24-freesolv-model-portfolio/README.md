# Frozen LLM source-model portfolio control

This run tests a source-model portfolio on ESOL -> continuous FreeSolv while
holding the target-only policy fixed to a previously frozen GPT-4o-mini skill.
The source candidate is the DeepSeek-generated source-outcome patch portfolio;
the paired GPT-4o-mini source run is archived separately in
`../2026-07-24-gpt4o-freesolv-branching/`. Selection still uses only the
30-seed calibration range and the 50 held-out seeds remain untouched until
replay.

The DeepSeek source route was rejected by the calibration gate: its calibration
mean was positive, but fold stability and non-loss rate did not meet the frozen
requirements. On held-out seeds, the raw source route was below the matched
GPT-4o-mini target-only skill:

| Route | Final best | Best-so-far AUC | Composite |
| --- | ---: | ---: | ---: |
| Raw DeepSeek source route | 86.6203 | 81.1688 | 167.7891 |
| Matched GPT-4o-mini target-only | 88.0497 | 80.1261 | 168.1758 |

This is a negative model-portfolio control, not a positive transfer result.
It shows that model selection must be done jointly with source-target
compatibility; a stronger target LLM does not guarantee an additive source
prior.

The archive contains one summary, one metrics table, 530 audit JSONL files,
and `SHA256SUMS`.
