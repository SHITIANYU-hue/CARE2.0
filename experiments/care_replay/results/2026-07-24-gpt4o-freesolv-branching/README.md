# GPT-4o-mini FreeSolv branching-skill transfer

This is a 30-calibration / 50-held-out extension using previously frozen
`gpt-4o-mini` records. The target-only LLM skill was
`llm_semantic_branching_effect_on_hydration`; the source-outcome patch record
was generated for ESOL -> FreeSolv. The target adapter is the real continuous
FreeSolv property table.

The target-only skill was stronger than the source-outcome route on the new
held-out range:

| Route | Final best | Best-so-far AUC | Composite |
| --- | ---: | ---: | ---: |
| Raw source-outcome router | 84.5302 | 77.0184 | 161.5486 |
| Matched target-only GPT-4o-mini skill | 85.8297 | 76.9630 | 162.7927 |

The calibration gate rejected source transfer because its paired deltas versus
the matched target-only skill were negative. This is not a positive transfer
result, but it is a useful model-control result: improving the target LLM
quality does not automatically make source transfer additive.

Artifacts include one summary, one per-seed metrics table, 530 audit JSONL
files, and `SHA256SUMS`.
