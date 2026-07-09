# LLM Prompt Follow-up

Date: 2026-07-09

This follow-up responds to the team feedback that the LLM output/prompt was too
close to generic AI-review language. The goal was not to make the prose nicer,
but to make the LLM behave more like a bounded CARE transfer-policy proposer:
it should propose candidate rules from the transfer card and revealed target
evidence, while the replay code verifies direction, support, and magnitude.

## Setup

- Source: `real_suzuki_miyaura`
- Target: `real_buchwald_hartwig`
- Source observations: 96
- Seeds: 0-4
- Initial target observations: 5
- Replay rounds: 10
- LLM endpoint: OpenAI-compatible CommonStack endpoint
- Model: `openai/gpt-5.5`

The API key was supplied through the environment and is not stored in the
repository.

## What Changed

The first prompt revision made the instruction more domain-aware and added a
parser guard so `prefer` must match a positive revealed target effect and
`penalize` must match a negative revealed target effect.

That long prompt was too verbose. It reduced some bad interventions, but it
also increased JSON parse errors. A shorter prompt then removed most of the
review-style language and asked for small experiment-policy adjustments only.

The useful change was the v3 calibration step. In v3, the LLM no longer gets to
directly add its requested weight to the acquisition score. Instead:

- the LLM proposes a small set of target factor rules;
- the parser checks support, sign, and effect direction against revealed target
  observations;
- the requested weight is treated only as an upper bound;
- the final magnitude is recalibrated from target support, target effect size,
  and transfer-card role weight;
- if a candidate matches several LLM rules, the signals are averaged instead of
  blindly summed.

This is closer to the CARE 2.0 direction we want: LLM as rule proposer, replay
and gate as verifier.

I also added a small reliability fix to the LLM client: `socket.timeout` is now
treated as retryable, so long LLM experiments are less likely to fail from one
slow API response.

## Main Result

| Run | Mode | Final best | AUC | Interventions | Bad interventions | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 |
| fixed baseline | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 |
| old prompt | `llm_transfer_gate_v1` | 87.1667 | 80.5335 | 2.8000 | 1.0000 | 0.4000 |
| v2 long prompt | `llm_transfer_gate_v1` | 84.4245 | 80.5090 | 2.2000 | 0.8000 | 2.8000 |
| v2 lite prompt | `llm_transfer_gate_v1` | 83.8706 | 80.4315 | 3.6000 | 1.4000 | 0.6000 |
| v3 calibrated, 700 tok | `llm_transfer_gate_v1` | 86.1182 | 80.8906 | 1.8000 | 0.6000 | 1.4000 |
| v3 strict, 700 tok | `llm_transfer_strict_gate_v1` | 85.7591 | 80.5356 | 0.8000 | 0.4000 | 2.4000 |
| v3 calibrated, 1200 tok | `llm_transfer_gate_v1` | 83.7647 | 79.3245 | 2.2000 | 0.8000 | 0.0000 |

The v3 calibrated prompt is the best safety/AUC trade-off among the new LLM
variants. It reduces bad interventions compared with the old prompt and has
the best LLM-proposer AUC in this follow-up. However, it still does not beat
the deterministic `transfer_gate_v1`, which remains the strongest result on
these five seeds.

Increasing `max_tokens` to 1200 eliminates parse errors in this run, but it
does not improve the scientific policy result. That means the remaining issue
is not just JSON truncation. The LLM is still not choosing transfer rules as
well as the deterministic card.

## Interpretation

The prompt feedback was useful, but prompt tuning alone is not enough. The LLM
can read the transfer card and propose plausible factor rules, but early target
evidence is sparse, and the model still overreacts to noisy small-sample
patterns. When we make it strict, it becomes safe but mostly collapses back to
incumbent behavior.

The next version should move the LLM one level up:

- propose rule artifacts, not direct candidate score changes;
- search over support thresholds, strictness, role weights, and descriptor
  whitelists;
- validate those proposals on calibration seeds;
- report held-out performance before adding a rule to the skill library.

In other words, the next step is rule evolution / policy selection, not another
round of prompt wording tweaks.

## Files

| Path | Contents |
| --- | --- |
| `llm_prompt_comparison.csv` | Compact aggregate comparison across old prompt, v2, v3, strict, and 1200-token retry runs |
| `tables/` | Per-seed metric CSVs for each new run |
| `summaries/` | Aggregate JSON summaries for each new run |
| `audit_summaries/` | Compact per-round audit summaries for LLM modes |
| `logs/` | LLM call traces with model name, timing, token usage, and retry status |

