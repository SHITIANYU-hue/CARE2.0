# LLM Open-Policy Follow-up

Date: 2026-07-10

This run follows the prompt/training note in `CARE2_Prompt_Optimization_and_Training_Strategy.docx`. The point was to test whether the LLM is weak because the prompt is bad, or because the downstream code gives it too little control.

## What Changed

I added two new modes without changing the old calibrated LLM modes:

- `llm_transfer_open_gate_v1`
- `llm_audit_transfer_open_gate_v1`

The old LLM proposer asks for a `weight`, but the replay code then hard-clips and recalibrates that weight. In practice, the LLM mostly chooses direction, while Python decides the magnitude.

The new open proposer instead asks for:

- `causal_confidence`
- `potential_confounder`
- a short evidence reason

The downstream score now uses that causal confidence directly. It still checks that `prefer` matches a positive revealed target effect and `penalize` matches a negative revealed target effect, but it allows sparse target support when the transfer role is plausible. Multiple LLM signals are summed and clipped at `0.12` instead of averaged and clipped at `0.08`.

The new open auditor uses a risk-reward prompt. It can approve sparse-evidence exploration if the transferred mechanism is plausible and the acquisition downside is acceptable. The approval threshold is `0.5` instead of `0.6`.

## Setup

- Source: `real_suzuki_miyaura`
- Target: `real_buchwald_hartwig`
- Source observations: 96
- Seeds: 0-4
- Initial target observations: 5
- Replay rounds: 10
- Model: `openai/gpt-5.5`
- Endpoint: OpenAI-compatible CommonStack endpoint

The API key was passed through the environment and is not stored in the repo.

## Main Result

| Run | Mode | Final best | AUC | Interventions | Bad interventions | LLM calls | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fixed transfer | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 | 0.0000 |
| previous v3 LLM | `llm_transfer_gate_v1` | 83.7647 | 79.3245 | 2.2000 | 0.8000 | 7.0000 | 0.0000 |
| previous v3 audit | `llm_audit_transfer_gate_v1` | 84.4245 | 80.4021 | 0.8000 | 0.2000 | 3.0000 | 0.0000 |
| open proposer | `llm_transfer_open_gate_v1` | 84.4245 | 80.4021 | 1.6000 | 0.2000 | 7.0000 | 0.0000 |
| open proposer + open auditor | `llm_audit_transfer_open_gate_v1` | 86.6878 | 80.5840 | 1.2000 | 0.0000 | 8.2000 | 0.0000 |

## Readout

The open proposer did what we expected mechanically: it produced more active score changes, used sparse evidence, and had no parse errors with `gpt-5.5` and a 1200-token budget. But by itself it did not beat the old audit result or the incumbent.

The open auditor helped. It removed bad interventions on these five seeds and moved final best above incumbent by about `+0.93`. That is a real improvement over the old LLM path, but it is still far below the fixed `transfer_gate_v1` baseline, which is about `+5.78` over incumbent on final best.

So the current answer is:

The LLM can participate safely when we let its causal confidence affect the policy and add a risk-reward auditor. But simply "letting the LLM do more" is not enough to beat the deterministic transfer rule. The stronger next move is to make the LLM optimize or edit the transfer rule itself, rather than replace the fixed rule with direct candidate-level score nudges.

## Practical Notes

The run exposed a cost issue. Open audit calls can carry large target-evidence prompts; one audit prompt was around 17k prompt tokens. If we keep this direction, the auditor needs a compressed evidence view instead of the full revealed-evidence payload.

The audit logs also show that `llm_audit_transfer_open_gate_v1` made 6 approved interventions across 50 replay rounds and had zero bad interventions. The plain open proposer made 8 interventions and had 1 bad intervention.

## Files

| Path | Contents |
| --- | --- |
| `llm_open_policy_comparison.csv` | Compact comparison against the previous v3 prompt run and fixed baselines |
| `tables/` | Per-seed metrics for the open-policy run |
| `summaries/` | Aggregate JSON summary for the open-policy run |
| `audit_summaries/` | Per-round compact audit/proposer behavior |
| `logs/` | LLM call trace with model, timing, retry, and token usage metadata |

## Next Step

The next experiment should be rule-level LLM transfer:

1. Start from `transfer_gate_v1`, not from a blank LLM proposer.
2. Ask the LLM to propose edits to thresholds, role weights, strictness, and descriptor whitelists.
3. Validate those edits on calibration seeds.
4. Freeze the best rule and evaluate on held-out seeds.

That would answer Tianyu's point more directly: can the LLM improve a strong known baseline, not just generate another weaker baseline.
