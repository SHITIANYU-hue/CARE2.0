# LLM Unlocked-Policy Follow-up

Date: 2026-07-10

This follow-up tests the obvious next question after the open-policy run: what happens if we loosen the LLM even more?

The short answer is that direct candidate-level unlocking did not help. It made the system more active, but also less stable and less safe.

## What Changed

I added three more permissive modes:

- `llm_transfer_unlocked_gate_v1`
- `llm_transfer_unlocked_no_gate_v1`
- `llm_audit_transfer_unlocked_gate_v1`

Compared with `llm_transfer_open_gate_v1`, the unlocked modes:

- start proposing at 5 revealed target observations instead of 8;
- allow up to 6 factor-level adjustments per round;
- allow `policy_weight` up to `0.20`;
- clip summed policy signal at `0.20` instead of `0.12`;
- allow weak direction overrides when `causal_confidence` and `exploration_value` are high;
- include a no-gate mode to estimate the upper bound of direct LLM scoring;
- include a permissive auditor with approval threshold `0.45`.

The LLM still only sees revealed target evidence, public candidate metadata, and the transfer card. Hidden target outcomes are not exposed.

## Setup

- Source: `real_suzuki_miyaura`
- Target: `real_buchwald_hartwig`
- Source observations: 96
- Seeds: 0-4
- Initial target observations: 5
- Replay rounds: 10
- Model: `openai/gpt-5.5`
- Max tokens: 1200
- Endpoint: OpenAI-compatible CommonStack endpoint

The API key was passed through the environment and is not stored in the repo.

## Main Result

| Run | Mode | Final best | AUC | Interventions | Bad interventions | LLM calls | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| fixed transfer | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 | 0.0000 |
| open proposer | `llm_transfer_open_gate_v1` | 84.4245 | 80.4021 | 1.6000 | 0.2000 | 7.0000 | 0.0000 |
| open proposer + open auditor | `llm_audit_transfer_open_gate_v1` | 86.6878 | 80.5840 | 1.2000 | 0.0000 | 8.2000 | 0.0000 |
| unlocked proposer + gate | `llm_transfer_unlocked_gate_v1` | 85.8218 | 79.4702 | 1.0000 | 0.6000 | 10.0000 | 2.4000 |
| unlocked proposer, no gate | `llm_transfer_unlocked_no_gate_v1` | 84.8214 | 80.0368 | 1.2000 | 0.6000 | 10.0000 | 2.8000 |
| unlocked proposer + permissive auditor | `llm_audit_transfer_unlocked_gate_v1` | 84.0181 | 76.1951 | 2.4000 | 1.4000 | 12.4000 | 1.8000 |

## Readout

This result is useful because it rules out a tempting hypothesis.

The previous open-policy run suggested that the LLM might be too constrained. I loosened the proposer and auditor further, but performance did not improve. The no-gate unlocked mode also did not reveal a hidden upside. It was not the gate alone holding the LLM back.

The unlocked modes made more aggressive proposals, started earlier, and affected more candidate scores. But they also produced more bad interventions and more empty/invalid LLM responses. The permissive auditor made this worse, not better.

So the current evidence says:

Directly relaxing candidate-level LLM scoring is not the right path. The LLM should move up one level and edit the transfer policy/rule, while deterministic replay evaluates whether that edited rule is actually better.

## Practical Notes

The unlocked run was much more expensive. It took roughly 45 minutes for this 5-seed sweep because unlocked proposer and audit calls were frequent and prompts were large. Several calls carried around 16k-17k prompt tokens. Some parse errors were empty final responses, likely because the model spent completion budget on internal reasoning and did not emit JSON.

That means future LLM experiments should use a compressed target-evidence view and a stricter response format before scaling seeds.

## Files

| Path | Contents |
| --- | --- |
| `llm_unlocked_policy_comparison.csv` | Compact comparison against open-policy and fixed baselines |
| `tables/` | Per-seed metrics for the unlocked-policy run |
| `summaries/` | Aggregate JSON summary for the unlocked-policy run |
| `audit_summaries/` | Per-round compact audit/proposer behavior |
| `logs/` | LLM call trace with model, timing, retry, and token usage metadata |

## Next Step

The next experiment should stop loosening direct candidate scoring and instead test rule-level LLM transfer:

1. Give the LLM the existing `transfer_gate_v1` rule and recent audit failures.
2. Ask it to propose a small patch to thresholds, role weights, direction handling, and descriptor whitelists.
3. Evaluate the proposed rule on calibration seeds.
4. Freeze the best rule and test on held-out seeds.

That would directly answer whether LLM can improve the strong baseline, rather than asking it to replace the baseline with noisier candidate-level suggestions.
