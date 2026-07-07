# Strong LLM Model Follow-up

Date: 2026-07-06

This follow-up checks whether the weak reaction-transfer LLM result was mainly
caused by using a small model. The experiment keeps the replay task, prompt,
gate, transfer card, seeds, and evaluation metrics fixed, and changes only the
LLM model behind the OpenAI-compatible CommonStack endpoint.

## Setup

Direction:

- Source: `real_suzuki_miyaura`
- Target: `real_buchwald_hartwig`
- Source observations: 96
- Seeds: 0-4
- Initial target observations: 5
- Replay rounds: 10

Modes:

- `no_care_random`
- `incumbent`
- `transfer_gate_v1`
- `llm_transfer_gate_v1`
- `llm_audit_transfer_gate_v1`

Models compared:

- `openai/gpt-4o-mini` first five seeds from the earlier 10-seed run
- `openai/gpt-5.5`
- `deepseek/deepseek-v3.2`

The API key was supplied through the environment and is not stored in the
repository.

## Summary

| Model | Mode | Final best | AUC | Interventions | Bad interventions | LLM calls | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `openai/gpt-4o-mini` first5 | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `openai/gpt-4o-mini` first5 | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 | 0.0000 |
| `openai/gpt-4o-mini` first5 | `llm_transfer_gate_v1` | 86.0649 | 80.9313 | 5.0000 | 2.0000 | 7.0000 | 0.0000 |
| `openai/gpt-5.5` | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `openai/gpt-5.5` | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 | 0.0000 |
| `openai/gpt-5.5` | `llm_transfer_gate_v1` | 87.1667 | 80.5335 | 2.8000 | 1.0000 | 7.0000 | 0.4000 |
| `deepseek/deepseek-v3.2` | `incumbent` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `deepseek/deepseek-v3.2` | `transfer_gate_v1` | 91.5394 | 81.1316 | 4.0000 | 0.4000 | 0.0000 | 0.0000 |
| `deepseek/deepseek-v3.2` | `llm_transfer_gate_v1` | 85.7591 | 80.4911 | 0.0000 | 0.0000 | 7.0000 | 3.8000 |

Full aggregate rows, including random and LLM-audit modes, are in
`strong_llm_model_comparison.csv`.

## Interpretation

Changing the model does matter. On the same five seeds, `openai/gpt-5.5`
improves the LLM proposer over the earlier `openai/gpt-4o-mini` run:

- final best increases from 86.0649 to 87.1667;
- bad interventions drop from 2.0000 to 1.0000;
- intervention count drops from 5.0000 to 2.8000, so the model is less noisy.

This is a useful signal that LLM quality affects transfer-policy quality.
However, the result is still not strong enough to replace the deterministic
transfer card. The fixed `transfer_gate_v1` remains much better on final best:
91.5394 versus 87.1667 for `gpt-5.5` LLM transfer.

`deepseek/deepseek-v3.2` is not a good fit for the current long-context
tool-call interface. The endpoint itself stayed alive, but about half of the
calls did not produce usable tool-call JSON in the full prompt setting. As a
result, the policy mostly collapsed back to incumbent behavior. The issue looks
like interface compatibility under long prompts, not a normal scientific-policy
failure.

The conclusion is therefore more precise than "use a bigger model": stronger
models can improve the current LLM proposer, but the current interface is still
too low-level. The LLM is being asked to emit small factor adjustments inside a
handwritten policy. The next experiment should move the LLM up one level and
ask it to propose a bounded rule update, for example:

- role-map edits;
- transfer-weight caps;
- support thresholds;
- strict versus non-strict gate selection;
- descriptor whitelist / blacklist;
- audit strictness.

Those proposed rule artifacts should then be evaluated by replay and held-out
seeds, rather than trusted directly.

## Files

- `strong_llm_model_comparison.csv`: compact model and mode comparison.
- `tables/`: per-seed metric CSVs for the new strong-model runs.
- `runs/`: summary JSON, audit JSONL, knowledge snapshots, and transfer cards.
- `logs/`: call-level trace logs with model name, timing, token usage, retry
  status, and tool-call counts. The trace logs do not store API keys.

