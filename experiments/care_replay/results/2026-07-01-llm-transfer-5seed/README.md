# LLM-in-the-loop BH -> Suzuki transfer replay

Date: 2026-07-01

This run adds actual LLM calls to the CARE 2.0 cross-domain transfer replay. The LLM sees the target revealed-observation evidence and the source-to-target transfer card, then proposes bounded target factor adjustments. The resulting adjustments are still passed through the replay gate.

## Setup

- Server: `420GP-253`
- Source dataset: `real_buchwald_hartwig`
- Target dataset: `real_suzuki_miyaura`
- Seeds: 5
- Initial target observations: 5
- Target reveal budget: 10 rounds
- LLM endpoint: CommonStack-compatible `/chat/completions`
- Model: `openai/gpt-4o-mini`
- Temperature: 0.0
- Max tokens: 500

## Run

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_buchwald_hartwig \
  --target-dataset real_suzuki_miyaura \
  --seeds 5 \
  --rounds 10 \
  --initial 5 \
  --source-observations 48 \
  --discount 0.65 \
  --modes no_care_random,incumbent,transfer_strict_gate_v1,llm_transfer_gate_v1,llm_transfer_strict_gate_v1 \
  --output-tag server_llm_transfer_5seed \
  --llm-max-tokens 500
```

## What the LLM did

The new `llm_transfer_gate_v1` and `llm_transfer_strict_gate_v1` modes call the LLM from round 3 onward, once each policy has at least 8 revealed target observations. Each LLM call receives:

- target factor evidence computed only from revealed target observations;
- the source-to-target transfer card;
- the role map from Buchwald-Hartwig roles to Suzuki roles;
- an output contract asking for factor-level score adjustments as JSON.

The audit logs include `llm_transfer_policy` with model name, token usage, raw response, parsed response, applied specs, rejected specs, and parse error status.

## Result

The LLM was called successfully: both LLM modes average 7 calls per seed and 0 parse errors.

The first LLM policy is not better than the rule-based strict transfer. Plain LLM transfer applies too broadly and creates more bad interventions. Strict LLM transfer narrows the candidate set and reduces bad interventions, but still trails the non-LLM strict rule on this 5-seed run.

Key aggregate numbers:

- `transfer_strict_gate_v1`: `final_best` 89.7358, `best_so_far_auc` 86.6963, `bad_intervention_count` 0.4, `llm_call_count` 0.0
- `llm_transfer_gate_v1`: `final_best` 87.2325, `best_so_far_auc` 85.1085, `bad_intervention_count` 2.8, `llm_call_count` 7.0
- `llm_transfer_strict_gate_v1`: `final_best` 87.9673, `best_so_far_auc` 85.1761, `bad_intervention_count` 1.4, `llm_call_count` 7.0

The interpretation is useful: actual LLM-in-the-loop transfer is now wired and auditable, but direct LLM factor proposals are currently noisier than the deterministic strict gate. The next improvement should use the LLM as a transfer-card auditor or explanation/reranking layer, while keeping deterministic safeguards for candidate authorization.

## Files

- `server_llm_transfer_5seed_metrics.csv`: per-seed metrics.
- `transfer_summary.csv`: compact aggregate table.
- Raw LLM audit logs and summary JSON are stored under `experiments/care_replay/outputs/runs/` with the `server_llm_transfer_5seed` tag.

