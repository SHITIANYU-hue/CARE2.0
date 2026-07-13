# LLM Prompt Risk-Capped Follow-Up

This snapshot tests the CommonStack `openai/gpt-5.5` path after adding a
prompt-optimized LLM rule-patch mode for Suzuki-Miyaura -> Buchwald-Hartwig
transfer.

The API key is not stored in the repository. Both 10-seed runs used real
CommonStack calls through the OpenAI-compatible endpoint.

## Question

The earlier LLM rule-patch result showed that guarded role interactions could
slightly beat the fixed deterministic transfer rule. The next question was
whether a more explicit prompt could make the LLM optimize the transfer skill
more aggressively and improve transfer gain.

## Runs

Aggressive prompt:

```bash
export CARE_LLM_API_KEY="..."
export CARE_LLM_TRACE_LOG="experiments/care_replay/outputs/logs/rule_patch_prompt_optimized_openai_gpt-5_5_suzuki_to_bh_10seed_calls.jsonl"
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_suzuki_miyaura \
  --target-dataset real_buchwald_hartwig \
  --source-observations 96 \
  --seeds 10 \
  --rounds 10 \
  --initial 5 \
  --modes incumbent,transfer_gate_v1,llm_rule_patch_guarded_confirmed_interaction_gate_v1,llm_rule_patch_prompt_optimized_confirmed_gate_v1 \
  --llm-model openai/gpt-5.5 \
  --llm-max-tokens 1200 \
  --output-tag rule_patch_prompt_optimized_openai_gpt-5_5_suzuki_to_bh_10seed
```

Risk-capped prompt:

```bash
export CARE_LLM_API_KEY="..."
export CARE_LLM_TRACE_LOG="experiments/care_replay/outputs/logs/rule_patch_prompt_optimized_risk_capped_openai_gpt-5_5_suzuki_to_bh_10seed_calls.jsonl"
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_suzuki_miyaura \
  --target-dataset real_buchwald_hartwig \
  --source-observations 96 \
  --seeds 10 \
  --rounds 10 \
  --initial 5 \
  --modes incumbent,transfer_gate_v1,llm_rule_patch_guarded_confirmed_interaction_gate_v1,llm_rule_patch_prompt_optimized_confirmed_gate_v1 \
  --llm-model openai/gpt-5.5 \
  --llm-max-tokens 1200 \
  --output-tag rule_patch_prompt_optimized_risk_capped_openai_gpt-5_5_suzuki_to_bh_10seed
```

## Result

| Run | Mode | Final best | AUC | Top-10 hit | Bad interventions | Parse errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| aggressive prompt | `transfer_gate_v1` | 90.0980 | 82.9136 | 0.20 | 1.10 | 0.00 |
| aggressive prompt | `llm_rule_patch_guarded_confirmed_interaction_gate_v1` | 90.6234 | 83.9926 | 0.10 | 1.40 | 0.00 |
| aggressive prompt | `llm_rule_patch_prompt_optimized_confirmed_gate_v1` | 88.4420 | 83.1054 | 0.00 | 2.30 | 0.00 |
| risk-capped prompt | `transfer_gate_v1` | 90.0980 | 82.9136 | 0.20 | 1.10 | 0.00 |
| risk-capped prompt | `llm_rule_patch_guarded_confirmed_interaction_gate_v1` | 91.1571 | 84.1105 | 0.10 | 1.60 | 0.00 |
| risk-capped prompt | `llm_rule_patch_prompt_optimized_confirmed_gate_v1` | 89.5885 | 83.8572 | 0.20 | 1.30 | 0.00 |

## Interpretation

The CommonStack key and structured LLM path work: all 20 calls in each 10-seed
run returned parseable rule patches.

The aggressive prompt made the LLM too intervention-seeking. It usually chose
`signal_cap=0.12`, two role interactions, and `negative_policy=allow`, which
raised bad interventions and hurt final best.

The risk-capped prompt repaired much of that behavior by capping `signal_cap` at
0.10, forcing at most one interaction, and converting `negative_policy=allow`
to `downweight`. This reduced bad interventions from 2.30 to 1.30 and improved
AUC over fixed transfer, but it still did not beat fixed `transfer_gate_v1` on
final best.

The best current LLM result remains the guarded confirmed interaction patch:
in the risk-capped rerun it reached final best 91.1571 and AUC 84.1105, versus
90.0980 and 82.9136 for fixed transfer. This suggests the promising direction
is not a more assertive prompt by itself, but a small guarded interaction
template with replay-level risk controls.

## Files

- `llm_prompt_risk_capped_comparison.csv`: compact comparison table.
- `tables/`: per-seed metrics for both 10-seed runs.
- `runs/`: aggregate summary JSON for both 10-seed runs.
