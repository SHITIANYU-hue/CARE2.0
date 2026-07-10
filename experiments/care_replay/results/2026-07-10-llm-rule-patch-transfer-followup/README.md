# LLM Rule-Patch Transfer Follow-Up

This follow-up tests a narrower way to increase LLM transfer gain. Earlier LLM
transfer modes asked the model to score or adjust target candidates directly;
those modes were noisy and stayed below the fixed deterministic
`transfer_gate_v1` rule. The change here moves the LLM one level up: the LLM
patches the transferable skill/rule once per seed, and deterministic CARE code
executes that patch under the existing gate.

## Setup

- Source: `real_suzuki_miyaura`
- Target: `real_buchwald_hartwig`
- Source observations: 96
- Target replay: 5 initial observations, 10 reveal rounds
- Model: `openai/gpt-5.5` through the CommonStack-compatible endpoint
- Hidden target outcomes are not included in LLM prompts. The LLM receives the
  source-to-target transfer card and revealed target evidence only.

## What Changed

Three LLM rule-patch variants were tested.

1. `llm_rule_patch_transfer_gate_v1`
   - One LLM call per seed.
   - LLM can tune support threshold, effect threshold, signal cap, aggregation,
     negative-signal policy, and role weight multipliers.
   - Result: mostly small ligand reweighting; underperformed fixed transfer.

2. `llm_rule_patch_interaction_gate_v1`
   - Adds LLM-selected transferable role interactions, such as
     `ligand-base` and `ligand-aryl_halide`.
   - The code estimates value-pair effects from revealed target rows only.
   - Result: better exploration/AUC in some runs, but too many bad
     interventions when sparse pair evidence acts alone.

3. `llm_rule_patch_guarded_interaction_gate_v1`
   - Same interaction idea, but guarded: a pair-interaction signal only applies
     when a single-field role-transfer signal agrees in direction.
   - Result: the first LLM mode in this line that beats fixed transfer on both
     final best and AUC in the 5-seed check, and keeps a smaller positive edge in
     the 10-seed check.

## Main Results

5-seed guarded interaction check:

| Mode | Final best | AUC | Bad interventions | LLM calls / seed |
| --- | ---: | ---: | ---: | ---: |
| `incumbent` | 85.7591 | 80.4911 | 0.0 | 0.0 |
| `transfer_gate_v1` | 91.5394 | 81.1316 | 0.4 | 0.0 |
| `llm_rule_patch_interaction_gate_v1` | 90.6576 | 82.7840 | 1.2 | 1.0 |
| `llm_rule_patch_guarded_interaction_gate_v1` | 91.9498 | 82.4969 | 0.8 | 1.0 |

10-seed robustness check:

| Mode | Final best | AUC | Bad interventions | LLM calls / seed |
| --- | ---: | ---: | ---: | ---: |
| `incumbent` | 86.6260 | 81.9846 | 0.0 | 0.0 |
| `transfer_gate_v1` | 90.0980 | 82.9136 | 1.1 | 0.0 |
| `llm_rule_patch_guarded_interaction_gate_v1` | 90.4106 | 83.7966 | 1.7 | 1.0 |

The gain is real but modest: in 10 seeds, guarded interaction improves final best
by +0.3126 and AUC by +0.8830 over fixed `transfer_gate_v1`. The trade-off is
more bad interventions, so the current result should be presented as a promising
direction rather than a finished win.

## Interpretation

The useful LLM role is not direct acquisition scoring. Direct candidate-level
LLM transfer remains too brittle. The more promising pattern is:

1. CARE compiles a role-level source-to-target transfer card.
2. The LLM acts as a skill optimizer and proposes mechanistic role interactions.
3. Deterministic target evidence computes the actual direction and magnitude.
4. The gate only allows bounded interventions.
5. The guarded version prevents sparse interaction evidence from overriding
   the base role-transfer signal by itself.

This gives us a clearer CARE 2.0 story: LLM transfer gain comes from evolving the
transferable skill, not from asking the model to guess hidden target outcomes.

## Files

- `llm_rule_patch_transfer_comparison.csv`: compact comparison across the rule
  patch runs.
- `tables/`: per-seed metrics CSVs.
- `runs/`: summary JSON files.
- `audits/`: 10-seed guarded interaction audit logs.
- `logs/`: LLM trace logs with model, timing, and token usage.

## Reproduction Command

```bash
COMMONSTACK_API_KEY=... \
CARE_LLM_TRACE_LOG=experiments/care_replay/outputs/logs/rule_patch_guarded_interaction_openai_gpt-5_5_suzuki_to_bh_10seed_calls.jsonl \
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_suzuki_miyaura \
  --target-dataset real_buchwald_hartwig \
  --source-observations 96 \
  --seeds 10 \
  --rounds 10 \
  --initial 5 \
  --modes incumbent,transfer_gate_v1,llm_rule_patch_guarded_interaction_gate_v1 \
  --llm-model openai/gpt-5.5 \
  --llm-max-tokens 1200 \
  --output-tag rule_patch_guarded_interaction_openai_gpt-5_5_suzuki_to_bh_10seed
```

## Next Step

The next useful experiment is to reduce the bad-intervention cost without losing
the AUC gain. Two obvious variants are worth testing next:

- damp interaction weights in guarded mode, for example multiply interaction
  contributions by 0.5;
- require interaction value-pairs to have support >= 2 after the first few
  rounds, while still allowing support 1 during the earliest sparse phase.
