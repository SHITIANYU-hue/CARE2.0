# Exploration-aware LLM policy preflight

This note records the implementation and preflight checks prompted by the
July 20 review of CARE 2.0 traces. It deliberately separates executable-policy
validation from actual LLM evidence.

## Problem

Earlier traces showed a conservative greedy pattern:

- confidence was concentrated near `0.70`;
- weights were concentrated near `0.05`;
- most proposals were `prefer` adjustments for already successful factors;
- adjusted Top-1 often matched the target-only incumbent.

This made the LLM stable but gave it little opportunity to reduce uncertainty
or test a counter-hypothesis.

## Implementation

The new `llm_explore_no_gate` and `llm_explore_gate_v1` modes add:

1. public factor coverage, including unseen and low-support values without
   exposing hidden outcomes;
2. an exploration prompt with explicit `explore`, `exploit`, and `avoid`
   intents;
3. a concise decision summary containing a testable hypothesis,
   counter-hypothesis, uncertainty target, and evidence for/against;
4. evidence-calibrated confidence and non-uniform adjustment weights;
5. a seeded probabilistic exploration gate with a decaying risk budget;
6. Top-1 change, exploration, penalization, confidence, and gate metrics;
7. an executable GP-UCB beta schedule (`gp_beta` to `gp_beta_end`);
8. target-only leave-one-out validation before a transfer-kernel expert can
   enter the router.

The legacy LLM modes remain unchanged as comparison arms.

## Deterministic policy preflight

To test the execution path without attributing behavior to a model, a local
deterministic proposer generated the required exploration/exploitation/avoid
JSON. This is **not an LLM result**.

Dataset: real Buchwald-Hartwig, 30 seeds, 8 initial observations, 6 replay
rounds.

| Mode | Final Best | Best-so-far AUC | Top-1 change rate | Interventions/seed | Bad interventions/seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 86.5224 | 82.6605 | 0.0000 | 0.0000 | 0.0000 |
| Exploration + probabilistic gate | 87.7209 | 84.3693 | 0.8167 | 1.2667 | 0.6333 |
| Exploration, no gate | 85.2503 | 83.9165 | 0.8222 | 4.9333 | 2.2333 |

The prompt contract can therefore create a different challenger, and the gate
removes most harmful interventions. The preflight does not establish LLM
generalization because the proposer was deterministic.

## Strong-baseline preflight

A three-patch exploration portfolio was also tested against GP-UCB, GP-EI, a
target acquisition portfolio, and a fixed transfer ensemble. The portfolio was
manually specified to test the dynamic-beta and router mechanics; it was not
presented as model output.

For Suzuki-Miyaura to Buchwald-Hartwig (10 seeds), target leave-one-out
calibration rejected all candidate-changing transfer proposals. The router
therefore matched the target acquisition portfolio: Final Best `86.4313`, AUC
`81.7289`. Relative to GP-UCB, the deltas were `-0.5484` Final Best and
`+0.2525` AUC. This is a safe fallback, not a transfer advantage.

For FreeSolv to Lipophilicity (5 seeds), strict routing also matched the target
portfolio: Final Best `89.2000`, AUC `87.5500`. GP-UCB reached `88.7000` and
`87.3000`; the fixed transfer ensemble reached `89.4250` and `87.4208`.

## Live-model status

The CommonStack models endpoint authenticated and listed current models,
including `openai/gpt-5.6-sol`. The first inference request returned HTTP 429
with `Insufficient available balance: 0`. No live-model result is reported in
this snapshot.

## Interpretation

The July 20 diagnosis was valid: fixed prompt anchors and a deterministic gate
were suppressing exploration. The revised policy resolves the mechanical
failure and exposes whether LLM proposals actually change a ranking. It does
not yet prove that an LLM beats strong target-only baselines across domains.
That claim requires funded live calls, frozen prompts, multiple source-target
pairs, and held-out seeds.

## Next experiment

Once inference is available, freeze one prompt and run:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset real_buchwald_hartwig \
  --seeds 30 --rounds 6 --initial 8 \
  --modes incumbent,llm_gate_v1,llm_explore_gate_v1,llm_explore_no_gate \
  --llm-model openai/gpt-5.6-sol \
  --llm-temperature 0.2 --llm-max-tokens 1000 \
  --output-tag exploration_gpt56_30seed
```

Then freeze the generated kernel-skill portfolio and evaluate it against
GP-UCB, GP-EI, the target acquisition portfolio, and fixed transfer on held-out
seeds. The report should include confidence/weight variance, explore/avoid
balance, Top-1 change rate, gate authorization, bad interventions, paired
confidence intervals, and per-pair wins/losses.
