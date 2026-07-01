# CARE 2.0 CommonStack LLM Generalization Sweep

Date: 2026-06-30

This snapshot records a nine-dataset CARE 2.0 replay sweep with real
CommonStack LLM calls. It is separate from the larger non-LLM 30-seed
generalization sweep.

Command:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset all \
  --seeds 5 \
  --rounds 6 \
  --initial 5 \
  --modes no_care_random,incumbent,llm_no_gate,llm_gate_v1 \
  --llm-model openai/gpt-4o-mini \
  --output-tag llm_commonstack_5seed
```

The API key was supplied through the process environment and is not stored in
the repository.

## Files

- `runs/all_datasets_summary.json`: full aggregate JSON for the LLM sweep.
- `runs/*_summary.json`: per-dataset aggregate summaries.
- `tables/*_metrics.csv`: per-seed metrics for each dataset.
- `llm_generalization_summary.csv`: compact comparison table with deltas
  against the incumbent.

Per-seed audit logs and knowledge snapshots are kept under
`experiments/care_replay/outputs/runs/*_llm_commonstack_5seed_*`.

## Call Audit

- Datasets: 9
- Seeds per dataset: 5
- Reveal rounds: 6
- LLM modes: `llm_no_gate`, `llm_gate_v1`
- Real LLM call records: 270
- Parse errors: 0
- Empty applied-spec records: 0

Each LLM seed has three actual model calls. Calls start only after at least
eight public observations are available, so the first three reveal rounds are
warm-up rounds.

## Main Result

The LLM proposer is useful, but not reliable by itself. The strongest signal is
that the gate can prevent or soften bad LLM proposals on several real datasets.
This is closer to the CARE 2.0 direction than a claim that LLM proposals alone
solve cross-domain optimization.

| Dataset | LLM No-Gate Delta Final | LLM Gate Delta Final | Interpretation |
| --- | ---: | ---: | --- |
| real_buchwald_hartwig | +0.1850 | +0.1850 | Small positive real HTE signal |
| real_suzuki_miyaura | -2.3329 | +0.0000 | Gate prevents a harmful LLM intervention |
| real_moleculenet_esol | +0.0000 | +0.0000 | Neutral |
| real_moleculenet_freesolv | -0.2866 | -0.2866 | Negative; current gate does not help |
| real_moleculenet_lipophilicity | -1.3250 | +1.4000 | Gate turns a bad no-gate run into a positive result |
| real_matbench_expt_gap | +2.8000 | +2.8000 | Positive vs incumbent, but random is still stronger |
| synthetic_chemlex_i | +0.5197 | +0.0000 | No-gate positive; gate conservative |
| synthetic_materials_i | -0.9524 | +0.1165 | Gate improves over bad no-gate proposals |
| synthetic_suzuki_i | -0.1434 | -0.1434 | Slightly negative |

## Interpretation

This run answers the immediate question of whether the CARE 2.0 replay can use
actual LLM calls. It can: the audit logs include raw responses, parsed JSON
policies, applied factor adjustments, model names, and usage metadata.

The scientific result is mixed. LLM no-gate proposals sometimes help, sometimes
hurt. `llm_gate_v1` is therefore best read as a risk-control layer around a
noisy proposer. The clearest examples are Suzuki-Miyaura and Lipophilicity,
where the no-gate LLM result is worse than the incumbent but the gated result is
neutral or positive.

The materials result needs careful wording. Matbench improves over the public
incumbent in this 5-seed LLM run, but `no_care_random` is still stronger. That
means the current materials observation model remains weak; this result should
motivate better descriptors or embeddings rather than be presented as a solved
materials optimization result.

## Next Experiments

1. Repeat the LLM sweep with more seeds after the prompt and gate are stabilized.
2. Add a stronger observation model based on molecular/material descriptors or
   embeddings.
3. Run a source-to-target transfer ablation, especially Buchwald-Hartwig to
   Suzuki-Miyaura, with no transfer, no-gate transfer, and gated transfer.
4. Add a critic/ranker interface so the LLM ranks a validated candidate menu
   instead of directly producing global factor adjustments.
