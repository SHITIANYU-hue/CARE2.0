# Materials Baseline and No-Gate Ablation

This snapshot records the first material-focused replay pass with an explicit
non-CARE baseline.

Run commands:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset synthetic_materials_i --seeds 30 --rounds 10 --initial 5
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py --dataset real_matbench_expt_gap --seeds 30 --rounds 10 --initial 5
```

## Modes

| Mode | Meaning |
| --- | --- |
| `no_care_random` | No CARE components. Randomly samples unseen candidates. |
| `incumbent` | Public-observation greedy baseline. No skill or gate intervention. |
| `no_gate` | Uses the CARE skill/challenger adjustment but accepts the challenger without gate checks. |
| `gate_v1` | Uses skill/challenger adjustment plus conservative gate checks. |
| `gate_v2` | Uses skill/challenger adjustment plus looser gate checks. |

## Aggregate Results

| Dataset | Mode | Final Best | AUC | Regret | Top-10 Hit | Interv. | Bad Interv. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_materials_i | no_care_random | 90.2415 | 88.4020 | 6.2324 | 0.4333 | 0.0000 | 0.0000 |
| synthetic_materials_i | incumbent | 93.3509 | 90.4718 | 3.1230 | 0.7000 | 0.0000 | 0.0000 |
| synthetic_materials_i | no_gate | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| synthetic_materials_i | gate_v1 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| synthetic_materials_i | gate_v2 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| real_matbench_expt_gap | no_care_random | 48.7417 | 43.7104 | 51.2583 | 0.0000 | 0.0000 | 0.0000 |
| real_matbench_expt_gap | incumbent | 44.8000 | 41.3633 | 55.2000 | 0.0000 | 0.0000 | 0.0000 |
| real_matbench_expt_gap | no_gate | 45.1250 | 41.4283 | 54.8750 | 0.0000 | 0.1000 | 0.0667 |
| real_matbench_expt_gap | gate_v1 | 44.8000 | 41.3633 | 55.2000 | 0.0000 | 0.0333 | 0.0333 |
| real_matbench_expt_gap | gate_v2 | 44.8000 | 41.3633 | 55.2000 | 0.0000 | 0.0333 | 0.0333 |

## Readout

The synthetic materials adapter shows the expected pattern: random search is
weakest, public-observation greedy is better, and skill/challenger modes are
best. On this adapter, `no_gate`, `gate_v1`, and `gate_v2` match exactly because
the challengers selected by the current material skills pass the gate checks.

The real Matbench experimental band-gap adapter is more conservative as a
claim. It proves that the harness can run a public real materials dataset, but
the current composition-only observation model is not strong enough: random
search slightly beats the current public greedy and CARE skill/gate variants.
This should be treated as a diagnostic result, not as evidence of CARE improving
real materials discovery yet.

The next material experiment should add a stronger public observation model:
composition embeddings, matminer-style descriptors, or a pretrained materials
property surrogate before applying the CARE gate.
