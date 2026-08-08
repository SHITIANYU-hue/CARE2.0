# Classical transfer-BO confirmation

## Scope

This run addresses the useful part of the 2026-08-06 review: CARE is compared
with classical transfer Bayesian optimization on source-target pairs that have
the same ordered feature space. It does not force same-space GP baselines onto
heterogeneous reaction tasks.

The protocol was frozen before held-out execution:

- execution commit: `03ebccc`;
- aggregation commit: `333b72a`;
- calibration seeds: `70000-70029`;
- held-out seeds: `71000-71099`;
- identical target initial observations and reveal budgets within every pair;
- classical arms: target-only GP-UCB, RGPE, and two-task ICM GP;
- no held-out model or hyperparameter selection.

## Classical transfer results

Held-out final-score deltas against target-only GP-UCB are:

| Source -> target | RGPE | Multi-task ICM GP |
| --- | ---: | ---: |
| FreeSolv -> Lipophilicity | -1.493 [-2.505, -0.480] | -0.413 [-1.311, 0.486] |
| Lipophilicity -> FreeSolv | -7.771 [-10.896, -4.646] | -0.124 [-3.002, 2.754] |
| Expt. gap -> Dielectric | -11.799 [-14.914, -8.685] | +1.387 [-1.397, 4.170] |
| Dielectric -> Expt. gap | +41.697 [37.089, 46.306] | +7.431 [0.973, 13.889] |
| Phonons -> Dielectric | -2.822 [-5.517, -0.127] | +2.127 [-0.523, 4.776] |

Classical transfer is therefore a strong baseline on the aligned
Dielectric-to-Band-Gap path, but it is not uniformly safe. RGPE significantly
hurts four of the five final-score comparisons.

## CARE against calibration-selected classical BO

The classical opponent is selected on calibration seeds and then frozen.
Positive values favor CARE.

| Source -> target | Frozen classical arm | CARE final delta | CARE AUC delta |
| --- | --- | ---: | ---: |
| FreeSolv -> Lipophilicity | target GP-UCB | +1.978 [0.958, 2.997] | +1.578 [0.621, 2.534] |
| Lipophilicity -> FreeSolv | target GP-UCB | +35.023 [32.100, 37.946] | +42.995 [40.287, 45.703] |
| Expt. gap -> Dielectric | multi-task ICM GP | +4.472 [0.937, 8.008] | +2.884 [0.321, 5.447] |
| Dielectric -> Expt. gap | RGPE | -33.250 [-38.511, -27.989] | -44.838 [-49.274, -40.403] |
| Phonons -> Dielectric | multi-task ICM GP | +4.329 [0.998, 7.659] | +0.045 [-1.911, 2.002] |

All five CARE source-transfer candidates were rejected by the frozen gate and
deployed the matched target-only fallback. These four positive CARE
comparisons must therefore be attributed to the target-only LLM/acquisition
policy, not to source transfer.

## Hybrid router

We also evaluated an offline portfolio that selects one frozen arm from CARE,
RGPE, multi-task ICM GP, and target GP-UCB using calibration seeds only.

| Source -> target | Frozen arm | Final delta vs target GP | AUC delta vs target GP |
| --- | --- | ---: | ---: |
| FreeSolv -> Lipophilicity | CARE fallback | +1.978 [0.958, 2.997] | +1.578 [0.621, 2.534] |
| Lipophilicity -> FreeSolv | CARE fallback | +35.023 [32.100, 37.946] | +42.995 [40.287, 45.703] |
| Expt. gap -> Dielectric | CARE fallback | +5.859 [2.241, 9.478] | +3.530 [0.883, 6.178] |
| Dielectric -> Expt. gap | RGPE | +41.697 [37.089, 46.306] | +49.757 [46.111, 53.403] |
| Phonons -> Dielectric | multi-task ICM GP | +2.127 [-0.523, 4.776] | +2.897 [1.020, 4.775] |

The hybrid portfolio has a positive held-out mean on all five pairs. Its AUC
gain is significant on all five; its final-score gain is significant on four.
This supports model routing as an offline benchmark result. It is not a
zero-target-cost gate and is marked `real_experiment_deployment_ready: false`.

## Evidence boundary

This experiment supports three conclusions:

1. classical transfer BO must be included whenever task spaces are compatible;
2. no single transfer family is reliable across all pairs, which motivates a
   calibrated router and exact fallback;
3. the current source-outcome and LLM-patch operators still have no independently
   confirmed post-initialization gain in this suite.

The next scientific target remains a calibration-free router or accumulated
transfer memory that can make the same choice without revealing target
outcomes in advance.

Raw metrics, summaries, canonical traces, skill artifacts, logs, and per-seed
audit archives are included in this directory. `SHA256SUMS` covers every
artifact and this interpretation record.
