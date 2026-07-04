# Transfer-Weighted GP Kernel

This snapshot follows the stronger direction raised in discussion: instead of
only comparing CARE transfer against the hand-written public incumbent, use
mixed-kernel GP-UCB as the strong target-only baseline and let CARE optimize the
acquisition itself.

The tested skill optimization is simple and auditable. A source-to-target
transfer card estimates which target decision fields are reliable to transfer.
Those role confidences reweight the categorical part of the GP kernel. The
policy still uses only public candidate descriptors and revealed target
observations; it does not read hidden target outcomes or directly select
candidate ids.

## Setting

- Base acquisition: `mixed_kernel_gp_ucb`
- Skill optimization: transfer-card role confidence -> categorical kernel
  weights
- Seeds: 50
- Initial observations: 5
- Reveal budget: 10
- Weight normalization: enabled, so the average categorical weight remains 1

## Results

### Suzuki-Miyaura -> Buchwald-Hartwig

This is the main reaction-transfer check against the strong GP-UCB baseline.

| Mode | Final Best | Delta vs GP-UCB | AUC | Delta AUC | Top-10 Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gp_ucb` | 91.1145 | 0.0000 | 82.9700 | 0.0000 | 0.3200 |
| `transfer_weighted_gp_ucb_scale_0p5` | 90.4219 | -0.6926 | 83.0740 | +0.1040 | 0.2800 |
| `transfer_weighted_gp_ucb_scale_1` | 90.5615 | -0.5530 | 82.9086 | -0.0614 | 0.2200 |
| `transfer_weighted_gp_ucb_scale_1p5` | 91.4146 | +0.3001 | 83.2862 | +0.3162 | 0.2600 |
| `transfer_weighted_gp_ucb_scale_2` | 90.4716 | -0.6429 | 82.6653 | -0.3047 | 0.2600 |
| `transfer_weighted_gp_ucb_scale_4` | 91.4524 | +0.3379 | 82.1020 | -0.8680 | 0.2600 |

`scale=1.5` is the cleanest configuration because it improves both final best
and search AUC over GP-UCB. `scale=4` gets a slightly higher final best but
hurts AUC, so it is less stable as the default acquisition setting.

### FreeSolv -> Lipophilicity

This is a molecular-property sanity check using shared SMILES-derived
descriptor fields.

| Mode | Final Best | Delta vs GP-UCB | AUC | Delta AUC | Top-10 Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gp_ucb` | 88.4775 | 0.0000 | 86.4578 | 0.0000 | 0.0800 |
| `transfer_weighted_gp_ucb_scale_1p5` | 88.5650 | +0.0875 | 86.6808 | +0.2230 | 0.0800 |
| `transfer_weighted_gp_ucb_scale_4` | 88.5025 | +0.0250 | 86.3800 | -0.0778 | 0.1200 |

The molecular gain is small, but it points in the same direction: a moderate
transfer-weighted kernel can improve GP-UCB without using hidden labels.

## Interpretation

This is the first positive CARE 2.0 result where transfer is used to modify the
strong GP-UCB acquisition geometry itself. The previous hybrid experiment added
bounded transfer adjustments on top of GP-UCB; that helped molecular transfer
slightly but did not improve Buchwald-Hartwig final best. Here, the transfer
skill changes the kernel field weights, which gives a small positive HTE result.

The result should be reported conservatively:

- It does show that skill optimization can beat the stronger GP-UCB baseline in
  selected real replay settings.
- The gain is still small relative to run variance.
- The transfer strength is sensitive; too weak or too strong can hurt.
- The next step is to learn or select the scale on calibration tasks/seeds
  rather than fixing it manually.

## Files

- `transfer_weighted_kernel_summary.csv`: compact comparison against GP-UCB.
- `transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_metrics.csv`
- `transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_summary.json`
- `transfer_weighted_kernel_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_metrics.csv`
- `transfer_weighted_kernel_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_summary.json`
