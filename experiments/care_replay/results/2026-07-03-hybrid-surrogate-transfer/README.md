# Hybrid Surrogate Transfer

This snapshot tests the next stricter CARE 2.0 setting: use mixed-kernel
GP-UCB as the incumbent acquisition, then let CARE transfer cards apply bounded
adjustments to that acquisition.

This is a stronger test than comparing transfer only against the hand-written
public incumbent. It directly addresses the question: can reusable CARE skills
improve a stronger target-only optimizer?

## Setting

- Base incumbent: `mixed_kernel_gp_ucb`
- Transfer mechanism: role-level transfer card, optionally plus shared
  descriptor value priors
- Gate: `gate_v1` for gated modes
- Seeds: 50
- Initial observations: 5
- Reveal budget: 10

## Readout

### FreeSolv -> Lipophilicity

Hybrid transfer gives a small positive gain over GP-UCB when shared descriptor
value priors are enabled.

| Mode | Final Best | Delta vs GP-UCB | AUC | Delta AUC | Top-10 Hit | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gp_ucb` | 88.4775 | 0.0000 | 86.4578 | 0.0000 | 0.0800 | 0.0000 |
| `hybrid_transfer_gp_ucb_gate_v1` | 88.3950 | -0.0825 | 86.4265 | -0.0313 | 0.0800 | 0.4800 |
| `hybrid_value_prior_gp_ucb_gate_v1` | 88.8775 | +0.4000 | 86.7270 | +0.2692 | 0.1600 | 3.4000 |

This is useful but not yet optimal. The earlier public-incumbent transfer run
reached final best 90.0625 and AUC 87.9660, so the value prior is still more
effective when it can directly shape the simpler incumbent. The hybrid result
shows the value prior can still help GP-UCB, but the interaction needs
calibration.

### Suzuki-Miyaura -> Buchwald-Hartwig

Hybrid transfer does not beat GP-UCB on final best, but slightly improves AUC.

| Mode | Final Best | Delta vs GP-UCB | AUC | Delta AUC | Top-10 Hit | Bad Interventions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gp_ucb` | 91.1145 | 0.0000 | 82.9700 | 0.0000 | 0.3200 | 0.0000 |
| `hybrid_transfer_gp_ucb_gate_v1` | 90.7886 | -0.3259 | 83.1863 | +0.2163 | 0.2600 | 0.7400 |

The reaction result is therefore not a clear win over GP-UCB. It suggests the
transfer adjustment may help earlier search trajectory, but can also pull the
final selection away from GP-UCB's best candidates. This should be treated as a
diagnostic for gate/acquisition calibration, not as a headline win.

## Interpretation

The current CARE 2.0 status is:

1. Transfer clearly works against the original public incumbent in selected
   directions.
2. Against a stronger GP-UCB incumbent, shared-descriptor molecular transfer has
   a small positive signal.
3. Reaction HTE transfer over GP-UCB is not yet strong enough; it improves AUC
   slightly but hurts final best.

The next technical step should be acquisition-level calibration: instead of
adding a fixed transfer adjustment, learn or tune when transfer should affect
posterior mean, uncertainty, exploration weight, or candidate filtering.

## Files

- `hybrid_surrogate_transfer_summary.csv`: compact comparison with GP-UCB and
  prior transfer references.
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_metrics.csv`
- `hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_summary.json`
- `hybrid_surrogate_transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_metrics.csv`
- `hybrid_surrogate_transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_summary.json`
