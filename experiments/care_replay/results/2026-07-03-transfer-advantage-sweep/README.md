# Transfer advantage sweep

Date: 2026-07-03

This snapshot is focused on making the transfer advantage easier to see. It
adds a conservative shared-vocabulary value-prior transfer mode and then runs
two 50-seed server sweeps:

- A molecular-property transfer case where source and target share public
  descriptor bins: FreeSolv to Lipophilicity.
- A reaction HTE transfer case using the existing role-level transfer card:
  Suzuki-Miyaura to Buchwald-Hartwig.

## What Changed

The new `transfer_value_prior_*` modes keep the original role-level transfer
card, but add an early source-guided prior for fields that share the same public
descriptor vocabulary. In this snapshot, that means MoleculeNet descriptor bins
such as `smiles_length_bin`, `hetero_atom_bin`, `aromatic_bin`, and related
SMILES-derived bins.

This mode is intentionally not enabled for reaction HTE labels such as `L00` or
`R00`, because those labels are dataset-local abbreviations and should not be
treated as the same chemical object across unrelated reaction datasets. For HTE,
the experiment still uses role-level transfer only.

## Main Results

### Molecular-property transfer: FreeSolv -> Lipophilicity

This is the clearest transfer-advantage result in this snapshot. Over 50 seeds,
the shared-descriptor value-prior transfer mode beats both incumbent and random
baselines by a large margin.

| Mode | Final best | Delta vs incumbent | Best-so-far AUC | AUC delta | Top10 hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 85.7650 | -1.5425 | 84.2917 | -1.2743 | 0.0200 |
| `incumbent` | 87.3075 | 0.0000 | 85.5660 | 0.0000 | 0.0400 |
| `transfer_value_prior_gate_v1` | 90.0625 | +2.7550 | 87.9660 | +2.4000 | 0.3000 |
| `transfer_value_prior_strict_gate_v1` | 88.1625 | +0.8550 | 86.1820 | +0.6160 | 0.1600 |

The important signal is not only the final-best gain. The top10 hit rate moves
from 4% for the incumbent to 30% for `transfer_value_prior_gate_v1`, which means
the transfer prior is helping the replay find genuinely better regions of the
target space.

The tradeoff is that the non-strict value-prior mode is aggressive. It has more
bad interventions than the stricter variant, so it is currently the "advantage"
mode rather than the final safety-calibrated policy.

### Reaction HTE transfer: Suzuki-Miyaura -> Buchwald-Hartwig

The reaction-transfer result is also stronger after increasing the stability run
to 50 seeds.

| Mode | Final best | Delta vs incumbent | Best-so-far AUC | AUC delta | Top10 hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 82.1728 | -4.4749 | 77.4347 | -2.5366 | 0.0600 |
| `incumbent` | 86.6477 | 0.0000 | 79.9713 | 0.0000 | 0.1600 |
| `transfer_gate_v1` | 89.0866 | +2.4389 | 81.0779 | +1.1066 | 0.2400 |
| `transfer_strict_gate_v1` | 87.6158 | +0.9681 | 80.2528 | +0.2815 | 0.1600 |

This gives a second, independent positive result: even without value-level
transfer, role-level Suzuki evidence improves Buchwald-Hartwig replay.

## Interpretation

The current story is now clearer:

- In shared descriptor spaces, CARE 2.0 can use source-domain value priors to
  create a large, visible transfer advantage.
- In reaction HTE, where direct value transfer would be unsafe, role-level
  transfer still gives a stable positive gain.
- The system exposes the safety tradeoff: aggressive transfer gives the largest
  advantage, while stricter gates reduce harmful interventions but leave some
  performance on the table.

The next useful experiment is gate calibration: keep the large value-prior
advantage while reducing bad interventions, probably by requiring either target
confirmation or an LLM audit that is less conservative than the current audit
prompt.

## Files

- `transfer_advantage_summary.csv`: compact table used for this README.
- `transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_server_value_prior_freesolv_to_lipo_50seed_metrics.csv`
- `transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_server_suzuki_to_bh_transfer_50seed_metrics.csv`

Canonical summary JSON files are also stored in `../../outputs/runs/`, and
canonical metric CSVs are stored in `../../outputs/tables/`.
