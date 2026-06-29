# CARE Replay Results: 2026-06-28 Document-Guided Sweep

This folder contains tracked summary outputs for a CARE 2.0 replay sweep guided
by the CARE 2.0 slide deck and AI4Science task-analysis notes. It is intended as
an internal, reproducible experiment record rather than a CARE 1.0 paper
reproduction.

## Run Command

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset all \
  --seeds 30 \
  --rounds 10 \
  --initial 5
```

## Scope

The sweep covers six adapters:

| Dataset | Type | Candidate Count | Purpose |
| --- | ---: | ---: | --- |
| `synthetic_suzuki_i` | synthetic reaction replay | 448 | Interface smoke test for Suzuki-like optimization |
| `synthetic_chemlex_i` | synthetic reaction replay | 1728 | ChemLex-shaped adapter smoke test; not real ChemLex |
| `synthetic_materials_i` | synthetic materials replay | 336 | Materials-shaped adapter smoke test |
| `real_buchwald_hartwig` | public real HTE | 3955 | Dreher-Doyle Buchwald-Hartwig yield replay |
| `real_suzuki_miyaura` | public real HTE | 5760 | Perera Suzuki-Miyaura yield replay; near-domain transfer probe |
| `real_moleculenet_esol` | public real molecular property | 1128 | MoleculeNet ESOL finite-pool molecular property replay |

Raw public data files downloaded by the script are not tracked in this folder.
The generated audit logs are also not tracked here; only aggregate summaries and
metrics tables are included.

## Aggregate Results

Values below are means over 30 seeds. `AUC` is the best-so-far area under the
replay curve. `Top-10` is the mean top-decile hit rate. `Interv.` and `Bad
Interv.` are mean gate-authorized interventions and adverse interventions.

| Dataset | Mode | Final Best | AUC | Regret | Top-10 | Interv. | Bad Interv. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_suzuki_i | incumbent | 81.8802 | 79.0852 | 5.5186 | 0.6000 | 0.0000 | 0.0000 |
| synthetic_suzuki_i | gate_v1 | 86.4365 | 85.1926 | 0.9622 | 0.9333 | 2.6333 | 0.5000 |
| synthetic_suzuki_i | gate_v2 | 86.4365 | 85.1926 | 0.9622 | 0.9333 | 2.6333 | 0.5000 |
| synthetic_chemlex_i | incumbent | 92.7274 | 88.8560 | 7.2726 | 0.2667 | 0.0000 | 0.0000 |
| synthetic_chemlex_i | gate_v1 | 99.6302 | 98.5616 | 0.3698 | 0.9000 | 6.8667 | 1.8000 |
| synthetic_chemlex_i | gate_v2 | 99.6302 | 98.5616 | 0.3698 | 0.9000 | 6.8667 | 1.8000 |
| synthetic_materials_i | incumbent | 93.3509 | 90.4718 | 3.1230 | 0.7000 | 0.0000 | 0.0000 |
| synthetic_materials_i | gate_v1 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| synthetic_materials_i | gate_v2 | 95.8318 | 94.0513 | 0.6421 | 0.9333 | 3.3667 | 0.7000 |
| real_buchwald_hartwig | incumbent | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.0000 | 0.0000 |
| real_buchwald_hartwig | gate_v1 | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.3000 | 0.1000 |
| real_buchwald_hartwig | gate_v2 | 86.8818 | 80.8275 | 13.1182 | 0.1667 | 0.3000 | 0.1000 |
| real_suzuki_miyaura | incumbent | 92.6085 | 87.5533 | 7.3915 | 0.1000 | 0.0000 | 0.0000 |
| real_suzuki_miyaura | gate_v1 | 92.6085 | 87.4647 | 7.3915 | 0.1000 | 0.1333 | 0.0667 |
| real_suzuki_miyaura | gate_v2 | 92.6085 | 87.4647 | 7.3915 | 0.1000 | 0.1333 | 0.0667 |
| real_moleculenet_esol | incumbent | 88.6412 | 86.1189 | 8.3588 | 0.2333 | 0.0000 | 0.0000 |
| real_moleculenet_esol | gate_v1 | 89.0674 | 86.1829 | 7.9326 | 0.3000 | 0.7333 | 0.1000 |
| real_moleculenet_esol | gate_v2 | 89.0674 | 86.1758 | 7.9326 | 0.3000 | 0.7667 | 0.1000 |

## Readout

- The synthetic adapters show clear positive movement from the gated
  skill/evidence loop. They are useful smoke tests, not scientific claims.
- The two real HTE adapters currently show mostly neutral results under the
  conservative public-evidence gate. This suggests the current gate is stable
  but not yet strong enough to demonstrate robust real-HTE gains.
- ESOL gives a small positive signal, showing that the same replay interface can
  be extended from HTE to molecular property search.
- Real ChemLex and Pfizer zero-inflated tables are still needed before making
  claims about those datasets.

## Files

- `runs/all_datasets_summary.json`: full aggregate JSON for all adapters.
- `runs/*_summary.json`: per-dataset aggregate summaries.
- `tables/*_metrics.csv`: per-dataset metric tables.

