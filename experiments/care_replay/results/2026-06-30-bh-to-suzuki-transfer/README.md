# BH to Suzuki Transfer Ablation

Date: 2026-06-30

This snapshot is the first explicit CARE 2.0 cross-domain transfer ablation.
It tests whether source evidence from Dreher-Doyle Buchwald-Hartwig replay can
be converted into a structured transfer card and used on the Perera
Suzuki-Miyaura replay.

Command:

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_buchwald_hartwig \
  --target-dataset real_suzuki_miyaura \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --source-observations 48 \
  --discount 0.65 \
  --output-tag bh_to_suzuki_30seed
```

## What Transfers

The transfer card does not copy a raw ligand or reagent rule from one dataset
to another. The public adapters anonymize factor values independently, so direct
factor-value transfer would be invalid.

Instead, the card transfers role-level evidence strength:

- BH `ligand` -> Suzuki `ligand`
- BH `base` -> Suzuki `reagent`
- BH `aryl_halide` -> Suzuki `reactant_1`
- BH `additive` -> Suzuki `solvent`

Target-side observed Suzuki evidence still decides the candidate-level
direction. This tests the CARE control interface for transfer, not a claim that
one specific BH chemical rule directly applies to Suzuki.

## Files

- `tables/transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_bh_to_suzuki_30seed_metrics.csv`
- `transfer_summary.csv`
- `runs/*_summary.json`
- `runs/*_transfer_card_seed*.json`

Per-seed audit logs and knowledge snapshots are kept under
`experiments/care_replay/outputs/runs/transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_bh_to_suzuki_30seed_*`.

## Main Result

This first transfer implementation is diagnostic and currently negative. The
transfer modes activate, but they underperform the target-only incumbent.

| Mode | Final Best Mean | AUC Mean | Bad Intervention Mean | Interpretation |
| --- | ---: | ---: | ---: | --- |
| no_care_random | 87.2205 | 83.3939 | 0.0000 | random baseline |
| incumbent | 92.6085 | 87.5533 | 0.0000 | strongest current target baseline |
| target_local_gate_v1 | 92.6085 | 87.4647 | 0.0667 | local CARE is neutral/slightly worse |
| transfer_no_gate | 91.3879 | 86.4187 | 1.6333 | transfer hurts |
| transfer_gate_v1 | 91.3879 | 86.4187 | 1.6333 | current gate does not block enough |
| transfer_plus_local_gate_v1 | 91.6643 | 86.6158 | 1.0333 | mixed, still below incumbent |

## Interpretation

This is still real cross-domain transfer work because the source task produces
a transfer card, the target task consumes it, and the result is compared against
matched no-transfer baselines. The result is not positive yet.

The main lesson is that role-level confidence transfer is too coarse. The next
iteration should transfer richer, chemically meaningful descriptors or use a
critic/ranker interface that scores a validated target candidate menu. The gate
also needs a transfer-specific risk check: in this run, transfer was active in
seven rounds per seed on average and still produced too many bad interventions.
