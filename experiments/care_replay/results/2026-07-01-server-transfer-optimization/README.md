# Server-side BH -> Suzuki transfer optimization

Date: 2026-07-01

This run moves the current transfer optimization experiments onto the GPU server and treats the server outputs as the canonical results. The target question is whether CARE 2.0 can transfer reusable evidence from a source HTE task into a different target HTE task without directly leaking target labels or source factor values.

## Setup

- Server: `420GP-253`
- Source dataset: `real_buchwald_hartwig`
- Target dataset: `real_suzuki_miyaura`
- Seeds: 30
- Initial target observations: 5
- Target reveal budget: 10 rounds
- Source observations used to build the transfer card: 48 per seed
- Transfer boundary: source outcomes are used only to estimate role-level evidence strength. The target hidden outcomes and direct source factor values are not transferred.
- Role map: `ligand -> ligand`, `base -> reagent`, `aryl_halide -> reactant_1`, `additive -> solvent`

## Runs

Main run:

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_buchwald_hartwig \
  --target-dataset real_suzuki_miyaura \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --source-observations 48 \
  --discount 0.65 \
  --modes no_care_random,incumbent,target_local_gate_v1,transfer_gate_v1,transfer_plus_local_gate_v1,transfer_strict_gate_v1,transfer_strict_plus_local_gate_v1 \
  --output-tag server_bh_to_suzuki_strict_30seed
```

Parameter check:

```bash
python3 experiments/care_replay/scripts/run_transfer_ablation.py \
  --source-dataset real_buchwald_hartwig \
  --target-dataset real_suzuki_miyaura \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --source-observations 48 \
  --discount 0.65 \
  --strict-min-positive-roles 3 \
  --modes incumbent,transfer_strict_gate_v1,transfer_strict_plus_local_gate_v1 \
  --output-tag server_bh_to_suzuki_strict_pos3_30seed
```

## Result

Plain transfer is worse than the incumbent because it applies too broadly: about 30k candidates receive transfer scores each seed, and bad interventions rise. The improvement appears only after adding a stricter transfer gate.

In the main run, `transfer_strict_gate_v1` is the best policy:

- `final_best`: 92.8770 vs. 92.6085 for incumbent
- `best_so_far_auc`: 87.7002 vs. 87.5533 for incumbent
- `transfer_scored_candidates_total`: 3597.4333 vs. 29997.2000 for plain transfer
- `bad_intervention_count`: 0.8333 vs. 1.6333 for plain transfer

This is a small but useful gain: the transfer card becomes helpful only when role-level evidence is filtered before it is allowed to influence target selection.

The stricter `min_positive_roles = 3` check is too narrow. It scores only about 245 candidates, but drops `final_best` to 91.2369 and increases bad interventions. For this source-target pair, the better setting is the default strict rule: confidence >= 0.18, at least 2 positive role signals, and 0 negative role signals.

## Files

- `server_bh_to_suzuki_strict_30seed_metrics.csv`: per-seed metrics for the main 7-policy comparison.
- `server_bh_to_suzuki_strict_pos3_30seed_metrics.csv`: per-seed metrics for the stricter parameter check.
- `transfer_summary.csv`: compact aggregate table.
- Raw audit logs, transfer cards, knowledge snapshots, and summary JSON are stored under `experiments/care_replay/outputs/runs/` with the same output tags.

