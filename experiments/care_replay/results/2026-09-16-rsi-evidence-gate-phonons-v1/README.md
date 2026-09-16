# Phonons trajectory-evidence + promotion-gate diagnostic

## Question

Can candidate-level experimental evidence plus a conservative incumbent/challenger gate improve the trained adapter's recursive updates, and prevent a harmful second update from being deployed?

This is a bounded diagnostic on `real_matbench_phonons`. Model weights are frozen. The pinned initial policy, development seeds, and final evaluation seeds are shared by both branches. `rich_gated` additionally spends 8 disjoint promotion seeds per update. All seed splits and promotion criteria were locked before generation.

## Design

- `sparse_ungated`: the previous feedback format (per-skill mean and per-seed AUC/final best), followed by unconditional deployment of the model-selected revision.
- `rich_gated`: the same aggregate metrics plus actually revealed candidate features, values, incremental improvements, matched-rule summaries, and strongest/weakest measured examples. A revision replaces the incumbent only if its paired promotion result has mean AUC delta at least `+1.0`, at least `5/8` wins, and nonnegative mean final-best delta.
- `fixed_initial`: the shared pinned policy before recursive updates.
- `gp_ucb`: the same target-only GP control used by the existing executor.

The final 10 evaluation seeds are never included in a generation request. Four valid model calls completed with no failed attempt. The run contains 104 replay trajectories.

## Final blind evaluation

| Arm | Mean AUC | Mean final best |
|---|---:|---:|
| `rich_gated` | **57.7436** | **74.2030** |
| `fixed_initial` | 54.1011 | 73.4158 |
| `sparse_ungated` | 54.1011 | 73.4158 |
| `gp_ucb` | 52.8740 | 64.8586 |

Paired results:

- `rich_gated - sparse_ungated`: AUC `+3.6424`, bootstrap 95% interval `[-4.9523, 12.3511]`, 6 wins / 2 ties / 2 losses.
- `rich_gated - fixed_initial`: the same result because the sparse final policy retained the initial policy's active rules and acquisition parameters under a new ID.
- `rich_gated - gp_ucb`: AUC `+4.8695`, bootstrap 95% interval `[-9.5725, 18.9432]`, 7 wins / 0 ties / 3 losses.

The intervals cross zero. This run supports a promising mechanism result on one task and one deterministic model draw; it does not establish cross-task or model-call generalization.

## What the gate did

| Update | Mean promotion AUC delta | Mean final-best delta | W/T/L | Decision |
|---|---:|---:|---:|---|
| 1 | `+2.1962` | `+2.6846` | 5 / 3 / 0 | accept challenger |
| 2 | `-1.7107` | `-11.1026` | 4 / 1 / 3 | retain incumbent |

The first accepted policy made the semantic prior more conservative while leaving the GP acquisition settings unchanged:

- semantic mass: `0.40 -> 0.30` at the start and `0.15 -> 0.10` at the end;
- rule weights: `0.6/0.5/0.4/0.3 -> 0.5/0.4/0.3/0.2`;
- `ucb_weight`, `gp_beta_start/end`, and `gp_xi` stayed unchanged.

The second challenger concentrated more weight on moderate concentration and mid atomic number. It suffered a large failure on one promotion seed and reduced mean final best by 11.10, so it was not deployed. This is the intended safety behavior: recursive revision remains possible, but a revision is not treated as an improvement until an external measurement split supports it.

## Leakage and integrity checks

- The development, promotion, and final evaluation seed sets are pairwise disjoint.
- Final evaluation seed IDs do not occur in any model request.
- Model requests contain no `objective_value`, raw target property column, `source_row`, or unobserved target label field.
- Candidate evidence is compiled only from `events` that the executor actually revealed on development seeds.
- All four first attempts passed schema compilation; no retry was selected using performance.
- `SHA256SUMS` covers the raw requests, raw responses, normalized decisions, all replay trajectories, locks, and summaries.

## Reproduce on the A800 environment

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /work/zeyuwang/care-rsi/venv/bin/python \
  experiments/care_replay/scripts/run_rsi_evidence_gate.py \
  --config experiments/care_replay/configs/rsi_evidence_gate_adapter_phonons_v1.json \
  --output-dir experiments/care_replay/results/2026-09-16-rsi-evidence-gate-phonons-v1
```

The config points to the existing local Qwen2.5-32B base cache and the selected epoch-2 adapter. No API credential is required for this run.
