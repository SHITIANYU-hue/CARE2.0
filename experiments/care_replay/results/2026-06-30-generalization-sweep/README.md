# CARE 2.0 Generalization Sweep

Date: 2026-06-30

This snapshot records a nine-dataset CARE 2.0 replay sweep after adding two
additional real MoleculeNet property datasets: FreeSolv and Lipophilicity.

Command:

```bash
python3 experiments/care_replay/scripts/run_synthetic_suzuki.py \
  --dataset all \
  --seeds 30 \
  --rounds 10 \
  --initial 5 \
  --modes no_care_random,incumbent,no_gate,gate_v1,gate_v2
```

## Files

- `runs/all_datasets_summary.json`: full aggregate JSON for all datasets.
- `runs/*_summary.json`: per-dataset aggregate summaries.
- `tables/*_metrics.csv`: per-seed metrics for each dataset.
- `generalization_summary.csv`: compact comparison table with deltas against
  the incumbent.

## Datasets

The sweep covers:

- synthetic Suzuki
- synthetic ChemLex-style acid-amine
- synthetic materials formulation
- real Buchwald-Hartwig HTE
- real Suzuki-Miyaura HTE
- real MoleculeNet ESOL
- real MoleculeNet FreeSolv
- real MoleculeNet Lipophilicity
- real Matbench experimental band gap

## Main Result

The current CARE 2.0 skill/gate loop generalizes as a software interface, but
not yet as a consistently positive scientific optimizer across real domains.

Synthetic tasks show strong gains from the skill/gate loop. Real HTE tasks are
mostly neutral under the current conservative public-evidence model. Molecular
property tasks show mixed behavior: ESOL is modestly positive, FreeSolv has a
small no-gate gain that the gate mostly removes, and Lipophilicity is slightly
negative. Real Matbench remains diagnostic: random search is still stronger than
the current composition-only public incumbent, so materials replay needs a
better observation model or descriptor layer.

## Aggregate Highlights

| Dataset | Best CARE2 Mode | Delta Final vs Incumbent | Delta AUC vs Incumbent | Interpretation |
| --- | ---: | ---: | ---: | --- |
| synthetic_suzuki_i | gate_v1/gate_v2 | +4.5563 | +6.1074 | Positive synthetic control |
| synthetic_chemlex_i | gate_v1/gate_v2 | +6.9028 | +9.7056 | Positive synthetic control |
| synthetic_materials_i | gate_v1/gate_v2 | +2.4809 | +3.5795 | Positive synthetic control |
| real_buchwald_hartwig | gate_v1/gate_v2 | +0.0000 | +0.0000 | Neutral real HTE |
| real_suzuki_miyaura | gate_v1/gate_v2 | +0.0000 | -0.0886 | Neutral/slightly worse real HTE |
| real_moleculenet_esol | gate_v1 | +0.4262 | +0.0640 | Small positive molecular signal |
| real_moleculenet_freesolv | no_gate | +0.3511 | +0.0352 | Small challenger signal, gate removes it |
| real_moleculenet_lipophilicity | gate_v1/gate_v2 | -0.0584 | -0.0058 | Slightly negative molecular signal |
| real_matbench_expt_gap | no_gate | +0.3250 | +0.0650 | Tiny gain, but random baseline is stronger |

## Interpretation

The strongest result is not that CARE 2.0 already improves every domain. It is
that the same replay protocol can now run across reaction HTE, molecular
properties, and materials properties with consistent baselines, no-gate
ablations, gated CARE2 modes, audit logs, and knowledge snapshots.

The weak point is the current observation model. It uses simple public
factor-level evidence over hand-built categorical descriptors. That is enough
for synthetic tasks and a small ESOL signal, but too weak for real materials and
mixed molecular-property tasks. This matches the CARE 2.0 direction: the next
step should be a stronger descriptor/embedding or surrogate observation layer,
not more synthetic datasets.

## Next Experiments

1. Add a stronger molecular/material descriptor layer, ideally embeddings or
   learned fingerprints, while keeping hidden targets out of the decision path.
2. Run a source-to-target transfer ablation, starting with Buchwald-Hartwig to
   Suzuki-Miyaura: no transfer, transfer without gate, and transfer with gate.
3. Add a real LLM proposer sweep when an API key is available in the environment,
   using the same `llm_no_gate` and `llm_gate_v1` modes.
4. Keep no-CARE random and incumbent baselines in every new dataset. Matbench
   shows why: without the random baseline, the current materials result would be
   easy to overread.
