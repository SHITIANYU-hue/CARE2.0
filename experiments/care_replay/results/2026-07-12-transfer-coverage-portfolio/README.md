# Transfer Coverage Portfolio Follow-Up

This run expands the transfer evaluation from a few headline pairs to 11 source-target pairs across reaction HTE, ChemLex-style acid-amine optimization, MoleculeNet molecular property replay, and a materials target stress test.

The main question is not only whether a hand-picked transfer policy can beat a public incumbent, but whether CARE 2.0 can decide when to use transfer on top of stronger target-only optimizers such as GP-UCB / GP-EI.

## What Was Added

- New 50-seed target-only surrogate baselines for `real_suzuki_miyaura`, `real_chemlex_acidamine`, and `real_matbench_expt_gap`.
- New 50-seed transfer coverage runs for reaction/ChemLex/materials pairs.
- New transfer-weighted GP-kernel runs for `BH -> Suzuki`, `Suzuki -> ChemLex`, and `ChemLex -> BH`.
- New hybrid GP-UCB transfer runs for `BH -> Suzuki`, `Suzuki -> ChemLex`, and `ChemLex -> BH`.
- A new portfolio summary script:

```bash
python3 experiments/care_replay/scripts/build_transfer_coverage_portfolio.py \
  --out-dir experiments/care_replay/results/2026-07-12-transfer-coverage-portfolio \
  --calibration-seed-count 25
```

The portfolio uses seeds `0-24` as calibration and seeds `25-49` as held-out evaluation. The default risk-aware selector only allows transfer if, on calibration seeds, it beats the fallback by at least `+1.0 final best` and `+1.0 AUC`.

## Headline Numbers

Across 11 source-target pairs:

| Criterion | Count |
| --- | ---: |
| Best transfer policy improves final best vs public incumbent | 7 / 11 |
| Best transfer policy improves final best vs GP-UCB | 4 / 11 |
| Best transfer policy improves final best vs best target-only baseline | 3 / 11 |
| Transfer-only calibration selector improves held-out final best vs public incumbent | 7 / 11 |
| Transfer-only calibration selector improves held-out final best vs GP-UCB | 2 / 11 |
| Risk-aware vs GP selector chooses transfer | 2 / 11 |
| Risk-aware vs GP selector is positive on held-out final best | 2 / 2 |

## Best Strong-Baseline Transfer Wins

The clearest strong-baseline wins are:

| Pair | Best transfer policy | Delta final vs GP-UCB | Delta AUC vs GP-UCB |
| --- | --- | ---: | ---: |
| `FreeSolv -> Lipophilicity` | `transfer_value_prior_gate_v1` | +1.5850 | +1.5082 |
| `Suzuki -> ChemLex` | `transfer_weighted_gp_ucb_scale_1` | +3.2395 | +1.7773 |
| `Suzuki -> Buchwald-Hartwig` | `transfer_weighted_gp_ucb_scale_1p5` | +0.3001 | +0.3162 |
| `BH -> Suzuki` | `hybrid_transfer_gp_ucb_no_gate` | +0.2847 | -0.0325 |

The first two are the strongest current cases. `Suzuki -> BH` is positive but small, and `BH -> Suzuki` only improves final best while AUC is basically flat/slightly negative.

## Interpretation

The results support a more careful CARE 2.0 story:

1. Transfer is real, but not universal.
   We can now show positive transfer beyond one cherry-picked pair, including a molecular property pair and a ChemLex target pair.

2. Strong baseline comparison is harder.
   Transfer beats public/incumbent baselines in most useful cases, but only beats GP-UCB or the best target-only surrogate in a minority of pairs.

3. The selector matters.
   A transfer-only selector is positive against public incumbent on 7 / 11 pairs, but only 2 / 11 against GP-UCB. A risk-aware selector is more conservative: it only picks transfer for 2 pairs, and both are held-out positive vs GP-UCB.

4. This is a better platform behavior than always transferring.
   CARE 2.0 should treat source knowledge as a reusable skill candidate, then validate it through target calibration and held-out replay before letting it affect acquisition.

## Files

- `pair_summary.csv`: one row per source-target pair, including best transfer policy, calibration-selected policy, and risk-aware selector deltas.
- `policy_summary.csv`: one row per policy/mode, with full, calibration, and held-out means.
- `summary.json`: compact count summary.
