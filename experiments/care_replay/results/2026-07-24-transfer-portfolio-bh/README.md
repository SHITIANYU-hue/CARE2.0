# Frozen Transfer Portfolio: ChemLex -> Buchwald-Hartwig

This archive records a 30-seed calibration / 50-seed held-out evaluation of
the frozen CARE 2.0 source-outcome transfer portfolio.

## Protocol

- Source: real ChemLex acid-amine wetlab data.
- Target: real Buchwald-Hartwig HTE data.
- Target budget: 2 initial observations + 13 reveal rounds.
- Candidate portfolio, fixed before replay: standard transfer, conservative
  transfer, and source-positive initial design.
- Calibration selected one candidate; held-out seeds did not affect selection.
- Fallback: matched target-only GP-UCB, because the frozen target-only anchor
  for this path is GP-UCB.
- Source history and all LLM patches were frozen before target replay.

## Result

Calibration selected `source_positive`. On 50 held-out seeds, the deployed
portfolio route beat both matched target-only GP-UCB and the strongest target
BO anchor:

| Metric | Delta | Approx. 95% CI | Win rate |
| --- | ---: | ---: | ---: |
| Final best | +10.2095 | [+7.2982, +13.1208] | 88% |
| Best-so-far AUC | +9.2076 | [+5.5913, +12.8239] | 68% |

An identical-seed warm-start-only control, with transfer mass fixed to zero,
reproduces this portfolio route exactly seed by seed. The gain should therefore
be described as source-informed initial-design transfer in this configuration;
the archive does not establish an additional continuous source-outcome gain.
The control is retained at
`results/2026-07-24-transfer-bh-warmstart-control/`.

The standard and conservative candidates were retained in the calibration
diagnostics but were not deployed. Every calibration and held-out candidate
route, baseline, and selector alias has a JSONL audit. `SHA256SUMS` verifies
the archived files.
