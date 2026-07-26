# ChemLex -> Buchwald-Hartwig Warm-Start Control

This archive records the matched 30-seed calibration / 50-seed held-out
control for the ChemLex -> Buchwald-Hartwig portfolio result.

The protocol is identical to the portfolio run: the same fixed ChemLex source
history, target-only GP-UCB anchor, target budget, and non-overlapping seeds.
The only change is `router_max_transfer_mass=0.0`; the source-positive initial
design is retained, but no source-outcome adjustment is allowed after the
initial observations.

On held-out seeds, the warm-start-only route exactly matches the portfolio
route seed by seed: Final best `99.6191`, best-so-far AUC `89.554`, and the
same per-seed metrics. Thus the portfolio gain versus the matched target-only
route is currently attributable to the source-informed initial design. This
is a valid cross-task warm-start transfer signal, but it is not evidence of an
additional continuous source-outcome gain.

All 530 audit files, raw metrics, summary, and hashes are retained here.
