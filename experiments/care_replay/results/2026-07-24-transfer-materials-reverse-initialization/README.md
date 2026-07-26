# Materials Reverse Direction: Initialization Audit

This archive contains a 10-seed calibration / 20-seed held-out audit for
Matbench dielectric -> experimental band gap, plus a matched source-informed
warm-start-only control.

## Protocol

- Source: Matbench dielectric.
- Target: Matbench experimental band gap.
- Target budget: 2 initial observations + 13 reveal rounds.
- The portfolio route used `source_extremes` for the initial probes.
- The control used the same source-informed initial design but fixed
  `router_max_transfer_mass=0.0`, so no source-outcome adjustment could be
  applied after initialization.
- Calibration and held-out seeds were identical between the two runs.

## Result and interpretation

Both routes reached Final best `100.0` and best-so-far AUC `100.0` on every
held-out seed. Their held-out metrics are identical seed by seed. The target
contains 12 candidates with the maximum normalized band-gap score, and the
source-informed initial design reliably places the replay in that region.

This is useful evidence that source outcomes can support a strong warm start,
but it is **not** evidence that continuous source-outcome transfer improves
the policy: the warm-start-only control explains the entire observed gain.
The pair is therefore excluded from the positive transfer count. All 460
audit files, raw metrics, summaries, and hashes are retained for review.
