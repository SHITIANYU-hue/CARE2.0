# Development screen

This directory preserves 47 exploratory summaries and their paired metrics.
They were used to decide the frozen pair set and policy settings before the
final seed ranges were opened. They are not pooled into the headline estimate.

The screen varied:

- source-informed initial designs: matched, positive, negative, extremes, and
  interior quantiles;
- target data regimes: 2+13 and 5+10 initial/reveal splits under the same
  total budget of 15;
- full source priors versus initial-design-only execution;
- alternative materials directions involving dielectric, experimental band
  gap, phonons, and log bulk modulus;
- molecular directions involving ESOL, FreeSolv, and Lipophilicity.

The screen retained ChemLex → Buchwald-Hartwig, dielectric → experimental band
gap, experimental band gap → dielectric, and FreeSolv → Lipophilicity for
source-outcome deployment. It also retained three deliberately difficult paths
to test the fallback mechanism. Log bulk-modulus and experimental band gap →
phonons candidates were not promoted because their final-quality and AUC
effects were unstable or traded off.

The formal report is based only on calibration seeds 40000-40049 and held-out
seeds 41000-41099 under `frozen_source_outcome_v1`.
