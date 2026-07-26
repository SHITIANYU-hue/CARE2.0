# CARE 2.0 Frozen Transfer Portfolio Report

This report summarizes frozen candidate portfolios. Candidate selection uses calibration seeds only; held-out seeds execute the selected route or the exact matched target-only fallback.

The deltas below are held-out point estimates. Inferential claims should use the per-archive paired summaries and audit files; no fallback is classified as a positive transfer gain.

| Source -> target | Cal / held-out | Deployed | Candidate | Final vs matched | AUC vs matched | Final vs strongest BO | AUC vs strongest BO |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| real_chemlex_acidamine -> real_buchwald_hartwig | 30 / 50 | source-informed warm start | positive | +10.2095 | +9.2076 | +6.8404 | +6.3181 |
| real_moleculenet_freesolv -> real_moleculenet_lipophilicity | 10 / 20 | exact fallback | - | +0.0000 | +0.0000 | +1.8875 | +1.0505 |
| real_matbench_expt_gap -> real_matbench_dielectric | 10 / 20 | exact fallback | - | +0.0000 | +0.0000 | -0.4344 | -3.2747 |

Source-informed deployments: 1/3; warm-start deployments: 1/3; continuous-transfer candidates: 0/3; exact fallbacks: 2/3.

The current evidence supports selective source-informed deployment: the router can deploy a strong warm-start route on a reaction pair and refuse unstable molecular/materials routes. Warm-start and continuous-transfer mechanisms are reported separately; it does not yet support the stronger claim that every domain or every source-target pair improves.

## Archives

- `experiments/care_replay/results/2026-07-24-transfer-portfolio-bh`
- `experiments/care_replay/results/2026-07-24-transfer-portfolio-freesolv-lipophilicity`
- `experiments/care_replay/results/2026-07-24-transfer-portfolio-materials`
