# LLM trajectory stability audit

## Result

The early audited calls and first frozen repeat both contain 6 wins, 2 ties, and 3 losses. However, 4 of 11 matched routes reverse sign, so the unchanged aggregate count hides route-level instability.
The mean AUC delta changes from +1.0135 to +1.7532, and the mean absolute paired change is 1.0479. These values are descriptive only.

## Interpretation

This audit does not test whether the frozen controller has a positive population effect. The early calls predate the confirmation lock, and only one frozen v1 trajectory is complete per route. The result instead demonstrates why single-call route labels and route-resampled confidence intervals are insufficient for a stochastic LLM controller.

The Suzuki route currently has one additional v2 frozen trajectory. Its three observed AUC deltas are +0.28 (early audit), +1.38 (v1 frozen trajectory), and +0.32 (v2 frozen trajectory). Direction is consistent, but magnitude is not yet precisely estimated.

## Claim boundary

No confirmatory transfer claim is evaluated until the predeclared repeated-trajectory protocol is complete. Infrastructure failures remain in the audit log and are excluded from scientific outcomes.
