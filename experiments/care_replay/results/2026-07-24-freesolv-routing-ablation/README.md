# FreeSolv routing ablations

These runs test whether source-informed initialization or a looser calibration
bound can turn the continuous FreeSolv extension into a reliable positive
transfer. They are exploratory ablations and are not folded into the headline
seven-pair result.

## Source-extremes initial design

The two real paths were rerun with `source_initial_strategy=source_extremes`
using 30 calibration and 50 held-out seeds. Both were rejected by the strict
matched-target-LLM gate. The raw transfer route improved over classical BO on
the held-out batch, but this did not establish a stable incremental gain over
the target-only LLM.

## `bound_v2`

The ESOL -> FreeSolv continuous path was rerun with 50 calibration and 100
held-out seeds. All original conditions were retained; only the composite CI
lower-bound threshold was predeclared as `-1.0` instead of `0.0`. The route was
still rejected because calibration evidence against the matched target-only
LLM was not positive enough. On the new held-out batch, the raw route was
slightly below the matched LLM on Final Best and only improved AUC, so this
bound was not promoted.

This ablation supports keeping the strict gate. Relaxing the confidence bound
does not reliably increase transfer wins and would weaken the no-negative-
transfer guarantee.

Artifacts are grouped by run:

- `source_extremes_30x50/`: 2 summaries, 2 metrics tables, and 1,060 audit JSONL files;
- `bound_v2_50x100/`: 1 summary, 1 metrics table, and 1,000 audit JSONL files;
- `SHA256SUMS`: integrity manifest.
