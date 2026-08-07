# Independent continuous materials transfer confirmation

This run independently tests the only positive post-initialization signal from
the matched-initial exploration: Matbench dielectric -> experimental band gap.
The complete policy was frozen before this run:

- identical target-only initial observations for every route;
- LLM-generated source-outcome patches frozen before replay;
- online source-quality threshold `0.15`;
- maximum transfer mass `0.45`;
- target-only fallback under the existing selector.

The confirmation uses 50 new calibration seeds (`68000-68049`) and 100 new
held-out seeds (`69000-69099`). None of these seeds appears in the exploratory
30+50 run.

## Result

The exploratory gain does not replicate.

| Comparison on 100 held-out seeds | Composite delta | Final-best delta | AUC delta |
| --- | ---: | ---: | ---: |
| Full LLM route vs identical-initialization control | +0.347 `[-2.066, +2.761]` | -0.198 `[-1.957, +1.562]` | +0.545 `[-0.395, +1.485]` |
| Full LLM route vs fixed data-only transfer | +1.305 `[-2.125, +4.734]` | +0.265 `[-2.112, +2.642]` | +1.040 `[-0.180, +2.259]` |

Both composite confidence intervals include zero. The 50 calibration seeds are
also negative relative to the matched target-only LLM, so the selector rejects
transfer and deploys exact fallback. The deployed route therefore has zero
delta relative to matched target-only LLM by construction.

The full route activates positive transfer mass on 13.9% of sequential rounds
and changes the selected candidate on 4.5% of rounds. These action changes do
not produce a statistically supported performance improvement.

## Conclusion

The matched-initial exploration showed that a continuous-transfer signal can
appear on one materials seed range. This independent confirmation shows that
the effect is not stable. It must not be presented as a confirmed CARE 2.0 or
LLM advantage.

The strongest supported result remains source-informed initial design on the
ChemLex -> Buchwald-Hartwig and dielectric -> band-gap routes. A robust
post-initialization source-outcome mechanism will require a better transferable
representation or surrogate, not a looser reporting threshold.

## Files

- `mechanism_controls.csv`, `.json`, and `.md`: compact confirmation readout.
- `raw_summaries/` and `raw_metrics/`: aggregate and per-seed results.
- `canonical_traces/`: selected-seed reasoning and execution trace.
- `transfer_skills/`: frozen skill artifact and provenance.
- `audit_archives/`: every calibration and held-out audit JSONL file.
- `logs/`: server driver and pair log.
- `run_manifest.json`: seed ranges and configuration provenance.
- `SHA256SUMS`: integrity checks for every archived artifact.

