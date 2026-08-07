# CARE 2.0 transfer mechanism controls

This run tests where the previously reported source-outcome transfer gain
actually comes from. It adds two matched controls to each of the seven frozen
source-target routes:

- `source_warmstart_only`: keep the source-informed initial design, then set
  transfer mass to zero for all sequential rounds.
- `fixed_data_only_transfer`: use measured source outcomes with a deterministic
  equal-role patch, without any LLM-generated patch fields.

The full route is evaluated against both controls on 50 calibration seeds and
100 disjoint held-out seeds. The LLM patch records were generated once with
`openai/gpt-4o-mini` and frozen before replay. No model is called per seed.

## Result

The new controls materially change the interpretation of the earlier result.
Only two routes pass the calibration selector on the new seed range. Both have
large held-out gains, but those gains are fully explained by source-informed
initialization. Once the initial design is held fixed, neither route has an
additional source-outcome or LLM-patch gain.

| Source -> target | Deployed route | Final delta vs strongest BO | Warm-start composite effect | Post-initialization composite effect | LLM increment vs fixed data-only |
| --- | --- | ---: | ---: | ---: | ---: |
| ChemLex -> Buchwald-Hartwig | transfer | +11.452 `[+9.025, +13.878]` | +22.460 `[+18.196, +26.723]` | 0 | 0 |
| Dielectric -> expt. gap | transfer | +42.461 `[+37.932, +46.990]` | +85.616 `[+75.374, +95.859]` | 0 | 0 |
| Expt. gap -> dielectric | fallback | +10.017 `[+6.562, +13.471]` | -0.912 `[-6.080, +4.255]` | 0 | +29.424 |
| Phonons -> dielectric | fallback | +3.369 `[-0.331, +7.069]` | +3.361 `[-1.565, +8.287]` | -1.296 `[-2.516, -0.075]` | -2.642 `[-6.142, +0.859]` |
| ESOL -> Lipophilicity | fallback | +1.089 `[+0.165, +2.013]` | +0.789 `[-0.686, +2.265]` | -0.145 `[-0.565, +0.275]` | -0.223 `[-1.573, +1.127]` |
| FreeSolv -> Lipophilicity | fallback | +2.126 `[+1.087, +3.165]` | +0.937 `[-0.586, +2.459]` | 0 | 0 |
| Lipophilicity -> FreeSolv | fallback | +31.760 `[+29.109, +34.411]` | -1.173 `[-1.408, -0.938]` | 0 | 0 |

The positive `+29.424` LLM increment for expt. gap -> dielectric does not mean
that the LLM improved over warm-start. The full LLM route and warm-start-only
route are identical there; the number is positive because the fixed data-only
patch is harmful. It is evidence that the LLM avoids that particular bad patch,
not evidence of a new continuous-transfer gain.

The fallback rows can still outperform a target-only BO baseline because the
deployed route is the frozen matched target-only LLM. Those deltas are not
counted as transfer gains.

## Boundary

This run supports a narrower claim than the previous report:

1. Source-informed initial design can produce strong gains on two routes.
2. The calibration selector rejects five unstable or harmful routes on this
   independent seed range.
3. There is no positive held-out evidence here for a post-initialization
   source-outcome component.
4. There is no positive held-out evidence here that an LLM-generated patch
   beats both warm-start-only and fixed data-only controls.

Calibration is offline replay model selection. It evaluates seven modes across
50 calibration seeds and is not a cost-feasible wet-lab deployment gate.

## Next protocol

`configs/source_outcome_continuous_exploration.json` removes the warm-start
advantage by giving every route exactly the same target-only initial points.
Source outcomes and LLM patches may influence only subsequent acquisition
rounds. That experiment uses fresh calibration and held-out seed ranges and is
reported separately from this confirmation run.

## Files

- `mechanism_controls.csv`, `.json`, and `.md`: compact seven-pair readout.
- `raw_summaries/`: complete per-pair aggregate reports.
- `raw_metrics/`: per-seed metrics for every evaluated mode.
- `canonical_traces/`: canonical selected-seed reasoning and execution traces.
- `transfer_skills/`: frozen executable skill artifacts and provenance.
- `audit_archives/`: all per-seed audit JSONL files, compressed by pair.
- `logs/`: pair logs and the server driver log.
- `run_manifest.json`: seed ranges, output tag, and code provenance.
- `SHA256SUMS`: integrity checks for every archived artifact.

