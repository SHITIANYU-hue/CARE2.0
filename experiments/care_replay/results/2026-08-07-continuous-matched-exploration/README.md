# Matched-initial continuous transfer exploration

This experiment removes the source-informed warm-start advantage from all
seven source-target routes. The full route, warm-start-only control, fixed
data-only control, and matched target-only LLM receive exactly the same target
initial observations. Source outcomes and LLM-generated patches can influence
only the subsequent acquisition rounds.

The exploratory protocol uses 30 calibration seeds and 50 disjoint held-out
seeds. It raises the online transfer-mass cap from 0.15 to 0.45 while retaining
the prequential source-quality gate and exact target-only fallback. The LLM
patches were generated once with `openai/gpt-4o-mini` and frozen before replay;
there are no per-seed model calls.

## Result

The calibration selector rejects transfer on all seven routes. The deployed
policy therefore remains matched target-only LLM for every pair. The raw full
route is still retained for mechanism analysis rather than being hidden behind
the fallback.

One route shows a positive held-out continuous-transfer signal after warm-start
is removed:

| Source -> target | Post-initialization composite effect | LLM increment vs fixed data-only | Action-change rate |
| --- | ---: | ---: | ---: |
| Dielectric -> expt. gap | +6.856 `[+0.107, +13.605]` | +6.880 `[+0.597, +13.164]` | 7.4% |

For this pair, `final_best` improves by +4.210 and `best_so_far_auc` by +2.646
relative to the identical-initialization control. The final-best confidence
interval narrowly crosses zero, while the AUC and composite confidence
intervals are positive. The LLM route also beats the deterministic data-only
patch on both final best and AUC.

This is exploratory evidence, not a deployment result. The 30 calibration
seeds did not predict the held-out gain, so the frozen selector chose fallback.
The route is therefore being rerun under a predeclared independent 50+100 seed
confirmation protocol.

## Other routes

| Source -> target | Post-initialization composite effect | LLM increment vs fixed data-only |
| --- | ---: | ---: |
| ChemLex -> Buchwald-Hartwig | +1.850 `[-1.611, +5.310]` | +1.510 `[-1.861, +4.880]` |
| Expt. gap -> dielectric | +1.231 `[-0.496, +2.959]` | -0.293 `[-3.017, +2.431]` |
| Phonons -> dielectric | +0.439 `[-3.483, +4.362]` | -0.500 `[-1.289, +0.290]` |
| ESOL -> Lipophilicity | +0.078 `[-0.109, +0.264]` | +0.011 `[-0.199, +0.221]` |
| FreeSolv -> Lipophilicity | -0.476 `[-1.153, +0.201]` | +0.059 `[-0.243, +0.362]` |
| Lipophilicity -> FreeSolv | 0 | 0 |

None of these six confidence intervals excludes zero. Their positive means are
not reported as established gains.

## Interpretation

The experiment changes the current CARE 2.0 conclusion in two ways:

1. The large gains in the earlier confirmation are initial-design gains, not
   continuous source-outcome transfer.
2. A continuous and LLM-specific signal is possible on the shared-descriptor
   materials route, but it is not yet calibration-stable or independently
   confirmed.

The action-change rate is diagnostic only. A route receives performance credit
only when changed actions improve held-out final best or best-so-far AUC.

## Files

- `mechanism_controls.csv`, `.json`, and `.md`: compact seven-pair readout.
- `raw_summaries/` and `raw_metrics/`: aggregate and per-seed results.
- `canonical_traces/`: selected-seed reasoning and execution traces.
- `transfer_skills/`: frozen executable skill artifacts and provenance.
- `audit_archives/`: every per-seed audit JSONL file, compressed by pair.
- `logs/`: server driver and pair logs.
- `run_manifest.json`: seed ranges and code provenance.
- `SHA256SUMS`: integrity checks for every archived artifact.

