# Experiment 3: approved offline execution v1

This execution applies the user-approved source-fit-only score conversion to the already locked 23-route panel. Historical protocols and data projections are unchanged. Outcome access is authorized only after FREEZE.json has been created and verified.

## Fixed procedure

- Each route uses exactly its frozen 1,024 source-fit variants, original G0 128-pool, matched G1 128-pool when supported, and frozen target pool (at most 5,000).
- Source-only score units: 100 × (DMS_score − source median) / source IQR, NumPy linear quantiles, no clipping. ProteinGym processed scores are already higher-is-better. Zero source IQR excludes and reports the route; other missing/invalid effects stop execution.
- Reuse legacy public protein features, shrunk additive mutation effects, mixed-kernel GP and rank tie handling. GP beta 1.5, numeric scale .45, categorical scale 4, noise .05; additive authority .5 to .1. Source inducing cap is immaterial because this fixed additive operator fits no source GP.
- Each trajectory reveals 3 initial and 12 additional outcomes. Mean best after each of 12 rounds is AUC. Always and Never share initial indices. Four-arm policy outcomes reuse their same two trajectories.
- G0/G1 use 100 paired calibration seeds per route. Unsupported G1 is Never and its unused budget is not reallocated. Legacy six-decimal episode AUC and six-decimal population-SD normal CI lower > 0 determine gate acceptance.
- All 23 route gate records and all calibration checkpoint hashes are locked before any heldout stage. Each route then runs 200 new paired seeds, without target-based recalibration. Each route has separate source units and fitted prior even when routes share an assay.
- Paired heldout full-precision AUC differences /100 are in source-IQR units. Normal 95% interval, primary practical margin .02; prespecified .00/.01/.05 sensitivities. Beneficial, harmful, near-zero and uncertain labels all remain reported.
- Whole candidate-dependency-group bootstrap: 10,000 resamples, fixed seed 917430281, route-weighted estimates and equal-group-weight sensitivity. Paired false-rate differences report valid bootstrap denominator counts. No noninferiority or equivalence claim; no multiplicity-adjusted route claims.
- Best-of-Always/Never regret is descriptive because its reference is estimated from the same heldout outcomes. This is within-assay distribution transfer, with 13 candidate dependency groups; remote family independence is not certified.

## Isolation and reproducibility

Public candidate objects contain NaN placeholders; the policy never reads them. It gets only a reveal callback for the selected index. The outcome loader numerically parses only declared role IDs: fit plus calibration before the gate lock, fit plus target afterward. It may scan raw CSV strings to identify sequence IDs. Different routes can share measurements, so the boundary is route-local access and nonadaptive frozen gates, not a claim that no measurement used as a target anywhere has ever been read as another route's source. The same OS user is not an independent custodian.

Network connect/connect_ex/create_connection/sendto are disabled in the runner. No API credentials are loaded. Atomic per-seed checkpoint envelopes contain the freeze hash, pool ID digest, seed, record checksum, both trajectories, observations and AUC. Resume refuses changed inputs or records. Full raw ZIP integrity is checked after the freeze; permissions return to 000 on process exit.

## Commands (WSL, repository .venv)

1. `python -B protocols/experiment3_execution_v1/preflight.py`
2. Review offline results, then `python -B protocols/experiment3_execution_v1/freeze.py`
3. `python -B protocols/experiment3_execution_v1/run.py all`
4. `python -B protocols/experiment3_execution_v1/analyze.py`

Outputs are in `outputs/experiment3_execution_v1/`. There are 3,600 calibration pairs and 4,600 heldout pairs (16,400 trajectories, 246,000 simulated observations), unless a source-IQR exclusion applies. These are local repeated observations of existing data, not new wet-lab measurements. All main-experiment API calls and cost are zero.
