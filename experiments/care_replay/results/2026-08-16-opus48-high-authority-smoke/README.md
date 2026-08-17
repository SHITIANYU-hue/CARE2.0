# High-Authority Smoke Diagnostics

These two-round runs isolated a deterministic tie-breaking error. The first
network run reported a `-0.7143` delta even when Opus selected the recorded
GP-rank-one candidate. `rank_map()` used unstable sorting, while the baseline
used the lowest candidate index for exact ties.

After matching that tie-break, the second run reproduced the target-only GP
result exactly (`0.0` AUC and final delta). The incomplete
`molecular_freesolv_to_esol` trace predates the fix and is retained only as an
audit artifact.

