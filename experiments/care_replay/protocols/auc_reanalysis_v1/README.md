# AUC reanalysis v1 — approved offline preparation

The user approved D1/D2/D3; see APPROVED_DECISIONS.json and config.json. This phase certifies inputs and freezes the normalization implementation. It does not compute new normalized effects on real data.

Preparation sequence: `python -B certify_pools.py`, `python -B certify_pools.py --historical`, `python -B prepare_tables.py`, `python -B freeze.py`. The last command runs all tests and verifies source hashes before writing an immutable FREEZE.json. Existing outputs are immutable; do not overwrite a different certification or regenerate a freeze silently.

Read outputs/auc_reanalysis_v1/PREPARATION_REPORT.md from the care_replay root for provenance and scope. Historical budget-conflict studies remain gaps outside the 32 core records. All network access is blocked in preparation processes; no optimizer or model API is executed.
