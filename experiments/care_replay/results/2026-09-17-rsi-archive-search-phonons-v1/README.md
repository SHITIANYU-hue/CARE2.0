# Phonons two-candidate archive-search follow-up

This follow-up reused the previously published trained-adapter initial policy for phonons and switched to new, disjoint development, promotion, and final evaluation seeds. Each of two archive rounds asked the frozen model for two skills. Both were compared with the incumbent on the same promotion seeds. Only a passing winner entered the short validated-policy archive carried to the next round. A matched sparse-feedback branch kept the previous unconditional model selection. This is a **new-seed follow-up on the same task**, not a new-target generalization test.

| Final blind arm | Mean best-so-far AUC | Mean final best |
|---|---:|---:|
| Archive search | **57.9008** | **77.0853** |
| Fixed initial | 49.0279 | 74.8103 |
| Sparse ungated | 49.0279 | 74.8103 |
| GP-UCB | 45.4788 | 60.2750 |

On 12 paired blind seeds, archive minus sparse/fixed AUC was `+8.8730`, bootstrap 95% interval `[-4.4360,+22.2690]`, with 9 wins and 3 losses. Final-best difference was `+2.2750`, interval `[-20.2489,+25.7035]`. Both intervals cross zero. The estimate supports further testing but not a robust superiority claim. The sparse final policy changed its ID but retained the initial policy's active rules and acquisition settings, explaining its exact tie with fixed initial.

The predeclared phonons gate required mean AUC `≥ +1.0`, at least 5 wins on 8 paired promotion seeds, and nonnegative mean final-best delta:

| Round | Candidate | Mean promotion AUC delta | W/T/L | Decision |
|---|---|---:|---:|---|
| 1 | `polarizable_oxide_transfer` | +26.188 | 6/1/1 | Promote |
| 1 | `light_element_counter_hypothesis` | -6.760 | 3/0/5 | Reject |
| 2 | `polarizable_oxide_transfer_v2` | -8.081 | 1/1/6 | Reject |
| 2 | `halogen_chalcogen_counter_hypothesis` | -3.208 | 3/0/5 | Reject |

The accepted strategy replaced the initial chalcogenide-positive rule with an oxide-positive rule, retained moderate concentration and mid atomic number, and added a no-lanthanide rule. Its acquisition settings (`semantic_mass`, UCB weight, GP beta, and xi) stayed at the initial values. The model's physical rationales are hypotheses, **not established mechanisms**; the measured evidence only supports a policy-level association on this finite pool. The archive stored the accepted skill and its promotion result as one compact card for round two. Round two produced no further accepted update.

The promoted skill was also the model's first-choice skill. Thus this run does not demonstrate that evaluating the second candidate improves selection; it demonstrates that the independent gate accepted one candidate and rejected three. The two-candidate promotion maximum is selection-biased. The archive arm also used more promotion trajectories than the sparse arm, so this is not a budget-matched algorithm comparison. Phonons was absent as an SFT target, but related materials schemas and dielectric source evidence were used in training.

All four model calls were valid on the first attempt. `protocol_lock.json` records the frozen configuration and implementation hashes; `code_snapshot/run_rsi_archive_search.py` preserves the exact runner; `SHA256SUMS` covers all evidence.
