# ESOL target-only archive-search diagnostic

The common initial policy was generated without source identity or source outcomes. The trained Qwen2.5-32B adapter then ran two recursive updates. The sparse branch used skill-level summary feedback and unconditionally deployed the model's selected skill. The archive branch used candidate-level development trajectories, measured both returned skills against the incumbent on fresh promotion seeds, and preserved only accepted skills as compact playbook cards. All model weights stayed frozen.

This study was the first execution of the archive protocol on a target absent from the 42-row SFT training set. It is not an independent domain test: ESOL appeared as source evidence in earlier training examples, and related FreeSolv examples were used in SFT.

| Final blind arm | Mean best-so-far AUC | Mean final best |
|---|---:|---:|
| Archive search | **93.1066** | **94.2500** |
| Fixed initial | **93.1066** | **94.2500** |
| Sparse ungated | 92.6756 | 93.4762 |
| GP-UCB | 92.8053 | 94.8095 |

On 12 paired final seeds, archive minus sparse AUC was `+0.4310`, bootstrap 95% interval `[+0.0536,+0.9012]`, with 5 wins, 6 ties, and 1 loss. Final-best difference was `+0.7738`, interval `[+0.0238,+1.5655]`. Archive minus fixed initial was exactly zero: **no new skill was promoted**. The main observed benefit was preventing an unhelpful rewrite, not improving the already strong initial policy. The AUC margin over GP-UCB was `+0.3013` with an interval crossing zero; GP-UCB's mean final best was higher.

Both generation rounds returned two skills. The four promotion comparisons were:

| Round | Candidate | Mean AUC delta vs incumbent | W/T/L | Accepted |
|---|---|---:|---:|---|
| 1 | `polar_hbond_donors_plus_compactness` | +0.573 | 1/6/1 | No |
| 1 | `hydrophobic_ring_rich_counter_hypothesis` | -2.569 | 2/1/5 | No |
| 2 | `polar_donor_plus_compact_surface` | +0.313 | 5/2/1 | No |
| 2 | `hydrophobic_ring_rich_counter_hypothesis` | -1.378 | 1/0/7 | No |

The predeclared gate required mean AUC `≥ +1.0`, at least 6 wins on 8 paired promotion seeds, and nonnegative mean final-best delta. The initial policy's AUC was already above 93 on this scale, which limited observable headroom. No accepted archive card existed for round two, so this run does not establish whether accumulating validated cards improves later proposals.

The original FreeSolv→ESOL full-transfer preflight stopped before any model call because this repository defines no role map for that direction. Rather than invent a mapping, this completed run used the generator's existing `target_only` mode. Raw requests confirm FreeSolv identity and outcomes were absent.

The archive used extra promotion trajectories, so the arm comparison is **not compute- or experimental-budget matched**. The candidate maximum on promotion seeds is selection-biased; only the held-out evaluation estimates the selected result. The 12-seed interval is conditional on this fixed target pool, initial draw, model checkpoint, and deterministic generation. It is not cross-task or model-call uncertainty.

All five model calls were valid on the first attempt. `protocol_lock.json` records code and config hashes, `code_snapshot/run_rsi_archive_search.py` preserves the exact runner used here, and `SHA256SUMS` covers every archived file.
