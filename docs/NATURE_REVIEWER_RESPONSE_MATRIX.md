# CARE 2.0: Nature Reviewer Response Matrix

Updated 2026-08-25. This is an internal writing and review aid, not manuscript
text. Every answer below is limited to evidence already present in the
repository. Planned Amylase and wet-lab work must not be described as results.

## Likely reviewer challenges

| Reviewer challenge | Evidence-based response | Evidence to cite | Wording boundary |
|---|---|---|---|
| The replay code can see hidden target labels. | The production-path intervention audit changed 2,725,689 unrevealed label positions across 170 archived decision states. Source priors, candidate menus, diagnostics, and prompts were invariant in all 170 states. | `experiments/care_replay/results/2026-08-24-hidden-target-noninterference-v1/` | This proves the tested implementation boundary. It does not prove scientific efficacy or eliminate every possible leakage route. |
| The transfer rule was tuned on the test outcomes. | Rhomax, IRED, and TrpB protocols were committed before their official raw files were downloaded. Configs, code hashes, seeds, LLM traces, and fallbacks were frozen. | Rhomax commit `4130dd4`; IRED commit `5389a9c`; TrpB commit `c5f4065`; each preregistration directory | Do not call retrospective chemistry or materials replays prospective. Amylase remains planned until its LLM record and lock are pushed. |
| The LLM is decorative and does not affect experiments. | The strongest LLM evidence is the offline compiler, not the first live online pilot. Outcome-blind Opus selected an executable additive skill that improved IRED AUC by +0.389 and TrpB AUC by +0.949. The live bounded-authority pilot selected GP rank one in all six calls and had zero online increment; that negative result is retained. | IRED and TrpB prospective result folders; `pilot_audit/decision_impact_audit.json` | Say that the LLM generated useful executable skills in two external families. Do not say the current online controller consistently improves actions. |
| A hand-written prior is stronger than the LLM. | On IRED, the fixed prior is indeed stronger (+0.864 versus +0.389), and this is reported. On TrpB, the LLM additive skill (+0.949) is stronger than the generic fixed prior (+0.154), RGPE, and ICM-BMA. | IRED and TrpB prospective comparison tables | The paper can claim nontrivial LLM skill generation, not universal LLM superiority. |
| The source outcomes may be irrelevant; geometry alone may explain the gain. | IRED showed a positive true-versus-mean-permutation difference, but its assignment test was p=0.07. TrpB preregistered the same test and rejected the 99-permutation null at p=0.01. | IRED falsification and TrpB source-outcome confirmation folders | Source-outcome attribution is confirmed for the TrpB additive skill only. It is not established across domains. |
| The safety gate is cherry-picked. | The same evidence table includes a correct harmful-transfer rejection (Rhomax), a correct positive deployment (IRED), and a false-negative hold (TrpB). Candidate and deployed effects are reported separately. | `results/2026-08-25-prospective-external-family-triad-v1/` | Do not claim that CARE eliminates negative transfer. The router has demonstrated precision but insufficient recall. |
| TrpB proves the full CARE system works. | It does not. The LLM skill improved by +0.949, but the frozen router selected HOLD, so deployed gain was exactly zero. The 0% calibration versus 99.97% deployment token coverage explains the power failure. | TrpB gate decision, calibration report, and triad figure | Always report `candidate +0.949; deployed 0.000`. Never substitute the candidate result for the complete-system result. |
| Three protein families are too few for generalization. | Agreed as a statistical boundary. The three-family synthesis is descriptive. Chemistry, molecular, and materials replay demonstrates interface breadth, but the clean prospective external evidence is currently protein-family transfer. | Supplementary evidence map and triad figure | Do not claim universal cross-domain transfer. A more distant untouched domain is still required. |
| Percent improvement alone is not significance. | Primary comparisons use paired seed-level AUC effects with 95% intervals. Mechanism claims use assignment-level randomization tests. Family count is not inflated by treating seeds as independent domains. | Route-effect CSVs, plot-stat JSONs, statistical methods in the supplement | State the unit of analysis beside every interval. Do not use `significant` for the three-family descriptive synthesis. |
| Replay is not a scientific discovery. | Replay is used to compare policies under a hidden-label sequential budget with exact counterfactual baselines. It is reproducible evidence about decision quality, not a new wet-lab discovery. | Replay harness, protocol locks, trace records | Do not use `discovered a new molecule/material/protein`. A matched prospective wet-lab campaign remains mandatory for that claim. |
| The method depends on one closed model or one lucky call. | Model, temperature, prompt, raw responses, hashes, usage, failures, and fallbacks are archived. Existing repeated-call and GLM protocols expose the gap, but neither is complete enough to establish model robustness. | Repeated-confirmation and GLM protocol documents | Report model dependence as an open limitation until frozen repetitions complete. |
| The experiment is not reproducible. | Code, configs, public dataset URLs, seeds, raw LLM traces, parsed decisions, route-level outputs, figure source data, and SHA-256 sidecars are versioned. | `docs/NATURE_SUBMISSION_READINESS.md`, appendix artifact map, result sidecars | Before submission, mint an immutable release and DOI. A mutable branch is not the final archival package. |

## Main-text requirements

The main text should make these distinctions on first use, not only in the
supplement:

1. **Candidate skill efficacy** asks whether an executable transferred skill
   would improve the matched target-only policy.
2. **Router deployment efficacy** asks whether the frozen complete system
   actually deployed that skill and improved the target campaign.
3. **Source-outcome attribution** asks whether measured source outcomes, rather
   than source geometry or a generic prior, caused the skill advantage.
4. **Prospective** means that the protocol and analysis were fixed before the
   official raw target file was downloaded. A retrospective split is not made
   prospective by changing seeds.
5. **Generalization** is assessed at the task-family level. One hundred target
   seeds quantify within-family policy stability; they do not create one
   hundred independent domains.

## Phrases to avoid

- "CARE 2.0 achieves universal cross-domain transfer."
- "The LLM autonomously discovers new scientific laws."
- "The gate eliminates negative transfer."
- "TrpB improves the complete CARE system by 0.949 AUC."
- "All external experiments are statistically significant."
- "The knowledge base self-evolves without human oversight."
- "Replay demonstrates wet-lab discovery."

## Defensible headline

> CARE 2.0 compiles source evidence into auditable LLM-generated scientific
> skills, evaluates them against matched target-only and transfer-BO baselines,
> and separates useful transfer, harmful transfer, and router abstention under
> preregistered finite experimental budgets.

This headline remains intentionally narrower than a claim of general autonomous
discovery. The missing evidence is explicit: a coverage-aware Amylase router
confirmation, replication in a more distant task family, and a paired
prospective wet-lab campaign.
