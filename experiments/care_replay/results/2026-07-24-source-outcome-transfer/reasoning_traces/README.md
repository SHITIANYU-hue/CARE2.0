# Reasoning trace guide

Each file contains the complete round-by-round JSONL trace for held-out seed
41000. The three modes are:

- `llm_transfer_router`: the raw source-outcome route, whether or not it was
  selected for deployment;
- `matched_target_only_llm`: the matched LLM baseline without source outcomes;
- `care_source_outcome_router`: the frozen deployed route. On rejected pairs
  this is an exact copy of the matched target-only trace.

Important fields:

- `source_initial_design`: which random target probes were retained, which
  source-informed probes replaced them, and the frozen source score used;
- `route_diagnostics`: neighbor, additive, interaction, kernel, and
  prequential-calibration diagnostics for every LLM patch;
- `router_gate`: target anchor, raw source candidate, authorization decision,
  risk budget, observed support, and rejection reason;
- `expert_weights` and `transfer_mass`: how much source evidence entered the
  deployed acquisition after calibration;
- `selected_candidate`, `revealed_value`, and `best_so_far`: the actual replay
  decision and its newly revealed outcome.

For example, the ChemLex → Buchwald-Hartwig trace starts with two target probes
chosen from fixed ChemLex source outcomes. At round 0 the source router and
target GP-UCB anchor agree, so the gate records
`router_matches_target_anchor`. Later rounds expose when calibrated source
priors alter the ranking. The ESOL → Lipophilicity raw trace still records its
source-informed probes and interactions, but the frozen deployed trace falls
back to the target-only LLM because calibration rejected that route.
