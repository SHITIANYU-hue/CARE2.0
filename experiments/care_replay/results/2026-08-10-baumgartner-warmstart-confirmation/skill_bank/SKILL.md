---
name: care2-baumgartner-initial-design
description: Evidence-bounded multi-source transfer for sequential scientific experiments.
---

# CARE 2.0 reusable transfer

Use this skill when completed experiments may help rank candidates for a new
sequential optimization task. Treat every source as a separate expert and
preserve an exact target-only fallback.

## Required inputs

- Public source schemas and measured source outcomes.
- Public target schema, bounds, constraints, and observations revealed so far.
- A declared development/evaluation task split.
- A fixed target-only acquisition policy for fallback.

## Procedure

### Source-guided diverse initial design

1. Validate the target variables, units, bounds, and categorical semantics against the source campaign contract.
2. Choose completed sources with the same substrate; if none exist, use sources with the same precatalyst; otherwise use all compatible sources.
3. Fit one GP expert per source and aggregate candidate predictions through median normalized ranks.
4. Select the first experiment by source rank, then select two more with 0.15 source-rank weight and 0.85 mixed-space diversity weight.
5. After the three initial outcomes are revealed, discard the source prior and run the frozen target-only GP-UCB for subsequent proposals.
6. Compare deployment against both random initial design and pure space filling; retain all negative-transfer evidence.

Abstain from this skill when:
- source and target decision spaces lack a declared mapping
- target constraints make any proposed condition invalid
- task-disjoint development evidence fails the frozen confidence and non-loss gate

Evidence: `baumgartner_campaign_space_contract`, `baumgartner_diverse_warmstart_development_v1`

## Non-negotiable boundaries

- Never read unrevealed target outcomes to choose a route or tune a threshold.
- Never merge incompatible source and target spaces without a declared mapping.
- Do not call a target-only fallback a successful source-transfer result.
- Keep rejected and negative-transfer evidence; it defines when to abstain.
- Evaluate an evolved skill on task-disjoint data before promoting it.

Detailed evidence and provenance are in `references/evidence.json`.
