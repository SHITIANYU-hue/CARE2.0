---
name: care2-wetlab-transfer
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

### Mixed-variable multi-source transfer

1. Validate that every source and the target share named variables, units, bounds, and categorical semantics.
2. Normalize continuous variables from the declared design bounds; never infer scaling from hidden target outcomes.
3. Fit one source expert per completed task instead of concatenating all source rows into one anonymous table.
4. Update source-versus-target expert weights only from target observations already returned by the wet lab.
5. Propose one unrevealed experiment and wait for tell(outcome) before the next proposal.
6. Deploy a source-informed route only when task-disjoint development evidence clears the frozen AUC confidence gate; otherwise execute the target-only fallback exactly.

Abstain from this skill when:
- source and target variables do not have a declared mapping
- all source experts receive negligible online weight
- the proposed condition violates target bounds or laboratory constraints

Evidence: `reizman_mixed_space_contract`, `separate_source_experts`, `exact_target_fallback`, `reizman_development_transfer_gate_v1`

## Non-negotiable boundaries

- Never read unrevealed target outcomes to choose a route or tune a threshold.
- Never merge incompatible source and target spaces without a declared mapping.
- Do not call a target-only fallback a successful source-transfer result.
- Keep rejected and negative-transfer evidence; it defines when to abstain.
- Evaluate an evolved skill on task-disjoint data before promoting it.

Detailed evidence and provenance are in `references/evidence.json`.
