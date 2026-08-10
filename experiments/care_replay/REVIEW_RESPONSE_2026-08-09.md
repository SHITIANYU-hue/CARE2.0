# Response to the 2026-08-09 CARE 2.0 discussion

## Changes adopted

1. **Persistent skill memory.** A versioned `SkillBank` now writes an
   agent-readable `SKILL.md`, a detailed evidence file, and a fingerprinted
   manifest. This addresses the valid criticism that the former pair-local
   `TransferSkill` was a structured policy for one source-target pair rather
   than accumulated reusable experience.
2. **Multi-source to new-task evaluation.** The new benchmark uses completed
   Reizman cases 1-3 as separate source experts and case 4 as the untouched
   target. It compares target GP-UCB, multi-source RGPE, ICM-BMA, and a
   source-rank skill prior under a matched budget.
3. **Strict development/confirmation split.** Route and weight selection occur
   only through leave-one-development-task-out replay. The confirmation runner
   verifies a configuration hash before loading case 4.
4. **Real wet-lab interface.** The adapter now represents catalyst as
   categorical and residence time, temperature, and loading as continuous. A
   stateful `ask()` / `tell()` queue keeps unrevealed outcomes outside policy
   state.
5. **Negative evidence is retained.** No tested source policy passed the
   development AUC gate. CARE therefore abstained and matched target GP-UCB on
   case 4. The individual negative and inconclusive transfer results remain in
   the archive.

## Suggestions treated as research hypotheses

- **Replacing the generator with a coding agent** is not, by itself, evidence
  of a better scientific skill. It would add a large confound unless the agent,
  tools, budget, and task-disjoint evaluation are fixed.
- **Trace2Skill-style automatic extraction** is a useful next step, but the
  current implementation deliberately starts with a deterministic compiler so
  evidence boundaries can be tested first.
- **SkillRL or a smaller trained meta-controller** requires a substantially
  larger task corpus than four reaction campaigns. It is not justified by the
  present sample size and is not claimed as implemented.
- **The company deployment history** is not used as scientific evidence because
  there is no executable trace showing that this protocol was run in a live
  laboratory.

## Remaining gap

The new code establishes the missing platform mechanics: persistent skill
artifacts, multi-source routing, mixed-variable wet-lab execution, and honest
new-task confirmation. It has not yet demonstrated positive accumulated-skill
transfer. The next defensible step is to add more completed campaigns and train
or calibrate a task-level router on development campaigns only, then confirm it
once on a new campaign.
