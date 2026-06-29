# CARE 2.0 Task Tracker

This tracker records the current project split from the CARE group sync. It is
kept in-repo so dataset blockers, experiment ideas, and review ownership remain
visible next to the code and results.

## Current Workstreams

| ID | Workstream | Owner | Status | Next Action |
| --- | --- | --- | --- | --- |
| `DATA-KIMI-LEX` | Add Kimi-Lex dataset and replay experiment | Franklin | blocked | Wait for Weiyi's download link and schema notes |
| `DATA-MATERIALS-PROJECT` | Add Materials Project pathway | Franklin | blocked | Confirm API/access, target property, and release-safe cache policy |
| `EXP-ABLATION-IDEAS` | Propose ablation experiments | Zhang Bo'er | waiting_review | Collect review ideas and convert into replay configs |
| `DOC-README-RESULTS` | Keep README and results explanation current | Franklin | active | Update docs after each tracked result snapshot |
| `BRANCH-CARRY` | Preserve current code and create active branch | Franklin | done | `carry1.0` and `carry2.0` branches created |
| `KB-CARD-TYPES` | Extend knowledge-base card model | Franklin | active | Add card types for result, run log, transfer, data request, review |
| `KB-REVIEW` | Review knowledge-base code and card model | Zhang Bo'er | waiting_review | Review schema and suggest changes |

## Immediate Dataset Intake Questions

### Kimi-Lex

Needed before implementation:

- Download URL or repository path.
- License / redistribution boundary.
- File format and schema.
- Candidate columns, observable fields, hidden target, and objective.
- Whether it should be treated as public, deidentified, or internal.

### Materials Project

Needed before implementation:

- Access mode: public dump, Materials Project API key, or curated subset.
- Target property: for example formation energy, band gap, stability, or
  synthesis success.
- Candidate representation: composition, structure, formula, or process.
- Raw-data policy: whether cached API results can be committed.
- Replay framing: finite pool size, public features, hidden target, and budget.

## Proposed Ablation Directions

These are ready for review:

1. No-transfer vs BH-to-Suzuki confidence-discount transfer.
2. Gate v1 vs gate v2 under identical challenger scores.
3. Factor-evidence ranker only vs risk-penalty only vs diversity only.
4. With/without public factor evidence model on real HTE datasets.
5. Synthetic-only skill priors vs evidence-derived skill adjustments.

