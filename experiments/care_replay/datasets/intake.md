# Dataset Intake Notes

This note tracks datasets requested for the next CARE 2.0 experiment pass. A
dataset is ready for adapter implementation only when its candidate schema,
observable fields, hidden target, objective, license, and evidence boundary are
clear.

## Kimi-Lex

Status: blocked on download link and schema.

Owner split from group sync:

- Weiyi: provide the Kimi-Lex download link.
- Franklin: build the dataset adapter and run replay once the link/schema are
  available.

Needed fields:

| Requirement | Status |
| --- | --- |
| Download URL or repository | missing |
| License / redistribution boundary | missing |
| File format | missing |
| Candidate columns | missing |
| Public observable fields | missing |
| Hidden target / objective | missing |
| Recommended replay budget | missing |

Adapter target:

```text
candidate_id
public feature columns
hidden target
optional group column
metadata
```

## Materials Project

Status: blocked on access and replay framing.

Materials Project can support a materials-discovery replay, but we should not
commit a raw API cache until the data policy is explicit.

Needed decisions:

| Requirement | Options / Notes |
| --- | --- |
| Access mode | API key, public dump, or curated release-safe subset |
| Candidate identity | material id, composition, structure, or formula |
| Public features | composition descriptors, element groups, known metadata |
| Hidden target | band gap, formation energy, stability, synthesis success, etc. |
| Replay objective | maximize/minimize/target-range property search |
| Cache policy | whether raw downloaded records can be tracked in git |

Initial adapter target:

```text
candidate_id = material_id or formula-derived id
group = composition / element-family bin
x1/x2/x3 = normalized public descriptors
objective_value = normalized target property
metadata = formula, source id, raw target name
```

## Ready-to-Implement Checklist

Before adding a new adapter to `run_synthetic_suzuki.py`, confirm:

1. The dataset can be loaded without private credentials, or credentials remain
   outside git.
2. The raw data can be tracked, or the adapter can reproduce the download.
3. The hidden target is not used by the policy before reveal.
4. Candidate count is reasonable for 30 seeds x 10 rounds.
5. The adapter maps cleanly into the existing `Candidate` / `DatasetAdapter`
   interface.

