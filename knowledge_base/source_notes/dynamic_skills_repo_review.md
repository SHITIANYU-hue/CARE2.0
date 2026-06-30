# CARE 1.0 Dynamic Skills Repo Review

Review date: 2026-06-30

This note summarizes the private CARE 1.0 dynamic-skills repository for CARE
2.0 planning. The local checkout inspected was branch `clean-main` at commit
`ea5ed510`.

## What This Repo Is

The repository is an HTE replay harness for studying how LLM-derived or
memory-derived signals can be used in finite-pool experimental optimization.
It is not a general cross-domain CARE 2.0 platform yet. It is mainly organized
around reaction HTE tasks:

- MINERVA/Olympus Suzuki I.
- ChemLex acid-amine wetlab.
- LLM-assisted BO.
- Direct LLM candidate selection diagnostics.
- Memory-based direct LLM selection diagnostics.

The most important design point is that the safer mainline keeps BO as the
primary optimizer. LLMs, memory, and static skills only provide bounded
auxiliary signals. Those signals must pass schema validation, leakage checks,
and a conservative gate before they can affect candidate ranking.

## Main Components

### Replay Boundary

`ReplayTables` separates visible candidate inputs from evaluator-owned
outcomes. `OfflineEvaluator` is the only component that can reveal target
values, and only after a candidate has been selected. Decision-facing artifacts
are sanitized so hidden outcomes, raw row indices, target-derived labels, and
oracle-like fields do not leak into selection.

This is a direct predecessor of the CARE 2.0 audit boundary.

### Static Skills

The repo implements two simple static guidance emitters:

- `HighPerformingRegionSkill`: gives a bounded bonus near the best observed
  local region.
- `RiskQCSkill`: gives a bounded penalty near the worst observed local region.

These are not domain-general scientific skills. They are examples of how an
observed pattern can be converted into a small, inspectable score adjustment.

### Structured Memory

Memory is represented as `MemoryEntry` objects, not as model fine-tuning. The
supported scopes are:

- `group_preference`
- `similarity_region`

`MemoryCompiler` turns active memory entries into `GuidancePatch` objects. The
patches can adjust candidate scores or group scores within hard limits. This is
the part most relevant to CARE 2.0 transfer: the transferable object should be a
structured, confidence-scored skill or hypothesis, not raw prompt text.

### Validator and Gate

`PatchValidator` rejects direct final selections, objective edits, target-like
fields, evaluator-only fields, unknown candidate IDs, and out-of-bounds
adjustments.

`ConservativeGate` compares BO-only and guided selections. It rejects guidance
if the raw BO acquisition loss is too high or the score modifier exceeds the
configured bound. In the default conservative setting, guidance cannot freely
override BO.

### LLM Decision Diagnostics

The repo also includes direct LLM policies where the model chooses a candidate
ID from a menu. The documentation is careful about the conclusion: fixed direct
LLM selection is unstable and sensitive to prompt, candidate order, and
history perturbations. The recommended future direction is not unrestricted
LLM selection, but interfaces such as:

- `RankThenSelect`
- `Memory + RankThenSelect`
- `Planner + RankThenSelect`
- `CriticGuardedRankThenSelect`

This matters for CARE 2.0 because cross-domain transfer should not mean letting
the LLM directly pick experiments across chemistry, materials, and molecules.
It should mean letting the LLM propose auditable, bounded skill artifacts that
external logic can verify.

## Local Smoke Check

I ran the repository's no-API smoke checks in a temporary venv with
`pandas` and `scikit-learn` installed.

MINERVA command:

```bash
/tmp/care-dynamic-skills-venv/bin/python -m hte_replay.experiments.run_minerva --seed 0
```

Output directory:

```text
results/hte_replay_minerva/minerva_seed0
```

Key metrics:

- rounds completed: 4
- total evaluations: 12
- oracle best: 89.2348
- best observed: 73.6110
- simple regret: 15.6238

ChemLex command:

```bash
/tmp/care-dynamic-skills-venv/bin/python -m hte_replay.experiments.run_chemlex --seed 0
```

Output directory:

```text
results/hte_replay_chemlex/chemlex_seed0
```

Key metrics:

- rounds completed: 4
- total evaluations: 12
- oracle best: 100.0
- best observed: 95.97
- simple regret: 4.03

These are smoke checks only. They confirm that the replay boundary and logging
path run locally; they are not paper-level results.

## What CARE 2.0 Should Borrow

CARE 2.0 should borrow the control pattern, not the narrow HTE assumptions:

```text
observed public evidence
-> structured memory / skill / hypothesis
-> bounded guidance patch
-> validator
-> gate
-> BO- or surrogate-centered final selection
-> evaluator reveal
-> audit log and knowledge update
```

For cross-domain transfer, the unit of transfer should contain:

- source domain and task.
- target domain and task.
- source evidence summary.
- candidate feature or factor scope.
- support count and counterexamples.
- confidence score.
- domain-distance discount.
- bounded action template.
- validator requirements.
- gate certificate fields.

This lets a skill learned in one task be reused in another task without
claiming that the raw chemical rule is universally true.

## Implications for CARE 2.0 Cross-Domain Transfer

The next CARE 2.0 milestone should explicitly test transfer rather than only
multi-dataset replay. The minimum useful experiment design is:

1. Learn or extract a structured skill from a source task.
2. Convert it into a domain-neutral transfer card.
3. Discount its confidence when moving to the target task.
4. Apply it only as bounded guidance.
5. Compare matched seeds against no-transfer and no-gate baselines.
6. Record whether the gate accepted, rejected, or corrected the transferred
   skill.

Near-domain transfer should come first:

- Buchwald-Hartwig to Suzuki-Miyaura.
- MINERVA Suzuki to public Suzuki-Miyaura.
- ChemLex acid-amine to another acid-amine or amide coupling task once a clean
  target table is available.

Then broader cross-domain transfer can be tested:

- reaction HTE to molecular property search.
- reaction HTE to materials property search.
- molecular property search to materials composition search.

The broad-domain experiments should be framed as testing whether the CARE
control interface transfers, not whether a reaction-specific chemical heuristic
directly transfers.

## Practical Design Decision

For CARE 2.0, the agent should not be described as "writing arbitrary skills
and directly using them." A more accurate description is:

The agent proposes structured, auditable skill updates from observed evidence
and retrieved knowledge. The system compiles those updates into bounded
candidate-score patches. A validator and gate decide whether each patch is safe
enough to affect selection. The final action remains controlled by the replay
or optimization harness.

That is the clean bridge from CARE 1.0 dynamic skills to CARE 2.0
cross-domain transfer.
