# CARE 2.0 Evaluation Completion (2026-07-28)

This package addresses the 7/28 review requests: broader baseline accounting,
iteration-level analysis, an explicit formula description of LLM-generated
rules, and additional publication-ready figures.

## What was added

### 1. Baseline comparison

`baseline_comparison_forest.png` compares the frozen CARE deployment against
two different primary references:

- the strongest target-only BO method for each source-target pair;
- the matched target-only LLM policy with identical target budget and seeds.

The distinction matters. The first panel measures the performance of the full
CARE deployment; the second isolates the gain attributable to source-outcome
transfer. On the formal seven-pair suite, transfer is deployed on four pairs
and beats both references on those four. The other three pairs use exact
target-only fallback and are therefore zero, not negative, against the matched
LLM comparator.

`baseline_protocol_matrix.png` inventories all implemented comparator
families: random search, public incumbent, GP-UCB, mixed-kernel GP-EI, target
UCB/EI portfolio, LLM direct prior, LLAMBO-style warm start, matched target-only
LLM, matched random rule, and the CARE source-outcome router. It separates
optimizer strength from source evidence and evaluation controls. The matrix is
an implementation inventory; historical controls were not all rerun on every
one of the seven frozen paths.

### 2. Iteration and experiment efficiency

`iteration_efficiency.png` reports, for every formal path:

- rounds saved to the matched target-only LLM's final quality;
- rounds saved to the global top-10 candidate set;
- best-so-far delta at rounds 1, 3, 5, and 13.

Each estimate is paired by held-out seed. Initial observations count as round
0, and a missed threshold is right-censored at replay budget plus one. This
makes the “fewer wet-lab rounds” claim explicit instead of inferring it from
final quality alone.

### 3. LLM rule formulas

`LLM_RULE_FORMULAS.md` maps every major rule object to its executed equation:

- role-weighted kernel geometry;
- GP-UCB, GP-EI, and target portfolio anchor;
- source-neighbor, additive, and interaction priors;
- leave-one-out target calibration;
- evidence-weighted expert routing;
- transfer mass and online safety gate;
- paired final, AUC, and round-efficiency metrics.

`llm_rule_formula_map.png` gives the same execution path as a compact diagram.
It also makes the division of labor explicit: the LLM proposes a bounded patch;
Python calibrates it from observed target evidence and executes it
deterministically.

## Data files

- `baseline_comparison.csv`: pair-level deltas and confidence intervals against
  strongest BO and matched target-only LLM.
- `iteration_efficiency.csv`: pair-level round savings and early-round deltas.
- `baseline_protocol_coverage.csv`: method-by-protocol coverage used in the
  baseline matrix.

All result values are read from the frozen source-outcome archive and goal
report. This package does not recompute or hand-enter held-out performance.

## Rebuild

```bash
python3 experiments/care_replay/scripts/build_20260728_evaluation_completion.py
```
