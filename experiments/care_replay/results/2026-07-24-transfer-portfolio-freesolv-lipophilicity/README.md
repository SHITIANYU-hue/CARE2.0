# Frozen Transfer Portfolio: FreeSolv -> Lipophilicity

This archive records a 10-seed calibration / 20-seed held-out portfolio
extension for a real MoleculeNet transfer path.

## Protocol

- Source: MoleculeNet FreeSolv.
- Target: MoleculeNet Lipophilicity.
- Target budget: 2 initial observations + 13 reveal rounds.
- Candidate portfolio, fixed before replay: standard transfer, conservative
  transfer, and source-positive initial design.
- Candidate selection used calibration outcomes only.
- Fallback: the frozen matched target-only LLM route.
- Source history, descriptor mappings, and LLM patches were frozen before
  target replay.

## Result

No candidate passed the joint calibration gate against both the matched
target-only LLM and the strongest target-only BO anchor. The deployed
held-out route therefore exactly fell back to target-only: final-best and
best-so-far AUC deltas versus the matched target-only LLM were both 0 over
20 held-out seeds.

The raw candidate metrics and all 260 calibration/held-out audit files are
retained here, with hashes in `SHA256SUMS`. This is a negative control for
automatic portfolio selection, not a positive source-outcome transfer claim.
It is deliberately reported alongside the earlier 50/100-seed
FreeSolv -> Lipophilicity value-prior result, which used a different frozen
protocol and remains the main molecular transfer result.
