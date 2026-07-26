# Frozen Transfer Portfolio: Matbench expt-gap -> dielectric

This archive records a 10-seed calibration / 20-seed held-out portfolio
extension for a real materials transfer path.

- Source: Matbench experimental band gap.
- Target: Matbench dielectric constant.
- Target budget: 2 initial observations + 13 reveal rounds.
- Candidate portfolio: standard transfer, conservative transfer, and
  source-extremes initial design.
- Candidate selection used calibration outcomes only.
- Fallback: the frozen matched target-only LLM route.

No candidate passed the joint calibration gate against both the matched
target-only LLM and the strongest target-only BO anchor. The deployed held-out
route therefore exactly fell back to target-only. This is a negative control for
the portfolio mechanism, not a positive transfer claim. The raw candidate
metrics and every audit are retained, with hashes in `SHA256SUMS`.
