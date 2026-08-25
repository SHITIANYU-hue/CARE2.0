# Archived LLM probability-calibration audit

The complete retrospective portfolio contains 170 decisions across 17 routes. 27 decisions improved the pre-round best (0.159).

| Probability source | Mean probability | Observed rate | Bias | Brier | Log loss | ECE | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final LLM decision | 0.219 | 0.159 | +0.060 | 0.133 | 0.486 | 0.060 | 0.679 |
| Production GP, same candidate | 0.182 | 0.159 | +0.023 | 0.134 | 0.561 | 0.023 | 0.628 |

Equal-route mean LLM-minus-GP Brier difference: -0.0016 (route-bootstrap 95% CI [-0.0132, +0.0090]). Negative values favor the LLM.
Equal-route mean LLM-minus-GP ROC-AUC difference: +0.0308 (route-bootstrap 95% CI [-0.0581, +0.1373]; 17 routes with both outcomes).

The route-level interval crosses zero, so the audit does not establish a calibration advantage for either probability source.

GP probability provenance: 110 recorded in the archived menu and 60 reconstructed from the archived GP posterior mean, standard deviation, and current best.

This comparison scores the LLM and GP probability assigned to the same executed candidate, so it tests probability estimation rather than action selection. The audit is retrospective, pools multiple model/controller versions, and does not validate a deployment gate.
