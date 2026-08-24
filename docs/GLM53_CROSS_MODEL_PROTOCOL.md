# GLM-5.3 Cross-Model Robustness Protocol

Updated 2026-08-24.

## Question

This experiment asks whether CARE 2.0's online transfer controller depends on
one foundation model. GLM-5.3 receives the same eleven source-target routes,
frozen initial target observations, candidate-menu policy, ten-reveal budget,
and same-initial target-only GP-UCB comparator as the separately frozen Opus
study. GLM and Opus results remain separate primary analyses.

The experiment isolates the online controller. The initial design and initial
hypothesis are held fixed across models; GLM-5.3 does not regenerate them. This
prevents initial-design variability from being mistaken for an online transfer
effect.

## Declared analysis

- Thirty independent API trajectories are declared for each of eleven routes.
- The primary estimand is the equal-route-weighted mean of within-route mean
  best-so-far AUC deltas against same-initial target-only GP-UCB.
- The primary success rule requires the two-sided hierarchical-bootstrap 95%
  interval to lie entirely above zero.
- Route-level sign tests are secondary and use Benjamini-Hochberg correction.
- Partial execution cannot produce a confirmatory claim.
- Negative scientific outcomes are never a reason to retry a trajectory.

## Provider adaptation history

Provider adaptation is versioned because it was introduced after observing an
operational failure. Earlier outputs are retained and are not reclassified.

| Version | Frozen change | Observed status |
|---|---|---|
| v1 | Same controller with the standard OpenAI-compatible client | Calls were valid initially, but HTTP 429 rate limits prevented a complete trajectory. |
| v2 | Added low-frequency request pacing, `reasoning_effort=low`, and in-place retry of an identical failed HTTP request | Rate limiting was controlled, but a critic returned the valid decision inside a persistent `answer` envelope; strict validation terminated the trajectory. |
| v3 | Added a transparent single-key JSON envelope adapter | The first execution exhausted eight in-place HTTP retries over approximately twelve minutes; a manually interrupted outer retry is retained separately. No scientific trajectory completed. |

The v3 adapter cannot rename fields, substitute candidates, coerce values,
fill missing fields, or repair scientific content. Responses with additional
top-level keys remain unchanged. This makes the adaptation a transport-schema
normalization rather than a scientific-policy change.

## Audit policy

Every API attempt, model-validation error, selected candidate, reveal, token
count, and fallback is retained. Infrastructure failures, model-format
failures, and negative transfer are reported separately:

- **Infrastructure failure:** the API did not return a usable response because
  of rate limit, timeout, connection error, or server error.
- **Model-format failure:** the API returned model content that still failed the
  frozen schema after the permitted repair.
- **Scientific loss:** a complete trajectory scored below the matched GP-UCB
  comparator. It remains a valid result and is never retried because it lost.

## Authoritative artifacts

- `experiments/care_replay/configs/online_llm_glm53_robustness_v1.json`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v2.json`
- `experiments/care_replay/configs/online_llm_glm53_robustness_v3.json`
- `experiments/care_replay/scripts/run_repeated_online_llm_confirmation_v3.py`
- `experiments/care_replay/scripts/run_repeated_online_llm_confirmation_v4.py`
- `experiments/care_replay/results/2026-08-24-glm53-robustness-v1/`
- `experiments/care_replay/results/2026-08-24-glm53-robustness-v2/`
- `experiments/care_replay/results/2026-08-24-glm53-robustness-v3/`

No API credential is stored in these artifacts.
