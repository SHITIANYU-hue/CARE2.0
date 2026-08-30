# CARE 2.0 on AstaBench

This directory adds a standards-based external evaluation path for CARE 2.0.
It does not replace the sequential transfer experiments. AstaBench measures
general scientific-agent performance; CARE's replay suite measures whether
source experience improves a target experiment under a matched budget.

## Why DiscoveryBench first

DiscoveryBench is the closest AstaBench task to CARE's scientific loop. The
agent must load supplied datasets, derive a specific hypothesis, and report the
workflow that supports it. The frozen CARE solver turns that into an auditable
sequence:

`schema -> candidate hypotheses -> executable tests -> falsification -> reroute -> supported claim`

The current confirmation protocol is
`configs/discoverybench_protocol_v4.json`. Validation-only development showed
that prompt instructions alone did not stop repeated Python repairs and that a
global 2,000-byte limit also truncated valid final submissions. The CARE
controller now enforces at most three Python calls, limits Python evidence to
2,000 bytes, and preserves final submissions up to 8,192 bytes. The agent must
reach a supported or unresolved submission within the matched 60,000-token
budget. No gold output or test trajectory was used for these changes.

The skill map is domain agnostic and contains no benchmark outcomes. The solver
uses only the tools already provided by AstaBench, so its submission category is
`standard` rather than a custom information-access toolset.

## Frozen validation check

The first frozen paired check used five DiscoveryBench validation samples and
one epoch per arm. CARE produced a strict two-key JSON submission on 5/5
samples, compared with 1/5 for stock ReAct, while using 176,412 versus 306,728
total tokens. This is a structural-completion and efficiency result only: the
official scorer model was unavailable on the configured endpoint, so no
scientific-quality score is reported. The auditable result package and native
Inspect logs are in
`results/2026-08-30-validation-paired5-v1/`.

## Reproducible setup

```bash
cd experiments/astabench
uv sync --group dev
uv run pytest
```

Official AstaBench data are gated. Accept the dataset license and set an
authenticated Hugging Face token before running the benchmark:

```bash
export HF_TOKEN="..."
export OPENAI_API_KEY="..."  # CommonStack or another compatible provider
export OPENAI_BASE_URL="https://api.commonstack.ai/v1"
```

Run the validation split first. The model name must be identical for CARE and
the ReAct control.

```bash
uv run inspect eval \
  astabench/discoverybench_validation \
  --solver care_astabench_solver.py@care_discovery_agent \
  --model openai/anthropic/claude-opus-5 \
  --model-base-url https://api.commonstack.ai/v1 \
  -M responses_api=false \
  --epochs 3 \
  --log-dir logs/care-discovery-validation
```

The official ReAct baseline lives in the pinned `allenai/agent-baselines`
repository recorded in `configs/discoverybench_protocol_v4.json`. Run it with
the same model and limits. Do not inspect the test split until the validation
comparison and solver code are frozen.

Generate the current internal quality-token audit from real CARE traces:

```bash
uv run python scripts/build_internal_quality_cost_audit.py
```

The resulting plot is based on recorded AUC effects and token counts. It is not
an AstaBench score, and tokens are not converted to dollars without a frozen
provider price snapshot.

## Evidence boundary

Passing DiscoveryBench would support general data-analysis and hypothesis-
formation ability. It would not by itself prove cross-domain experimental
transfer, wet-lab discovery, or superiority over other AI Scientist systems.
Those claims still require the preregistered CARE transfer experiments and a
prospective physical campaign.
