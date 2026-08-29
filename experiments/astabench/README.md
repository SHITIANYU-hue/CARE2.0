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

The skill map is domain agnostic and contains no benchmark outcomes. The solver
uses only the tools already provided by AstaBench, so its submission category is
`standard` rather than a custom information-access toolset.

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
export ANTHROPIC_API_KEY="..."  # or the key required by --model
```

Run the validation split first. The model name must be identical for CARE and
the ReAct control.

```bash
uv run inspect eval \
  astabench/discoverybench_validation \
  --solver care_astabench_solver.py@care_discovery_agent \
  --model anthropic/<frozen-model-id> \
  --epochs 3 \
  --log-dir logs/care-discovery-validation
```

The official ReAct baseline lives in the pinned `allenai/agent-baselines`
repository recorded in `configs/discoverybench_protocol_v1.json`. Run it with
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
