# Official EvE baseline

This integration runs the [official EvE engine](https://github.com/scaling-group/eve)
at `979065fd2cf1b3be9caf70596478fe27df7815ad` (package version 0.2.0).
The engine evolves both a solver population and agent guidance. CARE supplies
the finite-pool task and its evaluator; it does not replace EvE with a custom
search loop.

## Prepare without a key

Create a separate upstream checkout and use the pinned commit:

```sh
git clone https://github.com/scaling-group/eve.git /path/to/eve
git -C /path/to/eve checkout 979065fd2cf1b3be9caf70596478fe27df7815ad
uv sync --project /path/to/eve --no-dev
```

Use `UV_DEFAULT_INDEX=https://mirrors.osa.moe/pypi/web/simple/` if the campus mirror
is reachable, with `--refresh` when necessary. EvE has its own environment; its
dependencies need not be added to CARE's lockfile.

From the CARE repository, specify the exact model intended for the study:

```sh
uv run care.py eve prepare \
  --eve-repo /path/to/eve \
  --workspace runs/eve-development \
  --development configs/baselines/eve-development.json \
  --model YOUR_MODEL --iterations 5 --workers 2
```

Preparation writes a public application snapshot, guidance seed, worker prompts,
Hydra config, evaluation shell step, and a manifest with the upstream revision,
named development tasks, seeds, budgets, and launch command. It makes no model
calls. The seed solver is deliberately simple nearest-neighbor exploration;
running that seed is a wiring smoke test, not an EvE experimental result.

The development JSON contains `tasks` (the shared harness's task configs),
`seeds`, `initial_observations`, and `reveal_rounds`. Optional `solver_timeout`
limits each policy invocation (default 60 seconds). Preparation rejects an
already prepared workspace so it cannot overwrite an existing search.

## Evolve after authentication is available

Install/authenticate the coding-agent executable required by EvE. At the pinned
version, Codex >= 0.130.0 also requires upstream's one-time hook setup and trust:

```sh
cd /path/to/eve
uv run python -m scaling_evolve.providers.agent.codex_hooks
codex
```

In that interactive session follow upstream's `/hooks` trust procedure. Then,
from CARE, run:

```sh
uv run care.py eve launch --workspace runs/eve-development
```

Launch starts a localhost-only evaluation service and the real upstream runner
`scaling_evolve.algorithms.eve.runner`. The upstream environment owns model
authentication, token accounting, search traces, solver populations and guidance
populations. The experiment's `run/telemetry/` and upstream artifacts retain
those development costs. Workers can request extra development evaluations;
each is recorded in `development_evaluations/`, including initial/adaptive
observation counts and failed evaluations. These repeated development queries
are separate from the single fixed target-test budget.

The shell step sends solver source to the coordinator and writes the returned
JSON to `$EVE_EVAL_LOG_ROOT/score.yaml` (JSON is valid YAML). Score is mean
normalized best-so-far AUC over the named development episodes, using the same
metric as target replay; higher is better and failures score -1. The fixed
numeric `score` field drives upstream solver sampling and scalar Elo updates.

## Freeze and evaluate

Select a solver using **development results only**, from the official run's
`run/solver_workspaces/.../solver/solver.py`. Keep the original run and its
guidance artifacts; target execution only needs the frozen solver:

```sh
uv run care.py eve export \
  --solver /path/to/selected/solver.py \
  --workspace runs/eve-development \
  --output runs/eve-frozen
```

Export writes `solver.py` and `provenance.json` with a content hash and the
development manifest. The shared runner's `eve` method accepts this solver file.
The adapter verifies the hash and launches it in a fresh Python subprocess on
each decision, passing only the public view. Python and NumPy global random
generators are seeded from the campaign seed and round. The policy contract is:

```python
def select_candidate(view: dict) -> str | dict:
    # Return an unobserved candidate ID, or {"candidate_id": that_id}.
    ...
```

The view contains public candidates, source observations, revealed target
observations, budget and seed; it contains no hidden target result table. It
includes already observed candidate feature rows, so the policy filters those
IDs itself. Policies can use Python's standard library and packages installed
in the CARE interpreter. Fresh processes mean policies must reconstruct state
from `view`, rather than retaining module globals across rounds.

Development tasks must be chosen independently from the target-test tasks.
Different seeds on the same pool are not evidence of new-task generalization.
The adapter separates public policy inputs from the oracle, but it is not a
security sandbox against deliberately malicious Python. No paid search was
required for the integration tests.
