"""Usage: python -m care_harness.eve prepare|launch|export ..."""

import argparse
import json
from pathlib import Path

from .integration import export_solver, launch, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Prepare official EvE configs without model calls")
    setup.add_argument("--eve-repo", required=True)
    setup.add_argument("--workspace", required=True)
    setup.add_argument("--development", required=True, help="JSON with tasks, seeds, initial_observations, reveal_rounds")
    setup.add_argument("--model", required=True)
    setup.add_argument("--iterations", type=int, default=5)
    setup.add_argument("--workers", type=int, default=2)
    setup.add_argument("--reasoning-effort", default="medium")
    setup.add_argument("--rollout-max-turns", type=int, default=20)
    setup.add_argument("--timeout-seconds", type=int, default=1200)
    run = commands.add_parser("launch", help="Run official EvE; requires authenticated coding agent")
    run.add_argument("--workspace", required=True)
    run.add_argument("--uv", default="uv")
    export = commands.add_parser("export", help="Freeze a selected development solver for target replay")
    export.add_argument("--solver", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--workspace")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        config_path = Path(args.development).resolve()
        development = json.loads(config_path.read_text())
        for task in development["tasks"]:
            if isinstance(task, dict) and "task_file" in task:
                task["task_file"] = str((config_path.parent / task["task_file"]).resolve())
        result = prepare(args.eve_repo, args.workspace, development, args.model,
                         args.iterations, args.workers, args.reasoning_effort,
                         args.rollout_max_turns, args.timeout_seconds)
    elif args.command == "launch":
        return launch(args.workspace, args.uv)
    else:
        result = export_solver(args.solver, args.output, args.workspace)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
