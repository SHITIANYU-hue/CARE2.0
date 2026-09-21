"""Method-independent campaign execution and output accounting."""

import json
from pathlib import Path
import time
from typing import Protocol

from .environment import ReplayEnvironment, ReplayTask


class Agent(Protocol):
    def select(self, view: dict) -> dict: ...


def run_episode(task: ReplayTask, agent: Agent, seed: int = 0,
                initial_observations: int = 5, reveal_rounds: int = 10,
                initial_ids: list[str] | None = None, output_dir: str | Path | None = None) -> dict:
    env = ReplayEnvironment(task, seed, initial_observations, reveal_rounds, initial_ids)
    started = time.monotonic()
    trace, usage = [], {}
    output = Path(output_dir) if output_dir is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "initial.json").write_text(json.dumps(env.view(), indent=2))
        (output / "trace.jsonl").write_text("")
    try:
        while env.remaining_budget:
            decision = agent.select(env.view())
            if isinstance(decision, str):
                decision = {"candidate_id": decision}
            for key, value in decision.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            if output:
                # Persist the choice before outcome reveal.
                (output / f"decision_{len(trace):03d}.json").write_text(json.dumps(decision, indent=2))
            event = env.reveal(decision["candidate_id"])
            event["diagnostics"] = decision.get("diagnostics", {})
            event["usage"] = decision.get("usage", {})
            trace.append(event)
            if output:
                with (output / "trace.jsonl").open("a") as handle:
                    handle.write(json.dumps(event) + "\n")
        result = {"status": "complete", "task_id": task.task_id, "seed": seed,
                  "initial_ids": env.initial_ids, "initial_observations": env.observations[:initial_observations],
                  "metrics": env.metrics(), "usage": usage, "trace": trace,
                  "wall_seconds": time.monotonic() - started}
    except Exception as exc:
        if output:
            (output / "result.json").write_text(json.dumps({
                "status": "failed", "task_id": task.task_id, "seed": seed,
                "error": str(exc), "error_type": type(exc).__name__,
                "completed_rounds": len(trace), "usage": usage,
            }, indent=2))
        raise
    if output:
        (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
