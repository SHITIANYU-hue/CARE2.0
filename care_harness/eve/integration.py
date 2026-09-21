"""Configure the upstream engine; its evolution algorithm remains upstream-owned."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading

from . import UPSTREAM_REVISION, UPSTREAM_URL
from .broker import DevelopmentEvaluator, make_server
from care_harness.environment import ReplayEnvironment, load_task


WORKER_README = """# CARE finite-pool optimization

Improve `solver/solver.py`, defining `select_candidate(view) -> candidate_id`
or a dict with `candidate_id`. Every outcome is maximized. The solver is called
in a fresh Python subprocess each round; use the public history in `view`, not
module state. Python standard library and the CARE evaluator's installed NumPy
are available. All code must stay in this one file.

`view` contains task metadata, candidates with candidate_id/group/numeric_features/
metadata, observations with candidate_id/outcome, completed source observations,
round_index, remaining_budget, seed, initial_observations, reveal_rounds.
The candidate table includes observed rows; return an unobserved candidate ID.
`solver/example_view.json` illustrates the exact interface.

Optimize mean normalized best-so-far AUC on the named development tasks/seeds.
All target outcomes beyond the current observations belong to the evaluator.
Use source observations, observed target feedback, and candidate features. Do not
look for hidden results on the host or internet. No target test task participates
in evolution. Scores in reference logs are development feedback, not test results.

The official EvE loop evaluates your final solver after the worker finishes.
For additional development feedback, run `python solver/evaluate.py solver/solver.py`.
Every evaluation is logged, including extra evaluations you request.
Use `solver_examples/` and `guidance_examples/` to improve both the solver and
reusable guidance in `guidance/`. Keep model weights fixed. Finish autonomously.

{editable_files_block}
{editable_folders_block}
{immutable_overlay_block}
{solver_examples_block}
{optimizer_examples_block}
"""


def _revision(repository: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                          capture_output=True, text=True, check=True).stdout.strip()


def official_command(eve_repo: Path, workspace: Path, uv: str = "uv") -> list[str]:
    return [uv, "run", "--project", str(eve_repo), "python", "-m",
            "scaling_evolve.algorithms.eve.runner", "--config-dir", str(workspace / "config"),
            "--config-name", "care_replay"]


def prepare(eve_repo: str | Path, workspace: str | Path, development: dict,
            model: str, iterations: int = 5, workers: int = 2,
            reasoning_effort: str = "medium", rollout_max_turns: int = 20,
            timeout_seconds: int = 1200) -> dict:
    repository, work = Path(eve_repo).resolve(), Path(workspace).resolve()
    if (work / "manifest.json").exists():
        raise FileExistsError(f"Prepared experiment already exists: {work}; use a new workspace")
    revision = _revision(repository)
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Use the verified EvE checkout {UPSTREAM_REVISION}; got {revision}")
    if not development["tasks"] or not development["seeds"]:
        raise ValueError("Provide named development tasks and at least one development seed")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("seed", "config", "guidance", "immutable", "prompt"):
        (work / name).mkdir(exist_ok=True)
    module_dir = Path(__file__).parent
    shutil.copyfile(module_dir / "seed_solver.py", work / "seed" / "solver.py")
    shutil.copyfile(module_dir / "client.py", work / "seed" / "evaluate.py")
    tasks = [load_task(config) for config in development["tasks"]]
    task = tasks[0]
    development_task_ids = {task.task_id for task in tasks}
    development_task_ids.update(task.source["task"]["dataset_id"] for task in tasks
                                if task.source.get("task", {}).get("dataset_id"))
    example = ReplayEnvironment(task, development["seeds"][0],
                                development["initial_observations"], development["reveal_rounds"]).view()
    (work / "seed" / "example_view.json").write_text(json.dumps(example, indent=2))
    (work / "seed" / "README.md").write_text(
        "See example_view.json for the public policy input. Only solver.py is editable.\n")
    (work / "guidance" / "strategy.md").write_text(
        "Use observed feedback to balance exploitation with exploration. Compare source-informed "
        "and target-only reasoning on development tasks. Record useful findings for future workers.\n")
    (work / "immutable" / "README.md").write_text(WORKER_README)
    (work / "immutable" / "AGENTS.md").write_text("Read README.md before editing solver or guidance.\n")
    (work / "prompt" / "ENTRYPOINT.md").write_text(
        "Read README.md and the reference solvers/guidance. Improve solver/solver.py and guidance/. "
        "Use development evaluation to guide your changes, then finish.\n")
    (work / "prompt" / "BOUNDARY_REPAIR.md").write_text(
        "Keep solver changes within solver.py and reusable guidance within guidance/. "
        "Restore any other modified files and finish.\n")
    # The shell step executes under the CARE interpreter, not EvE's separate environment.
    shell = "#!/usr/bin/env bash\nset -eu\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(module_dir / "client.py")) + "\n"
    (work / "evaluation.sh").write_text(shell)
    failure = {"score": -1.0, "summary": "evaluation failed"}
    config = {
        "defaults": [{"runtime": "default"}, {"loop": "default"}, {"driver": "codex_max"},
                     {"logger": "many_loggers"}, "_self_"],
        "hydra": {"searchpath": ["file://" + str(repository / "configs" / "eve")]},
        "label": "care-replay-development", "run_root": str(work / "run"),
        "application": {"name": "care-replay", "path": str(work / "seed"),
                        "editable": {"files": ["solver.py"], "folders": []},
                        "boundary_failure_score": failure},
        "evaluation": {"steps": [str(work / "evaluation.sh")], "failure_score": failure,
                       "seed_solver_score": None, "seed_solver_skip_evaluation": False},
        "optimizer": {
            "initial_guidance": str(work / "guidance"),
            "workers": {
                "selection": {"_target_": "scaling_evolve.algorithms.eve.workspace.worker_selection.RandomWorkerSelector"},
                "items": [{"name": "care", "weight": 1.0,
                           "immutable": str(work / "immutable"), "prompt": str(work / "prompt")}]},
            "immutable_renderer": {"_target_": "scaling_evolve.algorithms.eve.workspace.immutable_renderers.default.DefaultRenderer"},
            "evaluation": {"_target_": "scaling_evolve.algorithms.eve.populations.evaluators.elo.ScalarEloEvaluator",
                           "k_factor": 32.0, "initial_score": {"elo": 1500.0}}},
        "loop": {"max_iterations": iterations, "n_workers_phase2": workers},
        "driver": {"model": model, "reasoning_effort": reasoning_effort,
                   "web_search": "disabled", "rollout_max_turns": rollout_max_turns,
                   "timeout_seconds": timeout_seconds},
        "logger": {"wandb": {"enabled": False}},
    }
    (work / "config" / "care_replay.yaml").write_text(json.dumps(config, indent=2))
    manifest = {"upstream_url": UPSTREAM_URL, "upstream_revision": revision,
                "eve_repo": str(repository), "development": development,
                "development_task_ids": sorted(development_task_ids),
                "model": model, "iterations": iterations, "workers": workers,
                "reasoning_effort": reasoning_effort, "rollout_max_turns": rollout_max_turns,
                "timeout_seconds": timeout_seconds,
                "protocol": "evolve on development tasks; freeze solver before target evaluation",
                "command": official_command(repository, work), "search_executed": False}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def launch(workspace: str | Path, uv: str = "uv") -> int:
    work = Path(workspace).resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    repository = Path(manifest["eve_repo"])
    revision = _revision(repository)
    if revision != manifest["upstream_revision"]:
        raise ValueError("EvE checkout changed after preparation; prepare the run again")
    evaluate = DevelopmentEvaluator(manifest["development"], work / "development_evaluations")
    server = make_server(evaluate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = dict(os.environ, PWD=str(repository),
                       CARE_EVE_EVALUATOR_URL=f"http://127.0.0.1:{server.server_port}/evaluate")
    command = official_command(repository, work, uv)
    manifest.update({"command": command, "search_executed": True})
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        completed = subprocess.run(command, cwd=repository, env=environment, check=False)
        manifest["returncode"] = completed.returncode
        return completed.returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        manifest["development_evaluations"] = evaluate.evaluations
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2))


def export_solver(solver: str | Path, output: str | Path, workspace: str | Path | None = None) -> dict:
    source = Path(solver).resolve()
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    (destination / "solver.py").write_bytes(content)
    provenance = {"solver_sha256": hashlib.sha256(content).hexdigest(),
                  "source_solver": str(source), "upstream_revision": UPSTREAM_REVISION,
                  "selection": "explicit solver file; choose using development results only"}
    if workspace is not None:
        provenance["evolution"] = json.loads((Path(workspace) / "manifest.json").read_text())
    else:
        provenance["evolution"] = None
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance
