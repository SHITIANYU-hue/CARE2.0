"""Development evaluation service, outside the agent's editable solver workspace."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from care_harness.agents.eve import EvEAgent
from care_harness.environment import load_task
from care_harness.runner import run_episode


class DevelopmentEvaluator:
    def __init__(self, config: dict, output_dir: str | Path):
        self.config = config
        self.tasks = [load_task(task) for task in config["tasks"]]
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluations = max((int(path.name.rsplit("_", 1)[1])
                                for path in self.output.glob("evaluation_*")), default=0)
        self.lock = threading.Lock()

    def __call__(self, source: str) -> dict:
        # Serialize each submitted policy's paired suite so logs/counts stay simple.
        with self.lock:
            self.evaluations += 1
            run_dir = self.output / f"evaluation_{self.evaluations:05d}"
            run_dir.mkdir()
            solver = run_dir / "solver.py"
            solver.write_text(source)
            episodes = []
            episode_directory = None
            try:
                for task in self.tasks:
                    for seed in self.config["seeds"]:
                        episode_directory = run_dir / f"episode_{len(episodes):04d}"
                        result = run_episode(
                            task, EvEAgent(solver, timeout=self.config.get("solver_timeout", 60)),
                            seed=seed, initial_observations=self.config["initial_observations"],
                            reveal_rounds=self.config["reveal_rounds"],
                            output_dir=episode_directory,
                        )
                        episodes.append({"task_id": task.task_id, "seed": seed,
                                         "score": result["metrics"]["normalized_best_so_far_auc"]})
                payload = {"status": "ok", "score": sum(e["score"] for e in episodes) / len(episodes),
                           "summary": f"Mean normalized best-so-far AUC on {len(episodes)} development episodes",
                           "development_episodes": episodes}
            except Exception as exc:
                payload = {"status": "error", "score": -1.0,
                           "summary": f"{type(exc).__name__}: {exc}"}
            payload["evaluation_index"] = self.evaluations
            payload["development_reveals"] = len(episodes) * self.config["reveal_rounds"]
            attempted_episodes = len(episodes)
            if payload["status"] == "error" and episode_directory is not None:
                attempted_episodes += 1
                failed_result = episode_directory / "result.json"
                if failed_result.exists():
                    payload["development_reveals"] += json.loads(failed_result.read_text()).get("completed_rounds", 0)
            payload["development_initial_observations"] = attempted_episodes * self.config["initial_observations"]
            (run_dir / "score.json").write_text(json.dumps(payload, indent=2))
            return payload


def make_server(evaluate: DevelopmentEvaluator) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = evaluate(request["solver"])
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)
