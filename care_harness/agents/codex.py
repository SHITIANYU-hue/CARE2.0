"""A Codex CLI coding-agent baseline using only the runner's public task view.

Each query starts a fresh CLI session; analysis code and notes persist in the
campaign workspace. The local CLI sandbox limits writes, not host reads: this
adapter stages public inputs but is not a filesystem isolation boundary.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["candidate_id", "rationale"],
    "additionalProperties": False,
}

PROMPT = """You are the Codex coding-agent baseline for a sequential experimental
optimization task. Read view.json for the task, all public candidate features,
source observations, target observations, and remaining experiment budget.
Choose exactly one as-yet unobserved target candidate to improve the best
observed outcome within the remaining budget. Follow the objective direction
given by the task. The harness, not you, executes the chosen experiment and
reveals its result on the next round.

You may write and run analysis or optimization code in this workspace, use
available local numerical libraries, and keep notes or reusable code across
rounds. You have a fresh conversation each round: persistent files are your
memory. The full current public state is always in view.json.

Use only the supplied data and your general knowledge. Do not read other task
directories, original benchmark datasets, repository results, or hidden target
labels. Do not browse the web or invoke additional model APIs. Source outcomes
and the target outcomes explicitly present in observations are available for
analysis. New target experiments may only be requested by your final answer.

Return the required JSON containing candidate_id and a short rationale.
"""


class CodexAgent:
    """Run ``codex exec`` once per requested target experiment.

    Config: ``command`` (argv prefix, default ``["codex"]``), ``model`` and
    ``reasoning_effort`` (optional), ``timeout_seconds`` (default 300).
    Authentication is the existing Codex CLI login. Its user configuration is
    ignored so the experiment does not inherit personal provider/MCP settings.
    Pass a dedicated workspace outside the repository for each campaign.
    """

    def __init__(self, config: dict, workspace: Path):
        self.config = dict(config)
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.command = config.get("command", ["codex"])
        if not isinstance(self.command, list) or not self.command:
            raise ValueError("Codex command must be a nonempty argv list")
        self.timeout = float(config.get("timeout_seconds", 300))

    def select(self, view: dict) -> dict:
        round_dir = self.workspace / "traces" / f"round-{view['round_index']:03d}"
        round_dir.mkdir(parents=True, exist_ok=False)
        view_text = json.dumps(view, ensure_ascii=False, indent=2, allow_nan=False)
        (self.workspace / "view.json").write_text(view_text + "\n", encoding="utf-8")
        (round_dir / "view.json").write_text(view_text + "\n", encoding="utf-8")
        schema_path = round_dir / "selection.schema.json"
        schema_path.write_text(json.dumps(SELECTION_SCHEMA, indent=2) + "\n", encoding="utf-8")
        final_path = round_dir / "selection.json"
        events_path = round_dir / "events.jsonl"
        stderr_path = round_dir / "stderr.log"
        command = [
            *self.command,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox", "workspace-write",
            "--json",
            "--color", "never",
            "--cd", str(self.workspace),
            "--output-schema", str(schema_path),
            "--output-last-message", str(final_path),
            "-c", 'web_search="disabled"',
            "-c", "features.apps=false",
            "-c", "mcp_servers={}",
            "-c", "sandbox_workspace_write.network_access=false",
        ]
        if self.config.get("model"):
            command.extend(["--model", self.config["model"]])
        if self.config.get("reasoning_effort"):
            command.extend([
                "-c", "model_reasoning_effort=" + json.dumps(self.config["reasoning_effort"])
            ])
        command.append("-")
        (round_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        (round_dir / "prompt.txt").write_text(PROMPT, encoding="utf-8")

        start = time.perf_counter()
        try:
            with events_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                process = subprocess.run(
                    command, input=PROMPT, text=True, cwd=self.workspace,
                    stdout=stdout, stderr=stderr, timeout=self.timeout, check=False,
                )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Codex exceeded {self.timeout:g}s; trace: {round_dir}") from error
        except FileNotFoundError as error:
            raise RuntimeError(f"Codex command not found: {self.command[0]}") from error
        elapsed = time.perf_counter() - start
        if process.returncode:
            raise RuntimeError(f"Codex exited with status {process.returncode}; see {stderr_path}")

        usage = {"agent_invocations": 1, "wall_seconds": elapsed}
        thread_id = None
        with events_path.open(encoding="utf-8") as events:
            for line in events:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") == "thread.started":
                    thread_id = event.get("thread_id")
                elif event.get("type") == "turn.failed":
                    raise RuntimeError(f"Codex turn failed: {event.get('error')}; trace: {round_dir}")
                elif event.get("type") == "turn.completed":
                    for key, value in event.get("usage", {}).items():
                        if isinstance(value, (int, float)):
                            usage[key] = usage.get(key, 0) + value

        if not final_path.is_file():
            raise RuntimeError(f"Codex did not produce a final selection; trace: {round_dir}")
        try:
            selection = json.loads(final_path.read_text(encoding="utf-8"))
            candidate_id = selection["candidate_id"]
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("candidate_id must be a nonempty string")
        except (ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"Invalid Codex selection in {final_path}: {error}") from error
        return {
            "candidate_id": candidate_id,
            "diagnostics": {
                "rationale": selection.get("rationale", ""),
                "thread_id": thread_id,
                "requested_model": self.config.get("model"),
                "trace_directory": str(round_dir),
            },
            "usage": usage,
        }
