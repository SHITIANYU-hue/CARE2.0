"""Deploy an EvE solver without exposing the replay oracle to its module."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class EvEAgent:
    """A frozen, single-file policy with select_candidate(public_view)."""

    def __init__(self, solver_path: str | Path, python_executable: str | None = None,
                 timeout: float = 60):
        path = Path(solver_path)
        content = path.read_bytes()
        self.source = content.decode()
        self.sha256 = hashlib.sha256(content).hexdigest()
        provenance_path = path.with_name("provenance.json")
        self.provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
        if self.provenance and self.provenance["solver_sha256"] != self.sha256:
            raise ValueError("Frozen solver changed after export; export it again before evaluation")
        self.python = python_executable or sys.executable
        self.timeout = timeout

    def select(self, view: dict) -> dict:
        worker = Path(__file__).parents[1] / "eve" / "worker.py"
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="care-eve-policy-") as directory:
            work = Path(directory)
            (work / "solver.py").write_text(self.source)
            (work / "view.json").write_text(json.dumps(view))
            completed = subprocess.run(
                [self.python, "-I", str(worker), "solver.py", "view.json", "decision.json"],
                cwd=work, capture_output=True, text=True, timeout=self.timeout,
            )
            if completed.returncode:
                raise RuntimeError(f"EvE solver failed: {completed.stderr[-4000:]}")
            decision = json.loads((work / "decision.json").read_text())
        if isinstance(decision, str):
            decision = {"candidate_id": decision}
        decision.setdefault("diagnostics", {}).update({"solver_sha256": self.sha256})
        if self.provenance:
            decision["diagnostics"]["upstream_revision"] = self.provenance["upstream_revision"]
            decision["diagnostics"]["evolution_record_available"] = self.provenance.get("evolution") is not None
        decision.setdefault("usage", {})["solver_seconds"] = time.monotonic() - started
        return decision
