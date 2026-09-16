#!/usr/bin/env python3
"""Single public entry point for the frozen CARE 2.0 method."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {
    "pair": ROOT / "scripts" / "run_calibrated_source_outcome_transfer.py",
    "suite": ROOT / "scripts" / "run_source_outcome_suite.py",
}


def usage() -> str:
    return (
        "Usage: run_care2.py {pair|suite} [runner arguments]\n"
        "  pair   Run one frozen source-target confirmation.\n"
        "  suite  Run a predeclared multi-pair confirmation suite.\n"
        "Use 'run_care2.py pair --help' or 'run_care2.py suite --help' for details."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in RUNNERS:
        print(usage(), file=sys.stderr)
        return 2
    command = sys.argv[1]
    completed = subprocess.run(
        [sys.executable, str(RUNNERS[command]), *sys.argv[2:]],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
