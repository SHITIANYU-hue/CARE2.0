#!/usr/bin/env python3
"""Small repository entry point; research implementations retain their paths."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "experiments" / "care_replay" / "scripts"
COMMANDS = {
    "smoke": (REPLAY / "run_synthetic_suzuki.py", [
        "--dataset", "synthetic_suzuki_i", "--seeds", "2", "--rounds", "3",
        "--initial", "3", "--modes", "no_care_random,incumbent,gate_v1",
        "--output-tag", "smoke",
    ]),
    "pair": (REPLAY / "run_care2.py", ["pair"]),
    "suite": (REPLAY / "run_care2.py", ["suite"]),
    "kb-build": (ROOT / "knowledge_base" / "build_kb.py", []),
    "kb-query": (ROOT / "knowledge_base" / "query_kb.py", []),
    "artifacts": (ROOT / "tools" / "artifacts.py", []),
    "pack": (ROOT / "tools" / "pack_run.py", []),
    "check": (ROOT / "tools" / "check_repository.py", []),
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        print("CARE 2.0: python care.py COMMAND [arguments]\n")
        print("  smoke       Small synthetic replay (no network or model API)")
        print("  pair/suite  Canonical frozen source-outcome transfer")
        print("  kb-build    Build the local knowledge base")
        print("  kb-query    Search the local knowledge base")
        print("  artifacts   List, verify or restore archived evidence")
        print("  pack        Archive a completed run without deleting it")
        print("  check       Check repository hygiene and canonical inputs")
        print("\nEach command accepts --help. See docs/PROJECT_MAINLINE.md.")
        return 0
    if args[0] not in COMMANDS:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        return 2
    script, defaults = COMMANDS[args[0]]
    return subprocess.run([sys.executable, str(script), *defaults, *args[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
