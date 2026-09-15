#!/usr/bin/env python3
"""Check staged/tracked layout and canonical suite inputs without running experiments."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = (
    "experiments/care_replay/results/", "experiments/care_replay/outputs/",
    "experiments/astabench/results/", "experiments/astabench/logs/",
    ".venv/", "knowledge_base/exports/", "knowledge_base/embeddings/",
)


def check(root: Path = ROOT) -> list[str]:
    errors = []
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    for name in filter(None, tracked):
        path = root / name
        if name.startswith(GENERATED):
            errors.append(f"Generated evidence must be archived: {name}")
        if Path(name).suffix.lower() in {".pptx", ".ppt", ".log", ".pyc"}:
            errors.append(f"Generated/binary output must be archived: {name}")
        if path.is_file() and path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"File exceeds regular GitHub file limit: {name}")
        if path.is_file() and path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Invalid Python source {name}: {exc}")
    replay = root / "experiments" / "care_replay"
    config = json.loads((replay / "configs" / "source_outcome_benchmark.json").read_text(encoding="utf-8"))
    for pair in config["pairs"]:
        for key in ("llm_record", "target_llm_record"):
            record = pair[key]
            if record.startswith(("results/", "outputs/")):
                errors.append(f"Canonical input depends on archived evidence: {record}")
            try:
                json.loads((replay / record).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"Missing/invalid canonical input {record}: {exc}")
    catalog = json.loads((root / "artifacts" / "catalog.json").read_text(encoding="utf-8"))
    seen = set()
    for entry in catalog["archives"]:
        if entry["id"] in seen:
            errors.append(f"Duplicate archive id: {entry['id']}")
        seen.add(entry["id"])
        archive = root / entry["path"]
        if not archive.is_file() or archive.stat().st_size != entry["size_bytes"]:
            errors.append(f"Missing/wrong-size archive: {entry['path']}")
    return errors


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors = check()
    if errors:
        print("\n".join(errors))
        return 1
    print("Repository layout, Python syntax, archive sizes and 7-pair frozen inputs: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
