#!/usr/bin/env python3
"""Run the frozen classical transfer-BO suite, one process per pair."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_classical_transfer_baselines.py"


def run_pair(
    config_path: Path,
    pair_id: str,
    output_dir: Path,
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{pair_id}.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(config_path),
                "--pair-id",
                pair_id,
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build_suite_summary(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for pair in config["pairs"]:
        summary_path = output_dir / f"{pair['pair_id']}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pairs[pair["pair_id"]] = summary["splits"]
    heldout_positive = {
        method: sum(
            1
            for result in pairs.values()
            if result["heldout"][f"{method}_vs_target_gp"]["final_best"][
                "normal_95ci_low"
            ]
            > 0.0
        )
        for method in ("rgpe", "multitask_gp")
    }
    return {
        "experiment": "care2_classical_transfer_bo_suite",
        "protocol": config["protocol"],
        "pair_count": len(pairs),
        "heldout_significant_positive_pair_count": heldout_positive,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "classical_transfer_benchmark_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel-pairs", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.parallel_pairs)
    ) as executor:
        futures = [
            executor.submit(
                run_pair,
                config_path,
                pair["pair_id"],
                args.output_dir.resolve(),
                log_dir.resolve(),
            )
            for pair in config["pairs"]
        ]
        for future in futures:
            future.result()
    summary = build_suite_summary(config, args.output_dir)
    (args.output_dir / "suite_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
