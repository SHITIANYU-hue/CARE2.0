#!/usr/bin/env python3
"""Run the predeclared CARE 2.0 source-outcome benchmark suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_calibrated_source_outcome_transfer.py"


def resolve_record(path: str) -> Path:
    return (ROOT / path).resolve()


def build_command(
    pair: dict[str, Any],
    protocol: dict[str, Any],
    strategy: dict[str, Any],
    calibration_seed_start: int,
    heldout_seed_start: int,
    pair_workers: int,
    output_tag: str,
) -> list[str]:
    frozen = pair.get("frozen_policy", {})
    command = [
        sys.executable,
        str(RUNNER),
        "--llm-record",
        str(resolve_record(pair["llm_record"])),
        "--target-llm-record",
        str(resolve_record(pair["target_llm_record"])),
        "--target-llm-mode",
        pair["target_llm_mode"],
        "--source-dataset",
        pair["source_dataset"],
        "--target-dataset",
        pair["target_dataset"],
        "--source-observations",
        str(pair["source_observations"]),
        "--source-seed",
        str(protocol["source_seed"]),
        "--initial",
        str(frozen.get(
            "initial_observations",
            protocol["initial_observations"],
        )),
        "--rounds",
        str(frozen.get("reveal_rounds", protocol["reveal_rounds"])),
        "--calibration-seed-start",
        str(calibration_seed_start),
        "--calibration-seeds",
        str(protocol["calibration_seed_count"]),
        "--heldout-seed-start",
        str(heldout_seed_start),
        "--heldout-seeds",
        str(protocol["heldout_seed_count"]),
        "--workers",
        str(pair_workers),
        "--source-initial-strategy",
        frozen.get(
            "source_initial_strategy",
            protocol["source_initial_strategy"],
        ),
        "--router-min-observations",
        str(frozen.get(
            "router_min_observations",
            strategy["router_min_observations"],
        )),
        "--router-min-quality",
        str(frozen.get(
            "router_min_quality",
            protocol["router_min_quality"],
        )),
        "--router-max-transfer-mass",
        str(frozen.get(
            "router_max_transfer_mass",
            protocol["router_max_transfer_mass"],
        )),
        "--output-tag",
        f"{output_tag}_{pair['pair_id']}",
    ]
    return command


def run_pair(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "source_outcome_benchmark.json",
    )
    parser.add_argument(
        "--strategy",
        choices=("full_source_outcome", "source_initial_only"),
        required=True,
    )
    parser.add_argument("--calibration-seed-start", type=int, required=True)
    parser.add_argument("--heldout-seed-start", type=int, required=True)
    parser.add_argument("--pair-workers", type=int, default=20)
    parser.add_argument("--parallel-pairs", type=int, default=1)
    parser.add_argument("--output-tag", required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    strategy = config["strategy_candidates"][args.strategy]
    jobs = []
    for pair in config["pairs"]:
        command = build_command(
            pair,
            protocol,
            strategy,
            args.calibration_seed_start,
            args.heldout_seed_start,
            args.pair_workers,
            args.output_tag,
        )
        log_path = (
            ROOT
            / "outputs"
            / "logs"
            / f"{args.output_tag}_{pair['pair_id']}.log"
        )
        jobs.append((command, log_path))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.parallel_pairs
    ) as executor:
        futures = [
            executor.submit(run_pair, command, log_path)
            for command, log_path in jobs
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
