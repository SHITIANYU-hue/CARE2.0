"""Summarize paired, officially scored DiscoveryBench Inspect logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-log", type=Path, required=True)
    parser.add_argument("--react-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_901)
    return parser.parse_args()


def load_scores(path: Path) -> tuple[dict[str, float], str, bool]:
    log = read_eval_log(str(path))
    if log.status != "success" or not log.samples:
        raise ValueError(f"Scored log is incomplete: {path}")

    values: dict[str, float] = {}
    judge_models: set[str] = set()
    official_flags: set[bool] = set()
    for sample in log.samples:
        if not sample.scores or len(sample.scores) != 1:
            raise ValueError(f"Expected one score for sample {sample.id} in {path}")
        score = next(iter(sample.scores.values()))
        values[str(sample.id)] = float(score.value)
        metadata = score.metadata or {}
        judge_models.add(str(metadata.get("judge_model", "")))
        official_flags.add(bool(metadata.get("official_judge", False)))

    if len(judge_models) != 1 or len(official_flags) != 1:
        raise ValueError(f"Mixed judge metadata in {path}")
    return values, judge_models.pop(), official_flags.pop()


def main() -> None:
    args = parse_args()
    care, care_model, care_official = load_scores(args.care_log)
    react, react_model, react_official = load_scores(args.react_log)
    if set(care) != set(react):
        raise ValueError("CARE and ReAct logs do not contain the same sample IDs")
    if (care_model, care_official) != (react_model, react_official):
        raise ValueError("CARE and ReAct were not scored with the same judge")

    rows = []
    for sample_id in sorted(care):
        delta = care[sample_id] - react[sample_id]
        outcome = "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie"
        rows.append(
            {
                "sample_id": sample_id,
                "care_hms": care[sample_id],
                "react_hms": react[sample_id],
                "paired_delta": delta,
                "outcome": outcome,
            }
        )

    deltas = np.asarray([row["paired_delta"] for row in rows], dtype=float)
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(
        0,
        len(deltas),
        size=(args.bootstrap_resamples, len(deltas)),
    )
    bootstrap_means = deltas[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    summary = {
        "schema_version": "care2.astabench.paired_hms.v1",
        "benchmark": "AstaBench 0.5.3 DiscoveryBench",
        "split": "validation",
        "samples": len(rows),
        "judge_model": care_model,
        "official_judge": care_official,
        "care_mean_hms": float(np.mean(list(care.values()))),
        "react_mean_hms": float(np.mean(list(react.values()))),
        "paired_mean_delta": float(deltas.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "wins": sum(row["outcome"] == "win" for row in rows),
        "ties": sum(row["outcome"] == "tie" for row in rows),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "quality_superiority_supported": bool(lower > 0),
        "claim_boundary": (
            "The official judge gives CARE a slightly higher mean HMS on the "
            "25-sample validation split, but the paired bootstrap interval "
            "crosses zero. This run does not establish quality superiority."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
