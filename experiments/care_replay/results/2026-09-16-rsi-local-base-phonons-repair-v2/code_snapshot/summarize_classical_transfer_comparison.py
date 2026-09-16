#!/usr/bin/env python3
"""Compare CARE with a calibration-selected classical transfer-BO arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CLASSICAL_MODES = ("target_gp_ucb", "rgpe", "multitask_gp_icm")
CARE_MODE = "care_source_outcome_router"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def calibration_score(rows: list[dict[str, Any]], mode: str) -> float:
    selected = [row for row in rows if row["split"] == "calibration" and row["mode"] == mode]
    if not selected:
        raise ValueError(f"No calibration rows for {mode}")
    return mean(float(row["final_best"]) + float(row["best_so_far_auc"]) for row in selected)


def select_classical_mode(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    scores = {mode: calibration_score(rows, mode) for mode in CLASSICAL_MODES}
    selected = max(CLASSICAL_MODES, key=lambda mode: (scores[mode], -CLASSICAL_MODES.index(mode)))
    return selected, {mode: round(score, 6) for mode, score in scores.items()}


def select_hybrid_mode(
    care_rows: list[dict[str, Any]],
    classical_rows: list[dict[str, Any]],
    care_calibration_mode: str = CARE_MODE,
) -> tuple[str, dict[str, float]]:
    scores = {
        CARE_MODE: calibration_score(care_rows, care_calibration_mode),
        **{
            mode: calibration_score(classical_rows, mode)
            for mode in CLASSICAL_MODES
        },
    }
    ordering = (CARE_MODE, *CLASSICAL_MODES)
    selected = max(ordering, key=lambda mode: (scores[mode], -ordering.index(mode)))
    return selected, {mode: round(score, 6) for mode, score in scores.items()}


def paired_comparison(
    challenger_rows: list[dict[str, Any]],
    challenger_mode: str,
    baseline_rows: list[dict[str, Any]],
    baseline_mode: str,
    split: str = "heldout",
) -> dict[str, Any]:
    challenger = {
        int(row["seed"]): row
        for row in challenger_rows
        if row["split"] == split and row["mode"] == challenger_mode
    }
    baseline = {
        int(row["seed"]): row
        for row in baseline_rows
        if row["split"] == split and row["mode"] == baseline_mode
    }
    seeds = sorted(challenger.keys() & baseline.keys())
    if not seeds:
        raise ValueError(
            f"No matched {split} seeds for {challenger_mode} vs {baseline_mode}"
        )
    output: dict[str, Any] = {
        "challenger": challenger_mode,
        "baseline": baseline_mode,
        "seed_count": len(seeds),
    }
    for field in ("final_best", "best_so_far_auc", "top10_hit"):
        deltas = [
            float(challenger[seed][field]) - float(baseline[seed][field])
            for seed in seeds
        ]
        delta_mean = mean(deltas)
        delta_std = pstdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * delta_std / math.sqrt(len(deltas))
        output[field] = {
            "mean_delta": round(delta_mean, 6),
            "normal_95ci_low": round(delta_mean - half_width, 6),
            "normal_95ci_high": round(delta_mean + half_width, 6),
            "win_rate": round(sum(delta > 0.0 for delta in deltas) / len(deltas), 6),
            "non_loss_rate": round(sum(delta >= 0.0 for delta in deltas) / len(deltas), 6),
        }
    return output


def summarize_pair(
    pair_id: str,
    classical_metrics: Path,
    care_metrics: Path,
    care_summary: Path,
) -> dict[str, Any]:
    classical_rows = read_rows(classical_metrics)
    care_rows = read_rows(care_metrics)
    care_record = json.loads(care_summary.read_text(encoding="utf-8"))
    care_calibration_mode = str(care_record["selection"]["selected_mode"])
    selected_mode, calibration_scores = select_classical_mode(classical_rows)
    selected_hybrid_mode, hybrid_calibration_scores = select_hybrid_mode(
        care_rows,
        classical_rows,
        care_calibration_mode,
    )
    comparisons = {
        mode: paired_comparison(care_rows, CARE_MODE, classical_rows, mode)
        for mode in CLASSICAL_MODES
    }
    if selected_hybrid_mode == CARE_MODE:
        hybrid_vs_target_gp = comparisons["target_gp_ucb"]
    else:
        hybrid_vs_target_gp = paired_comparison(
            classical_rows,
            selected_hybrid_mode,
            classical_rows,
            "target_gp_ucb",
        )
    return {
        "pair_id": pair_id,
        "classical_selection_scope": "calibration_only",
        "selected_classical_mode": selected_mode,
        "classical_calibration_composite_scores": calibration_scores,
        "care_vs_selected_classical_heldout": comparisons[selected_mode],
        "care_vs_each_classical_heldout": comparisons,
        "hybrid_portfolio": {
            "selection_scope": "calibration_only",
            "selected_mode": selected_hybrid_mode,
            "care_calibration_mode": care_calibration_mode,
            "calibration_composite_scores": hybrid_calibration_scores,
            "heldout_vs_target_gp": hybrid_vs_target_gp,
            "real_experiment_deployment_ready": False,
        },
    }


def locate_care_metrics(care_tables: Path, tag: str, pair_id: str) -> Path:
    matches = sorted(care_tables.glob(f"*{tag}_{pair_id}_metrics.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CARE metrics file for {pair_id}; found {len(matches)}: {matches}"
        )
    return matches[0]


def locate_care_summary(care_runs: Path, tag: str, pair_id: str) -> Path:
    matches = sorted(care_runs.glob(f"*{tag}_{pair_id}_summary.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one CARE summary for {pair_id}; found {len(matches)}: {matches}"
        )
    return matches[0]


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CARE vs classical transfer BO",
        "",
        "The classical opponent is selected on calibration seeds and frozen before held-out comparison.",
        "",
        "| Pair | Selected classical arm | CARE final delta [95% CI] | CARE AUC delta [95% CI] |",
        "| --- | --- | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        comparison = pair["care_vs_selected_classical_heldout"]
        final = comparison["final_best"]
        auc = comparison["best_so_far_auc"]
        lines.append(
            "| {pair} | {mode} | {final:.3f} [{final_low:.3f}, {final_high:.3f}] | "
            "{auc:.3f} [{auc_low:.3f}, {auc_high:.3f}] |".format(
                pair=pair["pair_id"],
                mode=pair["selected_classical_mode"],
                final=final["mean_delta"],
                final_low=final["normal_95ci_low"],
                final_high=final["normal_95ci_high"],
                auc=auc["mean_delta"],
                auc_low=auc["normal_95ci_low"],
                auc_high=auc["normal_95ci_high"],
            )
        )
    lines.extend(
        [
            "",
            "## Calibration-selected CARE/classical portfolio",
            "",
            "| Pair | Frozen portfolio arm | Final delta vs target GP [95% CI] | AUC delta vs target GP [95% CI] |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for pair in summary["pairs"]:
        portfolio = pair["hybrid_portfolio"]
        comparison = portfolio["heldout_vs_target_gp"]
        final = comparison["final_best"]
        auc = comparison["best_so_far_auc"]
        lines.append(
            "| {pair} | {mode} | {final:.3f} [{final_low:.3f}, {final_high:.3f}] | "
            "{auc:.3f} [{auc_low:.3f}, {auc_high:.3f}] |".format(
                pair=pair["pair_id"],
                mode=portfolio["selected_mode"],
                final=final["mean_delta"],
                final_low=final["normal_95ci_low"],
                final_high=final["normal_95ci_high"],
                auc=auc["mean_delta"],
                auc_low=auc["normal_95ci_low"],
                auc_high=auc["normal_95ci_high"],
            )
        )
    lines.extend(
        [
            "",
            "A positive delta favors CARE. This table does not credit LLM causality; that requires the separate fixed data-only control.",
            "The hybrid portfolio is an offline calibration result and is not a zero-target-cost deployment gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--classical-dir", type=Path, required=True)
    parser.add_argument("--care-tables", type=Path, required=True)
    parser.add_argument("--care-runs", type=Path, required=True)
    parser.add_argument("--care-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = []
    for pair in config["pairs"]:
        pair_id = pair["pair_id"]
        pairs.append(
            summarize_pair(
                pair_id,
                args.classical_dir / f"{pair_id}_metrics.csv",
                locate_care_metrics(args.care_tables, args.care_tag, pair_id),
                locate_care_summary(args.care_runs, args.care_tag, pair_id),
            )
        )
    summary = {
        "experiment": "care2_vs_calibration_selected_classical_transfer_bo",
        "protocol_version": config["protocol"]["version"],
        "care_mode": CARE_MODE,
        "pairs": pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "care_vs_classical.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "care_vs_classical.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    print(render_markdown(summary))


if __name__ == "__main__":
    main()
