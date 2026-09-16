#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRIC_FIELDS = ("final_best", "best_so_far_auc", "top10_hit")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "mode": row["mode"],
                    "seed": int(row["seed"]),
                    "final_best": float(row["final_best"]),
                    "best_so_far_auc": float(row["best_so_far_auc"]),
                    "top10_hit": float(row["top10_hit"]),
                }
            )
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in METRIC_FIELDS:
        values = [float(row[field]) for row in rows]
        out[f"{field}_mean"] = round(mean(values), 4)
        out[f"{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
    return out


def policy_table(
    rows: list[dict[str, Any]],
    calibration_seed_count: int,
    baseline_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    if baseline_mode not in by_mode:
        raise ValueError(f"Baseline mode {baseline_mode!r} not present in metrics")

    table: list[dict[str, Any]] = []
    for mode, mode_rows in sorted(by_mode.items()):
        cal_rows = [row for row in mode_rows if row["seed"] < calibration_seed_count]
        eval_rows = [row for row in mode_rows if row["seed"] >= calibration_seed_count]
        if not cal_rows or not eval_rows:
            raise ValueError(
                f"Mode {mode!r} has {len(cal_rows)} calibration rows and "
                f"{len(eval_rows)} evaluation rows; check --calibration-seed-count."
            )
        cal_summary = summarize(cal_rows)
        eval_summary = summarize(eval_rows)
        table.append(
            {
                "mode": mode,
                "calibration_seeds": len(cal_rows),
                "evaluation_seeds": len(eval_rows),
                "cal_final_best_mean": cal_summary["final_best_mean"],
                "cal_best_so_far_auc_mean": cal_summary["best_so_far_auc_mean"],
                "cal_top10_hit_mean": cal_summary["top10_hit_mean"],
                "eval_final_best_mean": eval_summary["final_best_mean"],
                "eval_best_so_far_auc_mean": eval_summary["best_so_far_auc_mean"],
                "eval_top10_hit_mean": eval_summary["top10_hit_mean"],
            }
        )

    cal_baseline = next(row for row in table if row["mode"] == baseline_mode)
    eval_baseline = cal_baseline
    for row in table:
        row["cal_delta_final_vs_baseline"] = round(row["cal_final_best_mean"] - cal_baseline["cal_final_best_mean"], 4)
        row["cal_delta_auc_vs_baseline"] = round(row["cal_best_so_far_auc_mean"] - cal_baseline["cal_best_so_far_auc_mean"], 4)
        row["cal_delta_top10_vs_baseline"] = round(row["cal_top10_hit_mean"] - cal_baseline["cal_top10_hit_mean"], 4)
        row["eval_delta_final_vs_baseline"] = round(row["eval_final_best_mean"] - eval_baseline["eval_final_best_mean"], 4)
        row["eval_delta_auc_vs_baseline"] = round(row["eval_best_so_far_auc_mean"] - eval_baseline["eval_best_so_far_auc_mean"], 4)
        row["eval_delta_top10_vs_baseline"] = round(row["eval_top10_hit_mean"] - eval_baseline["eval_top10_hit_mean"], 4)
        row["cal_balanced_score"] = round(
            row["cal_final_best_mean"] + row["cal_best_so_far_auc_mean"] + row["cal_top10_hit_mean"],
            4,
        )

    selectors = {
        "balanced": {
            "description": "Selects the mode with the largest calibration final_best + AUC + top10_hit.",
            "key": lambda row: (
                row["cal_balanced_score"],
                row["cal_final_best_mean"],
                row["cal_best_so_far_auc_mean"],
                row["cal_top10_hit_mean"],
            ),
        },
        "final_priority": {
            "description": "Selects the mode with the largest calibration final best, then AUC.",
            "key": lambda row: (
                row["cal_final_best_mean"],
                row["cal_best_so_far_auc_mean"],
                row["cal_top10_hit_mean"],
            ),
        },
        "auc_priority": {
            "description": "Selects the mode with the largest calibration AUC, then final best.",
            "key": lambda row: (
                row["cal_best_so_far_auc_mean"],
                row["cal_final_best_mean"],
                row["cal_top10_hit_mean"],
            ),
        },
    }
    selected: dict[str, dict[str, Any]] = {}
    for selector_name, selector in selectors.items():
        chosen = max(table, key=selector["key"])
        selected[selector_name] = {
            "selector": selector_name,
            "description": selector["description"],
            **chosen,
        }
    return table, selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration/held-out summaries for transfer-weighted GP kernels.")
    parser.add_argument("--metrics", type=Path, required=True, help="Metrics CSV from run_transfer_weighted_kernel.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="", help="Short label used in output filenames.")
    parser.add_argument("--calibration-seed-count", type=int, default=50)
    parser.add_argument("--baseline-mode", default="gp_ucb")
    args = parser.parse_args()

    label = args.label or args.metrics.stem
    rows = read_rows(args.metrics)
    table, selected = policy_table(rows, args.calibration_seed_count, args.baseline_mode)
    table.sort(
        key=lambda row: (
            row["cal_balanced_score"],
            row["cal_final_best_mean"],
            row["cal_best_so_far_auc_mean"],
        ),
        reverse=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{label}_policy_calibration_summary.csv", table)
    payload = {
        "label": label,
        "metrics": str(args.metrics),
        "baseline_mode": args.baseline_mode,
        "calibration_seed_count": args.calibration_seed_count,
        "total_rows": len(rows),
        "selected": selected,
    }
    (args.out_dir / f"{label}_selected_policies.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
