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
OUTPUT_TABLES = ROOT / "outputs" / "tables"
RESULTS = ROOT / "results"


def read_rows(path: Path, family: str, modes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["mode"] not in modes:
                continue
            rows.append(
                {
                    "policy": f"{family}:{row['mode']}",
                    "family": family,
                    "mode": row["mode"],
                    "seed": int(row["seed"]),
                    "final_best": float(row["final_best"]),
                    "best_so_far_auc": float(row["best_so_far_auc"]),
                    "top10_hit": float(row["top10_hit"]),
                }
            )
    return rows


def summarize_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "final_best_mean": round(mean(row["final_best"] for row in rows), 4),
        "final_best_std": round(pstdev(row["final_best"] for row in rows), 4) if len(rows) > 1 else 0.0,
        "best_so_far_auc_mean": round(mean(row["best_so_far_auc"] for row in rows), 4),
        "best_so_far_auc_std": round(pstdev(row["best_so_far_auc"] for row in rows), 4) if len(rows) > 1 else 0.0,
        "top10_hit_mean": round(mean(row["top10_hit"] for row in rows), 4),
    }


def split_summary(rows: list[dict[str, Any]], calibration_seed_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)

    policy_rows: list[dict[str, Any]] = []
    for policy, items in sorted(by_policy.items()):
        cal = [row for row in items if row["seed"] < calibration_seed_count]
        eval_rows = [row for row in items if row["seed"] >= calibration_seed_count]
        if len(cal) != calibration_seed_count or not eval_rows:
            continue
        cal_summary = summarize_values(cal)
        eval_summary = summarize_values(eval_rows)
        policy_rows.append(
            {
                "policy": policy,
                "calibration_seeds": len(cal),
                "evaluation_seeds": len(eval_rows),
                "cal_final_best_mean": cal_summary["final_best_mean"],
                "cal_best_so_far_auc_mean": cal_summary["best_so_far_auc_mean"],
                "cal_top10_hit_mean": cal_summary["top10_hit_mean"],
                "eval_final_best_mean": eval_summary["final_best_mean"],
                "eval_best_so_far_auc_mean": eval_summary["best_so_far_auc_mean"],
                "eval_top10_hit_mean": eval_summary["top10_hit_mean"],
            }
        )

    policy_rows.sort(
        key=lambda row: (
            row["cal_final_best_mean"],
            row["cal_best_so_far_auc_mean"],
            row["cal_top10_hit_mean"],
        ),
        reverse=True,
    )
    selected = policy_rows[0]
    gp_ucb = next(row for row in policy_rows if row["policy"] == "target_only_surrogate:mixed_kernel_gp_ucb")
    public_incumbent = next(row for row in policy_rows if row["policy"] == "target_only_surrogate:public_incumbent")
    selected_with_delta = dict(selected)
    selected_with_delta.update(
        {
            "eval_delta_final_vs_gp_ucb": round(selected["eval_final_best_mean"] - gp_ucb["eval_final_best_mean"], 4),
            "eval_delta_auc_vs_gp_ucb": round(selected["eval_best_so_far_auc_mean"] - gp_ucb["eval_best_so_far_auc_mean"], 4),
            "eval_delta_top10_vs_gp_ucb": round(selected["eval_top10_hit_mean"] - gp_ucb["eval_top10_hit_mean"], 4),
            "eval_delta_final_vs_public_incumbent": round(selected["eval_final_best_mean"] - public_incumbent["eval_final_best_mean"], 4),
            "eval_delta_auc_vs_public_incumbent": round(selected["eval_best_so_far_auc_mean"] - public_incumbent["eval_best_so_far_auc_mean"], 4),
            "eval_delta_top10_vs_public_incumbent": round(selected["eval_top10_hit_mean"] - public_incumbent["eval_top10_hit_mean"], 4),
        }
    )
    return policy_rows, selected_with_delta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibrated CARE skill portfolio summary tables.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "2026-07-04-calibrated-skill-portfolio")
    parser.add_argument("--calibration-seed-count", type=int, default=25)
    args = parser.parse_args()

    lipophilicity_rows: list[dict[str, Any]] = []
    lipophilicity_rows.extend(
        read_rows(
            OUTPUT_TABLES / "surrogate_baselines_real_moleculenet_lipophilicity_50seed_metrics.csv",
            "target_only_surrogate",
            {"random", "public_incumbent", "mixed_kernel_gp_ucb", "mixed_kernel_gp_ei", "knn_ucb"},
        )
    )
    lipophilicity_rows.extend(
        read_rows(
            OUTPUT_TABLES / "transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_server_value_prior_freesolv_to_lipo_50seed_metrics.csv",
            "freesolv_value_prior",
            {"transfer_value_prior_gate_v1", "transfer_value_prior_strict_gate_v1"},
        )
    )
    lipophilicity_rows.extend(
        read_rows(
            OUTPUT_TABLES / "transfer_real_moleculenet_esol_to_real_moleculenet_lipophilicity_esol_to_lipo_value_prior_50seed_metrics.csv",
            "esol_value_prior",
            {"transfer_value_prior_gate_v1", "transfer_value_prior_strict_gate_v1"},
        )
    )
    lipophilicity_rows.extend(
        read_rows(
            OUTPUT_TABLES / "transfer_weighted_kernel_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_metrics.csv",
            "freesolv_weighted_kernel",
            {"transfer_weighted_gp_ucb_scale_1p5", "transfer_weighted_gp_ucb_scale_4"},
        )
    )
    policy_rows, selected = split_summary(lipophilicity_rows, args.calibration_seed_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "lipophilicity_policy_calibration_summary.csv", policy_rows)
    (args.out_dir / "selected_lipophilicity_policy.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    freesolv_rows: list[dict[str, Any]] = []
    freesolv_rows.extend(
        read_rows(
            OUTPUT_TABLES / "surrogate_baselines_real_moleculenet_freesolv_50seed_metrics.csv",
            "target_only_surrogate",
            {"random", "public_incumbent", "mixed_kernel_gp_ucb", "mixed_kernel_gp_ei", "knn_ucb"},
        )
    )
    freesolv_rows.extend(
        read_rows(
            OUTPUT_TABLES / "transfer_real_moleculenet_lipophilicity_to_real_moleculenet_freesolv_lipo_to_freesolv_value_prior_50seed_metrics.csv",
            "lipo_value_prior",
            {"transfer_value_prior_gate_v1", "transfer_value_prior_strict_gate_v1"},
        )
    )
    freesolv_rows.extend(
        read_rows(
            OUTPUT_TABLES / "transfer_real_moleculenet_esol_to_real_moleculenet_freesolv_esol_to_freesolv_value_prior_50seed_metrics.csv",
            "esol_value_prior",
            {"transfer_value_prior_gate_v1", "transfer_value_prior_strict_gate_v1"},
        )
    )
    freesolv_policy_rows, freesolv_selected = split_summary(freesolv_rows, args.calibration_seed_count)
    write_csv(args.out_dir / "freesolv_policy_calibration_summary.csv", freesolv_policy_rows)
    (args.out_dir / "selected_freesolv_policy.json").write_text(
        json.dumps(freesolv_selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"lipophilicity_selected": selected, "freesolv_selected": freesolv_selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
