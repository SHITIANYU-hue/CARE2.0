#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLES = ROOT / "outputs" / "tables"
RESULTS = ROOT / "results"


METRICS = ("final_best", "best_so_far_auc", "top10_hit")


@dataclass(frozen=True)
class PolicySource:
    source_dataset: str
    target_dataset: str
    family: str
    path: Path
    include_modes: set[str] | None = None
    exclude_modes: set[str] | None = None


def default_sources() -> list[PolicySource]:
    scale_ensemble_dir = RESULTS / "2026-07-14-scale-ensemble-transfer"
    return [
        # Target-only strong baselines.
        PolicySource("target_only", "real_buchwald_hartwig", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_buchwald_hartwig_50seed_metrics.csv"),
        PolicySource("target_only", "real_suzuki_miyaura", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_suzuki_miyaura_50seed_metrics.csv"),
        PolicySource("target_only", "real_chemlex_acidamine", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_chemlex_acidamine_50seed_metrics.csv"),
        PolicySource("target_only", "real_matbench_expt_gap", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_matbench_expt_gap_50seed_metrics.csv"),
        PolicySource("target_only", "real_moleculenet_freesolv", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_moleculenet_freesolv_50seed_metrics.csv"),
        PolicySource("target_only", "real_moleculenet_lipophilicity", "target_only", OUTPUT_TABLES / "surrogate_baselines_real_moleculenet_lipophilicity_50seed_metrics.csv"),
        # Public-evidence transfer gates and value priors.
        PolicySource("real_suzuki_miyaura", "real_buchwald_hartwig", "transfer_gate", OUTPUT_TABLES / "transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_server_suzuki_to_bh_transfer_50seed_metrics.csv"),
        PolicySource("real_buchwald_hartwig", "real_suzuki_miyaura", "transfer_gate", OUTPUT_TABLES / "transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_coverage_50seed_metrics.csv"),
        PolicySource("real_suzuki_miyaura", "real_chemlex_acidamine", "transfer_gate", OUTPUT_TABLES / "transfer_real_suzuki_miyaura_to_real_chemlex_acidamine_coverage_50seed_metrics.csv"),
        PolicySource("real_chemlex_acidamine", "real_suzuki_miyaura", "transfer_gate", OUTPUT_TABLES / "transfer_real_chemlex_acidamine_to_real_suzuki_miyaura_coverage_50seed_metrics.csv"),
        PolicySource("real_buchwald_hartwig", "real_chemlex_acidamine", "transfer_gate", OUTPUT_TABLES / "transfer_real_buchwald_hartwig_to_real_chemlex_acidamine_coverage_50seed_metrics.csv"),
        PolicySource("real_chemlex_acidamine", "real_buchwald_hartwig", "transfer_gate", OUTPUT_TABLES / "transfer_real_chemlex_acidamine_to_real_buchwald_hartwig_coverage_50seed_metrics.csv"),
        PolicySource("synthetic_materials_i", "real_matbench_expt_gap", "transfer_gate", OUTPUT_TABLES / "transfer_synthetic_materials_i_to_real_matbench_expt_gap_coverage_50seed_metrics.csv"),
        PolicySource("real_moleculenet_freesolv", "real_moleculenet_lipophilicity", "value_prior", OUTPUT_TABLES / "transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_server_value_prior_freesolv_to_lipo_50seed_metrics.csv"),
        PolicySource("real_moleculenet_lipophilicity", "real_moleculenet_freesolv", "value_prior", OUTPUT_TABLES / "transfer_real_moleculenet_lipophilicity_to_real_moleculenet_freesolv_lipo_to_freesolv_value_prior_50seed_metrics.csv"),
        PolicySource("real_moleculenet_esol", "real_moleculenet_freesolv", "value_prior", OUTPUT_TABLES / "transfer_real_moleculenet_esol_to_real_moleculenet_freesolv_esol_to_freesolv_value_prior_50seed_metrics.csv"),
        PolicySource("real_moleculenet_esol", "real_moleculenet_lipophilicity", "value_prior", OUTPUT_TABLES / "transfer_real_moleculenet_esol_to_real_moleculenet_lipophilicity_esol_to_lipo_value_prior_50seed_metrics.csv"),
        # Acquisition-level transfer over GP-UCB.
        PolicySource("real_suzuki_miyaura", "real_buchwald_hartwig", "weighted_kernel", OUTPUT_TABLES / "transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_metrics.csv"),
        PolicySource("real_buchwald_hartwig", "real_suzuki_miyaura", "weighted_kernel", OUTPUT_TABLES / "transfer_weighted_kernel_real_buchwald_hartwig_to_real_suzuki_miyaura_coverage_50seed_metrics.csv"),
        PolicySource("real_suzuki_miyaura", "real_chemlex_acidamine", "weighted_kernel", OUTPUT_TABLES / "transfer_weighted_kernel_real_suzuki_miyaura_to_real_chemlex_acidamine_coverage_50seed_metrics.csv"),
        PolicySource("real_chemlex_acidamine", "real_buchwald_hartwig", "weighted_kernel", OUTPUT_TABLES / "transfer_weighted_kernel_real_chemlex_acidamine_to_real_buchwald_hartwig_coverage_50seed_metrics.csv"),
        PolicySource("real_moleculenet_freesolv", "real_moleculenet_lipophilicity", "weighted_kernel", OUTPUT_TABLES / "transfer_weighted_kernel_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_metrics.csv"),
        # Robust scale ensembles over acquisition-level transfer. These average
        # normalized candidate ranks across several transfer-weight scales.
        PolicySource("real_suzuki_miyaura", "real_buchwald_hartwig", "weighted_kernel_ensemble", scale_ensemble_dir / "transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_scale_ensemble_mid_50seed_metrics.csv"),
        PolicySource("real_suzuki_miyaura", "real_chemlex_acidamine", "weighted_kernel_ensemble", scale_ensemble_dir / "transfer_weighted_kernel_real_suzuki_miyaura_to_real_chemlex_acidamine_scale_ensemble_mid_50seed_metrics.csv"),
        PolicySource("real_chemlex_acidamine", "real_buchwald_hartwig", "weighted_kernel_ensemble", scale_ensemble_dir / "transfer_weighted_kernel_real_chemlex_acidamine_to_real_buchwald_hartwig_scale_ensemble_high_50seed_metrics.csv"),
        # Hybrid transfer over GP-UCB.
        PolicySource("real_suzuki_miyaura", "real_buchwald_hartwig", "hybrid_gp", OUTPUT_TABLES / "hybrid_surrogate_transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_50seed_metrics.csv"),
        PolicySource("real_buchwald_hartwig", "real_suzuki_miyaura", "hybrid_gp", OUTPUT_TABLES / "hybrid_surrogate_transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_coverage_50seed_metrics.csv"),
        PolicySource("real_suzuki_miyaura", "real_chemlex_acidamine", "hybrid_gp", OUTPUT_TABLES / "hybrid_surrogate_transfer_real_suzuki_miyaura_to_real_chemlex_acidamine_coverage_50seed_metrics.csv"),
        PolicySource("real_chemlex_acidamine", "real_buchwald_hartwig", "hybrid_gp", OUTPUT_TABLES / "hybrid_surrogate_transfer_real_chemlex_acidamine_to_real_buchwald_hartwig_coverage_50seed_metrics.csv"),
        PolicySource("real_moleculenet_freesolv", "real_moleculenet_lipophilicity", "hybrid_gp", OUTPUT_TABLES / "hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_50seed_metrics.csv"),
    ]


def read_source(source: PolicySource) -> list[dict[str, Any]]:
    if not source.path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            if source.include_modes is not None and mode not in source.include_modes:
                continue
            if source.exclude_modes is not None and mode in source.exclude_modes:
                continue
            if source.family != "target_only" and mode in {"incumbent", "no_care_random", "random", "gp_ucb"}:
                continue
            rows.append(
                {
                    "source_dataset": source.source_dataset,
                    "target_dataset": source.target_dataset,
                    "family": source.family,
                    "mode": mode,
                    "policy": f"{source.family}:{mode}",
                    "seed": int(row["seed"]),
                    "final_best": float(row["final_best"]),
                    "best_so_far_auc": float(row["best_so_far_auc"]),
                    "top10_hit": float(row["top10_hit"]),
                    "path": str(source.path.relative_to(ROOT)),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in METRICS:
        values = [float(row[field]) for row in rows]
        out[f"{prefix}{field}_mean"] = round(mean(values), 4)
        out[f"{prefix}{field}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
    out[f"{prefix}balanced_score"] = round(
        out[f"{prefix}final_best_mean"] + out[f"{prefix}best_so_far_auc_mean"] + out[f"{prefix}top10_hit_mean"],
        4,
    )
    return out


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["source_dataset"], row["target_dataset"], row["family"], row["mode"])
        grouped[key].append(row)
    return grouped


def policy_summaries(rows: list[dict[str, Any]], calibration_seed_count: int) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for (source_dataset, target_dataset, family, mode), items in sorted(group_rows(rows).items()):
        if len(items) < calibration_seed_count + 1:
            continue
        cal = [row for row in items if row["seed"] < calibration_seed_count]
        heldout = [row for row in items if row["seed"] >= calibration_seed_count]
        if not cal or not heldout:
            continue
        full_summary = summarize(items)
        cal_summary = summarize(cal, "cal_")
        heldout_summary = summarize(heldout, "heldout_")
        summaries.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "family": family,
                "mode": mode,
                "policy": f"{family}:{mode}",
                "seeds": len(items),
                "calibration_seeds": len(cal),
                "heldout_seeds": len(heldout),
                **full_summary,
                **cal_summary,
                **heldout_summary,
            }
        )
    return summaries


def best_by(rows: list[dict[str, Any]], metric_prefix: str, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    choices = candidates if candidates is not None else rows
    return max(
        choices,
        key=lambda row: (
            row[f"{metric_prefix}balanced_score"],
            row[f"{metric_prefix}final_best_mean"],
            row[f"{metric_prefix}best_so_far_auc_mean"],
            row[f"{metric_prefix}top10_hit_mean"],
        ),
    )


def risk_aware_selection(
    transfer_candidate: dict[str, Any],
    fallback: dict[str, Any],
    final_margin: float,
    auc_margin: float,
) -> dict[str, Any]:
    if (
        transfer_candidate["cal_final_best_mean"] >= fallback["cal_final_best_mean"] + final_margin
        and transfer_candidate["cal_best_so_far_auc_mean"] >= fallback["cal_best_so_far_auc_mean"] + auc_margin
    ):
        return transfer_candidate
    return fallback


def build_pair_summary(
    summaries: list[dict[str, Any]],
    transfer_final_margin: float,
    transfer_auc_margin: float,
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    target_baselines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        if row["family"] == "target_only":
            target_baselines[row["target_dataset"]].append(row)
        else:
            by_pair[(row["source_dataset"], row["target_dataset"])].append(row)

    out: list[dict[str, Any]] = []
    for (source_dataset, target_dataset), transfer_rows in sorted(by_pair.items()):
        baselines = target_baselines.get(target_dataset, [])
        if not baselines:
            continue
        public = next((row for row in baselines if row["mode"] == "public_incumbent"), None)
        gp_ucb = next((row for row in baselines if row["mode"] == "mixed_kernel_gp_ucb"), None)
        best_target_full = best_by(baselines, "")
        best_target_heldout = best_by(baselines, "heldout_")
        best_target_cal = best_by(baselines, "cal_")
        best_transfer_full = best_by(transfer_rows, "")
        selected_transfer = best_by(transfer_rows, "cal_")
        candidate_rows = [*baselines, *transfer_rows]
        selected = best_by(candidate_rows, "cal_")
        selected_baseline = public or best_target_full
        gp_baseline = gp_ucb or best_target_full
        risk_gp = risk_aware_selection(selected_transfer, gp_baseline, transfer_final_margin, transfer_auc_margin)
        risk_best = risk_aware_selection(selected_transfer, best_target_cal, transfer_final_margin, transfer_auc_margin)
        out.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "best_transfer_policy": best_transfer_full["policy"],
                "best_transfer_final_mean": best_transfer_full["final_best_mean"],
                "best_transfer_auc_mean": best_transfer_full["best_so_far_auc_mean"],
                "best_transfer_top10_mean": best_transfer_full["top10_hit_mean"],
                "delta_transfer_final_vs_public": round(best_transfer_full["final_best_mean"] - selected_baseline["final_best_mean"], 4),
                "delta_transfer_auc_vs_public": round(best_transfer_full["best_so_far_auc_mean"] - selected_baseline["best_so_far_auc_mean"], 4),
                "delta_transfer_final_vs_gp_ucb": round(best_transfer_full["final_best_mean"] - gp_baseline["final_best_mean"], 4),
                "delta_transfer_auc_vs_gp_ucb": round(best_transfer_full["best_so_far_auc_mean"] - gp_baseline["best_so_far_auc_mean"], 4),
                "delta_transfer_final_vs_best_target": round(best_transfer_full["final_best_mean"] - best_target_full["final_best_mean"], 4),
                "delta_transfer_auc_vs_best_target": round(best_transfer_full["best_so_far_auc_mean"] - best_target_full["best_so_far_auc_mean"], 4),
                "selected_by_calibration": selected["policy"],
                "selected_family": selected["family"],
                "selected_heldout_final_mean": selected["heldout_final_best_mean"],
                "selected_heldout_auc_mean": selected["heldout_best_so_far_auc_mean"],
                "selected_heldout_top10_mean": selected["heldout_top10_hit_mean"],
                "heldout_delta_selected_final_vs_public": round(selected["heldout_final_best_mean"] - selected_baseline["heldout_final_best_mean"], 4),
                "heldout_delta_selected_auc_vs_public": round(selected["heldout_best_so_far_auc_mean"] - selected_baseline["heldout_best_so_far_auc_mean"], 4),
                "heldout_delta_selected_final_vs_gp_ucb": round(selected["heldout_final_best_mean"] - gp_baseline["heldout_final_best_mean"], 4),
                "heldout_delta_selected_auc_vs_gp_ucb": round(selected["heldout_best_so_far_auc_mean"] - gp_baseline["heldout_best_so_far_auc_mean"], 4),
                "heldout_delta_selected_final_vs_best_target": round(selected["heldout_final_best_mean"] - best_target_heldout["heldout_final_best_mean"], 4),
                "heldout_delta_selected_auc_vs_best_target": round(selected["heldout_best_so_far_auc_mean"] - best_target_heldout["heldout_best_so_far_auc_mean"], 4),
                "selected_transfer_by_calibration": selected_transfer["policy"],
                "selected_transfer_heldout_final_mean": selected_transfer["heldout_final_best_mean"],
                "selected_transfer_heldout_auc_mean": selected_transfer["heldout_best_so_far_auc_mean"],
                "selected_transfer_heldout_top10_mean": selected_transfer["heldout_top10_hit_mean"],
                "heldout_delta_transfer_selector_final_vs_public": round(selected_transfer["heldout_final_best_mean"] - selected_baseline["heldout_final_best_mean"], 4),
                "heldout_delta_transfer_selector_auc_vs_public": round(selected_transfer["heldout_best_so_far_auc_mean"] - selected_baseline["heldout_best_so_far_auc_mean"], 4),
                "heldout_delta_transfer_selector_final_vs_gp_ucb": round(selected_transfer["heldout_final_best_mean"] - gp_baseline["heldout_final_best_mean"], 4),
                "heldout_delta_transfer_selector_auc_vs_gp_ucb": round(selected_transfer["heldout_best_so_far_auc_mean"] - gp_baseline["heldout_best_so_far_auc_mean"], 4),
                "heldout_delta_transfer_selector_final_vs_best_target": round(selected_transfer["heldout_final_best_mean"] - best_target_heldout["heldout_final_best_mean"], 4),
                "heldout_delta_transfer_selector_auc_vs_best_target": round(selected_transfer["heldout_best_so_far_auc_mean"] - best_target_heldout["heldout_best_so_far_auc_mean"], 4),
                "risk_aware_vs_gp_selected": risk_gp["policy"],
                "risk_aware_vs_gp_family": risk_gp["family"],
                "risk_aware_vs_gp_heldout_final_mean": risk_gp["heldout_final_best_mean"],
                "risk_aware_vs_gp_heldout_auc_mean": risk_gp["heldout_best_so_far_auc_mean"],
                "heldout_delta_risk_gp_final_vs_gp_ucb": round(risk_gp["heldout_final_best_mean"] - gp_baseline["heldout_final_best_mean"], 4),
                "heldout_delta_risk_gp_auc_vs_gp_ucb": round(risk_gp["heldout_best_so_far_auc_mean"] - gp_baseline["heldout_best_so_far_auc_mean"], 4),
                "risk_aware_vs_best_target_selected": risk_best["policy"],
                "risk_aware_vs_best_target_family": risk_best["family"],
                "risk_aware_vs_best_target_heldout_final_mean": risk_best["heldout_final_best_mean"],
                "risk_aware_vs_best_target_heldout_auc_mean": risk_best["heldout_best_so_far_auc_mean"],
                "heldout_delta_risk_best_final_vs_best_target": round(risk_best["heldout_final_best_mean"] - best_target_heldout["heldout_final_best_mean"], 4),
                "heldout_delta_risk_best_auc_vs_best_target": round(risk_best["heldout_best_so_far_auc_mean"] - best_target_heldout["heldout_best_so_far_auc_mean"], 4),
                "best_target_policy": best_target_full["policy"],
                "best_target_final_mean": best_target_full["final_best_mean"],
                "best_target_auc_mean": best_target_full["best_so_far_auc_mean"],
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize transfer coverage and calibrated policy selection.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "2026-07-12-transfer-coverage-portfolio")
    parser.add_argument("--calibration-seed-count", type=int, default=25)
    parser.add_argument("--transfer-final-margin", type=float, default=1.0)
    parser.add_argument("--transfer-auc-margin", type=float, default=1.0)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in default_sources():
        source_rows = read_source(source)
        if source_rows:
            rows.extend(source_rows)
        else:
            missing.append(str(source.path.relative_to(ROOT)))

    summaries = policy_summaries(rows, args.calibration_seed_count)
    pair_rows = build_pair_summary(summaries, args.transfer_final_margin, args.transfer_auc_margin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "policy_summary.csv", summaries)
    write_csv(args.out_dir / "pair_summary.csv", pair_rows)

    transfer_positive_vs_public = sum(1 for row in pair_rows if row["delta_transfer_final_vs_public"] > 0)
    transfer_positive_vs_gp = sum(1 for row in pair_rows if row["delta_transfer_final_vs_gp_ucb"] > 0)
    transfer_positive_vs_best_target = sum(1 for row in pair_rows if row["delta_transfer_final_vs_best_target"] > 0)
    selected_transfer = sum(1 for row in pair_rows if row["selected_family"] != "target_only")
    selected_positive_vs_public = sum(1 for row in pair_rows if row["heldout_delta_selected_final_vs_public"] > 0)
    selected_positive_vs_gp = sum(1 for row in pair_rows if row["heldout_delta_selected_final_vs_gp_ucb"] > 0)
    transfer_selector_positive_vs_public = sum(1 for row in pair_rows if row["heldout_delta_transfer_selector_final_vs_public"] > 0)
    transfer_selector_positive_vs_gp = sum(1 for row in pair_rows if row["heldout_delta_transfer_selector_final_vs_gp_ucb"] > 0)
    transfer_selector_positive_vs_best_target = sum(1 for row in pair_rows if row["heldout_delta_transfer_selector_final_vs_best_target"] > 0)
    risk_gp_selected_transfer = sum(1 for row in pair_rows if row["risk_aware_vs_gp_family"] != "target_only")
    risk_gp_positive = sum(1 for row in pair_rows if row["heldout_delta_risk_gp_final_vs_gp_ucb"] > 0)
    risk_best_selected_transfer = sum(1 for row in pair_rows if row["risk_aware_vs_best_target_family"] != "target_only")
    risk_best_positive = sum(1 for row in pair_rows if row["heldout_delta_risk_best_final_vs_best_target"] > 0)

    payload = {
        "experiment": "transfer_coverage_portfolio_summary",
        "calibration_seed_count": args.calibration_seed_count,
        "transfer_final_margin": args.transfer_final_margin,
        "transfer_auc_margin": args.transfer_auc_margin,
        "pair_count": len(pair_rows),
        "policy_count": len(summaries),
        "missing_inputs": missing,
        "counts": {
            "best_transfer_positive_final_vs_public": transfer_positive_vs_public,
            "best_transfer_positive_final_vs_gp_ucb": transfer_positive_vs_gp,
            "best_transfer_positive_final_vs_best_target": transfer_positive_vs_best_target,
            "calibrated_selector_chose_transfer": selected_transfer,
            "calibrated_selector_positive_heldout_final_vs_public": selected_positive_vs_public,
            "calibrated_selector_positive_heldout_final_vs_gp_ucb": selected_positive_vs_gp,
            "transfer_only_selector_positive_heldout_final_vs_public": transfer_selector_positive_vs_public,
            "transfer_only_selector_positive_heldout_final_vs_gp_ucb": transfer_selector_positive_vs_gp,
            "transfer_only_selector_positive_heldout_final_vs_best_target": transfer_selector_positive_vs_best_target,
            "risk_aware_vs_gp_selected_transfer": risk_gp_selected_transfer,
            "risk_aware_vs_gp_positive_heldout_final_vs_gp_ucb": risk_gp_positive,
            "risk_aware_vs_best_target_selected_transfer": risk_best_selected_transfer,
            "risk_aware_vs_best_target_positive_heldout_final_vs_best_target": risk_best_positive,
        },
        "pair_summary_csv": "pair_summary.csv",
        "policy_summary_csv": "policy_summary.csv",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
