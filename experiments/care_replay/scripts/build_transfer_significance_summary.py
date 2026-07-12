#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "2026-07-11-transfer-significance-audit"


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    metrics: Path
    baseline_mode: str
    transfer_mode: str
    note: str


DEFAULT_COMPARISONS = (
    ComparisonSpec(
        label="molecule_value_prior_vs_incumbent_50seed",
        metrics=ROOT
        / "results"
        / "2026-07-03-transfer-advantage-sweep"
        / "transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_server_value_prior_freesolv_to_lipo_50seed_metrics.csv",
        baseline_mode="incumbent",
        transfer_mode="transfer_value_prior_gate_v1",
        note="Shared MoleculeNet descriptor value-prior transfer over the public incumbent.",
    ),
    ComparisonSpec(
        label="reaction_role_transfer_vs_incumbent_50seed",
        metrics=ROOT
        / "results"
        / "2026-07-03-transfer-advantage-sweep"
        / "transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_server_suzuki_to_bh_transfer_50seed_metrics.csv",
        baseline_mode="incumbent",
        transfer_mode="transfer_gate_v1",
        note="Suzuki-to-Buchwald-Hartwig role-level transfer over the public incumbent.",
    ),
    ComparisonSpec(
        label="reaction_transfer_weighted_kernel_vs_gpucb_100seed",
        metrics=ROOT
        / "results"
        / "2026-07-05-calibrated-transfer-weighted-kernel"
        / "tables"
        / "transfer_weighted_kernel_real_suzuki_miyaura_to_real_buchwald_hartwig_calibrated_grid_100seed_metrics.csv",
        baseline_mode="gp_ucb",
        transfer_mode="transfer_weighted_gp_ucb_scale_1p5",
        note="Acquisition-geometry transfer over mixed-kernel GP-UCB; included as a strong-baseline boundary check.",
    ),
    ComparisonSpec(
        label="molecule_hybrid_value_prior_vs_gpucb_lowbudget5_100seed",
        metrics=ROOT
        / "results"
        / "2026-07-04-hybrid-transfer-budget-sweep"
        / "hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_lowbudget5_100seed_metrics.csv",
        baseline_mode="gp_ucb",
        transfer_mode="hybrid_value_prior_gp_ucb_gate_v1",
        note="Low-budget shared value-prior transfer over GP-UCB; included as a strong-baseline boundary check.",
    ),
    ComparisonSpec(
        label="molecule_target_calibrated_hybrid_vs_gpucb_lowbudget5_100seed",
        metrics=ROOT
        / "results"
        / "2026-07-11-transfer-significance-audit"
        / "tables"
        / "hybrid_surrogate_transfer_real_moleculenet_freesolv_to_real_moleculenet_lipophilicity_target_calibrated_value_prior_lowbudget5_100seed_metrics.csv",
        baseline_mode="gp_ucb",
        transfer_mode="hybrid_value_prior_gp_ucb_target_calibrated_gate_v1",
        note="Safer target-calibrated shared value-prior transfer over GP-UCB; reduces bad interventions but is not a headline gain.",
    ),
    ComparisonSpec(
        label="llm_rule_patch_confirmed_vs_fixed_transfer_10seed",
        metrics=ROOT
        / "results"
        / "2026-07-10-llm-rule-patch-transfer-followup"
        / "tables"
        / "transfer_real_suzuki_miyaura_to_real_buchwald_hartwig_rule_patch_guarded_risk_control_openai_gpt-5_5_suzuki_to_bh_10seed_metrics.csv",
        baseline_mode="transfer_gate_v1",
        transfer_mode="llm_rule_patch_guarded_confirmed_interaction_gate_v1",
        note="LLM rule-patch transfer over fixed deterministic transfer; exploratory 10-seed signal.",
    ),
)


def parse_comparison(raw: str) -> ComparisonSpec:
    parts = raw.split(":", 4)
    if len(parts) < 4:
        raise ValueError(
            "--comparison must be label:metrics_csv:baseline_mode:transfer_mode[:note]"
        )
    label, metrics, baseline_mode, transfer_mode = parts[:4]
    note = parts[4] if len(parts) == 5 else ""
    return ComparisonSpec(label, Path(metrics), baseline_mode, transfer_mode, note)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paired_values(
    rows: list[dict[str, str]],
    baseline_mode: str,
    transfer_mode: str,
    metric: str,
) -> list[float]:
    by_key = {
        (row["mode"], int(row["seed"])): float(row[metric])
        for row in rows
        if row.get("seed", "").lstrip("-").isdigit()
        and row.get("mode") in {baseline_mode, transfer_mode}
        and row.get(metric, "") != ""
    }
    seeds = sorted(
        {seed for mode, seed in by_key if mode == baseline_mode}
        & {seed for mode, seed in by_key if mode == transfer_mode}
    )
    return [by_key[(transfer_mode, seed)] - by_key[(baseline_mode, seed)] for seed in seeds]


def bootstrap_ci(values: list[float], samples: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    boot: list[float] = []
    count = len(values)
    for _ in range(samples):
        boot.append(mean(values[rng.randrange(count)] for _ in range(count)))
    boot.sort()
    return boot[int(samples * 0.025)], boot[int(samples * 0.975)]


def summarize_comparison(spec: ComparisonSpec, samples: int, rng: random.Random) -> dict[str, Any]:
    rows = load_rows(spec.metrics)
    metric_rows: dict[str, Any] = {}
    seed_count = 0
    for metric in ("final_best", "best_so_far_auc", "top10_hit", "bad_intervention_count"):
        if metric not in rows[0]:
            continue
        deltas = paired_values(rows, spec.baseline_mode, spec.transfer_mode, metric)
        if not deltas:
            continue
        seed_count = max(seed_count, len(deltas))
        lower, upper = bootstrap_ci(deltas, samples, rng)
        metric_rows[metric] = {
            "mean_delta": round(mean(deltas), 4),
            "std_delta": round(pstdev(deltas), 4) if len(deltas) > 1 else 0.0,
            "bootstrap_ci_low": round(lower, 4),
            "bootstrap_ci_high": round(upper, 4),
            "wins": sum(1 for value in deltas if value > 0),
            "losses": sum(1 for value in deltas if value < 0),
            "ties": sum(1 for value in deltas if value == 0),
            "ci_excludes_zero_positive": lower > 0.0,
        }
    return {
        "label": spec.label,
        "metrics": str(spec.metrics.relative_to(ROOT)),
        "baseline_mode": spec.baseline_mode,
        "transfer_mode": spec.transfer_mode,
        "paired_seed_count": seed_count,
        "note": spec.note,
        "metrics_summary": metric_rows,
    }


def flatten(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in summary_rows:
        for metric, metric_summary in item["metrics_summary"].items():
            flat.append(
                {
                    "label": item["label"],
                    "metric": metric,
                    "baseline_mode": item["baseline_mode"],
                    "transfer_mode": item["transfer_mode"],
                    "paired_seed_count": item["paired_seed_count"],
                    "mean_delta": metric_summary["mean_delta"],
                    "bootstrap_ci_low": metric_summary["bootstrap_ci_low"],
                    "bootstrap_ci_high": metric_summary["bootstrap_ci_high"],
                    "wins": metric_summary["wins"],
                    "losses": metric_summary["losses"],
                    "ties": metric_summary["ties"],
                    "ci_excludes_zero_positive": metric_summary["ci_excludes_zero_positive"],
                    "note": item["note"],
                    "metrics_file": item["metrics"],
                }
            )
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired transfer significance summaries.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="Optional label:metrics_csv:baseline_mode:transfer_mode[:note]. Defaults cover current headline checks.",
    )
    args = parser.parse_args()

    specs = [parse_comparison(raw) for raw in args.comparison] if args.comparison else list(DEFAULT_COMPARISONS)
    rng = random.Random(args.seed)
    summaries = [summarize_comparison(spec, args.samples, rng) for spec in specs]
    flat = flatten(summaries)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "transfer_significance_summary.csv", flat)
    (args.out_dir / "transfer_significance_summary.json").write_text(
        json.dumps(
            {
                "bootstrap_samples": args.samples,
                "random_seed": args.seed,
                "comparisons": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(flat)}, indent=2))


if __name__ == "__main__":
    main()
