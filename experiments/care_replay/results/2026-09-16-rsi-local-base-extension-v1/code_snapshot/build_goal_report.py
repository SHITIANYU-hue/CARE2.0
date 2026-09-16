#!/usr/bin/env python3
"""Build a conservative cross-domain CARE 2.0 goal report.

The report separates three claims that are often conflated:

* source-outcome CARE route versus target-only BO/LLM baselines;
* deployment fallback, which proves non-loss but not transfer gain;
* matched random-rule null controls, which test whether semantic structure alone
  reproduces a result.

Only held-out summaries are consumed for comparisons. Development and
calibration summaries are retained as protocol metadata and never classified as
final evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import cross_task_router


TARGET_BO_MODES = (
    "gp_ucb",
    "mixed_kernel_gp_ei",
    "target_acquisition_portfolio",
)
PRIMARY_METRICS = ("final_best", "best_so_far_auc")


def normal_ci(mean_value: float, sd_value: float, count: int) -> tuple[float, float]:
    if count <= 1:
        return mean_value, mean_value
    half_width = 1.96 * sd_value / math.sqrt(count)
    return mean_value - half_width, mean_value + half_width


def classify_pairwise(pairwise: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for baseline, metrics in pairwise.items():
        primary_lows = [
            float(metrics[metric]["normal_95ci_low"])
            for metric in PRIMARY_METRICS
        ]
        primary_highs = [
            float(metrics[metric]["normal_95ci_high"])
            for metric in PRIMARY_METRICS
        ]
        result[baseline] = {
            "final_best_mean_delta": float(metrics["final_best"]["mean"]),
            "final_best_ci": [
                float(metrics["final_best"]["normal_95ci_low"]),
                float(metrics["final_best"]["normal_95ci_high"]),
            ],
            "auc_mean_delta": float(metrics["best_so_far_auc"]["mean"]),
            "auc_ci": [
                float(metrics["best_so_far_auc"]["normal_95ci_low"]),
                float(metrics["best_so_far_auc"]["normal_95ci_high"]),
            ],
            "primary_gain": any(value > 0.0 for value in primary_lows),
            "primary_harm": all(value < 0.0 for value in primary_highs),
        }
    return result


def strongest_target_bo(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    heldout = summary["heldout"]
    available = [mode for mode in TARGET_BO_MODES if mode in heldout]
    if not available:
        raise ValueError("Source-outcome summary has no target-only BO modes")
    selected = max(
        available,
        key=lambda mode: (
            float(heldout[mode]["composite"]),
            mode,
        ),
    )
    return selected, {
        "final_best": {"mean": float(heldout[selected]["final_best"])},
        "best_so_far_auc": {"mean": float(heldout[selected]["best_so_far_auc"])},
    }


def load_source_outcome(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    selection = summary["selection"]
    strongest_mode, strongest_metrics = strongest_target_bo(summary)
    pairwise = classify_pairwise(summary["heldout_pairwise"])
    source_transfer = bool(selection["selected_source_outcome_transfer"])
    return {
        "source_dataset": summary["source_dataset"],
        "target_dataset": summary["target_dataset"],
        "schema_route_proposal": summary.get(
            "schema_route_proposal",
            cross_task_router.propose_route(
                summary["source_dataset"],
                summary["target_dataset"],
            ).as_dict(),
        ),
        "heldout_seed_count": int(summary["heldout_seed_count"]),
        "selected_mode": selection["selected_mode"],
        "selected_source_outcome_transfer": bool(
            source_transfer
        ),
        "strongest_target_bo_mode": strongest_mode,
        "strongest_target_bo_final_best_mean": float(
            strongest_metrics["final_best"]["mean"]
        ),
        "strongest_target_bo_auc_mean": float(
            strongest_metrics["best_so_far_auc"]["mean"]
        ),
        "pairwise": pairwise,
        "deployment_non_loss": not source_transfer,
        "source_outcome_beats_strongest_bo": bool(
            source_transfer
            and pairwise.get(strongest_mode, {}).get("primary_gain", False)
        ),
        "source_outcome_beats_matched_llm": bool(
            source_transfer
            and pairwise.get("matched_target_only_llm", {}).get("primary_gain", False)
        ),
    }


def load_random_null_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    heldout = summary["selected_heldout"]
    count = int(heldout["seed_count"])
    final_sd = float(heldout["final_best_sd_delta"])
    auc_sd = float(heldout.get("auc_sd_delta", 0.0))
    final_low, final_high = normal_ci(
        float(heldout["final_best_mean_delta"]), final_sd, count
    )
    auc_low, auc_high = normal_ci(
        float(heldout["auc_mean_delta"]), auc_sd, count
    )
    return {
        "target_dataset": summary["protocol"]["target_dataset"],
        "selected_mode": summary["selected_on_calibration"],
        "heldout_seed_count": count,
        "final_best_mean_delta": float(heldout["final_best_mean_delta"]),
        "final_best_ci": [final_low, final_high],
        "auc_mean_delta": float(heldout["auc_mean_delta"]),
        "auc_ci": [auc_low, auc_high],
        "final_best_win_rate": float(heldout["final_best_win_rate"]),
        "stable_primary_gain": final_low > 0.0 or auc_low > 0.0,
    }


def load_random_null(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    heldout = summary["selected_heldout"]
    if "auc_sd_delta" not in heldout:
        raw_path = path.parent / "raw_metrics.csv"
        if raw_path.exists():
            rows = list(csv.DictReader(raw_path.open(encoding="utf-8", newline="")))
            selected = str(summary["selected_on_calibration"])
            heldout_seeds = {
                int(row["seed"])
                for row in rows
                if row.get("split") == "heldout"
            }
            selected_rows = {
                int(row["seed"]): row
                for row in rows
                if row.get("mode") == selected and int(row["seed"]) in heldout_seeds
            }
            baseline_rows = {
                int(row["seed"]): row
                for row in rows
                if row.get("mode") == "gp_ucb" and int(row["seed"]) in heldout_seeds
            }
            auc_deltas = [
                float(selected_rows[seed]["best_so_far_auc"])
                - float(baseline_rows[seed]["best_so_far_auc"])
                for seed in sorted(set(selected_rows) & set(baseline_rows))
            ]
            heldout["auc_sd_delta"] = (
                pstdev(auc_deltas) if len(auc_deltas) > 1 else 0.0
            )
    return load_random_null_from_summary(summary)


def load_traditional_csv(spec: str) -> dict[str, Any]:
    """Summarize a fixed traditional transfer CSV against its GP-UCB rows."""
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    baseline = [row for row in rows if row.get("mode") == "gp_ucb"]
    transfer_modes = sorted({
        row["mode"] for row in rows
        if row.get("mode", "").startswith("transfer_")
    })
    if not baseline or not transfer_modes:
        raise ValueError(f"No GP-UCB and transfer rows in {path}")
    by_seed = {int(row["seed"]): row for row in baseline}
    comparisons = []
    for mode in transfer_modes:
        by_mode = {
            int(row["seed"]): row
            for row in rows
            if row.get("mode") == mode
        }
        seeds = sorted(set(by_seed) & set(by_mode))
        final = [
            float(by_mode[seed]["final_best"])
            - float(by_seed[seed]["final_best"])
            for seed in seeds
        ]
        auc = [
            float(by_mode[seed]["best_so_far_auc"])
            - float(by_seed[seed]["best_so_far_auc"])
            for seed in seeds
        ]
        final_mean = mean(final)
        auc_mean = mean(auc)
        final_low, final_high = normal_ci(
            final_mean, pstdev(final) if len(final) > 1 else 0.0, len(final)
        )
        auc_low, auc_high = normal_ci(
            auc_mean, pstdev(auc) if len(auc) > 1 else 0.0, len(auc)
        )
        comparisons.append({
            "mode": mode,
            "seed_count": len(seeds),
            "final_best_mean_delta": final_mean,
            "final_best_ci": [final_low, final_high],
            "auc_mean_delta": auc_mean,
            "auc_ci": [auc_low, auc_high],
            "primary_gain": final_low > 0.0 or auc_low > 0.0,
        })
    return {
        "label": label,
        "path": str(path),
        "comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CARE 2.0 Cross-Domain Goal Report",
        "",
        "This report separates source-outcome transfer, target-only baselines, "
        "LLM routes, and matched random nulls. All source-outcome comparisons "
        "use held-out rows from frozen summaries.",
        "",
        "## Source-Outcome Paths",
        "",
        "| Source -> target | Route candidate | Deployed | Strongest target BO | Beats BO | Beats matched LLM |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["source_outcome"]:
        lines.append(
            f"| {item['source_dataset']} -> {item['target_dataset']} | "
            f"{item['schema_route_proposal']['recommended_candidate']} | "
            f"{item['selected_mode']} | {item['strongest_target_bo_mode']} | "
            f"{'yes' if item['source_outcome_beats_strongest_bo'] else 'no'} | "
            f"{'yes' if item['source_outcome_beats_matched_llm'] else 'no'} |"
        )
    lines.extend([
        "",
        "Route proposals are schema-only candidates; the calibration gate still "
        "decides whether source-outcome transfer is deployed.",
        "",
        "## Random Nulls",
        "",
        "| Target | Selected null | Final delta | AUC delta | Win rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for item in report["random_nulls"]:
        lines.append(
            f"| {item['target_dataset']} | {item['selected_mode']} | "
            f"{item['final_best_mean_delta']:.4f} | "
            f"{item['auc_mean_delta']:.4f} | "
            f"{item['final_best_win_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Traditional Transfer Baselines",
        "",
        "These are fixed weighted-kernel transfer runs, reported descriptively "
        "because their task pairs and seed counts are not the same as the frozen "
        "seven-pair CARE suite.",
        "",
        "| Pair | Mode | Final delta | AUC delta | Primary gain |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for item in report["traditional_transfer"]:
        for comparison in item["comparisons"]:
            lines.append(
                f"| {item['label']} | {comparison['mode']} | "
                f"{comparison['final_best_mean_delta']:.4f} | "
                f"{comparison['auc_mean_delta']:.4f} | "
                f"{'yes' if comparison['primary_gain'] else 'no'} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The report does not classify a fallback as a positive transfer gain. "
        "A random null with a positive mean but a confidence interval crossing "
        "zero is also not treated as stable generalization. A warm-start null can "
        "still beat GP-UCB, which is why initialization controls are reported separately. "
        "Traditional weighted-kernel "
        "rows are kept separate from the frozen CARE claim because their confidence "
        "intervals cross zero in this descriptive batch.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outcome-dir", type=Path, required=True)
    parser.add_argument("--random-null-summary", action="append", default=[])
    parser.add_argument("--traditional-metrics", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_paths = sorted((args.source_outcome_dir / "raw_summaries").glob("*.json"))
    source_outcome = [load_source_outcome(path) for path in summary_paths]
    random_nulls = [load_random_null(Path(path)) for path in args.random_null_summary]
    traditional = [load_traditional_csv(spec) for spec in args.traditional_metrics]
    report = {
        "study": "care2_cross_domain_goal",
        "source_outcome": source_outcome,
        "random_nulls": random_nulls,
        "traditional_transfer": traditional,
        "aggregate": {
            "source_outcome_pair_count": len(source_outcome),
            "source_outcome_beats_strongest_bo_count": sum(
                item["source_outcome_beats_strongest_bo"] for item in source_outcome
            ),
            "source_outcome_beats_matched_llm_count": sum(
                item["source_outcome_beats_matched_llm"] for item in source_outcome
            ),
            "random_null_count": len(random_nulls),
            "random_null_stable_gain_count": sum(
                item["stable_primary_gain"] for item in random_nulls
            ),
            "traditional_transfer_suite_count": len(traditional),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "goal_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "goal_report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
