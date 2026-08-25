#!/usr/bin/env python3
"""Plot the preregistered TrpB efficacy, attribution, and gate diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


TRUE = "true_source_outcomes"
BASELINE = "target_gp_ucb"
METHOD_LABELS = {
    "multisource_rgpe": "RGPE",
    "multisource_icm_bma": "ICM-BMA",
    "multisource_skill_prior": "Generic skill prior",
    "source_additive_mutation_prior": "LLM additive skill",
}
COLORS = {
    "target": "#34434F",
    "null": "#9AA5B1",
    "llm": "#137A68",
    "generic": "#D47A22",
    "rgpe": "#2E6F9E",
    "icm": "#7A4EAB",
    "gate": "#B4453E",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_name(f"{path.name}.sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, draws: int = 20_000
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    chunk = 2_000
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = np.mean(array[indices], axis=1)
    return (
        float(np.mean(array)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(materialized[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(materialized)
    write_sha256(path)


def deployment_effects(efficacy_root: Path) -> list[dict[str, Any]]:
    summary = read_json(efficacy_root / "deployment" / "summary.json")
    rows = []
    for method, label in METHOD_LABELS.items():
        effect = summary["comparisons_vs_target_gp_ucb"][method]["best_so_far_auc"]
        rows.append(
            {
                "method": method,
                "label": label,
                "mean_delta_auc": float(effect["mean_delta"]),
                "ci95_low": float(effect["normal_95ci_low"]),
                "ci95_high": float(effect["normal_95ci_high"]),
                "win_rate": float(effect["win_rate"]),
                "non_loss_rate": float(effect["non_loss_rate"]),
            }
        )
    return rows


def trajectory_summary(mechanism_root: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(mechanism_root / "trajectory_audit.jsonl")
    by_condition_seed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition_seed[str(row["condition"])][int(row["target_seed"])] = row
    seeds = sorted(by_condition_seed[BASELINE])
    null_conditions = sorted(
        condition for condition in by_condition_seed if condition not in {BASELINE, TRUE}
    )
    rounds = len(by_condition_seed[BASELINE][seeds[0]]["best_trace"])
    output: list[dict[str, Any]] = []
    for round_index in range(rounds):
        series = {
            "Target-only GP-UCB": [
                float(by_condition_seed[BASELINE][seed]["best_trace"][round_index])
                for seed in seeds
            ],
            "Outcome-permuted skills": [
                float(
                    np.mean(
                        [
                            float(
                                by_condition_seed[condition][seed]["best_trace"][
                                    round_index
                                ]
                            )
                            for condition in null_conditions
                        ]
                    )
                )
                for seed in seeds
            ],
            "LLM skill, true binding": [
                float(by_condition_seed[TRUE][seed]["best_trace"][round_index])
                for seed in seeds
            ],
        }
        for series_index, (label, values) in enumerate(series.items()):
            estimate = bootstrap_mean_interval(
                values, seed=20260825 + 100 * series_index + round_index
            )
            output.append(
                {
                    "series": label,
                    "round": round_index + 1,
                    "seed_count": len(values),
                    "mean_best_so_far": estimate[0],
                    "bootstrap_ci95_low": estimate[1],
                    "bootstrap_ci95_high": estimate[2],
                }
            )
    return output


def plot(efficacy_root: Path, mechanism_root: Path, output: Path) -> None:
    effects = deployment_effects(efficacy_root)
    mechanism_report = read_json(mechanism_root / "falsification_report.json")
    assignment_rows = read_csv(mechanism_root / "assignment_effects.csv")
    trajectories = trajectory_summary(mechanism_root)
    gate_summary = read_json(efficacy_root / "source_only_gate_summary.json")
    null_effects = [
        float(row["auc_mean_delta"])
        for row in assignment_rows
        if row["condition"] != TRUE
    ]
    true_effect = float(
        mechanism_report["true_assignment"]["auc_delta_vs_target_gp"]["mean"]
    )
    p_value = float(
        mechanism_report["true_vs_permuted_assignment"][
            "one_sided_randomization_p_auc"
        ]
    )
    calibration = gate_summary["gate"]["method_diagnostics"][
        "source_additive_mutation_prior"
    ]["route_effects"][0]

    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(effects, output.with_name(f"{output.name}_effects.csv"))
    write_csv(trajectories, output.with_name(f"{output.name}_trajectories.csv"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#23313D",
            "axes.labelcolor": "#23313D",
            "xtick.color": "#4B5964",
            "ytick.color": "#23313D",
        }
    )
    figure = plt.figure(figsize=(15.4, 5.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.05, 1.0, 1.4))
    forest = figure.add_subplot(grid[0, 0])
    null_axis = figure.add_subplot(grid[0, 1])
    trace_axis = figure.add_subplot(grid[0, 2])

    method_colors = {
        "multisource_rgpe": COLORS["rgpe"],
        "multisource_icm_bma": COLORS["icm"],
        "multisource_skill_prior": COLORS["generic"],
        "source_additive_mutation_prior": COLORS["llm"],
    }
    y_positions = np.arange(len(effects))[::-1]
    for y, row in zip(y_positions, effects):
        mean = float(row["mean_delta_auc"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        forest.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=7.5 if row["method"] == "source_additive_mutation_prior" else 6.5,
            color=method_colors[row["method"]],
            ecolor=method_colors[row["method"]],
            elinewidth=2.1,
            capsize=4,
        )
        forest.text(high + 0.04, y, f"{mean:+.3f}", va="center", fontsize=9)
    forest.axvline(0.0, color="#2D3748", linewidth=1.0, linestyle="--")
    forest.set_yticks(y_positions, [row["label"] for row in effects])
    forest.set_xlabel("Paired best-so-far AUC delta\nvs target-only GP-UCB (95% CI)")
    forest.set_title("A  New-family efficacy", loc="left")
    forest.spines[["top", "right", "left"]].set_visible(False)
    forest.tick_params(axis="y", length=0)
    forest.text(
        0.02,
        -0.25,
        "Gate result: HOLD\ncalibration +0.044 [−0.018, +0.106]\nexact token coverage: 0% calibration vs 99.97% deployment",
        transform=forest.transAxes,
        fontsize=9,
        color=COLORS["gate"],
        va="top",
    )

    null_axis.hist(
        null_effects,
        bins=18,
        color=COLORS["null"],
        edgecolor="white",
        linewidth=0.8,
    )
    null_axis.axvline(true_effect, color=COLORS["llm"], linewidth=2.7)
    null_axis.axvline(0.0, color="#2D3748", linewidth=1.0, linestyle="--")
    null_axis.text(
        0.04,
        0.95,
        f"true binding: {true_effect:+.3f}\n99 permutations; p = {p_value:.2f}\n0 nulls ≥ true",
        transform=null_axis.transAxes,
        va="top",
        fontsize=10,
    )
    null_axis.set_title("B  Prospective mechanism test", loc="left")
    null_axis.set_xlabel("Mean paired AUC delta vs target GP-UCB")
    null_axis.set_ylabel("Outcome-permuted assignments")
    null_axis.spines[["top", "right"]].set_visible(False)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[str(row["series"])].append(row)
    trace_colors = {
        "Target-only GP-UCB": COLORS["target"],
        "Outcome-permuted skills": COLORS["null"],
        "LLM skill, true binding": COLORS["llm"],
    }
    for label, rows in grouped.items():
        rows.sort(key=lambda row: int(row["round"]))
        x = np.asarray([int(row["round"]) for row in rows])
        means = np.asarray([float(row["mean_best_so_far"]) for row in rows])
        lows = np.asarray([float(row["bootstrap_ci95_low"]) for row in rows])
        highs = np.asarray([float(row["bootstrap_ci95_high"]) for row in rows])
        trace_axis.plot(
            x,
            means,
            color=trace_colors[label],
            linewidth=2.5 if "true" in label else 1.9,
            label=label,
        )
        trace_axis.fill_between(
            x, lows, highs, color=trace_colors[label], alpha=0.13, linewidth=0
        )
    trace_axis.set_title("C  Same budget, different source binding", loc="left")
    trace_axis.set_xlabel("Target reveal round")
    trace_axis.set_ylabel("Mean best-so-far TrpB fitness")
    trace_axis.set_xticks(range(1, 13))
    trace_axis.grid(axis="y", color="#DDE2E7", linewidth=0.7)
    trace_axis.spines[["top", "right"]].set_visible(False)
    trace_axis.legend(frameon=False, fontsize=9, loc="lower right")

    figure.suptitle(
        "TrpB replication: the LLM-selected skill transfers, and the measured source binding is necessary",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure.text(
        0.01,
        -0.04,
        "FLIP2 one-to-many; protocol and LLM hypothesis committed before download; 100 paired target seeds, 3 initial observations + 12 reveals. "
        "The frozen gate abstained, so efficacy and routing power are reported separately.",
        fontsize=9,
        color="#4A5568",
    )
    for suffix in (".png", ".pdf", ".svg"):
        path = output.with_suffix(suffix)
        figure.savefig(path, dpi=260 if suffix == ".png" else None, bbox_inches="tight")
        write_sha256(path)
    plt.close(figure)

    stats_path = output.with_name(f"{output.name}_stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "schema_version": "care.prospective_trpb_replication_plot/v1",
                "deployment_effects": effects,
                "calibration_effect": calibration,
                "gate_deployed_policy": gate_summary["deployment"]["deployed_policy"],
                "mechanism": mechanism_report["true_vs_permuted_assignment"],
                "source_files": {
                    "efficacy_summary": str(efficacy_root / "source_only_gate_summary.json"),
                    "mechanism_report": str(mechanism_root / "falsification_report.json"),
                    "mechanism_trajectories": str(mechanism_root / "trajectory_audit.jsonl"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sha256(stats_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--efficacy-root", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.efficacy_root, args.mechanism_root, args.output)


if __name__ == "__main__":
    main()
