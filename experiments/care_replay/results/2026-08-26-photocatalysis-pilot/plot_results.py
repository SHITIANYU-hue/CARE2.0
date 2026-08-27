#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPLAY_ROOT = HERE.parents[1]
SCRIPTS = REPLAY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


COLORS = {
    "random": "#A8ADB3",
    "public_incumbent": "#767D85",
    "mixed_kernel_gp_ucb": "#2563A6",
    "mixed_kernel_gp_ei": "#5B8DB8",
    "knn_ucb": "#B66A3C",
    "llm": "#18866B",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats(values: list[float]) -> tuple[float, float]:
    average = mean(values)
    ci = 1.96 * pstdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return average, ci


def write_rule_associations(adapter: replay.DatasetAdapter) -> None:
    record = json.loads((HERE / "llm_target_only_skills_opus5_v2.json").read_text())
    catalog = semantic.semantic_field_catalog(adapter)
    skills = semantic.normalize_skills(
        {"skills": record["normalized_skills"]}, catalog, max_skills=100
    )
    rows: list[dict[str, object]] = []
    overall = mean(candidate.objective_value for candidate in adapter.candidates)
    for skill in skills:
        for rule in skill.rules:
            matched = [
                candidate.objective_value
                for candidate in adapter.candidates
                if semantic.rule_matches(rule, candidate)
            ]
            rows.append({
                "skill_id": skill.skill_id,
                "rule_id": rule.rule_id,
                "conditions": json.dumps(dict(rule.conditions), sort_keys=True),
                "llm_weight": rule.weight,
                "matched_candidates": len(matched),
                "matched_mean_her_umol_h": round(mean(matched), 6) if matched else "",
                "delta_vs_pool_mean_umol_h": round(mean(matched) - overall, 6) if matched else "",
            })
    with (HERE / "rule_associations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    baseline_rows = read_csv(HERE / "baselines" / "metrics.csv")
    calibrated = json.loads(
        (HERE / "calibrated_warmstart" / "summary.json").read_text(encoding="utf-8")
    )
    adapter = replay.real_photocatalytic_hydrogen_evolution_adapter()
    write_rule_associations(adapter)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.1), constrained_layout=True)

    modes = [
        "random",
        "public_incumbent",
        "mixed_kernel_gp_ucb",
        "mixed_kernel_gp_ei",
        "knn_ucb",
    ]
    labels = ["Random", "Incumbent", "GP-UCB", "GP-EI", "kNN-UCB"]
    grouped = {
        mode: [float(row["final_best"]) for row in baseline_rows if row["mode"] == mode]
        for mode in modes
    }
    means, cis = zip(*(stats(grouped[mode]) for mode in modes))
    x = np.arange(len(modes))
    axes[0].bar(
        x,
        means,
        yerr=cis,
        capsize=4,
        color=[COLORS[mode] for mode in modes],
        width=0.72,
    )
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("Best HER after 20 reveals (µmol h⁻¹)")
    axes[0].set_title("A. Target-only baselines (30 seeds)", loc="left", fontweight="bold")
    axes[0].axhline(adapter.candidates and max(c.objective_value for c in adapter.candidates), color="#30363D", linestyle=":", linewidth=1.2)
    axes[0].text(4.45, max(c.objective_value for c in adapter.candidates) + 0.35, "finite-pool maximum", ha="right", fontsize=8, color="#30363D")
    axes[0].set_ylim(0, 36)

    vs_gp = calibrated["strategy_router_pairwise"]["gp_ucb"]
    metrics = ["final_best", "best_so_far_auc"]
    effect_labels = ["Final best", "Search AUC"]
    effect_means = [float(vs_gp[metric]["mean"]) for metric in metrics]
    effect_low = [float(vs_gp[metric]["normal_95ci_low"]) for metric in metrics]
    effect_high = [float(vs_gp[metric]["normal_95ci_high"]) for metric in metrics]
    y = np.arange(len(metrics))
    axes[1].errorbar(
        effect_means,
        y,
        xerr=[
            [value - low for value, low in zip(effect_means, effect_low)],
            [high - value for value, high in zip(effect_means, effect_high)],
        ],
        fmt="o",
        markersize=8,
        color=COLORS["llm"],
        ecolor=COLORS["llm"],
        capsize=5,
        linewidth=2,
    )
    axes[1].axvline(0.0, color="#30363D", linewidth=1.2)
    axes[1].set_yticks(y, effect_labels)
    axes[1].invert_yaxis()
    axes[1].set_ylim(1.5, -0.5)
    axes[1].set_xlabel("Paired delta vs GP-UCB (µmol h⁻¹)")
    axes[1].set_title("B. Opus skill warm-start (50 held-out seeds)", loc="left", fontweight="bold")
    axes[1].text(
        0.98,
        0.50,
        "Positive means; 95% CIs cross zero",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=9,
        color="#4D545B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )

    p10_order = ["low", "medium", "high", "very_high"]
    cysteine_order = ["low", "medium", "high", "very_high"]
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for candidate in adapter.candidates:
        cells[(
            str(candidate.metadata["p10_mix1_bin"]),
            str(candidate.metadata["l_cysteine_100gl_bin"]),
        )].append(candidate.objective_value)
    matrix = np.asarray([
        [mean(cells[(p10, cysteine)]) for cysteine in cysteine_order]
        for p10 in p10_order
    ])
    image = axes[2].imshow(matrix, cmap="viridis", vmin=0.0, vmax=16.0, aspect="auto")
    for row_index, p10 in enumerate(p10_order):
        for column_index, cysteine in enumerate(cysteine_order):
            value = matrix[row_index, column_index]
            count = len(cells[(p10, cysteine)])
            axes[2].text(
                column_index,
                row_index,
                f"{value:.1f}\nn={count}",
                ha="center",
                va="center",
                color="white" if value < 7.0 or value > 13.0 else "#17212B",
                fontsize=8,
                fontweight="bold",
            )
    axes[2].set_xticks(range(4), cysteine_order, rotation=24, ha="right")
    axes[2].set_yticks(range(4), p10_order)
    axes[2].set_xlabel("L-cysteine amount bin")
    axes[2].set_ylabel("P10-MIX1 amount bin")
    axes[2].set_title("C. Measured formulation landscape", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axes[2], shrink=0.82)
    colorbar.set_label("Mean HER (µmol h⁻¹)")

    figure.savefig(HERE / "photocatalysis_pilot_results.png", dpi=300)
    figure.savefig(HERE / "photocatalysis_pilot_results.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
