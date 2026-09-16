#!/usr/bin/env python3
"""Synthesize the three preregistered external-family gate outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = (
    RESULTS
    / "2026-08-25-prospective-external-family-triad-v1"
    / "prospective_external_family_triad"
)
RHOMAX_ROOT = RESULTS / "2026-08-25-flip2-rhomax-prospective-gate-v1"
IRED_ROOT = RESULTS / "2026-08-25-flip2-ired-prospective-positive-transfer-v1"
TRPB_EFFICACY_ROOT = (
    RESULTS / "2026-08-25-flip2-trpb-prospective-positive-transfer-v1"
)
TRPB_MECHANISM_ROOT = (
    RESULTS / "2026-08-25-flip2-trpb-source-outcome-confirmation-v1"
)


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256_path(path)}  {path.name}\n", encoding="utf-8"
    )


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    write_sha256(path)


def build_rows() -> list[dict[str, Any]]:
    rhomax_summary = read_json(RHOMAX_ROOT / "source_only_gate_summary.json")
    rhomax_effects = read_csv(RHOMAX_ROOT / "route_effects.csv")
    rhomax_method = str(rhomax_summary["gate"]["ungated_source_only_selection"])
    rhomax_effect = next(
        row
        for row in rhomax_effects
        if row["stage"] == "external_deployment" and row["method"] == rhomax_method
    )

    ired_summary = read_json(IRED_ROOT / "source_only_gate_summary.json")
    ired_effects = read_csv(
        IRED_ROOT / "prospective_ired_positive_transfer_effects.csv"
    )
    ired_effect = next(
        row
        for row in ired_effects
        if row["route_id"] == "official_test"
        and row["method"] == "source_additive_mutation_prior"
    )

    trpb_summary = read_json(
        TRPB_EFFICACY_ROOT / "source_only_gate_summary.json"
    )
    trpb_effects = read_csv(
        TRPB_MECHANISM_ROOT / "prospective_trpb_replication_effects.csv"
    )
    trpb_effect = next(
        row
        for row in trpb_effects
        if row["method"] == "source_additive_mutation_prior"
    )

    rows = [
        {
            "family": "Rhomax",
            "task": "Rhodopsin peak wavelength",
            "candidate_policy": "Fixed skill prior",
            "candidate_mean_delta_auc": float(rhomax_effect["mean_auc_delta"]),
            "candidate_ci95_low": float(rhomax_effect["ci95_low"]),
            "candidate_ci95_high": float(rhomax_effect["ci95_high"]),
            "candidate_ci_method": "paired normal",
            "gate_decision": "HOLD",
            "deployed_policy": str(rhomax_summary["gate"]["deployed_policy"]),
            "deployed_mean_delta_auc": float(
                rhomax_summary["deployment"]["deployed_auc_delta_vs_target_gp"]
            ),
            "router_outcome": "Correct reject",
            "evidence_role": "Prospective safety",
            "deployment_seed_count": int(
                rhomax_summary["deployment"]["heldout_seed_count"]
            ),
        },
        {
            "family": "IRED",
            "task": "Imine-reductase activity",
            "candidate_policy": "LLM additive skill",
            "candidate_mean_delta_auc": float(ired_effect["mean_delta_auc"]),
            "candidate_ci95_low": float(ired_effect["bootstrap_ci95_low"]),
            "candidate_ci95_high": float(ired_effect["bootstrap_ci95_high"]),
            "candidate_ci_method": "paired bootstrap",
            "gate_decision": "TRANSFER",
            "deployed_policy": str(ired_summary["gate"]["deployed_policy"]),
            "deployed_mean_delta_auc": float(
                ired_summary["deployment"]["deployed_auc_delta_vs_target_gp"]
            ),
            "router_outcome": "Correct deploy",
            "evidence_role": "Prospective efficacy",
            "deployment_seed_count": int(
                ired_summary["deployment"]["heldout_seed_count"]
            ),
        },
        {
            "family": "TrpB",
            "task": "Tryptophan-synthase growth fitness",
            "candidate_policy": "LLM additive skill",
            "candidate_mean_delta_auc": float(trpb_effect["mean_delta_auc"]),
            "candidate_ci95_low": float(trpb_effect["ci95_low"]),
            "candidate_ci95_high": float(trpb_effect["ci95_high"]),
            "candidate_ci_method": "paired normal",
            "gate_decision": "HOLD",
            "deployed_policy": str(trpb_summary["gate"]["deployed_policy"]),
            "deployed_mean_delta_auc": float(
                trpb_summary["deployment"]["deployed_auc_delta_vs_target_gp"]
            ),
            "router_outcome": "False-negative hold",
            "evidence_role": "Prospective efficacy + attribution",
            "deployment_seed_count": int(
                trpb_summary["deployment"]["heldout_seed_count"]
            ),
        },
    ]
    return rows


def plot(output_stem: Path) -> None:
    rows = build_rows()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    write_csv(csv_path, rows)

    summary = {
        "schema_version": "care.prospective_external_family_triad/v1",
        "analysis_status": "descriptive_synthesis_of_preregistered_results",
        "family_count": len(rows),
        "correct_router_decisions": 2,
        "false_negative_holds": 1,
        "source_files": {
            "rhomax": str(RHOMAX_ROOT / "source_only_gate_summary.json"),
            "ired": str(IRED_ROOT / "source_only_gate_summary.json"),
            "trpb_efficacy": str(
                TRPB_EFFICACY_ROOT / "source_only_gate_summary.json"
            ),
            "trpb_attribution": str(
                TRPB_MECHANISM_ROOT / "falsification_report.json"
            ),
        },
        "claim_boundary": (
            "This is a descriptive synthesis of three preregistered external "
            "protein-family replays. Three families are too few to estimate a "
            "population generalization rate, and no result is a wet-lab discovery."
        ),
    }
    stats_path = output_stem.with_name(f"{output_stem.name}_stats.json")
    stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_sha256(stats_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": "#263640",
            "axes.labelcolor": "#263640",
            "xtick.color": "#53616B",
            "ytick.color": "#263640",
        }
    )
    figure = plt.figure(figsize=(13.8, 6.7), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.42, 1.0))
    figure.subplots_adjust(
        left=0.10, right=0.97, bottom=0.16, top=0.76, wspace=0.28
    )
    effect_axis = figure.add_subplot(grid[0, 0])
    decision_axis = figure.add_subplot(grid[0, 1])

    y_positions = np.arange(len(rows))[::-1]
    colors = {
        "Rhomax": "#B34A42",
        "IRED": "#137A68",
        "TrpB": "#0E6578",
    }
    for y, row in zip(y_positions, rows):
        mean = float(row["candidate_mean_delta_auc"])
        low = float(row["candidate_ci95_low"])
        high = float(row["candidate_ci95_high"])
        effect_axis.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            color=colors[str(row["family"])],
            elinewidth=2.5,
            capsize=4.0,
            zorder=3,
        )
        effect_axis.scatter(
            [float(row["deployed_mean_delta_auc"])],
            [y - 0.20],
            marker="D",
            s=54,
            color="#27343C",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        effect_axis.text(
            high + 0.06,
            y,
            f"{mean:+.3f}",
            va="center",
            fontsize=9.5,
            color="#263640",
        )
    effect_axis.axvline(0.0, color="#27343C", linestyle="--", linewidth=1.1)
    effect_axis.set_yticks(
        y_positions,
        [
            "Rhomax\nfixed prior",
            "IRED\nLLM skill",
            "TrpB\nLLM skill",
        ],
    )
    effect_axis.set_xlim(-2.25, 1.35)
    effect_axis.set_ylim(-0.65, 2.45)
    effect_axis.set_xlabel("Paired best-so-far AUC effect vs target-only GP-UCB")
    effect_axis.set_title(
        "A. Candidate efficacy and complete-router outcome",
        loc="left",
        fontweight="bold",
        pad=20,
    )
    effect_axis.grid(axis="x", color="#D9E0E4", linewidth=0.75)
    effect_axis.spines[["top", "right", "left"]].set_visible(False)
    effect_axis.tick_params(axis="y", length=0)
    effect_axis.scatter(
        [-2.10], [2.30], marker="o", s=58, color="#137A68", zorder=5
    )
    effect_axis.text(
        -2.02, 2.30, "Candidate effect (95% CI)", va="center", fontsize=9
    )
    effect_axis.scatter(
        [-0.92], [2.30], marker="D", s=48, color="#27343C", zorder=5
    )
    effect_axis.text(
        -0.84, 2.30, "Deployed router effect", va="center", fontsize=9
    )

    decision_axis.set_xlim(0.0, 1.0)
    decision_axis.set_ylim(0.0, 1.0)
    decision_axis.axis("off")
    decision_axis.set_title(
        "B. What the frozen router did",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    row_y = [0.72, 0.43, 0.14]
    for y, row in zip(row_y, rows):
        family = str(row["family"])
        decision = str(row["gate_decision"])
        outcome = str(row["router_outcome"])
        decision_color = "#137A68" if outcome != "False-negative hold" else "#B34A42"
        decision_axis.text(
            0.0,
            y + 0.09,
            family,
            fontsize=13,
            fontweight="bold",
            color=colors[family],
        )
        decision_axis.text(
            0.23,
            y + 0.09,
            f"{decision}  |  {outcome}",
            fontsize=12,
            fontweight="bold",
            color=decision_color,
        )
        decision_axis.text(
            0.0,
            y - 0.01,
            str(row["evidence_role"]),
            fontsize=9.5,
            color="#53616B",
        )
        decision_axis.plot(
            [0.0, 0.98], [y - 0.10, y - 0.10], color="#D9E0E4", linewidth=0.8
        )
    decision_axis.text(
        0.0,
        0.0,
        "Lesson: routing must condition on deployment support,\nnot only on a low-overlap calibration replay.",
        fontsize=12.5,
        fontweight="bold",
        color="#263640",
    )

    figure.suptitle(
        "Three preregistered external families separate skill efficacy from routing quality",
        x=0.10,
        y=0.96,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#1E2B33",
    )
    figure.text(
        0.10,
        0.885,
        "Rhomax: harmful transfer correctly rejected. IRED: useful LLM skill deployed. TrpB: useful skill missed by the gate.",
        ha="left",
        fontsize=11.2,
        color="#53616B",
    )
    figure.text(
        0.10,
        0.045,
        "100 paired deployment seeds per family. IRED uses its preregistered bootstrap CI; Rhomax and TrpB use paired normal CIs. Descriptive only; n=3 is not a population estimate.",
        ha="left",
        fontsize=9,
        color="#596773",
    )

    for suffix in (".png", ".pdf", ".svg"):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        if suffix == ".svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        write_sha256(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.output_stem)


if __name__ == "__main__":
    main()
