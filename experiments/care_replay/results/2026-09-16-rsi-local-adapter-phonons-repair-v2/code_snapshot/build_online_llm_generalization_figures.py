#!/usr/bin/env python3
"""Build publication-ready figures for the online LLM generalization study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TASK_LABELS = {
    "materials_expt_gap_to_dielectric": "Band gap -> dielectric",
    "materials_phonons_to_bulk_modulus": "Phonons -> bulk modulus",
    "molecular_freesolv_to_lipophilicity": "FreeSolv -> lipophilicity",
    "molecular_lipophilicity_to_freesolv": "Lipophilicity -> FreeSolv",
    "materials_bulk_to_shear_modulus": "Bulk -> shear modulus",
    "molecular_freesolv_to_esol": "FreeSolv -> ESOL",
    "materials_dielectric_to_jdft2d": "Dielectric -> exfoliation",
    "materials_phonons_to_perovskites": "Phonons -> perovskites",
    "materials_expt_gap_to_steels": "Band gap -> steel strength",
    "molecular_freesolv_to_bace": "FreeSolv -> BACE",
    "materials_expt_gap_to_mp_gap": "Band gap -> MP gap",
    "materials_perovskites_to_mp_e_form": "Perovskites -> MP formation",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "online_auc_delta_vs_same_initial_gp",
            "full_auc_delta_vs_frozen_control",
            "llm_decision_authority_rate",
        ):
            row[field] = float(row[field])
    return rows


def build_figure(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, (delta_ax, authority_ax) = plt.subplots(
        1,
        2,
        figsize=(13.6, 7.2),
        gridspec_kw={"width_ratios": [1.7, 1]},
    )

    positions = list(range(len(rows)))
    labels = [
        f"{TASK_LABELS.get(str(row['case']), str(row['case']))}  [{row['protocol_version']}]"
        for row in rows
    ]
    online = [float(row["online_auc_delta_vs_same_initial_gp"]) for row in rows]
    full = [float(row["full_auc_delta_vs_frozen_control"]) for row in rows]
    bar_height = 0.34
    delta_ax.barh(
        [position - bar_height / 2 for position in positions],
        online,
        height=bar_height,
        color="#147D92",
        label="Online LLM effect vs same-initial GP-UCB",
    )
    delta_ax.barh(
        [position + bar_height / 2 for position in positions],
        full,
        height=bar_height,
        color="#D56A3A",
        label="Full system vs frozen source-diverse control",
    )
    delta_ax.axvline(0, color="#333333", linewidth=1)
    delta_ax.set_yticks(positions, labels)
    delta_ax.invert_yaxis()
    delta_ax.set_xlabel("Best-so-far AUC delta (percentage points)")
    delta_ax.set_title("Route-level transfer outcomes", loc="left", pad=12)
    delta_ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    delta_ax.legend(frameon=False, loc="lower right")

    domain_colors = {
        "molecular_property": "#3A7D44",
        "materials_property": "#6B5B95",
    }
    for row in rows:
        authority_ax.scatter(
            float(row["llm_decision_authority_rate"]),
            float(row["online_auc_delta_vs_same_initial_gp"]),
            color=domain_colors[str(row["domain"])],
            edgecolor="white",
            linewidth=0.7,
            s=70,
            alpha=0.95,
        )
    authority_ax.axhline(0, color="#333333", linewidth=1)
    authority_ax.set_xlim(-0.04, 1.04)
    authority_ax.set_xlabel("LLM decision-authority rate")
    authority_ax.set_ylabel("Online LLM AUC delta vs same-initial GP-UCB")
    authority_ax.set_title("More authority did not guarantee gain", loc="left", pad=12)
    authority_ax.grid(color="#DDDDDD", linewidth=0.7)
    authority_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=8,
                label=label,
            )
            for label, color in (
                ("Molecular property", domain_colors["molecular_property"]),
                ("Materials property", domain_colors["materials_property"]),
            )
        ],
        frameon=False,
        loc="lower right",
    )

    figure.suptitle(
        "Claude Opus 5 online-scientist generalization study",
        fontsize=15,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    figure.text(
        0.06,
        0.925,
        "12 real target tasks, 10 online rounds each; v1-v5 are sequential protocol extensions on new targets.",
        color="#4A4A4A",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.9))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "generalization_route_outcomes.png", dpi=240)
    figure.savefig(output_dir / "generalization_route_outcomes.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_figure(read_rows(args.metrics), args.output_dir)


if __name__ == "__main__":
    main()
