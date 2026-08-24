#!/usr/bin/env python3
"""Plot the frozen source-outcome permutation falsification study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PAIR_LABELS = {
    "reaction_chemlex_to_buchwald_hartwig": "ChemLex -> Buchwald-Hartwig",
    "materials_dielectric_to_expt_gap": "Dielectric -> experimental gap",
    "molecular_lipophilicity_to_freesolv": "Lipophilicity -> FreeSolv",
}

PAIR_COLORS = {
    "reaction_chemlex_to_buchwald_hartwig": "#1F5A7A",
    "materials_dielectric_to_expt_gap": "#B75D2A",
    "molecular_lipophilicity_to_freesolv": "#2B7A55",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def load_rows(report_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    rows = []
    for pair in report["pairs"]:
        pair_id = str(pair["pair_id"])
        target = pair["true_outcomes_minus_target_only"]["best_so_far_auc"]
        permuted = pair["true_outcomes_minus_mean_permutation"]["best_so_far_auc"]
        rows.append({
            "pair_id": pair_id,
            "pair": PAIR_LABELS.get(pair_id, pair_id),
            "target_seed_count": int(pair["target_seed_count"]),
            "permutation_count": int(pair["permutation_count"]),
            "true_minus_target_mean": float(target["mean"]),
            "true_minus_target_ci_low": float(target["normal_95ci_low"]),
            "true_minus_target_ci_high": float(target["normal_95ci_high"]),
            "true_minus_permuted_mean": float(permuted["mean"]),
            "true_minus_permuted_ci_low": float(permuted["normal_95ci_low"]),
            "true_minus_permuted_ci_high": float(permuted["normal_95ci_high"]),
            "empirical_p": float(
                pair["randomization_test"]["best_so_far_auc"]
                ["one_sided_empirical_p"]
            ),
        })
    return rows, protocol


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_sha256(path)


def configure_plot() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#FAFAF8",
        "axes.facecolor": "#FAFAF8",
        "savefig.facecolor": "#FAFAF8",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def errorbar(ax: Any, rows: list[dict[str, Any]], prefix: str) -> None:
    for index, row in enumerate(rows):
        value = row[f"{prefix}_mean"]
        low = row[f"{prefix}_ci_low"]
        high = row[f"{prefix}_ci_high"]
        color = PAIR_COLORS.get(row["pair_id"], "#4A5568")
        ax.errorbar(
            value,
            index,
            xerr=[[value - low], [high - value]],
            fmt="o",
            markersize=7,
            capsize=4,
            color=color,
            ecolor=color,
            elinewidth=2,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )
        label_on_left = high > 35.0
        ax.text(
            low - 0.8 if label_on_left else high + 0.8,
            index - 0.18,
            f"{value:+.2f}  [{low:+.2f}, {high:+.2f}]",
            color="#30353B",
            fontsize=8.5,
            va="center",
            ha="right" if label_on_left else "left",
        )
    ax.axvline(0, color="#73777C", linewidth=1, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#E2E3E1", linewidth=0.8, zorder=0)
    ax.set_ylim(len(rows) - 0.45, -0.65)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)


def plot(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    output: Path,
    *,
    presentation: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    configure_plot()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 7.05) if presentation else (12.8, 4.8),
        sharey=True,
    )
    errorbar(axes[0], rows, "true_minus_target")
    errorbar(axes[1], rows, "true_minus_permuted")

    axes[0].set_yticks(range(len(rows)), [row["pair"] for row in rows])
    axes[0].set_title("A  True source outcomes vs target-only", loc="left", fontweight="bold")
    axes[1].set_title("B  True outcomes vs matched permutations", loc="left", fontweight="bold")
    axes[0].set_xlabel("Paired best-so-far AUC difference")
    axes[1].set_xlabel("Paired best-so-far AUC difference")

    for index, row in enumerate(rows):
        p_value = row["empirical_p"]
        axes[1].text(
            0.985,
            index + 0.20,
            f"p={p_value:.2f}",
            transform=axes[1].get_yaxis_transform(),
            ha="right",
            va="center",
            color="#5A6066",
            fontsize=9,
            bbox={"facecolor": "#FAFAF8", "edgecolor": "none", "pad": 0.6},
        )

    if presentation:
        figure.text(
            0.24,
            0.035,
            (
                f"{protocol['executed_target_seed_count']} target seeds · "
                f"{protocol['executed_permutation_count']} matched permutations per route · "
                "paired 95% CI · one-sided empirical p"
            ),
            ha="left",
            va="bottom",
            fontsize=9,
            color="#5A6066",
        )
    else:
        figure.suptitle(
            "Do measured source outcomes carry information beyond schema and initialization?",
            x=0.06,
            y=1.02,
            ha="left",
            fontsize=15,
            fontweight="bold",
            color="#202428",
        )
        figure.text(
            0.06,
            -0.015,
            (
                f"Frozen retrospective replay: {protocol['executed_target_seed_count']} target seeds; "
                f"{protocol['executed_permutation_count']} outcome permutations per route. "
                "Points are paired means; bars are normal 95% CIs. p is a one-sided empirical "
                "randomization test."
            ),
            ha="left",
            va="top",
            fontsize=9,
            color="#5A6066",
        )
    figure.subplots_adjust(
        left=0.24,
        right=0.985,
        top=0.92 if presentation else 0.82,
        bottom=0.14 if presentation else 0.22,
        wspace=0.22,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 320}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output.with_suffix(suffix)
        figure.savefig(path, bbox_inches=None if presentation else "tight", **kwargs)
        write_sha256(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Render to the deck's 16:9 evidence-frame aspect ratio.",
    )
    args = parser.parse_args()
    rows, protocol = load_rows(args.report)
    if not rows:
        raise ValueError("The falsification report contains no source-target pairs")
    expected_seed_count = int(protocol["target_seed_count"])
    expected_permutation_count = len(protocol["permutation_seeds"])
    if any(row["target_seed_count"] != expected_seed_count for row in rows):
        raise ValueError("The report is incomplete: not every route has all target seeds")
    if any(row["permutation_count"] != expected_permutation_count for row in rows):
        raise ValueError("The report is incomplete: not every route has all permutations")
    plot(rows, protocol, args.output, presentation=args.presentation)
    write_plot_data(args.output.with_name(f"{args.output.name}_data.csv"), rows)


if __name__ == "__main__":
    main()
