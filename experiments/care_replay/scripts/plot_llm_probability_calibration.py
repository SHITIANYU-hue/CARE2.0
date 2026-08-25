#!/usr/bin/env python3
"""Plot the archived LLM probability-calibration audit from versioned results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROUTE_LABELS = {
    "molecular_freesolv_to_esol": "FreeSolv -> ESOL",
    "molecular_freesolv_to_lipophilicity": "FreeSolv -> Lipophilicity",
    "materials_expt_gap_to_dielectric": "Band gap -> Dielectric",
    "materials_phonons_to_bulk_modulus": "Phonons -> Bulk modulus",
    "materials_dielectric_to_jdft2d": "Dielectric -> JDFT2D",
    "materials_expt_gap_to_mp_gap": "Band gap -> MP gap",
    "baumgartner_aniline_to_phenethylamine_alphos": "Aniline -> phenethylamine (AlPhos)",
    "baumgartner_aniline_to_benzamide_tbuxphos": "Aniline -> benzamide (tBuXPhos)",
    "baumgartner_aniline_to_phenethylamine_tbubrettphos": "Aniline -> phenethylamine (BrettPhos)",
    "baumgartner_benzamide_tbubrettphos_to_alphos": "Benzamide: BrettPhos -> AlPhos",
    "reizman_cases_123_to_case4": "Suzuki cases 1-3 -> case 4",
    "molecular_lipophilicity_to_freesolv": "Lipophilicity -> FreeSolv",
    "materials_bulk_to_shear_modulus": "Bulk -> Shear modulus",
    "materials_phonons_to_perovskites": "Phonons -> Perovskites",
    "materials_expt_gap_to_steels": "Band gap -> Steels",
    "molecular_freesolv_to_bace": "FreeSolv -> BACE",
    "materials_perovskites_to_mp_e_form": "Perovskites -> Formation energy",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def plot(report: dict[str, Any], output: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#FAFAF8",
        "axes.facecolor": "#FAFAF8",
        "savefig.facecolor": "#FAFAF8",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    pooled = report["pooled"]
    comparison = report["route_level_comparison"]
    routes = sorted(report["routes"], key=lambda row: float(row["llm_minus_gp_brier"]))
    figure = plt.figure(figsize=(12.8, 7.2))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[0.95, 1.55],
        left=0.07,
        right=0.98,
        top=0.82,
        bottom=0.12,
        wspace=0.48,
    )
    reliability = figure.add_subplot(grid[0, 0])
    route_axis = figure.add_subplot(grid[0, 1])

    reliability.plot([0, 1], [0, 1], color="#7A7F84", linestyle="--", linewidth=1.2)
    for label, key, color, marker in (
        ("LLM final decision", "llm", "#D95F02", "o"),
        ("GP, same candidate", "gp_same_candidate", "#1B6CA8", "s"),
    ):
        bins = [row for row in pooled[key]["bins"] if row["count"]]
        x = [float(row["mean_probability"]) for row in bins]
        y = [float(row["observed_improvement_rate"]) for row in bins]
        sizes = [40 + 2 * int(row["count"]) for row in bins]
        reliability.plot(x, y, color=color, linewidth=1.6, alpha=0.85)
        reliability.scatter(
            x,
            y,
            s=sizes,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    reliability.set_xlim(-0.02, 1.02)
    reliability.set_ylim(-0.02, 1.02)
    reliability.set_xlabel("Predicted probability of improving current best")
    reliability.set_ylabel("Observed improvement rate")
    reliability.set_title("A  Reliability diagram", loc="left", fontweight="bold")
    reliability.grid(color="#E1E3E2", linewidth=0.8)
    reliability.legend(frameon=False, loc="upper left", markerscale=0.7)
    for spine in reliability.spines.values():
        spine.set_visible(False)

    y = np.arange(len(routes))
    values = [float(row["llm_minus_gp_brier"]) for row in routes]
    colors = [
        "#C95D4B" if value > 0 else "#2A9D8F" if value < 0 else "#7A7F84"
        for value in values
    ]
    route_axis.barh(y, values, color=colors, height=0.64, alpha=0.9)
    route_axis.axvline(0.0, color="#24282C", linewidth=1.1)
    route_axis.set_yticks(
        y,
        [ROUTE_LABELS.get(str(row["case_id"]), str(row["case_id"])) for row in routes],
    )
    route_axis.set_xlabel("Brier difference: LLM - GP (negative favors LLM)")
    route_axis.set_title("B  Route-level calibration comparison", loc="left", fontweight="bold")
    route_axis.grid(axis="x", color="#E1E3E2", linewidth=0.8)
    route_axis.tick_params(axis="y", length=0, labelsize=6.9, pad=3)
    for spine in route_axis.spines.values():
        spine.set_visible(False)

    llm = pooled["llm"]
    gp = pooled["gp_same_candidate"]
    figure.suptitle(
        "Can the LLM estimate whether its next experiment will improve the current best?",
        x=0.04,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#202428",
    )
    figure.text(
        0.04,
        0.885,
        (
            f"{pooled['decision_count']} archived decisions / {pooled['route_count']} routes; "
            f"observed improvement rate {llm['observed_improvement_rate']:.3f}. "
            f"Brier: LLM {llm['brier_score']:.3f}, same-candidate GP {gp['brier_score']:.3f}. "
            f"Equal-route difference {comparison['equal_route_mean_llm_minus_gp_brier']:+.3f} "
            f"[{comparison['route_bootstrap_95ci_low']:+.3f}, {comparison['route_bootstrap_95ci_high']:+.3f}]."
        ),
        ha="left",
        va="center",
        fontsize=10,
        color="#50565C",
    )
    figure.text(
        0.5,
        0.035,
        (
            "Real archived decisions; no generated values. Probabilities are scored on the executed candidate. "
            "Retrospective calibration evidence, not prospective gate validation."
        ),
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#5A6066",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 320}), (".pdf", {}), (".svg", {})):
        path = output.with_suffix(suffix)
        figure.savefig(path, **kwargs)
        if suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
        write_sha256(path)
    plt.close(figure)


def write_plot_data(path: Path, report: dict[str, Any]) -> None:
    fields = [
        "series",
        "bin_index",
        "bin_low",
        "bin_high",
        "count",
        "mean_probability",
        "observed_improvement_rate",
    ]
    rows: list[dict[str, Any]] = []
    for series, key in (("llm", "llm"), ("gp_same_candidate", "gp_same_candidate")):
        for row in report["pooled"][key]["bins"]:
            rows.append({"series": series, **row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plot(report, args.output)
    write_plot_data(args.output.with_name(args.output.name + "_data.csv"), report)


if __name__ == "__main__":
    main()
