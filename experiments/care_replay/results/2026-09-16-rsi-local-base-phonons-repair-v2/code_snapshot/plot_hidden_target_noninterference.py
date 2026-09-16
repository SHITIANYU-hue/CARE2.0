#!/usr/bin/env python3
"""Plot the real hidden-target non-interference audit results."""

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
    "baumgartner_aniline_to_phenethylamine_alphos": "CN aniline -> phenethylamine (AlPhos)",
    "baumgartner_aniline_to_benzamide_tbuxphos": "CN aniline -> benzamide (tBuXPhos)",
    "baumgartner_aniline_to_phenethylamine_tbubrettphos": "CN aniline -> phenethylamine (BrettPhos)",
    "baumgartner_benzamide_tbubrettphos_to_alphos": "CN benzamide: BrettPhos -> AlPhos",
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


def configure_plot() -> None:
    import matplotlib as mpl

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


def plot(report: dict[str, Any], output: Path, *, presentation: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    configure_plot()
    routes = report["routes"]
    states = report["states"]
    case_ids = [str(route["case_id"]) for route in routes]
    route_index = {case_id: index for index, case_id in enumerate(case_ids)}
    rounds = max(int(state["round_index"]) for state in states) + 1
    matrix = np.full((len(routes), rounds), np.nan, dtype=float)
    pass_matrix = np.zeros((len(routes), rounds), dtype=bool)
    for state in states:
        row = route_index[str(state["case_id"])]
        column = int(state["round_index"])
        matrix[row, column] = max(1, int(state["changed_hidden_labels"]))
        pass_matrix[row, column] = bool(state["pass"])

    figure = plt.figure(figsize=(12.8, 7.05) if presentation else (12.8, 7.8))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[2.25, 1.0],
        left=0.235,
        right=0.985,
        top=0.86 if presentation else 0.87,
        bottom=0.16 if presentation else 0.15,
        wspace=0.16,
    )
    heat = figure.add_subplot(grid[0, 0])
    bars = figure.add_subplot(grid[0, 1], sharey=heat)

    image = heat.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=max(1, float(np.nanmin(matrix))), vmax=float(np.nanmax(matrix))),
        interpolation="nearest",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not pass_matrix[row, column]:
                heat.text(
                    column,
                    row,
                    "x",
                    color="#B3261E",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
    heat.set_xticks(range(rounds), [str(index + 1) for index in range(rounds)])
    heat.set_yticks(
        range(len(routes)),
        [ROUTE_LABELS.get(case_id, case_id) for case_id in case_ids],
    )
    heat.set_xlabel("Online decision round")
    heat.set_title("A  Hidden-label intervention coverage", loc="left", fontweight="bold")
    heat.tick_params(length=0)
    for spine in heat.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=heat, fraction=0.028, pad=0.02)
    colorbar.set_label("Unrevealed labels permuted per state")
    colorbar.outline.set_visible(False)

    totals = np.array([int(route["changed_hidden_labels"]) for route in routes])
    y = np.arange(len(routes))
    bars.barh(y, totals, height=0.62, color="#2E6F86", alpha=0.88)
    bars.set_xscale("log")
    bars.set_xlabel("Total hidden-label interventions")
    bars.set_title("B  Route-level audit", loc="left", fontweight="bold")
    bars.grid(axis="x", color="#E1E3E2", linewidth=0.8)
    bars.tick_params(axis="y", left=False, labelleft=False)
    for row, route in enumerate(routes):
        bars.text(
            float(totals[row]) * 1.08,
            row,
            f"{route['passing_states']}/{route['decision_states']} pass",
            va="center",
            ha="left",
            fontsize=8.4,
            color="#30353B",
        )
    maximum = float(np.max(totals))
    bars.set_xlim(max(1.0, float(np.min(totals)) * 0.7), maximum * 7.5)
    for spine in bars.spines.values():
        spine.set_visible(False)

    aggregate = report["aggregate"]
    figure.suptitle(
        "Unrevealed target labels do not change the production decision state",
        x=0.035,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#202428",
    )
    figure.text(
        0.035,
        0.91,
        (
            f"{aggregate['passing_states']}/{aggregate['decision_states']} states across "
            f"{aggregate['passing_routes']} routes passed exact source-prior, menu, and prompt invariance; "
            f"{aggregate['changed_hidden_labels']:,} unrevealed labels were permuted."
        ),
        ha="left",
        va="center",
        fontsize=10,
        color="#52585E",
    )
    figure.text(
        0.61,
        0.055 if presentation else 0.045,
        (
            "Real archived states; no generated values. Revealed outcomes and all public candidate attributes remain fixed.\n"
            "A pass supports implementation non-interference, not transfer efficacy."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#5A6066",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 320}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = output.with_suffix(suffix)
        figure.savefig(path, **kwargs)
        write_sha256(path)
    plt.close(figure)


def write_plot_data(path: Path, states: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "round_index",
        "observed_count",
        "unrevealed_count",
        "changed_hidden_labels",
        "public_candidate_fingerprint_equal",
        "source_prior_equal",
        "candidate_menu_invariant",
        "menu_diagnostics_invariant",
        "llm_prompt_invariant",
        "saved_prompt_declared_hash_valid",
        "pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in states)
    write_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--presentation", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not report["aggregate"]["all_states_pass"]:
        raise ValueError("Non-interference audit is incomplete or contains failures")
    plot(report, args.output, presentation=args.presentation)
    write_plot_data(
        args.output.with_name(f"{args.output.name}_data.csv"),
        report["states"],
    )


if __name__ == "__main__":
    main()
