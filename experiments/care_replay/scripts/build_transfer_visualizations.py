#!/usr/bin/env python3
"""Build publication-ready CARE 2.0 transfer visualizations.

The figures are generated from the frozen source-outcome replay archive. They
keep deployed transfer, rejected negative transfer, and inconclusive fallback
visually separate instead of turning every fallback into a zero gain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


PAIR_LABELS = {
    ("real_chemlex_acidamine", "real_buchwald_hartwig"): "ChemLex -> BH",
    ("real_matbench_dielectric", "real_matbench_expt_gap"): "Dielectric -> Gap",
    ("real_matbench_expt_gap", "real_matbench_dielectric"): "Gap -> Dielectric",
    ("real_matbench_phonons", "real_matbench_dielectric"): "Phonons -> Dielectric",
    ("real_moleculenet_esol", "real_moleculenet_lipophilicity"): "ESOL -> Lipo",
    ("real_moleculenet_freesolv", "real_moleculenet_lipophilicity"): "FreeSolv -> Lipo",
    ("real_moleculenet_lipophilicity", "real_moleculenet_freesolv"): "Lipo -> FreeSolv",
}

DATASET_LABELS = {
    "real_chemlex_acidamine": "ChemLex",
    "real_buchwald_hartwig": "BH",
    "real_matbench_dielectric": "Dielectric",
    "real_matbench_expt_gap": "Expt gap",
    "real_matbench_phonons": "Phonons",
    "real_moleculenet_esol": "ESOL",
    "real_moleculenet_freesolv": "FreeSolv",
    "real_moleculenet_lipophilicity": "Lipophilicity",
}

DOMAIN_BY_DATASET = {
    "real_chemlex_acidamine": "Reaction",
    "real_buchwald_hartwig": "Reaction",
    "real_matbench_dielectric": "Materials",
    "real_matbench_expt_gap": "Materials",
    "real_matbench_phonons": "Materials",
    "real_moleculenet_esol": "Molecular",
    "real_moleculenet_freesolv": "Molecular",
    "real_moleculenet_lipophilicity": "Molecular",
}

DOMAIN_COLORS = {
    "Reaction": "#2B6CB0",
    "Materials": "#C05621",
    "Molecular": "#2F855A",
}


def pair_label(source: str, target: str) -> str:
    return PAIR_LABELS.get((source, target), f"{source} -> {target}")


def short_field(field: str) -> str:
    return (
        field.replace("_flag", "")
        .replace("_bin", "")
        .replace("_", " ")
        .title()
    )


def load_records(root: Path) -> list[dict[str, Any]]:
    headline_path = root / "headline_results.csv"
    with headline_path.open(newline="", encoding="utf-8") as handle:
        headline = list(csv.DictReader(handle))

    summary_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (root / "raw_summaries").glob("*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary_by_pair[(summary["source_dataset"], summary["target_dataset"])] = summary

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    negative_by_pair = {
        (item["source"], item["target"]): item
        for item in manifest.get("negative_results", [])
    }
    records: list[dict[str, Any]] = []
    for row in headline:
        source = row["source"] if "source" in row else row["source_dataset"]
        target = row["target"] if "target" in row else row["target_dataset"]
        pair = (source, target)
        summary = summary_by_pair[pair]
        selected = str(row.get("selected_skill", "")) == "source_outcome_router"
        negative = negative_by_pair.get(pair)
        if selected:
            status = "deployed_positive"
            raw_composite = float(row["delta_final_best"]) + float(row["delta_auc"])
            raw_low = float(row["final_ci_low"]) + float(row["auc_ci_low"])
            raw_high = float(row["final_ci_high"]) + float(row["auc_ci_high"])
        elif negative and "negative" in str(negative.get("case", "")):
            status = "rejected_negative"
            match = re.search(r"composite delta ([+-]?\d+(?:\.\d+)?)", str(negative["result"]))
            ci = re.search(r"95% CI \[([+-]?\d+(?:\.\d+)?), ([+-]?\d+(?:\.\d+)?)\]", str(negative["result"]))
            raw_composite = float(match.group(1)) if match else math.nan
            raw_low = float(ci.group(1)) if ci else math.nan
            raw_high = float(ci.group(2)) if ci else math.nan
        else:
            status = "fallback_uncertain"
            match = re.search(r"composite delta ([+-]?\d+(?:\.\d+)?)", str((negative or {}).get("result", "")))
            ci = re.search(r"95% CI \[([+-]?\d+(?:\.\d+)?), ([+-]?\d+(?:\.\d+)?)\]", str((negative or {}).get("result", "")))
            raw_composite = float(match.group(1)) if match else math.nan
            raw_low = float(ci.group(1)) if ci else math.nan
            raw_high = float(ci.group(2)) if ci else math.nan

        patches = summary.get("patches", [])
        fields = sorted({
            str(field)
            for patch in patches
            for field in patch.get("role_multipliers", {})
        })
        role_weights = {
            field: round(mean(
                float(patch["role_multipliers"][field])
                for patch in patches
                if field in patch.get("role_multipliers", {})
            ), 6)
            for field in fields
        }
        records.append({
            "source": source,
            "target": target,
            "pair": pair_label(source, target),
            "domain": DOMAIN_BY_DATASET.get(target, "Other"),
            "status": status,
            "selected_transfer": selected,
            "final_delta": float(row["delta_final_best"]),
            "auc_delta": float(row["delta_auc"]),
            "composite_delta": raw_composite,
            "composite_ci_low": raw_low,
            "composite_ci_high": raw_high,
            "rounds_saved_top10": float(row["rounds_saved_top10"]),
            "role_weights": role_weights,
            "patch_count": len(patches),
        })
    return records


def configure_plot() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#FBFBF9",
        "axes.facecolor": "#FBFBF9",
        "savefig.facecolor": "#FBFBF9",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def plot_weight_heatmap(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    fields = sorted({field for record in records for field in record["role_weights"]})
    values = np.full((len(records), len(fields)), np.nan)
    for row_index, record in enumerate(records):
        for column_index, field in enumerate(fields):
            if field in record["role_weights"]:
                values[row_index, column_index] = record["role_weights"][field]
    finite = values[np.isfinite(values)]
    low = min(0.5, float(np.min(finite))) if finite.size else 0.5
    high = max(1.5, float(np.max(finite))) if finite.size else 1.5
    norm = TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=high)
    fig, ax = plt.subplots(figsize=(max(12, len(fields) * 0.7), 6.3))
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(fields)), [short_field(field) for field in fields], rotation=45, ha="right")
    row_labels = []
    for record in records:
        prefix = {"deployed_positive": "+", "rejected_negative": "-", "fallback_uncertain": "~"}[record["status"]]
        row_labels.append(f"{prefix} {record['pair']}")
    ax.set_yticks(range(len(records)), row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(records)):
        for j in range(len(fields)):
            if math.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean role multiplier across candidate LLM patches\n1.00 = neutral weight")
    ax.set_title("CARE 2.0 transfer weights by source-target pair", loc="left", fontweight="bold", pad=16)
    ax.text(
        0,
        1.02,
        "+ deployed transfer   - raw negative route rejected   ~ unstable route rejected",
        transform=ax.transAxes,
        color="#5B6168",
        fontsize=9,
    )
    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def plot_transfer_matrix(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    datasets = [
        "real_chemlex_acidamine",
        "real_buchwald_hartwig",
        "real_matbench_dielectric",
        "real_matbench_expt_gap",
        "real_matbench_phonons",
        "real_moleculenet_esol",
        "real_moleculenet_freesolv",
        "real_moleculenet_lipophilicity",
    ]
    record_map = {(r["source"], r["target"]): r for r in records}
    matrix = [[0 for _ in datasets] for _ in datasets]
    annotations: dict[tuple[int, int], tuple[str, int]] = {}
    for record in records:
        i = datasets.index(record["source"])
        j = datasets.index(record["target"])
        status_value = 1 if record["status"] == "deployed_positive" else -1 if record["status"] == "rejected_negative" else 0
        matrix[i][j] = status_value
        if record["status"] == "deployed_positive":
            text = f"+{record['final_delta']:.1f}\nAUC +{record['auc_delta']:.1f}"
        elif record["status"] == "rejected_negative":
            text = f"{record['composite_delta']:.1f}\nrejected"
        else:
            value = record["composite_delta"]
            text = f"~{value:+.1f}\nuncertain" if math.isfinite(value) else "~\nfallback"
        annotations[(i, j)] = (text, status_value)
    fig, ax = plt.subplots(figsize=(10.5, 8.8))
    image = ax.imshow(matrix, cmap=ListedColormap(["#D95555", "#D9D9D4", "#2F855A"]), vmin=-1, vmax=1)
    ax.set_xticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets], rotation=45, ha="right")
    ax.set_yticks(range(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.set_xlabel("Target task")
    ax.set_ylabel("Source task")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            annotation = annotations.get((i, j))
            if not annotation:
                continue
            text, status_value = annotation
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#FFFFFF" if status_value else "#55575E", fontweight="bold" if status_value else "normal")
    ax.set_title("CARE 2.0 transfer matrix", loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.02, "Cells show held-out final-best / AUC deltas for deployed routes; rejected cells show the raw route signal.", transform=ax.transAxes, color="#5B6168", fontsize=9)
    ax.legend(handles=[
        Patch(facecolor="#2F855A", label="Positive transfer deployed"),
        Patch(facecolor="#D95555", label="Negative transfer rejected"),
        Patch(facecolor="#D9D9D4", label="Unstable / fallback"),
    ], loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, output)
    plt.close(fig)


def plot_transfer_graph(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    positions = {
        "real_chemlex_acidamine": (0.08, 0.82),
        "real_buchwald_hartwig": (0.90, 0.82),
        "real_matbench_phonons": (0.08, 0.48),
        "real_matbench_expt_gap": (0.48, 0.48),
        "real_matbench_dielectric": (0.90, 0.48),
        "real_moleculenet_esol": (0.08, 0.14),
        "real_moleculenet_freesolv": (0.48, 0.14),
        "real_moleculenet_lipophilicity": (0.90, 0.14),
    }
    fig, ax = plt.subplots(figsize=(13, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for record in records:
        start = positions[record["source"]]
        end = positions[record["target"]]
        status = record["status"]
        color = {"deployed_positive": "#2F855A", "rejected_negative": "#C53030", "fallback_uncertain": "#8B8D98"}[status]
        magnitude = abs(record["composite_delta"]) if math.isfinite(record["composite_delta"]) else 1.0
        width = 1.4 + min(3.8, magnitude / 12.0)
        reverse = (record["source"], record["target"]) in {
            ("real_matbench_dielectric", "real_matbench_expt_gap"),
            ("real_matbench_expt_gap", "real_matbench_dielectric"),
        }
        if record["source"] == "real_matbench_phonons" and record["target"] == "real_matbench_dielectric":
            rad = 0.24
        else:
            rad = 0.16 if record["source"] == "real_matbench_dielectric" and reverse else -0.16 if reverse else 0.0
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=width,
            color=color,
            alpha=0.82,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=38,
            shrinkB=38,
        )
        ax.add_patch(arrow)
        label_value = record["composite_delta"]
        if status == "deployed_positive":
            label = f"+{label_value:.1f}"
        elif status == "rejected_negative":
            label = f"{label_value:.1f}"
        else:
            label = f"~{label_value:+.1f}" if math.isfinite(label_value) else "~?"
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + (0.035 if rad >= 0 else -0.035)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold", bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FBFBF9", "edgecolor": "none", "alpha": 0.92})

    for dataset, (x, y) in positions.items():
        domain = DOMAIN_BY_DATASET[dataset]
        color = DOMAIN_COLORS[domain]
        box = FancyBboxPatch((x - 0.085, y - 0.035), 0.17, 0.07, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor="#FFFFFF", edgecolor=color, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x, y + 0.002, DATASET_LABELS[dataset], ha="center", va="center", fontsize=10, fontweight="bold", color="#263238")
        ax.text(x, y - 0.055, domain, ha="center", va="center", fontsize=8, color=color)
    ax.text(0.02, 0.98, "CARE 2.0 transfer graph", transform=ax.transAxes, fontsize=17, fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.945, "Arrow direction: source -> target. Width reflects the magnitude of the composite held-out signal.", transform=ax.transAxes, fontsize=10, color="#5B6168", ha="left", va="top")
    ax.legend(handles=[
        Patch(facecolor="#2F855A", label="Positive transfer deployed"),
        Patch(facecolor="#C53030", label="Negative raw route rejected"),
        Patch(facecolor="#8B8D98", label="Unstable route rejected / fallback"),
        Patch(facecolor=DOMAIN_COLORS["Reaction"], label="Reaction task"),
        Patch(facecolor=DOMAIN_COLORS["Materials"], label="Materials task"),
        Patch(facecolor=DOMAIN_COLORS["Molecular"], label="Molecular task"),
    ], loc="lower left", bbox_to_anchor=(0.02, 0.01), ncol=3, frameon=False, fontsize=9)
    save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outcome-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_plot()
    records = load_records(args.source_outcome_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "transfer_visualization_data.json").write_text(
        json.dumps({"source": str(args.source_outcome_root), "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_weight_heatmap(records, args.output_dir / "transfer_role_weight_heatmap")
    plot_transfer_matrix(records, args.output_dir / "transfer_matrix")
    plot_transfer_graph(records, args.output_dir / "transfer_graph")
    print(json.dumps({"records": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
