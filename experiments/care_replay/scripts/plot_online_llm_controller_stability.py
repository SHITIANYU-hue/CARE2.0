#!/usr/bin/env python3
"""Plot development-route selection stability and disjoint-route effects."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROUTE_LABELS = {
    "molecular_lipophilicity_to_freesolv": "Lipo -> FreeSolv",
    "materials_bulk_to_shear_modulus": "Bulk -> shear",
    "materials_phonons_to_perovskites": "Phonons -> perovskites",
    "materials_expt_gap_to_steels": "Expt. gap -> steels",
    "molecular_freesolv_to_bace": "FreeSolv -> BACE",
    "materials_perovskites_to_mp_e_form": "Perovskites -> formation E",
}

POLICY_LABELS = {
    "bounded_authority_r1": "One-round bounded authority",
    "prediction_error_r2_t9999_hard1": "Two-round hard-abstention gate",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_sidecar(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    selection_rows = read_jsonl(args.selection)
    evaluation_rows = read_jsonl(args.evaluation)
    if len(selection_rows) != 11:
        raise ValueError("Expected 11 leave-one-route-out selection folds.")
    if len(evaluation_rows) != 6:
        raise ValueError("Expected six route-disjoint evaluation routes.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = Counter(row["selected_policy_id"] for row in selection_rows)
    policies = sorted(counts, key=lambda policy: (-counts[policy], policy))
    case_ids = [row["case_id"] for row in evaluation_rows]
    values = [float(row["policy_minus_target_gp_auc"]) for row in evaluation_rows]
    colors = [
        "#1B7A62" if value > 1e-9 else "#B14C55" if value < -1e-9 else "#77838C"
        for value in values
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.6),
        gridspec_kw={"width_ratios": [0.8, 1.45], "wspace": 0.38},
    )

    policy_labels = [POLICY_LABELS.get(policy, policy) for policy in policies]
    policy_values = [counts[policy] for policy in policies]
    left.barh(
        range(len(policies)),
        policy_values,
        color=["#2F6FB6", "#AAB4BC"][: len(policies)],
        height=0.55,
    )
    left.set_yticks(range(len(policies)))
    left.set_yticklabels(policy_labels)
    left.invert_yaxis()
    left.set_xlim(0, 11.6)
    left.set_xticks(range(0, 12, 2))
    left.set_xlabel("LOO folds selecting the controller")
    left.set_title("A. Development-route stability", loc="left", fontweight="bold")
    for index, value in enumerate(policy_values):
        left.text(value + 0.2, index, f"{value}/11", va="center", fontweight="bold")

    right.axvline(0.0, color="#8B979F", linestyle="--", linewidth=1.1)
    right.barh(range(len(case_ids)), values, color=colors, height=0.58)
    right.set_yticks(range(len(case_ids)))
    right.set_yticklabels([ROUTE_LABELS.get(case_id, case_id) for case_id in case_ids])
    right.invert_yaxis()
    right.set_xlim(-1.8, 3.35)
    right.set_xlabel("AUC delta versus same-start target-only GP-UCB")
    right.set_title("B. Frozen route-disjoint evaluation", loc="left", fontweight="bold")
    for index, value in enumerate(values):
        offset = 0.08 if value >= 0 else -0.08
        alignment = "left" if value >= 0 else "right"
        right.text(value + offset, index, f"{value:+.2f}", va="center", ha=alignment)

    for axis in (left, right):
        axis.grid(axis="x", color="#E3E7EA", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    figure.suptitle(
        "One-round LLM authority is selection-stable, but one disjoint route still loses",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.015,
        "LOO uses only the 11 development routes. The six evaluation routes are disjoint retrospective trajectories; positive AUC delta is better.",
        fontsize=8.5,
        color="#58646C",
    )
    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.82)

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_root / "controller_selection_stability.png",
        args.output_root / "controller_selection_stability.pdf",
        args.output_root / "controller_selection_stability.svg",
    ]
    for output in outputs:
        metadata = {"CreationDate": None} if output.suffix == ".pdf" else None
        figure.savefig(output, dpi=260, bbox_inches="tight", metadata=metadata)
        if output.suffix == ".svg":
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text().splitlines())
                + "\n",
                encoding="utf-8",
            )
        add_sidecar(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
