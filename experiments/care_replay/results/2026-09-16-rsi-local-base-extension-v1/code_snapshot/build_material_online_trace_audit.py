#!/usr/bin/env python3
"""Rebuild and plot an audited online-materials trajectory from frozen data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_multisource_warmstart as warmstart
import run_synthetic_suzuki as replay


SCHEMA_VERSION = "care.material_online_trace_audit/v1"
TOLERANCE = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def require_close(label: str, observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, atol=TOLERANCE, rtol=0.0):
        raise ValueError(
            f"{label} mismatch: rebuilt={observed:.8f}, summary={expected:.8f}"
        )


def rebuild_same_initial_gp(
    summary: Mapping[str, Any],
    case_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_id = str(summary["target_task"])
    target = replay.DATASET_BUILDERS[target_id]()
    candidate_index = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    initial_ids = str(
        summary["metrics"]["online_llm_scientist"][
            "executed_initial_candidate_ids"
        ]
    ).split(";")
    missing = [candidate_id for candidate_id in initial_ids if candidate_id not in candidate_index]
    if missing:
        raise ValueError(f"Initial candidates missing from target adapter: {missing}")
    initial_indices = [candidate_index[candidate_id] for candidate_id in initial_ids]
    protocol = case_config["protocol"]
    metrics, audit = warmstart.run_target_gp(
        target,
        initial_indices,
        int(summary["requested_reveal_rounds"]),
        "llm_initial_target_gp",
        -1,
        protocol["kernel"],
        summary["source_tasks"],
        {"initial_policy": "llm"},
    )
    expected = summary["metrics"]["llm_initial_target_gp"]
    require_close(
        "same-initial GP AUC",
        float(metrics["best_so_far_auc"]),
        float(expected["best_so_far_auc"]),
    )
    require_close(
        "same-initial GP final best",
        float(metrics["final_best"]),
        float(expected["final_best"]),
    )
    return metrics, audit


def build_rows(
    trace: Sequence[Mapping[str, Any]],
    baseline_audit: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    online = {
        int(event["round_index"]): event
        for event in trace
        if event.get("event") == "target_reveal"
    }
    baseline = {
        int(event["round_index"]): event
        for event in baseline_audit
        if event.get("event") == "target_gp_reveal"
    }
    if set(online) != set(baseline):
        raise ValueError("Online and same-initial GP rounds do not match.")
    rows = []
    for round_index in sorted(online):
        llm = online[round_index]
        gp = baseline[round_index]
        rows.append(
            {
                "round": round_index + 1,
                "llm_candidate": llm["selected_candidate"],
                "llm_composition": llm.get("public_conditions", {}).get(
                    "composition", ""
                ),
                "llm_revealed_value": float(llm["revealed_value"]),
                "llm_best_so_far": float(llm["best_so_far"]),
                "llm_gp_rank_at_selection": int(llm["gp_rank_at_selection"]),
                "llm_overrode_gp_rank_one": int(llm["gp_rank_at_selection"]) > 1,
                "gp_candidate": gp["selected_candidate"],
                "gp_composition": gp.get("public_conditions", {}).get(
                    "composition", ""
                ),
                "gp_revealed_value": float(gp["revealed_value"]),
                "gp_best_so_far": float(gp["best_so_far"]),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_case(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    rounds = np.asarray([row["round"] for row in rows], dtype=float)
    llm_best = np.asarray([row["llm_best_so_far"] for row in rows], dtype=float)
    gp_best = np.asarray([row["gp_best_so_far"] for row in rows], dtype=float)
    gp_ranks = np.asarray([row["llm_gp_rank_at_selection"] for row in rows], dtype=float)
    overrides = np.asarray([bool(row["llm_overrode_gp_rank_one"]) for row in rows])
    delta = float(
        summary["deltas"]["online_llm_increment_over_same_initial_gp"][
            "best_so_far_auc"
        ]
    )
    final_delta = float(
        summary["deltas"]["online_llm_increment_over_same_initial_gp"][
            "final_best"
        ]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#353B40",
            "axes.linewidth": 0.8,
            "xtick.color": "#353B40",
            "ytick.color": "#353B40",
            "text.color": "#20262B",
        }
    )
    fig = plt.figure(figsize=(11.6, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.05, 1.0))
    ax = fig.add_subplot(grid[0])
    rank_ax = fig.add_subplot(grid[1], sharex=ax)

    llm_color = "#167D8D"
    gp_color = "#E58B2A"
    negative = "#B23A48"
    ax.step(rounds, llm_best, where="mid", color=llm_color, linewidth=2.8, label="CARE 2.0 online LLM")
    ax.scatter(rounds, llm_best, color=llm_color, s=30, zorder=3)
    ax.step(rounds, gp_best, where="mid", color=gp_color, linewidth=2.4, linestyle="--", label="Same-initial target-only GP-UCB")
    ax.scatter(rounds, gp_best, facecolor="white", edgecolor=gp_color, linewidth=1.5, s=30, zorder=3)
    ax.set_ylabel("Best normalized bulk-modulus score")
    if final_delta > TOLERANCE:
        title = "Materials case: the LLM improves discovery speed and final score"
    elif final_delta < -TOLERANCE:
        title = "Materials case: earlier gains coexist with a lower final score"
    else:
        title = "Materials case: the LLM improves discovery speed, not the final score"
    ax.set_title(title)
    ax.grid(axis="y", color="#DDE2E5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.text(
        0.01,
        0.96,
        f"AUC delta vs same-initial GP-UCB: {delta:+.3f}",
        transform=ax.transAxes,
        va="top",
        fontweight="bold",
        color=llm_color,
    )

    improved = np.flatnonzero(np.diff(np.r_[llm_best[0], llm_best]) > 0)
    annotated = []
    if len(improved):
        annotated.append(int(improved[0]))
    if len(improved) > 1:
        annotated.append(int(improved[-1]))
    for annotation_index, hit in enumerate(dict.fromkeys(annotated)):
        label = str(rows[hit]["llm_composition"] or rows[hit]["llm_candidate"])
        right_edge = rounds[hit] >= rounds[-1] - 1
        x_offset = -2.3 if right_edge else 0.7
        y_offset = -0.85 if right_edge else 0.45
        ax.annotate(
            f"{label}: {llm_best[hit]:.2f} normalized score\nfound in round {int(rounds[hit])}",
            xy=(rounds[hit], llm_best[hit]),
            xytext=(rounds[hit] + x_offset, llm_best[hit] + y_offset),
            arrowprops={"arrowstyle": "->", "color": llm_color, "lw": 1.3},
            fontsize=9,
            color=llm_color,
        )

    colors = np.where(overrides, negative, "#69747C")
    rank_ax.scatter(rounds, gp_ranks, c=colors, s=55, zorder=3)
    rank_ax.plot(rounds, gp_ranks, color="#B7BFC4", linewidth=1.2, zorder=1)
    rank_ax.axhline(1, color="#69747C", linestyle="--", linewidth=1.0)
    rank_ax.set_ylabel("GP rank chosen\nby the LLM")
    rank_ax.set_xlabel("Online target reveal round")
    rank_ax.set_xticks(rounds)
    rank_ax.set_yticks(sorted(set(int(value) for value in gp_ranks)))
    rank_ax.invert_yaxis()
    rank_ax.grid(axis="x", color="#EDF0F2", linewidth=0.8)
    rank_ax.spines[["top", "right"]].set_visible(False)
    rank_ax.text(
        0.99,
        0.92,
        f"LLM overrode GP rank 1 in {int(overrides.sum())}/{len(rows)} rounds",
        transform=rank_ax.transAxes,
        ha="right",
        va="top",
        color=negative,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.16, hspace=0.2)
    fig.text(
        0.01,
        0.025,
        "Source: frozen Matbench phonons -> bulk-modulus replay. Score = log10(K_VRH) / 3 x 100. Unselected target outcomes remained hidden until reveal.",
        fontsize=8.5,
        color="#5D666D",
    )
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = load_json(args.summary)
    trace = load_trace(args.trace)
    case_config = load_json(args.case_config)
    baseline_metrics, baseline_audit = rebuild_same_initial_gp(summary, case_config)
    rows = build_rows(trace, baseline_audit)
    online_auc = float(np.mean([row["llm_best_so_far"] for row in rows]))
    baseline_auc = float(np.mean([row["gp_best_so_far"] for row in rows]))
    online_summary = summary["metrics"]["online_llm_scientist"]
    require_close("online AUC", online_auc, float(online_summary["best_so_far_auc"]))
    require_close(
        "same-initial GP AUC",
        baseline_auc,
        float(baseline_metrics["best_so_far_auc"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "materials_online_trace.csv"
    figure_path = args.output_dir / "materials_online_trace.png"
    figure_pdf_path = args.output_dir / "materials_online_trace.pdf"
    audit_path = args.output_dir / "materials_online_trace_audit.json"
    write_csv(csv_path, rows)
    plot_case(rows, summary, figure_path)
    plot_case(rows, summary, figure_pdf_path)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "case_id": "materials_phonons_to_bulk_modulus",
        "source_summary": {
            "path": str(args.summary),
            "sha256": sha256(args.summary),
        },
        "source_trace": {"path": str(args.trace), "sha256": sha256(args.trace)},
        "case_config": {
            "path": str(args.case_config),
            "sha256": sha256(args.case_config),
        },
        "reconstruction_checks": {
            "online_auc": round(online_auc, 6),
            "same_initial_gp_auc": round(baseline_auc, 6),
            "auc_delta": round(online_auc - baseline_auc, 6),
            "online_final_best": round(float(rows[-1]["llm_best_so_far"]), 6),
            "same_initial_gp_final_best": round(
                float(rows[-1]["gp_best_so_far"]), 6
            ),
            "llm_gp_rank_one_overrides": sum(
                bool(row["llm_overrode_gp_rank_one"]) for row in rows
            ),
        },
        "claim_boundary": (
            "Single development trajectory. The figure demonstrates an auditable "
            "decision mechanism and search-efficiency gain, not confirmatory transfer."
        ),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in (csv_path, figure_path, figure_pdf_path, audit_path):
        write_sha256(path)


if __name__ == "__main__":
    main()
