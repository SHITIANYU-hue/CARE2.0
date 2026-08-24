#!/usr/bin/env python3
"""Replay classical transfer-BO baselines from an online LLM trajectory's start."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import run_classical_transfer_baselines as classical
import run_multisource_transfer_baselines as multisource
import run_synthetic_suzuki as replay
import run_transfer_ablation as transfer


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "care.online_llm_matched_classical_audit/v1"
TOLERANCE = 1e-6
ROUTE_LABELS = {
    "molecular_freesolv_to_lipophilicity": "FreeSolv → Lipophilicity",
    "materials_phonons_to_bulk_modulus": "Phonons → bulk modulus",
    "reizman_cases_123_to_case4": "Suzuki cases 1–3 → case 4",
}
METHOD_LABELS = {
    "target_gp_ucb": "Target-only GP-UCB",
    "rgpe": "RGPE",
    "multisource_rgpe": "RGPE",
    "multitask_gp_icm": "Multitask GP",
    "multisource_icm_bma": "Multitask GP",
}
METHOD_COLORS = {
    "Target-only GP-UCB": "#68737D",
    "RGPE": "#E58B2A",
    "Multitask GP": "#2F6FA3",
}
METHOD_MARKERS = {
    "Target-only GP-UCB": "o",
    "RGPE": "s",
    "Multitask GP": "D",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def plot_comparisons(
    rows: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    route_order = [str(report["case_id"]) for report in reports]
    strongest = {
        str(report["case_id"]): str(
            report["strongest_realized_baseline_posthoc"]["method"]
        )
        for report in reports
    }
    offsets = {
        "Target-only GP-UCB": -0.22,
        "RGPE": 0.0,
        "Multitask GP": 0.22,
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#30363B",
            "axes.linewidth": 0.8,
            "text.color": "#20262B",
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.axvline(0.0, color="#B23A48", linewidth=1.2, linestyle="--", zorder=0)
    for route_index, route_id in enumerate(route_order):
        route_rows = [row for row in rows if row["case_id"] == route_id]
        for row in route_rows:
            method = str(row["method"])
            label = METHOD_LABELS[method]
            value = float(row["online_llm_minus_baseline_auc"])
            is_strongest = method == strongest[route_id]
            ax.scatter(
                value,
                route_index + offsets[label],
                s=105 if is_strongest else 72,
                marker=METHOD_MARKERS[label],
                color=METHOD_COLORS[label],
                edgecolor="#15191C" if is_strongest else "white",
                linewidth=1.5 if is_strongest else 0.8,
                zorder=3,
                label=label if route_index == 0 else None,
            )
            ax.text(
                value + (0.28 if value >= 0 else -0.28),
                route_index + offsets[label],
                f"{value:+.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=9,
                color=METHOD_COLORS[label],
                fontweight="bold" if is_strongest else "normal",
            )
    ax.set_yticks(range(len(route_order)))
    ax.set_yticklabels([ROUTE_LABELS.get(route, route) for route in route_order])
    ax.invert_yaxis()
    ax.set_xlabel("Δ best-so-far AUC (online LLM − baseline)")
    ax.set_title("Same-start stress test: the LLM beats every baseline on 2 of 3 routes", loc="left")
    ax.grid(axis="x", color="#DDE2E5", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, ncol=3)
    fig.text(
        0.01,
        0.01,
        "Exact same initial candidates, pool and 10-reveal budget. Black outline marks the strongest realized baseline per route. Post-hoc, n=1 trajectory per route; no confidence interval.",
        fontsize=8.5,
        color="#5D6870",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    outputs = [
        output_root / "online_llm_vs_matched_baselines.png",
        output_root / "online_llm_vs_matched_baselines.pdf",
        output_root / "online_llm_vs_matched_baselines.svg",
    ]
    for output in outputs:
        fig.savefig(output, dpi=220, bbox_inches="tight")
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{sha256(output)}  {output.name}\n",
            encoding="utf-8",
        )
    plt.close(fig)
    return outputs


def require_close(label: str, observed: float, expected: float) -> None:
    if abs(observed - expected) > TOLERANCE:
        raise ValueError(
            f"{label} mismatch: rebuilt={observed:.8f}, summary={expected:.8f}"
        )


def initial_indices(
    target: replay.DatasetAdapter,
    summary: Mapping[str, Any],
) -> list[int]:
    candidate_index = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    ids = str(
        summary["metrics"]["online_llm_scientist"][
            "executed_initial_candidate_ids"
        ]
    ).split(";")
    missing = [candidate_id for candidate_id in ids if candidate_id not in candidate_index]
    if missing:
        raise ValueError(f"Initial candidates are missing from target adapter: {missing}")
    return [candidate_index[candidate_id] for candidate_id in ids]


def source_posteriors(
    sources: Sequence[replay.DatasetAdapter],
    target: replay.DatasetAdapter,
    protocol: Mapping[str, Any],
    kernel: Mapping[str, Any],
) -> list[classical.SourcePosterior]:
    posteriors = []
    for source in sources:
        observed = transfer.source_observations(
            source,
            int(protocol["source_seed"]),
            len(source.candidates),
        )
        posteriors.append(
            classical.build_source_posterior(
                source,
                target,
                observed,
                int(protocol["source_inducing_limit"]),
                float(kernel["numeric_length_scale"]),
                float(kernel["categorical_length_scale"]),
                float(kernel["gp_noise"]),
            )
        )
    return posteriors


def run_case(
    case: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = ROOT / str(case["summary"])
    config_path = ROOT / str(case["config"])
    summary = load_json(summary_path)
    config = load_json(config_path)
    kernel = config["protocol"]["kernel"]
    target = replay.DATASET_BUILDERS[str(summary["target_task"])]()
    sources = [
        replay.DATASET_BUILDERS[str(dataset_id)]()
        for dataset_id in summary["source_tasks"]
    ]
    for source in sources:
        classical.validate_compatible_spaces(source, target)
    fixed_initial = initial_indices(target, summary)
    posteriors = source_posteriors(sources, target, protocol, kernel)
    initial_count = len(fixed_initial)
    rounds = int(summary["requested_reveal_rounds"])
    rows = []
    if len(sources) == 1:
        modes = classical.MODES
        for mode in modes:
            metrics, audit = classical.run_seed(
                sources[0],
                target,
                posteriors[0],
                seed=int(protocol["baseline_seed"]),
                initial=initial_count,
                rounds=rounds,
                mode=mode,
                gp_beta=float(kernel["gp_beta"]),
                numeric_length_scale=float(kernel["numeric_length_scale"]),
                categorical_length_scale=float(kernel["categorical_length_scale"]),
                gp_noise=float(kernel["gp_noise"]),
                rgpe_draws=int(protocol["rgpe_draws"]),
                rho_grid=tuple(float(value) for value in protocol["multitask_rho_grid"]),
                fixed_initial_indices=fixed_initial,
            )
            rows.append(metrics)
            write_jsonl(output_root / str(case["case_id"]) / f"{mode}.jsonl", audit)
    else:
        modes = ("target_gp_ucb", "multisource_rgpe", "multisource_icm_bma")
        for mode in modes:
            metrics, audit = multisource.run_seed(
                sources,
                target,
                posteriors,
                seed=int(protocol["baseline_seed"]),
                initial=initial_count,
                rounds=rounds,
                mode=mode,
                gp_beta=float(kernel["gp_beta"]),
                numeric_length_scale=float(kernel["numeric_length_scale"]),
                categorical_length_scale=float(kernel["categorical_length_scale"]),
                gp_noise=float(kernel["gp_noise"]),
                rgpe_draws=int(protocol["rgpe_draws"]),
                rho_grid=tuple(float(value) for value in protocol["multitask_rho_grid"]),
                bma_temperature=float(protocol["bma_temperature"]),
                fixed_initial_indices=fixed_initial,
            )
            rows.append(metrics)
            write_jsonl(output_root / str(case["case_id"]) / f"{mode}.jsonl", audit)

    target_gp = next(row for row in rows if row["mode"] == "target_gp_ucb")
    expected_gp = summary["metrics"]["llm_initial_target_gp"]
    require_close(
        f"{case['case_id']} target GP AUC",
        float(target_gp["best_so_far_auc"]),
        float(expected_gp["best_so_far_auc"]),
    )
    require_close(
        f"{case['case_id']} target GP final",
        float(target_gp["final_best"]),
        float(expected_gp["final_best"]),
    )
    online = summary["metrics"]["online_llm_scientist"]
    comparisons = []
    for row in rows:
        comparisons.append(
            {
                "method": row["mode"],
                "baseline_best_so_far_auc": float(row["best_so_far_auc"]),
                "baseline_final_best": float(row["final_best"]),
                "online_llm_minus_baseline_auc": round(
                    float(online["best_so_far_auc"])
                    - float(row["best_so_far_auc"]),
                    6,
                ),
                "online_llm_minus_baseline_final": round(
                    float(online["final_best"]) - float(row["final_best"]),
                    6,
                ),
            }
        )
    strongest = max(comparisons, key=lambda row: row["baseline_best_so_far_auc"])
    case_report = {
        "case_id": case["case_id"],
        "source_tasks": summary["source_tasks"],
        "target_task": summary["target_task"],
        "summary_path": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "fixed_initial_candidate_ids": [
            target.candidates[index].candidate_id for index in fixed_initial
        ],
        "reveal_rounds": rounds,
        "online_llm": {
            "best_so_far_auc": float(online["best_so_far_auc"]),
            "final_best": float(online["final_best"]),
        },
        "comparisons": comparisons,
        "strongest_realized_baseline_posthoc": strongest,
        "online_llm_beats_every_realized_baseline_auc": (
            strongest["online_llm_minus_baseline_auc"] > 0
        ),
    }
    write_json(output_root / str(case["case_id"]) / "summary.json", case_report)
    flat_rows = [
        {
            "case_id": case["case_id"],
            "source_task_count": len(sources),
            **comparison,
        }
        for comparison in comparisons
    ]
    return case_report, flat_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    protocol = config["protocol"]
    reports = []
    rows = []
    for case in config["cases"]:
        report, case_rows = run_case(case, protocol, args.output_root)
        reports.append(report)
        rows.extend(case_rows)
    strongest_deltas = [
        float(report["strongest_realized_baseline_posthoc"]["online_llm_minus_baseline_auc"])
        for report in reports
    ]
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "completed_routes": len(reports),
        "routes_where_online_llm_beats_every_realized_baseline_auc": sum(
            int(report["online_llm_beats_every_realized_baseline_auc"])
            for report in reports
        ),
        "equal_route_mean_online_llm_minus_strongest_realized_baseline_auc": round(
            mean(strongest_deltas),
            6,
        ),
        "route_reports": reports,
        "claim_boundary": (
            "This post-hoc descriptive audit replays all named baselines from the "
            "same initial candidates and budget. The strongest realized baseline "
            "is selected after observing these trajectories and is therefore an "
            "oracle stress test, not a confirmatory model-selection result."
        ),
    }
    csv_path = args.output_root / "comparisons.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_path.with_suffix(csv_path.suffix + ".sha256").write_text(
        f"{sha256(csv_path)}  {csv_path.name}\n",
        encoding="utf-8",
    )
    figures = plot_comparisons(rows, reports, args.output_root)
    aggregate["figure_files"] = [
        {
            "path": str(path),
            "sha256": sha256(path),
            "source_data": str(csv_path),
        }
        for path in figures
    ]
    write_json(args.output_root / "aggregate.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
