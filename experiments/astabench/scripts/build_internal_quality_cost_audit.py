#!/usr/bin/env python3
"""Build an AstaBench-style quality/token audit from real CARE trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "experiments/care_replay/results/2026-08-23-online-llm-repeated-confirmation-v1"
    / "aggregate/route_statistics.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/astabench/results/2026-08-29-internal-quality-cost-audit-v1"
)
REQUIRED_COLUMNS = {
    "case_id",
    "domain",
    "evidence_tier",
    "completed_replicates",
    "expected_replicates",
    "mean_auc_delta",
    "mean_total_tokens",
}


def read_routes(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = []
        for row in reader:
            if not row["mean_auc_delta"] or not row["mean_total_tokens"]:
                continue
            auc_delta = float(row["mean_auc_delta"])
            total_tokens = int(float(row["mean_total_tokens"]))
            rows.append(
                {
                    "case_id": row["case_id"],
                    "domain": row["domain"],
                    "evidence_tier": row["evidence_tier"],
                    "completed_replicates": int(row["completed_replicates"]),
                    "expected_replicates": int(row["expected_replicates"]),
                    "auc_delta_vs_same_initial_gp": auc_delta,
                    "total_tokens": total_tokens,
                    "auc_delta_per_100k_tokens": (
                        auc_delta * 100000.0 / total_tokens if total_tokens else 0.0
                    ),
                    "outcome": (
                        "win"
                        if auc_delta > 1e-12
                        else "loss"
                        if auc_delta < -1e-12
                        else "tie"
                    ),
                }
            )
    if not rows:
        raise ValueError(f"No complete route rows found in {path}")
    return rows


def bootstrap_mean_interval(
    values: list[float], iterations: int = 20000, seed: int = 20260829
) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(iterations)
    )
    low_index = int(0.025 * (iterations - 1))
    high_index = int(0.975 * (iterations - 1))
    return draws[low_index], draws[high_index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = [float(row["auc_delta_vs_same_initial_gp"]) for row in rows]
    low, high = bootstrap_mean_interval(effects)
    completed = sum(int(row["completed_replicates"]) for row in rows)
    expected = sum(int(row["expected_replicates"]) for row in rows)
    return {
        "route_count": len(rows),
        "completed_trajectories": completed,
        "expected_trajectories": expected,
        "completion_fraction": completed / expected if expected else 0.0,
        "mean_auc_delta_vs_same_initial_gp": statistics.fmean(effects),
        "route_bootstrap_95ci": [low, high],
        "median_total_tokens": statistics.median(
            int(row["total_tokens"]) for row in rows
        ),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "wins": sum(row["outcome"] == "win" for row in rows),
        "ties": sum(row["outcome"] == "tie" for row in rows),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "claim_boundary": (
            "Descriptive first-trajectory quality/token audit. The frozen repeated "
            "confirmation remains incomplete, and no monetary cost is inferred "
            "without a frozen provider price snapshot."
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def short_label(case_id: str) -> str:
    replacements = {
        "molecular_": "Mol: ",
        "materials_": "Mat: ",
        "baumgartner_": "C-N: ",
        "reizman_": "Suzuki: ",
        "_to_": " -> ",
        "_": " ",
    }
    label = case_id
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def scatter_label(case_id: str) -> str:
    labels = {
        "molecular_freesolv_to_esol": "Mol F->E",
        "molecular_freesolv_to_lipophilicity": "Mol F->L",
        "materials_expt_gap_to_dielectric": "Mat Eg->Diel",
        "materials_phonons_to_bulk_modulus": "Mat Ph->K",
        "materials_dielectric_to_jdft2d": "Mat Diel->Jdft",
        "materials_expt_gap_to_mp_gap": "Mat Eg->MPg",
        "baumgartner_aniline_to_phenethylamine_alphos": "C-N A->P/Al",
        "baumgartner_aniline_to_benzamide_tbuxphos": "C-N A->B/tBuX",
        "baumgartner_aniline_to_phenethylamine_tbubrettphos": "C-N A->P/tBuB",
        "baumgartner_benzamide_tbubrettphos_to_alphos": "C-N B/tBuB->Al",
        "reizman_cases_123_to_case4": "Suz 1-3->4",
    }
    return labels.get(case_id, case_id)


def build_figure(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"win": "#14866D", "tie": "#68727D", "loss": "#C34A36"}
    ordered = sorted(rows, key=lambda row: float(row["auc_delta_vs_same_initial_gp"]))
    labels = [short_label(str(row["case_id"])) for row in ordered]
    effects = [float(row["auc_delta_vs_same_initial_gp"]) for row in ordered]
    tokens_k = [float(row["total_tokens"]) / 1000.0 for row in ordered]
    point_colors = [colors[str(row["outcome"])] for row in ordered]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    fig.subplots_adjust(left=0.31, right=0.98, bottom=0.17, top=0.84, wspace=0.24)

    axes[0].barh(range(len(ordered)), effects, color=point_colors, height=0.68)
    axes[0].axvline(0.0, color="#202124", linewidth=1.0)
    axes[0].set_yticks(range(len(ordered)), labels)
    axes[0].set_xlabel("Best-so-far AUC delta vs same-initial target GP")
    axes[0].set_title("Observed route effect")
    axes[0].grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)

    axes[1].scatter(
        tokens_k, effects, c=point_colors, s=72, edgecolor="white", linewidth=0.8
    )
    axes[1].axhline(0.0, color="#202124", linewidth=1.0)
    axes[1].set_xlabel("Recorded model tokens (thousands)")
    axes[1].set_ylabel("Best-so-far AUC delta vs same-initial target GP")
    axes[1].set_title("Quality-token frontier")
    axes[1].grid(color="#D9DEE3", linewidth=0.6, alpha=0.8)
    axes[1].set_xlim(min(tokens_k) - 7, max(tokens_k) + 13)
    axes[1].set_ylim(min(effects) - 0.35, max(effects) + 0.45)
    scatter_labels = [scatter_label(str(row["case_id"])) for row in ordered]
    offsets = {
        "Mol F->L": (-62, 8),
        "Mol F->E": (-46, 8),
        "C-N A->B/tBuX": (-88, -14),
        "C-N B/tBuB->Al": (7, 7),
        "Mat Diel->Jdft": (7, -13),
        "C-N A->P/Al": (7, 6),
        "Mat Eg->MPg": (7, 11),
        "C-N A->P/tBuB": (7, -13),
    }
    for x, y, label in zip(tokens_k, effects, scatter_labels):
        axes[1].annotate(
            label,
            (x, y),
            xytext=offsets.get(label, (7, 5)),
            textcoords="offset points",
            fontsize=8.5,
        )

    fig.suptitle(
        "CARE 2.0 online LLM: effect and recorded inference volume",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.035,
        "Real first trajectories from the frozen 11-route panel; one trajectory per route. "
        "Tokens are not converted to dollars because no provider price snapshot was frozen.",
        fontsize=8.5,
        color="#4F5963",
        ha="center",
    )
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"quality_token_audit.{extension}", dpi=220)
    plt.close(fig)


def write_markdown(
    rows: list[dict[str, Any]], summary: dict[str, Any], input_path: Path, path: Path
) -> None:
    lines = [
        "# CARE 2.0 quality-token audit",
        "",
        "This audit adopts AstaBench's quality-cost reporting principle for the",
        "existing CARE online-controller evidence. It does not report an AstaBench",
        "score and must not be presented as one.",
        "",
        f"- Source: `{input_path.relative_to(REPO_ROOT)}`",
        f"- Routes: {summary['route_count']}",
        f"- Completed frozen trajectories: {summary['completed_trajectories']} / {summary['expected_trajectories']}",
        f"- Mean AUC delta versus same-initial target GP: {summary['mean_auc_delta_vs_same_initial_gp']:+.4f}",
        f"- Route-bootstrap 95% interval: [{summary['route_bootstrap_95ci'][0]:+.4f}, {summary['route_bootstrap_95ci'][1]:+.4f}]",
        f"- Win / tie / loss: {summary['wins']} / {summary['ties']} / {summary['losses']}",
        f"- Recorded model tokens: {summary['total_tokens']:,} total; {summary['median_total_tokens']:,.0f} median per route",
        "",
        "## Interpretation",
        "",
        "The first trajectory on several molecular, materials, and reaction routes",
        "improved early discovery relative to the same-initial target-only GP, but",
        "the route-level interval remains descriptive and the 30-trajectory-per-route",
        "confirmation is incomplete. Negative routes are retained. Token counts expose",
        "a substantial inference burden and motivate the external AstaBench comparison",
        "against simple ReAct under matched resource limits.",
        "",
        "No monetary cost is reported: historical provider prices were not frozen, so",
        "converting these tokens to dollars now would create a post-hoc and potentially",
        "incorrect estimate.",
        "",
        "## Route table",
        "",
        "| Route | Domain | AUC delta | Tokens | Delta / 100k tokens | Outcome |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: str(item["case_id"])):
        lines.append(
            "| {case_id} | {domain} | {auc_delta_vs_same_initial_gp:+.4f} | "
            "{total_tokens:,} | {auc_delta_per_100k_tokens:+.4f} | {outcome} |".format(
                **row
            )
        )
    lines.extend(["", f"**Claim boundary:** {summary['claim_boundary']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def build(input_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_routes(input_path)
    summary = summarize(rows)
    summary.update(
        {
            "schema_version": "care2.astabench.internal_quality_cost_audit.v1",
            "input_path": str(input_path.relative_to(REPO_ROOT)),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        }
    )

    csv_path = output_dir / "route_quality_tokens.csv"
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "RESULTS.md"
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, summary, input_path, markdown_path)
    build_figure(rows, output_dir)

    for path in (
        csv_path,
        json_path,
        markdown_path,
        output_dir / "quality_token_audit.png",
        output_dir / "quality_token_audit.pdf",
    ):
        write_sha256(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.input.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
