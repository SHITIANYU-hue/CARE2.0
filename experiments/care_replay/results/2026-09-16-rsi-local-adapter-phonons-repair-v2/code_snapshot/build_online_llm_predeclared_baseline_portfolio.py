#!/usr/bin/env python3
"""Build a same-start, predeclared-comparator audit across the full LLM portfolio."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import build_online_llm_classical_baseline_audit as matched


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "care.online_llm_predeclared_baseline_portfolio/v1"
TOLERANCE = 1e-9
ROUTE_LABELS = {
    "molecular_freesolv_to_esol": "FreeSolv → ESOL",
    "molecular_freesolv_to_lipophilicity": "FreeSolv → Lipophilicity",
    "materials_expt_gap_to_dielectric": "Expt. gap → dielectric",
    "materials_phonons_to_bulk_modulus": "Phonons → bulk modulus",
    "materials_dielectric_to_jdft2d": "Dielectric → JDFT2D",
    "materials_expt_gap_to_mp_gap": "Expt. gap → MP gap",
    "baumgartner_aniline_to_phenethylamine_alphos": "C–N: aniline → phenethylamine (AlPhos)",
    "baumgartner_aniline_to_benzamide_tbuxphos": "C–N: aniline → benzamide (tBuXPhos)",
    "baumgartner_aniline_to_phenethylamine_tbubrettphos": "C–N: aniline → phenethylamine (tBuBrettPhos)",
    "baumgartner_benzamide_tbubrettphos_to_alphos": "C–N: tBuBrettPhos → AlPhos (benzamide)",
    "reizman_cases_123_to_case4": "Suzuki cases 1–3 → case 4",
    "molecular_lipophilicity_to_freesolv": "Lipophilicity → FreeSolv",
    "materials_bulk_to_shear_modulus": "Bulk → shear modulus",
    "materials_phonons_to_perovskites": "Phonons → perovskites",
    "materials_expt_gap_to_steels": "Expt. gap → steel yield strength",
    "molecular_freesolv_to_bace": "FreeSolv → BACE",
    "materials_perovskites_to_mp_e_form": "Perovskites → MP formation energy",
}
DOMAIN_LABELS = {
    "molecular_property": "Molecular properties",
    "materials_property": "Materials",
    "reaction_optimization_cn": "C–N optimization",
    "reaction_optimization_suzuki": "Suzuki optimization",
}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    low = estimates[max(0, math.floor(0.025 * (samples - 1)))]
    high = estimates[min(samples - 1, math.ceil(0.975 * (samples - 1)))]
    return low, high


def exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    wins = sum(value > TOLERANCE for value in values)
    losses = sum(value < -TOLERANCE for value in values)
    ties = len(values) - wins - losses
    n = wins + losses
    if n == 0:
        p_value = None
    else:
        tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {"wins": wins, "ties": ties, "losses": losses, "two_sided_p": p_value}


def comparison(report: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    for row in report["comparisons"]:
        if row["method"] == method:
            return row
    raise ValueError(f"{report['case_id']} does not contain comparator {method}")


def load_portfolio_config(path: Path) -> dict[str, Any]:
    config = matched.load_json(path)
    base_path = config.get("base_config")
    if not base_path:
        return config
    base = matched.load_json(ROOT / str(base_path))
    protocol = dict(base["protocol"])
    protocol.update(config.get("protocol", {}))
    return {
        "protocol": protocol,
        "cases": [*base["cases"], *config.get("cases", [])],
    }


def summarize(values: Sequence[float], protocol: Mapping[str, Any], seed_offset: int) -> dict[str, Any]:
    low, high = bootstrap_mean_ci(
        values,
        samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["bootstrap_seed"]) + seed_offset,
    )
    signs = exact_sign_test(values)
    return {
        "route_count": len(values),
        "equal_route_mean_auc_delta": round(mean(values), 6),
        "route_bootstrap_95ci_low": round(low, 6),
        "route_bootstrap_95ci_high": round(high, 6),
        "route_wins": signs["wins"],
        "route_ties": signs["ties"],
        "route_losses": signs["losses"],
        "negative_transfer_route_rate": round(signs["losses"] / len(values), 6),
        "two_sided_route_sign_test_p": signs["two_sided_p"],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{matched.sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def plot_portfolio(rows: Sequence[Mapping[str, Any]], output_root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#30363B",
            "axes.linewidth": 0.8,
            "text.color": "#20262B",
        }
    )
    figure, axis = plt.subplots(figsize=(11.8, 7.4))
    y_values = list(range(len(rows)))
    axis.axvline(0.0, color="#A23B45", linestyle="--", linewidth=1.2, zorder=1)
    for index, row in enumerate(rows):
        target_delta = float(row["llm_minus_target_gp_auc"])
        transfer_delta = float(row["llm_minus_predeclared_transfer_auc"])
        axis.plot(
            [target_delta, transfer_delta],
            [index, index],
            color="#C9CFD3",
            linewidth=1.5,
            zorder=1,
        )
        axis.scatter(target_delta, index, s=62, color="#2775B6", edgecolor="white", linewidth=0.8, zorder=3)
        axis.scatter(transfer_delta, index, s=62, marker="s", color="#E28735", edgecolor="white", linewidth=0.8, zorder=3)
    axis.set_yticks(y_values)
    axis.set_yticklabels([ROUTE_LABELS.get(str(row["case_id"]), str(row["case_id"])) for row in rows])
    axis.invert_yaxis()
    axis.grid(axis="x", color="#E1E5E8", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Δ best-so-far AUC (online LLM − comparator)")
    axis.set_title(
        f"{len(rows)}-route portfolio: same-start comparison to rule-fixed baselines",
        loc="left",
    )
    axis.scatter([], [], s=62, color="#2775B6", label="vs target-only GP-UCB")
    axis.scatter([], [], s=62, marker="s", color="#E28735", label="vs rule-fixed RGPE")
    axis.legend(loc="lower right", frameon=False, ncol=2)

    domain_starts: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        domain_starts[str(row["domain"])].append(index)
    for indices in domain_starts.values():
        axis.axhspan(min(indices) - 0.46, max(indices) + 0.46, color="#F5F7F8", alpha=0.55, zorder=0)
    for left, right in zip(rows, rows[1:]):
        if left["domain"] != right["domain"]:
            axis.axhline(rows.index(right) - 0.5, color="#B8C0C5", linewidth=0.8)

    figure.text(
        0.01,
        0.01,
        "Real online LLM trajectories; exact same initial candidates, target pool and 10-reveal budget. One completed trajectory per route. Error bars are intentionally absent: this is retrospective route coverage, not repeated-run inference.",
        fontsize=8.2,
        color="#5D6870",
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    outputs = [
        output_root / "predeclared_baseline_portfolio.png",
        output_root / "predeclared_baseline_portfolio.pdf",
        output_root / "predeclared_baseline_portfolio.svg",
    ]
    for output in outputs:
        figure.savefig(output, dpi=220, bbox_inches="tight")
        if output.suffix == ".svg":
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
        output.with_suffix(output.suffix + ".sha256").write_text(
            f"{matched.sha256(output)}  {output.name}\n",
            encoding="utf-8",
        )
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_portfolio_config(args.config)
    protocol = config["protocol"]
    reports = []
    rows = []
    for case in config["cases"]:
        report, _ = matched.run_case(case, protocol, args.output_root / "baselines")
        source_count = len(report["source_tasks"])
        transfer_method = (
            protocol["primary_comparators"]["single_source_classical_transfer"]
            if source_count == 1
            else protocol["primary_comparators"]["multisource_classical_transfer"]
        )
        target = comparison(report, protocol["primary_comparators"]["no_transfer"])
        transfer = comparison(report, transfer_method)
        row = {
            "case_id": case["case_id"],
            "domain": case["domain"],
            "evidence_tier": case["evidence_tier"],
            "source_task_count": source_count,
            "predeclared_transfer_method": transfer_method,
            "online_llm_auc": report["online_llm"]["best_so_far_auc"],
            "online_llm_final": report["online_llm"]["final_best"],
            "target_gp_auc": target["baseline_best_so_far_auc"],
            "predeclared_transfer_auc": transfer["baseline_best_so_far_auc"],
            "llm_minus_target_gp_auc": target["online_llm_minus_baseline_auc"],
            "llm_minus_predeclared_transfer_auc": transfer["online_llm_minus_baseline_auc"],
            "llm_minus_target_gp_final": target["online_llm_minus_baseline_final"],
            "llm_minus_predeclared_transfer_final": transfer["online_llm_minus_baseline_final"],
        }
        report["domain"] = case["domain"]
        report["evidence_tier"] = case["evidence_tier"]
        report["predeclared_comparators"] = {
            "no_transfer": target,
            "classical_transfer": transfer,
        }
        reports.append(report)
        rows.append(row)

    no_transfer_values = [float(row["llm_minus_target_gp_auc"]) for row in rows]
    transfer_values = [float(row["llm_minus_predeclared_transfer_auc"]) for row in rows]
    strongest_values = [
        float(report["strongest_realized_baseline_posthoc"]["online_llm_minus_baseline_auc"])
        for report in reports
    ]
    domains: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        domains[str(row["domain"])].append(row)
    domain_summary = {}
    for domain, domain_rows in domains.items():
        domain_summary[domain] = {
            "label": DOMAIN_LABELS.get(domain, domain),
            "route_count": len(domain_rows),
            "mean_llm_minus_target_gp_auc": round(mean(float(row["llm_minus_target_gp_auc"]) for row in domain_rows), 6),
            "mean_llm_minus_predeclared_transfer_auc": round(mean(float(row["llm_minus_predeclared_transfer_auc"]) for row in domain_rows), 6),
        }

    csv_path = args.output_root / "route_comparisons.csv"
    write_csv(csv_path, rows)
    figures = plot_portfolio(rows, args.output_root)
    route_count = len(rows)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "config_path": str(args.config),
        "config_sha256": matched.sha256(args.config),
        "completed_routes": route_count,
        "primary_no_transfer_comparison": summarize(no_transfer_values, protocol, 0),
        "primary_classical_transfer_comparison": summarize(transfer_values, protocol, 1),
        "posthoc_strongest_realized_baseline_diagnostic": summarize(strongest_values, protocol, 2),
        "domain_summary": domain_summary,
        "route_reports": reports,
        "route_table": rows,
        "figure_files": [
            {
                "path": str(path),
                "sha256": matched.sha256(path),
                "source_data": str(csv_path),
            }
            for path in figures
        ],
        "claim_boundary": (
            "This audit covers every route declared in the analysis configuration and applies one comparator rule uniformly. "
            "However, the online LLM trajectories were observed before this audit, each route has one completed trajectory, and route-bootstrap intervals quantify benchmark-route variation rather than LLM stochastic variation. "
            "Confirmatory claims require the already frozen 30-replicate protocol or a prospectively collected wet-lab test."
        ),
    }
    matched.write_json(args.output_root / "aggregate.json", aggregate)
    target_summary = aggregate["primary_no_transfer_comparison"]
    transfer_summary = aggregate["primary_classical_transfer_comparison"]
    strongest_summary = aggregate["posthoc_strongest_realized_baseline_diagnostic"]
    results_lines = [
        "# Online LLM portfolio versus rule-fixed baselines",
        "",
        f"This audit replays all {route_count} routes declared in the analysis configuration from the exact online-LLM initial candidates, target pool and 10-reveal budget.",
        "",
        "## Aggregate results",
        "",
        "| Comparator | Mean AUC delta | Route-bootstrap 95% CI | Win / tie / loss | Route sign-test p |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Target-only GP-UCB | {target_summary['equal_route_mean_auc_delta']:+.3f} | "
            f"[{target_summary['route_bootstrap_95ci_low']:+.3f}, {target_summary['route_bootstrap_95ci_high']:+.3f}] | "
            f"{target_summary['route_wins']} / {target_summary['route_ties']} / {target_summary['route_losses']} | "
            f"{target_summary['two_sided_route_sign_test_p']:.4f} |"
        ),
        (
            f"| Rule-fixed RGPE | {transfer_summary['equal_route_mean_auc_delta']:+.3f} | "
            f"[{transfer_summary['route_bootstrap_95ci_low']:+.3f}, {transfer_summary['route_bootstrap_95ci_high']:+.3f}] | "
            f"{transfer_summary['route_wins']} / {transfer_summary['route_ties']} / {transfer_summary['route_losses']} | "
            f"{transfer_summary['two_sided_route_sign_test_p']:.4f} |"
        ),
        (
            f"| Strongest realized baseline (post-hoc diagnostic) | {strongest_summary['equal_route_mean_auc_delta']:+.3f} | "
            f"[{strongest_summary['route_bootstrap_95ci_low']:+.3f}, {strongest_summary['route_bootstrap_95ci_high']:+.3f}] | "
            f"{strongest_summary['route_wins']} / {strongest_summary['route_ties']} / {strongest_summary['route_losses']} | "
            f"{strongest_summary['two_sided_route_sign_test_p']:.4f} |"
        ),
        "",
        "## Interpretation",
        "",
        (
            f"The online LLM controller has an equal-route mean of {target_summary['equal_route_mean_auc_delta']:+.3f} against target-only GP-UCB, "
            f"with {target_summary['route_wins']} wins, {target_summary['route_ties']} ties and {target_summary['route_losses']} losses. "
            "The route-bootstrap interval and exact route sign test must be interpreted together because the portfolio is small and the effect is heterogeneous."
        ),
        "",
        (
            f"Against the rule-fixed classical transfer comparator, the equal-route mean is {transfer_summary['equal_route_mean_auc_delta']:+.3f}; "
            f"the interval is [{transfer_summary['route_bootstrap_95ci_low']:+.3f}, {transfer_summary['route_bootstrap_95ci_high']:+.3f}] "
            f"and {transfer_summary['route_losses']} routes lose. The evidence does not establish stable superiority over classical transfer BO."
        ),
        "",
        "## Claim boundary",
        "",
        str(aggregate["claim_boundary"]),
        "",
    ]
    results_path = args.output_root / "RESULTS.md"
    results_path.write_text("\n".join(results_lines), encoding="utf-8")
    results_path.with_suffix(results_path.suffix + ".sha256").write_text(
        f"{matched.sha256(results_path)}  {results_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
