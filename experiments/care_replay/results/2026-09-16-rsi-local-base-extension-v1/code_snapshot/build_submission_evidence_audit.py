#!/usr/bin/env python3
"""Build a conservative, reproducible submission-evidence audit.

The input is the route-level output of ``build_opus_generalization_report.py``.
This audit intentionally separates retrospective development routes from the
post-freeze chemistry extension and avoids treating route counts as repeated
LLM trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any


SCHEMA_VERSION = "care.submission_evidence_audit/v1"
DELTA_FIELD = "online_auc_delta_vs_same_initial_gp"
TOLERANCE = 1e-9
PHASE_LABELS = {
    "development": "Retrospective development",
    "frozen_extension": "Post-freeze chemistry extension",
}
PHASE_COLORS = {
    "development": "#2F6B7C",
    "frozen_extension": "#C05A47",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No route rows found in {path}")
    required = {"phase", "case_id", "domain", DELTA_FIELD}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    count = len(values)
    bootstrap_means = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return (
        percentile(bootstrap_means, 0.025),
        percentile(bootstrap_means, 0.975),
    )


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign-test p-value, excluding ties."""

    trials = wins + losses
    if trials == 0:
        return None
    smaller = min(wins, losses)
    lower_tail = sum(
        math.comb(trials, successes)
        for successes in range(smaller + 1)
    ) / (2**trials)
    return min(1.0, 2.0 * lower_tail)


def leave_one_out_mean_range(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    estimates = [
        mean([*values[:index], *values[index + 1 :]])
        for index in range(len(values))
    ]
    return min(estimates), max(estimates)


def outcome(value: float) -> str:
    if value > TOLERANCE:
        return "win"
    if value < -TOLERANCE:
        return "loss"
    return "tie"


def summarize_values(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    wins = sum(value > TOLERANCE for value in values)
    ties = sum(abs(value) <= TOLERANCE for value in values)
    losses = sum(value < -TOLERANCE for value in values)
    ci_low, ci_high = bootstrap_mean_ci(
        values,
        bootstrap_samples,
        random.Random(bootstrap_seed),
    )
    loo_range = leave_one_out_mean_range(values)
    return {
        "route_count": len(values),
        "mean_auc_delta": round(mean(values), 6),
        "median_auc_delta": round(median(values), 6),
        "minimum_auc_delta": round(min(values), 6),
        "maximum_auc_delta": round(max(values), 6),
        "bootstrap_mean_95ci_low": round(ci_low, 6),
        "bootstrap_mean_95ci_high": round(ci_high, 6),
        "bootstrap_ci_excludes_zero": bool(
            len(values) > 1 and (ci_low > 0.0 or ci_high < 0.0)
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "nonloss_rate": round((wins + ties) / len(values), 6),
        "exact_two_sided_sign_test_p_excluding_ties": exact_two_sided_sign_p(
            wins, losses
        ),
        "leave_one_route_out_mean_low": (
            round(loo_range[0], 6) if loo_range is not None else None
        ),
        "leave_one_route_out_mean_high": (
            round(loo_range[1], 6) if loo_range is not None else None
        ),
    }


def build_audit(
    rows: Sequence[Mapping[str, str]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    route_rows: list[dict[str, Any]] = []
    by_phase: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in rows:
        phase = str(row["phase"])
        value = float(row[DELTA_FIELD])
        all_values.append(value)
        by_phase[phase].append(value)
        by_domain[str(row["domain"])].append(value)
        route_rows.append(
            {
                "phase": phase,
                "evidence_tier": (
                    "post_freeze_extension"
                    if phase == "frozen_extension"
                    else "retrospective_development"
                ),
                "case_id": str(row["case_id"]),
                "domain": str(row["domain"]),
                "target_task": str(row.get("target_task", "")),
                "online_auc_delta_vs_same_initial_gp": round(value, 6),
                "outcome": outcome(value),
            }
        )

    phase_statistics = {
        phase: summarize_values(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index + 1,
        )
        for index, (phase, values) in enumerate(sorted(by_phase.items()))
    }
    domain_statistics = {
        domain: summarize_values(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100 + index,
        )
        for index, (domain, values) in enumerate(sorted(by_domain.items()))
    }
    overall = summarize_values(
        all_values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    frozen = phase_statistics.get("frozen_extension")
    confirmatory_positive = bool(
        frozen
        and frozen["bootstrap_mean_95ci_low"] > 0.0
        and frozen["exact_two_sided_sign_test_p_excluding_ties"] is not None
        and frozen["exact_two_sided_sign_test_p_excluding_ties"] < 0.05
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "primary_question": (
            "Does the online LLM controller improve best-so-far AUC over the same "
            "LLM initial design followed by target-only GP-UCB under the same reveal budget?"
        ),
        "primary_comparator": "same_llm_initial_design_plus_target_only_gp_ucb",
        "unit_of_analysis": "source-target route",
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "method": "nonparametric percentile bootstrap over route-level deltas",
            "limitation": (
                "Routes are not repeated LLM trials and share task families; the interval "
                "is a descriptive route-level sensitivity analysis, not hierarchical inference."
            ),
        },
        "overall_descriptive": overall,
        "phases": phase_statistics,
        "domains": domain_statistics,
        "confirmatory_positive_transfer_supported": confirmatory_positive,
        "claim_boundary": {
            "supported": [
                "The LLM made an executable choice on every online round in this suite.",
                "Six of eleven routes improved best-so-far AUC relative to the matched same-initial GP comparator.",
                "The post-freeze chemistry extension produced three wins, one tie, and one loss.",
            ],
            "not_supported": [
                "Statistically significant general positive transfer across domains.",
                "Universal positive transfer or elimination of negative transfer.",
                "Superiority of the complete LLM system over the fixed initializer on unseen tasks.",
                "Independent wet-lab validation or model-weight self-improvement.",
            ],
            "recommended_manuscript_language": (
                "The controller produced route-specific positive transfer signals under "
                "matched budgets, including a majority-positive post-freeze chemistry "
                "extension, but current route-level uncertainty does not support a universal "
                "or statistically confirmatory cross-domain claim."
            ),
        },
        "submission_readiness": [
            {
                "criterion": "Matched target-only comparator",
                "status": "pass",
                "evidence": "Same initial observations and reveal budget; only the online controller differs.",
            },
            {
                "criterion": "Target-outcome leakage control",
                "status": "pass",
                "evidence": "Prompts expose target outcomes only after the selected experiment is revealed.",
            },
            {
                "criterion": "Post-freeze task extension",
                "status": "partial",
                "evidence": "Five chemistry routes were added after the high-authority controller was frozen.",
            },
            {
                "criterion": "Repeated stochastic LLM trials",
                "status": "fail",
                "evidence": "Each route currently has one online Opus trajectory for this evidence suite.",
            },
            {
                "criterion": "Confirmatory route-level significance",
                "status": "fail",
                "evidence": "Frozen-extension bootstrap interval crosses zero and the sign test is not significant.",
            },
            {
                "criterion": "Full-system superiority",
                "status": "fail",
                "evidence": "The complete system is less stable because outcome-blind LLM initial design can hurt performance.",
            },
            {
                "criterion": "Independent wet-lab validation",
                "status": "fail",
                "evidence": "The present evidence is finite-pool replay on real datasets, not a new physical campaign.",
            },
        ],
        "routes": route_rows,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )


def friendly_route(case_id: str) -> str:
    replacements = {
        "molecular_": "Molecular: ",
        "materials_": "Materials: ",
        "baumgartner_": "C-N: ",
        "reizman_": "Suzuki: ",
        "_to_": " -> ",
        "_": " ",
    }
    label = case_id
    for before, after in replacements.items():
        label = label.replace(before, after)
    return label


def svg_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_route_deltas_svg(routes: Sequence[Mapping[str, Any]], output: Path) -> None:
    ordered = sorted(
        routes,
        key=lambda row: (
            0 if row["phase"] == "frozen_extension" else 1,
            float(row["online_auc_delta_vs_same_initial_gp"]),
        ),
    )
    width = 1500
    height = 900
    label_x = 40
    plot_left = 620
    plot_right = 1425
    plot_top = 150
    plot_bottom = 820
    values = [float(row["online_auc_delta_vs_same_initial_gp"]) for row in ordered]
    lower = min(-3.0, min(values) - 0.5)
    upper = max(7.5, max(values) + 0.5)

    def x_position(value: float) -> float:
        return plot_left + (value - lower) / (upper - lower) * (plot_right - plot_left)

    zero_x = x_position(0.0)
    row_gap = (plot_bottom - plot_top) / max(1, len(ordered) - 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="40" y="52" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#24292F">Route-level effect is heterogeneous</text>',
        '<text x="40" y="84" font-family="Arial, sans-serif" font-size="17" fill="#5B6168">Post-freeze chemistry routes are red; development routes are teal. Values are AUC deltas against the same-initial target-only GP-UCB comparator.</text>',
    ]
    for tick in range(math.floor(lower), math.ceil(upper) + 1):
        x = x_position(float(tick))
        color = "#343A40" if tick == 0 else "#E2E4E7"
        dash = ' stroke-dasharray="8 7"' if tick == 0 else ""
        lines.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="835" stroke="{color}" stroke-width="{2 if tick == 0 else 1}"{dash}/>')
        lines.append(f'<text x="{x:.1f}" y="865" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#5B6168">{tick:+d}</text>')

    for index, row in enumerate(ordered):
        y = plot_top + index * row_gap
        value = float(row["online_auc_delta_vs_same_initial_gp"])
        color = PHASE_COLORS[str(row["phase"])]
        value_x = x_position(value)
        lines.extend([
            f'<text x="{label_x}" y="{y + 6:.1f}" font-family="Arial, sans-serif" font-size="16" fill="#30363D">{svg_text(friendly_route(str(row["case_id"])))}</text>',
            f'<line x1="{min(zero_x, value_x):.1f}" y1="{y:.1f}" x2="{max(zero_x, value_x):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="8" stroke-linecap="round" opacity="0.82"/>',
            f'<circle cx="{value_x:.1f}" cy="{y:.1f}" r="8" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>',
        ])
        anchor = "start" if value >= 0 else "end"
        text_x = value_x + (15 if value >= 0 else -15)
        lines.append(f'<text x="{text_x:.1f}" y="{y + 6:.1f}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="{color}">{value:+.2f}</text>')
    lines.extend([
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="895" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#30363D">Best-so-far AUC delta</text>',
        '</svg>',
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_phase_statistics_svg(audit: Mapping[str, Any], output: Path) -> None:
    labels = ["All routes", "Development", "Post-freeze chemistry"]
    summaries = [
        audit["overall_descriptive"],
        audit["phases"]["development"],
        audit["phases"]["frozen_extension"],
    ]
    colors = ["#3D3D3D", PHASE_COLORS["development"], PHASE_COLORS["frozen_extension"]]
    width = 1500
    height = 690
    effect_left = 330
    effect_right = 900
    count_left = 1080
    count_right = 1435
    plot_top = 230
    row_gap = 125
    effect_lower = -1.2
    effect_upper = 5.0

    def effect_x(value: float) -> float:
        return effect_left + (value - effect_lower) / (effect_upper - effect_lower) * (effect_right - effect_left)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="40" y="52" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#24292F">Current evidence is descriptive, not confirmatory</text>',
        '<text x="40" y="84" font-family="Arial, sans-serif" font-size="17" fill="#5B6168">Route-level bootstrap intervals cross zero; repeated LLM trajectories and independent campaigns remain necessary.</text>',
        '<text x="330" y="150" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#30363D">Mean effect and bootstrap 95% interval</text>',
        '<text x="1080" y="150" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#30363D">Win / tie / loss</text>',
    ]
    for tick in range(-1, 6):
        x = effect_x(float(tick))
        color = "#343A40" if tick == 0 else "#E2E4E7"
        dash = ' stroke-dasharray="8 7"' if tick == 0 else ""
        lines.append(f'<line x1="{x:.1f}" y1="185" x2="{x:.1f}" y2="530" stroke="{color}" stroke-width="{2 if tick == 0 else 1}"{dash}/>')
        lines.append(f'<text x="{x:.1f}" y="562" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#5B6168">{tick:+d}</text>')

    max_count = max(int(summary["route_count"]) for summary in summaries)
    for index, (label, summary, color) in enumerate(zip(labels, summaries, colors)):
        y = plot_top + index * row_gap
        estimate = float(summary["mean_auc_delta"])
        low = float(summary["bootstrap_mean_95ci_low"])
        high = float(summary["bootstrap_mean_95ci_high"])
        low_x = effect_x(low)
        high_x = effect_x(high)
        estimate_x = effect_x(estimate)
        lines.extend([
            f'<text x="40" y="{y + 6:.1f}" font-family="Arial, sans-serif" font-size="18" fill="#30363D">{svg_text(label)}</text>',
            f'<line x1="{low_x:.1f}" y1="{y:.1f}" x2="{high_x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="5"/>',
            f'<line x1="{low_x:.1f}" y1="{y - 11:.1f}" x2="{low_x:.1f}" y2="{y + 11:.1f}" stroke="{color}" stroke-width="3"/>',
            f'<line x1="{high_x:.1f}" y1="{y - 11:.1f}" x2="{high_x:.1f}" y2="{y + 11:.1f}" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{estimate_x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>',
            f'<text x="{min(960, high_x + 18):.1f}" y="{y + 6:.1f}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="{color}">{estimate:+.2f}</text>',
        ])
        start_x = count_left
        for key, segment_color in (("wins", "#2F855A"), ("ties", "#8B8D98"), ("losses", "#C53030")):
            count = int(summary[key])
            segment_width = (count / max_count) * (count_right - count_left)
            if count:
                lines.append(f'<rect x="{start_x:.1f}" y="{y - 22:.1f}" width="{segment_width:.1f}" height="44" fill="{segment_color}"/>')
                lines.append(f'<text x="{start_x + segment_width / 2:.1f}" y="{y + 6:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#FFFFFF">{count}</text>')
            start_x += segment_width
    lines.extend([
        '<rect x="1080" y="575" width="18" height="18" fill="#2F855A"/><text x="1108" y="590" font-family="Arial, sans-serif" font-size="15" fill="#30363D">Win</text>',
        '<rect x="1180" y="575" width="18" height="18" fill="#8B8D98"/><text x="1208" y="590" font-family="Arial, sans-serif" font-size="15" fill="#30363D">Tie</text>',
        '<rect x="1270" y="575" width="18" height="18" fill="#C53030"/><text x="1298" y="590" font-family="Arial, sans-serif" font-size="15" fill="#30363D">Loss</text>',
        '<text x="40" y="650" font-family="Arial, sans-serif" font-size="15" fill="#5B6168">Intervals resample source-target routes; they do not replace repeated LLM trajectories or independent wet-lab campaigns.</text>',
        '</svg>',
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def markdown_report(audit: Mapping[str, Any], metrics_path: Path) -> str:
    overall = audit["overall_descriptive"]
    frozen = audit["phases"]["frozen_extension"]
    readiness_rows = [
        f"| {item['criterion']} | {item['status'].upper()} | {item['evidence']} |"
        for item in audit["submission_readiness"]
    ]
    route_rows = [
        (
            f"| {row['evidence_tier']} | {row['case_id']} | {row['domain']} | "
            f"{row['online_auc_delta_vs_same_initial_gp']:+.4f} | {row['outcome']} |"
        )
        for row in audit["routes"]
    ]
    return "\n".join(
        [
            "# CARE 2.0 Submission Evidence Audit",
            "",
            "## Primary estimand",
            "",
            audit["primary_question"],
            "",
            "The matched comparator receives the same LLM-generated initial observations and the same target-evaluation budget. The contrast therefore isolates the online LLM controller from the initial-design effect.",
            "",
            "## Current result",
            "",
            (
                f"Across all {overall['route_count']} routes, the mean route-level AUC delta is "
                f"{overall['mean_auc_delta']:+.4f} with a descriptive bootstrap 95% interval "
                f"[{overall['bootstrap_mean_95ci_low']:+.4f}, {overall['bootstrap_mean_95ci_high']:+.4f}]. "
                f"The routes contain {overall['wins']} wins, {overall['ties']} ties, and "
                f"{overall['losses']} losses; the exact two-sided sign-test p-value excluding "
                f"ties is {overall['exact_two_sided_sign_test_p_excluding_ties']:.4f}."
            ),
            (
                f"The post-freeze chemistry extension contains {frozen['wins']} wins, "
                f"{frozen['ties']} tie, and {frozen['losses']} loss. Its mean delta is "
                f"{frozen['mean_auc_delta']:+.4f}, bootstrap 95% interval "
                f"[{frozen['bootstrap_mean_95ci_low']:+.4f}, {frozen['bootstrap_mean_95ci_high']:+.4f}], "
                f"and exact two-sided sign-test p={frozen['exact_two_sided_sign_test_p_excluding_ties']:.4f}."
            ),
            "",
            "**Decision:** the current result is a route-specific, majority-positive signal. It is not a statistically confirmatory claim of general cross-domain transfer.",
            "",
            "## Claim boundary",
            "",
            f"> {audit['claim_boundary']['recommended_manuscript_language']}",
            "",
            "Claims currently supported:",
            "",
            *[f"- {item}" for item in audit["claim_boundary"]["supported"]],
            "",
            "Claims not currently supported:",
            "",
            *[f"- {item}" for item in audit["claim_boundary"]["not_supported"]],
            "",
            "## Submission readiness",
            "",
            "| Criterion | Status | Evidence |",
            "|---|---|---|",
            *readiness_rows,
            "",
            "## Route-level evidence",
            "",
            "| Evidence tier | Route | Domain | Online AUC delta | Outcome |",
            "|---|---|---|---:|---|",
            *route_rows,
            "",
            "## Statistical boundary",
            "",
            "The bootstrap resamples source-target routes and assumes they are exchangeable. Several routes share task families and data-generation structure, so this interval is a sensitivity analysis rather than a hierarchical population estimate. Each route currently contributes one online Opus trajectory; repeated stochastic LLM runs are required to estimate within-route variance.",
            "",
            "## Minimum next evidence package",
            "",
            "1. Freeze the controller, prompt, candidate-menu policy, and analysis code before any new target is revealed.",
            "2. Run repeated LLM trajectories per route with predeclared seeds and matched GP controls.",
            "3. Add at least one fresh task family not used during prompt or controller development.",
            "4. Evaluate the complete system and the online-controller increment separately.",
            "5. Run a paired wet-lab campaign or an independently held prospective campaign before claiming practical scientific acceleration.",
            "6. Report all negative routes, abstentions, model failures, token cost, and experiment count.",
            "",
            "## Provenance",
            "",
            f"Source route metrics: `{metrics_path}`",
            "",
            "Generated vector figures: `route_level_auc_deltas.svg` and `phase_evidence_summary.svg`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    args = parser.parse_args()

    rows = read_rows(args.metrics)
    audit = build_audit(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = args.output_dir / "submission_evidence.json"
    write_json(audit_path, audit)
    route_path = args.output_dir / "route_evidence.csv"
    write_csv(route_path, audit["routes"])
    phase_rows = [
        {"phase": "all_routes", **audit["overall_descriptive"]},
        *(
            {"phase": phase, **summary}
            for phase, summary in audit["phases"].items()
        ),
    ]
    phase_path = args.output_dir / "phase_statistics.csv"
    write_csv(phase_path, phase_rows)

    route_figure = args.output_dir / "route_level_auc_deltas.svg"
    phase_figure = args.output_dir / "phase_evidence_summary.svg"
    write_route_deltas_svg(audit["routes"], route_figure)
    write_phase_statistics_svg(audit, phase_figure)

    report_path = args.output_dir / "EVIDENCE_AUDIT.md"
    report_path.write_text(
        markdown_report(audit, args.metrics),
        encoding="utf-8",
    )
    outputs = [
        audit_path,
        route_path,
        phase_path,
        route_figure,
        phase_figure,
        report_path,
    ]
    for path in outputs:
        write_sha256(path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
