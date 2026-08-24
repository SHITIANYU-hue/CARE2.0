#!/usr/bin/env python3
"""Compare audited LLM trajectories without overstating repeatability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_VERSION = "care.llm_trajectory_stability_audit/v1"
EARLY_FIELD = "online_auc_delta_vs_same_initial_gp"
REPEAT_FIELD = "auc_delta_vs_same_initial_gp"
TOLERANCE = 1e-9


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return rows


def sign(value: float) -> str:
    if value > TOLERANCE:
        return "positive"
    if value < -TOLERANCE:
        return "negative"
    return "tie"


def outcome_counts(values: Sequence[float]) -> dict[str, int]:
    labels = [sign(value) for value in values]
    return {
        "wins": labels.count("positive"),
        "ties": labels.count("tie"),
        "losses": labels.count("negative"),
    }


def build_audit(
    early_rows: Sequence[Mapping[str, str]],
    repeat_rows: Sequence[Mapping[str, str]],
    second_repeat_rows: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    early = {str(row["case_id"]): row for row in early_rows}
    repeat = {str(row["case_id"]): row for row in repeat_rows}
    if set(early) != set(repeat):
        missing_repeat = sorted(set(early) - set(repeat))
        missing_early = sorted(set(repeat) - set(early))
        raise ValueError(
            "Route sets do not match: "
            f"missing_repeat={missing_repeat}, missing_early={missing_early}"
        )
    second_repeat = {str(row["case_id"]): row for row in second_repeat_rows}
    unknown_second = sorted(set(second_repeat) - set(early))
    if unknown_second:
        raise ValueError(f"Second-repeat routes are not in the paired audit: {unknown_second}")

    routes: list[dict[str, Any]] = []
    for case_id, early_row in early.items():
        repeat_row = repeat[case_id]
        early_value = float(early_row[EARLY_FIELD])
        repeat_value = float(repeat_row[REPEAT_FIELD])
        early_sign = sign(early_value)
        repeat_sign = sign(repeat_value)
        strict_flip = {early_sign, repeat_sign} == {"positive", "negative"}
        second_row = second_repeat.get(case_id)
        second_value = (
            float(second_row[REPEAT_FIELD]) if second_row is not None else None
        )
        routes.append(
            {
                "case_id": case_id,
                "domain": str(early_row["domain"]),
                "evidence_tier": str(early_row["evidence_tier"]),
                "early_audit_auc_delta": round(early_value, 6),
                "frozen_repeat_auc_delta": round(repeat_value, 6),
                "second_frozen_repeat_auc_delta": (
                    round(second_value, 6) if second_value is not None else None
                ),
                "early_sign": early_sign,
                "frozen_repeat_sign": repeat_sign,
                "exact_sign_agreement": early_sign == repeat_sign,
                "strict_sign_flip": strict_flip,
                "absolute_change": round(abs(repeat_value - early_value), 6),
            }
        )

    early_values = [float(row["early_audit_auc_delta"]) for row in routes]
    repeat_values = [float(row["frozen_repeat_auc_delta"]) for row in routes]
    strict_flips = sum(bool(row["strict_sign_flip"]) for row in routes)
    exact_agreements = sum(bool(row["exact_sign_agreement"]) for row in routes)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_of_analysis": "source-target route",
        "comparison": (
            "early audited online-LLM call versus the first trajectory under the "
            "frozen repeated-confirmation protocol"
        ),
        "route_count": len(routes),
        "exact_sign_agreement_count": exact_agreements,
        "strict_sign_flip_count": strict_flips,
        "strict_sign_flip_rate": round(strict_flips / len(routes), 6),
        "mean_absolute_auc_change": round(
            mean(abs(after - before) for before, after in zip(early_values, repeat_values)),
            6,
        ),
        "early_mean_auc_delta": round(mean(early_values), 6),
        "frozen_repeat_mean_auc_delta": round(mean(repeat_values), 6),
        "early_outcomes": outcome_counts(early_values),
        "frozen_repeat_outcomes": outcome_counts(repeat_values),
        "second_frozen_repeat_route_count": len(second_repeat),
        "claim_decision": "descriptive_only_incomplete_frozen_repetition",
        "limitations": [
            "The early audited calls predate the repeated-confirmation lock and are not interchangeable confirmatory replicates.",
            "Each route currently has only one completed trajectory under the v1 frozen protocol.",
            "The v2 protocol currently contributes a second frozen trajectory for one route only.",
            "No route-level confidence interval is estimable from these completed trajectories.",
        ],
        "routes": routes,
    }


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


def write_stability_svg(audit: Mapping[str, Any], output: Path) -> None:
    routes = sorted(
        audit["routes"],
        key=lambda row: (
            0 if row["evidence_tier"] == "post_freeze_extension" else 1,
            str(row["case_id"]),
        ),
    )
    values = [
        float(value)
        for row in routes
        for value in (
            row["early_audit_auc_delta"],
            row["frozen_repeat_auc_delta"],
            row["second_frozen_repeat_auc_delta"],
        )
        if value is not None
    ]
    lower = min(-3.0, min(values) - 0.6)
    upper = max(7.5, max(values) + 0.6)
    width = 1500
    height = 990
    plot_left = 610
    plot_right = 1405
    plot_top = 205
    plot_bottom = 850

    def x(value: float) -> float:
        return plot_left + (value - lower) / (upper - lower) * (plot_right - plot_left)

    row_gap = (plot_bottom - plot_top) / max(1, len(routes) - 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="40" y="52" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#24292F">Single-call route direction is not yet stable</text>',
        (
            f'<text x="40" y="88" font-family="Arial, sans-serif" font-size="19" fill="#C53030">'
            f'{audit["strict_sign_flip_count"]} of {audit["route_count"]} routes reversed sign between the early audit and first frozen repeat.</text>'
        ),
        '<circle cx="45" cy="132" r="8" fill="#FFFFFF" stroke="#343A40" stroke-width="3"/>',
        '<text x="65" y="138" font-family="Arial, sans-serif" font-size="16" fill="#30363D">Early audited call</text>',
        '<circle cx="265" cy="132" r="8" fill="#2F6B7C"/>',
        '<text x="285" y="138" font-family="Arial, sans-serif" font-size="16" fill="#30363D">First frozen repeat</text>',
        '<polygon points="500,121 511,132 500,143 489,132" fill="#D68A2F"/>',
        '<text x="520" y="138" font-family="Arial, sans-serif" font-size="16" fill="#30363D">Second frozen repeat (Suzuki only)</text>',
    ]
    for tick in range(int(lower), int(upper) + 1):
        tick_x = x(float(tick))
        color = "#343A40" if tick == 0 else "#E2E4E7"
        dash = ' stroke-dasharray="8 7"' if tick == 0 else ""
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="170" x2="{tick_x:.1f}" y2="875" '
            f'stroke="{color}" stroke-width="{2 if tick == 0 else 1}"{dash}/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="905" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="15" fill="#5B6168">{tick:+d}</text>'
        )

    for index, row in enumerate(routes):
        y = plot_top + index * row_gap
        before = float(row["early_audit_auc_delta"])
        after = float(row["frozen_repeat_auc_delta"])
        line_color = "#C53030" if row["strict_sign_flip"] else "#9AA0A6"
        lines.extend(
            [
                f'<text x="40" y="{y + 6:.1f}" font-family="Arial, sans-serif" font-size="16" fill="#30363D">{svg_text(friendly_route(str(row["case_id"])))}</text>',
                f'<line x1="{x(before):.1f}" y1="{y:.1f}" x2="{x(after):.1f}" y2="{y:.1f}" stroke="{line_color}" stroke-width="5" opacity="0.85"/>',
                f'<circle cx="{x(before):.1f}" cy="{y:.1f}" r="8" fill="#FFFFFF" stroke="#343A40" stroke-width="3"/>',
                f'<circle cx="{x(after):.1f}" cy="{y:.1f}" r="8" fill="#2F6B7C" stroke="#FFFFFF" stroke-width="2"/>',
            ]
        )
        second = row["second_frozen_repeat_auc_delta"]
        if second is not None:
            second_x = x(float(second))
            points = (
                f"{second_x:.1f},{y - 11:.1f} {second_x + 11:.1f},{y:.1f} "
                f"{second_x:.1f},{y + 11:.1f} {second_x - 11:.1f},{y:.1f}"
            )
            lines.append(f'<polygon points="{points}" fill="#D68A2F" stroke="#FFFFFF" stroke-width="2"/>')
        if row["strict_sign_flip"]:
            lines.append(
                f'<text x="1435" y="{y + 6:.1f}" font-family="Arial, sans-serif" '
                f'font-size="15" font-weight="700" fill="#C53030">SIGN FLIP</text>'
            )

    lines.extend(
        [
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="935" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#30363D">Best-so-far AUC delta vs same-initial target-only GP-UCB</text>',
            '<text x="40" y="972" font-family="Arial, sans-serif" font-size="15" fill="#5B6168">Descriptive audit only: early calls predate the confirmation lock; no route-level confidence interval is estimable before repeated frozen trajectories complete.</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sha256(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def markdown_report(audit: Mapping[str, Any]) -> str:
    early = audit["early_outcomes"]
    repeat = audit["frozen_repeat_outcomes"]
    return "\n".join(
        [
            "# LLM trajectory stability audit",
            "",
            "## Result",
            "",
            (
                f"The early audited calls and first frozen repeat both contain "
                f"{early['wins']} wins, {early['ties']} ties, and {early['losses']} losses. "
                f"However, {audit['strict_sign_flip_count']} of {audit['route_count']} matched "
                "routes reverse sign, so the unchanged aggregate count hides route-level instability."
            ),
            (
                f"The mean AUC delta changes from {audit['early_mean_auc_delta']:+.4f} to "
                f"{audit['frozen_repeat_mean_auc_delta']:+.4f}, and the mean absolute paired "
                f"change is {audit['mean_absolute_auc_change']:.4f}. These values are descriptive only."
            ),
            "",
            "## Interpretation",
            "",
            "This audit does not test whether the frozen controller has a positive population effect. The early calls predate the confirmation lock, and only one frozen v1 trajectory is complete per route. The result instead demonstrates why single-call route labels and route-resampled confidence intervals are insufficient for a stochastic LLM controller.",
            "",
            "The Suzuki route currently has one additional v2 frozen trajectory. Its three observed AUC deltas are +0.28 (early audit), +1.38 (v1 frozen trajectory), and +0.32 (v2 frozen trajectory). Direction is consistent, but magnitude is not yet precisely estimated.",
            "",
            "## Claim boundary",
            "",
            "No confirmatory transfer claim is evaluated until the predeclared repeated-trajectory protocol is complete. Infrastructure failures remain in the audit log and are excluded from scientific outcomes.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-route-csv", type=Path, required=True)
    parser.add_argument("--frozen-v1-csv", type=Path, required=True)
    parser.add_argument("--frozen-v2-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    early_rows = read_csv(
        args.early_route_csv,
        {"case_id", "domain", "evidence_tier", EARLY_FIELD},
    )
    repeat_rows = read_csv(
        args.frozen_v1_csv,
        {"case_id", "domain", "evidence_tier", REPEAT_FIELD},
    )
    second_rows = (
        read_csv(args.frozen_v2_csv, {"case_id", REPEAT_FIELD})
        if args.frozen_v2_csv
        else []
    )
    audit = build_audit(early_rows, repeat_rows, second_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "stability_audit.json",
        args.output_dir / "paired_route_stability.csv",
        args.output_dir / "paired_route_stability.svg",
        args.output_dir / "RESULTS.md",
    ]
    write_json(outputs[0], audit)
    write_csv(outputs[1], audit["routes"])
    write_stability_svg(audit, outputs[2])
    outputs[3].write_text(markdown_report(audit), encoding="utf-8")
    for output in outputs:
        write_sha256(output)
    print(json.dumps({key: audit[key] for key in (
        "route_count",
        "strict_sign_flip_count",
        "early_mean_auc_delta",
        "frozen_repeat_mean_auc_delta",
        "claim_decision",
    )}, indent=2))


if __name__ == "__main__":
    main()
