#!/usr/bin/env python3
"""Summarize CARE 2.0 mechanism-control runs across source-target pairs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def scalar(summary: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def effect_fields(prefix: str, effect: dict[str, Any] | None) -> dict[str, Any]:
    composite = (effect or {}).get("composite", {})
    return {
        f"{prefix}_mean": composite.get("mean"),
        f"{prefix}_ci_low": composite.get("normal_95ci_low"),
        f"{prefix}_ci_high": composite.get("normal_95ci_high"),
        f"{prefix}_win_rate": composite.get("win_rate"),
        f"{prefix}_exact_seed_match_rate": (effect or {}).get(
            "exact_seed_match_rate"
        ),
    }


def summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    mechanism = summary.get("mechanism_attribution", {})
    selection = summary.get("selection", {})
    pairwise = summary.get("heldout_pairwise", {})
    strongest_mode = selection.get("target_anchor_mode")
    strongest = pairwise.get(strongest_mode, {})
    matched = pairwise.get("matched_target_only_llm", {})
    actions = mechanism.get("full_route_action_diagnostics", {})
    cost = summary.get("offline_selection_cost", {})
    row: dict[str, Any] = {
        "source_dataset": summary.get("source_dataset"),
        "target_dataset": summary.get("target_dataset"),
        "selected_mode": selection.get("selected_mode"),
        "selected_source_outcome_transfer": selection.get(
            "selected_source_outcome_transfer"
        ),
        "mechanism_classification": mechanism.get("classification"),
        "strongest_target_bo_mode": strongest_mode,
        "heldout_seed_count": summary.get("heldout_seed_count"),
        "calibration_seed_count": summary.get("calibration_seed_count"),
        "deployment_ready": selection.get("real_experiment_deployment_ready"),
        "selected_final_delta_vs_strongest_bo": scalar(
            strongest, "final_best", "mean"
        ),
        "selected_final_ci_low_vs_strongest_bo": scalar(
            strongest, "final_best", "normal_95ci_low"
        ),
        "selected_final_ci_high_vs_strongest_bo": scalar(
            strongest, "final_best", "normal_95ci_high"
        ),
        "selected_final_delta_vs_matched_llm": scalar(
            matched, "final_best", "mean"
        ),
        "source_initial_active_seed_rate": actions.get(
            "source_initial_active_seed_rate"
        ),
        "positive_transfer_mass_round_rate": actions.get(
            "positive_transfer_mass_round_rate"
        ),
        "post_initialization_action_change_rate": actions.get(
            "post_initialization_action_change_rate"
        ),
        "offline_gross_reveal_equivalents": cost.get(
            "gross_reveal_equivalents"
        ),
        "summary_path": str(path),
    }
    row.update(
        effect_fields(
            "warmstart_effect",
            mechanism.get("source_informed_initial_design_effect"),
        )
    )
    row.update(
        effect_fields(
            "post_initialization_effect",
            mechanism.get("post_initialization_source_outcome_effect"),
        )
    )
    row.update(
        effect_fields(
            "llm_increment",
            mechanism.get("llm_patch_increment_over_fixed_data_only"),
        )
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formatted(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def effect_cell(row: dict[str, Any], prefix: str) -> str:
    return (
        f"{formatted(row[f'{prefix}_mean'])} "
        f"[{formatted(row[f'{prefix}_ci_low'])}, "
        f"{formatted(row[f'{prefix}_ci_high'])}]"
    )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CARE 2.0 Mechanism-Control Summary",
        "",
        "All effects are paired held-out composite deltas with normal 95% "
        "confidence intervals. Positive values favor the left-hand route.",
        "",
        "| Source -> target | Deployed | vs strongest BO (final) | "
        "Warm-start effect | Post-init source-outcome effect | "
        "LLM increment vs fixed data-only | Action change |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pair = f"{row['source_dataset']} -> {row['target_dataset']}"
        bo = (
            f"{formatted(row['selected_final_delta_vs_strongest_bo'])} "
            f"[{formatted(row['selected_final_ci_low_vs_strongest_bo'])}, "
            f"{formatted(row['selected_final_ci_high_vs_strongest_bo'])}]"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    pair,
                    str(row["selected_mode"]),
                    bo,
                    effect_cell(row, "warmstart_effect"),
                    effect_cell(row, "post_initialization_effect"),
                    effect_cell(row, "llm_increment"),
                    formatted(row["post_initialization_action_change_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Warm-start effect compares the same target-only policy after "
            "replacing its initial design with the source-informed design.",
            "- Post-init source-outcome effect compares the full route with the "
            "same source-informed initialization but zero transfer mass after "
            "initialization.",
            "- LLM increment compares the full LLM patch with a deterministic "
            "equal-role data-only transfer patch.",
            "- These runs are offline replay model selection. A positive result "
            "does not make the calibration protocol wet-lab deployment ready.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob",
        default=str(ROOT / "outputs" / "runs" / "*_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = [Path(value) for value in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise SystemExit(f"No summary files matched: {args.input_glob}")
    rows = [summary_row(path) for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mechanism_controls.csv", rows)
    (args.output_dir / "mechanism_controls.json").write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "mechanism_controls.md", rows)
    print(f"Wrote {len(rows)} pair summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
