#!/usr/bin/env python3
"""Run and summarize a frozen repeated online-LLM confirmation protocol."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import html
import io
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence

import numpy as np

import run_online_llm_scientist as online


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "care.repeated_online_confirmation/v1"
TOLERANCE = 1e-9


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def replicate_ids(suite: Mapping[str, Any]) -> list[int]:
    protocol = suite["protocol"]
    start = int(protocol["replicate_start"])
    count = int(protocol["replicates_per_route"])
    return list(range(start, start + count))


def validate_suite(
    suite: Mapping[str, Any], *, check_paths: bool = True
) -> None:
    protocol = suite.get("protocol", {})
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("Repeated confirmation requires a frozen protocol.")
    if int(protocol.get("replicates_per_route", 0)) < 2:
        raise ValueError("At least two trajectories per route are required.")
    if int(protocol.get("bootstrap_samples", 0)) < 100:
        raise ValueError("At least 100 bootstrap samples are required.")
    if not protocol.get("primary_estimand") or not protocol.get(
        "primary_success_rule"
    ):
        raise ValueError("Primary estimand and success rule must be declared.")
    cases = list(suite.get("cases", []))
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if not cases or any(not case_id for case_id in case_ids):
        raise ValueError("Every confirmation case requires a case_id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Confirmation case IDs must be unique.")
    required_runner = {
        "model",
        "base_url",
        "api_key_env",
        "api_mode",
        "temperature",
        "rounds",
        "decision_policy",
        "deliberation_mode",
    }
    missing_runner = required_runner - set(suite.get("runner", {}))
    if missing_runner:
        raise ValueError(f"Missing runner fields: {sorted(missing_runner)}")
    gate = suite.get("calibration_gate", {})
    if gate.get("enabled"):
        gate_round = int(gate.get("evaluation_after_reveals", 0))
        force_fallback = bool(gate.get("force_fallback_after_evaluation", False))
        raw_threshold = gate.get("prediction_mae_threshold")
        if not 1 <= gate_round <= int(suite["runner"]["rounds"]):
            raise ValueError("Calibration gate round must be within runner rounds.")
        if raw_threshold is None and not force_fallback:
            raise ValueError(
                "Calibration gate requires a prediction MAE threshold unless "
                "bounded-authority fallback is forced."
            )
        if raw_threshold is not None and float(raw_threshold) < 0.0:
            raise ValueError("Calibration gate threshold must be non-negative.")
    if check_paths:
        for case in cases:
            for key in ("config", "initial_record"):
                path = resolve(str(case[key]))
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {key}: {path}")


def protocol_lock(
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "suite_config": {
            "path": str(suite_config),
            "sha256": file_sha256(suite_config),
        },
        "runner_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "online_runner_script": {
            "path": str(Path(online.__file__).resolve()),
            "sha256": file_sha256(Path(online.__file__).resolve()),
        },
    }
    for case in suite["cases"]:
        case_id = str(case["case_id"])
        for key in ("config", "initial_record"):
            path = resolve(str(case[key]))
            artifacts[f"{case_id}:{key}"] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite["suite"],
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "expected_case_ids": [str(case["case_id"]) for case in suite["cases"]],
        "expected_replicate_ids": replicate_ids(suite),
        "artifacts": artifacts,
    }


def lock_identity(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": lock["schema_version"],
        "suite": lock["suite"],
        "expected_case_ids": lock["expected_case_ids"],
        "expected_replicate_ids": lock["expected_replicate_ids"],
        "artifacts": lock["artifacts"],
    }


def ensure_protocol_lock(
    output_root: Path,
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_root / "protocol_lock.json"
    candidate = protocol_lock(suite_config, suite)
    if path.exists():
        existing = load_json(path)
        if lock_identity(existing) != lock_identity(candidate):
            raise ValueError(
                "Frozen protocol artifacts changed after the output lock was created."
            )
        return existing
    write_json(path, candidate)
    online.write_fingerprint(path)
    return candidate


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence.")
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
    values: Sequence[float], samples: int, seed: int
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    smaller = min(wins, losses)
    lower_tail = sum(
        math.comb(trials, successes) for successes in range(smaller + 1)
    ) / (2**trials)
    return min(1.0, 2.0 * lower_tail)


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(p_values) if value is not None]
    if not indexed:
        return [None] * len(p_values)
    indexed.sort(key=lambda item: float(item[1]))
    count = len(indexed)
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index, value = indexed[rank_index]
        rank = rank_index + 1
        running = min(running, float(value) * count / rank)
        adjusted[original_index] = min(1.0, running)
    return [adjusted.get(index) for index in range(len(p_values))]


def summarize_values(
    values: Sequence[float], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "bootstrap_95ci_low": None,
            "bootstrap_95ci_high": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "two_sided_sign_test_p": None,
        }
    wins = sum(value > TOLERANCE for value in values)
    ties = sum(abs(value) <= TOLERANCE for value in values)
    losses = sum(value < -TOLERANCE for value in values)
    low, high = bootstrap_mean_ci(values, bootstrap_samples, bootstrap_seed)
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "standard_deviation": (
            round(stdev(values), 6) if len(values) > 1 else None
        ),
        "bootstrap_95ci_low": round(low, 6) if low is not None else None,
        "bootstrap_95ci_high": round(high, 6) if high is not None else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "two_sided_sign_test_p": exact_two_sided_sign_p(wins, losses),
    }


def hierarchical_bootstrap(
    values_by_route: Mapping[str, Sequence[float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    usable = {
        route: [float(value) for value in values]
        for route, values in values_by_route.items()
        if values
    }
    if len(usable) < 2 or any(len(values) < 2 for values in usable.values()):
        return {
            "route_count": len(usable),
            "mean_of_route_means": (
                round(mean(mean(values) for values in usable.values()), 6)
                if usable
                else None
            ),
            "bootstrap_95ci_low": None,
            "bootstrap_95ci_high": None,
        }
    route_names = list(usable)
    rng = np.random.default_rng(seed)
    within = np.vstack(
        [
            rng.choice(np.asarray(usable[name], dtype=np.float64), size=(samples, len(usable[name])), replace=True).mean(axis=1)
            for name in route_names
        ]
    )
    route_indices = rng.integers(0, len(route_names), size=(samples, len(route_names)))
    sample_indices = np.arange(samples, dtype=np.int64)[:, None]
    estimates = within[route_indices, sample_indices].mean(axis=1)
    return {
        "route_count": len(route_names),
        "mean_of_route_means": round(
            mean(mean(values) for values in usable.values()), 6
        ),
        "bootstrap_95ci_low": round(float(np.quantile(estimates, 0.025)), 6),
        "bootstrap_95ci_high": round(float(np.quantile(estimates, 0.975)), 6),
    }


def trajectory_dir(output_root: Path, case_id: str, replicate_id: int) -> Path:
    return output_root / case_id / f"trajectory_{replicate_id:04d}"


def run_args(case: Mapping[str, Any], suite: Mapping[str, Any], output: Path) -> argparse.Namespace:
    runner = suite["runner"]
    menu = suite["menu"]
    gate = suite.get("calibration_gate", {})
    gate_enabled = bool(gate.get("enabled", False))
    return argparse.Namespace(
        config=resolve(str(case["config"])),
        initial_record=resolve(str(case["initial_record"])),
        output_dir=output,
        rounds=int(runner["rounds"]),
        initial_design_mode=str(runner["initial_design_mode"]),
        menu_gp_count=int(menu["gp_count"]),
        menu_source_count=int(menu["source_count"]),
        menu_consensus_count=int(menu["consensus_count"]),
        max_transfer_gp_rank=int(menu["max_transfer_gp_rank"]),
        safety_fallback_gp_count=int(menu["safety_fallback_gp_count"]),
        force_first_consensus=bool(menu["force_first_consensus"]),
        menu_diversity_count=int(menu["diversity_count"]),
        decision_policy=str(runner["decision_policy"]),
        deliberation_mode=str(runner["deliberation_mode"]),
        fail_on_llm_error=bool(runner["fail_on_llm_error"]),
        llm_base_url=str(runner["base_url"]),
        llm_model=str(runner["model"]),
        llm_api_key_env=str(runner["api_key_env"]),
        llm_api_mode=str(runner["api_mode"]),
        llm_temperature=float(runner["temperature"]),
        llm_max_tokens=int(runner["max_tokens"]),
        llm_repair_attempts=int(runner["repair_attempts"]),
        calibration_gate_round=(
            int(gate["evaluation_after_reveals"]) if gate_enabled else None
        ),
        calibration_gate_mae_threshold=(
            float(gate["prediction_mae_threshold"])
            if gate_enabled and gate.get("prediction_mae_threshold") is not None
            else None
        ),
        calibration_gate_hard_abstention=bool(
            gate.get("hard_abstention", True)
        ),
        calibration_gate_force_fallback=bool(
            gate.get("force_fallback_after_evaluation", False)
        ),
    )


def run_one(
    case: Mapping[str, Any],
    suite: Mapping[str, Any],
    output: Path,
    replicate_id: int,
    lock: Mapping[str, Any],
) -> None:
    metadata_path = output / "trajectory_metadata.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite["suite"],
        "case_id": case["case_id"],
        "replicate_id": replicate_id,
        "status": "running",
        "started_at": utc_now(),
        "model": suite["runner"]["model"],
        "temperature": suite["runner"]["temperature"],
        "protocol_lock_sha256": file_sha256(output.parents[1] / "protocol_lock.json"),
        "git_commit": lock.get("git_commit"),
    }
    write_json(metadata_path, metadata)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            online.run_online(run_args(case, suite, output))
    except Exception as exc:
        metadata.update(
            {
                "status": "error",
                "finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )
        write_json(metadata_path, metadata)
        write_json(
            output / "error.json",
            {
                "case_id": case["case_id"],
                "replicate_id": replicate_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            },
        )
        online.write_fingerprint(metadata_path)
        online.write_fingerprint(output / "error.json")
        raise
    metadata.update({"status": "complete", "finished_at": utc_now()})
    write_json(metadata_path, metadata)
    online.write_fingerprint(metadata_path)


def collect_rows(
    output_root: Path, suite: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in suite["cases"]:
        case_id = str(case["case_id"])
        for replicate_id in replicate_ids(suite):
            output = trajectory_dir(output_root, case_id, replicate_id)
            summary_path = output / "summary.json"
            error_path = output / "error.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                metrics = summary["metrics"][online.MODE]
                delta = summary["deltas"][
                    "online_llm_increment_over_same_initial_gp"
                ]
                rows.append(
                    {
                        "case_id": case_id,
                        "domain": case["domain"],
                        "evidence_tier": case["evidence_tier"],
                        "replicate_id": replicate_id,
                        "auc_delta_vs_same_initial_gp": float(
                            delta["best_so_far_auc"]
                        ),
                        "final_delta_vs_same_initial_gp": float(
                            delta["final_best"]
                        ),
                        "participation_rate": float(
                            metrics["llm_participation_rate"]
                        ),
                        "decision_authority_rate": float(
                            metrics["llm_decision_authority_rate"]
                        ),
                        "gp_override_rate": float(metrics["llm_gp_override_rate"]),
                        "critic_revision_rate": float(
                            metrics["llm_critic_revision_rate"]
                        ),
                        "source_transfer_active_rate": float(
                            metrics["source_transfer_active_rate"]
                        ),
                        "calibration_gate_triggered": bool(
                            metrics.get("calibration_gate_triggered", False)
                        ),
                        "calibration_gate_prediction_mae": metrics.get(
                            "calibration_gate_prediction_mae"
                        ),
                        "calibration_gate_fallback_rounds": int(
                            metrics.get("calibration_gate_fallback_rounds", 0)
                        ),
                        "calibration_gate_llm_rounds_saved": int(
                            metrics.get("calibration_gate_llm_rounds_saved", 0)
                        ),
                        "calibration_gate_nominal_llm_calls_avoided": int(
                            metrics.get(
                                "calibration_gate_nominal_llm_calls_avoided", 0
                            )
                        ),
                        "total_tokens": int(summary["usage"]["total_tokens"]),
                        "summary_sha256": file_sha256(summary_path),
                        "trace_sha256": file_sha256(output / "llm_trace.jsonl"),
                    }
                )
            elif error_path.exists():
                failures.append(load_json(error_path))
    return rows, failures


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_forest_svg(route_rows: Sequence[Mapping[str, Any]], output: Path, complete: bool) -> None:
    width = 1500
    height = max(520, 150 + 70 * len(route_rows))
    plot_left = 610
    plot_right = 1430
    values = [
        float(value)
        for row in route_rows
        for value in (
            row.get("auc_bootstrap_95ci_low"),
            row.get("auc_bootstrap_95ci_high"),
            row.get("mean_auc_delta"),
        )
        if value is not None
    ]
    lower = min([-1.0, *values])
    upper = max([1.0, *values])
    margin = max(0.5, (upper - lower) * 0.08)
    lower -= margin
    upper += margin

    def x(value: float) -> float:
        return plot_left + (value - lower) / (upper - lower) * (plot_right - plot_left)

    zero_x = x(0.0)
    status = "complete frozen protocol" if complete else "partial execution; inference disabled"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="40" y="48" font-family="Arial, sans-serif" font-size="29" font-weight="700" fill="#24292F">Repeated online-LLM trajectories</text>',
        f'<text x="40" y="82" font-family="Arial, sans-serif" font-size="17" fill="#5B6168">{html.escape(status)}. Points are route means; intervals bootstrap trajectories within route.</text>',
        f'<line x1="{zero_x:.1f}" y1="110" x2="{zero_x:.1f}" y2="{height - 65}" stroke="#343A40" stroke-width="2" stroke-dasharray="8 7"/>',
    ]
    for index, row in enumerate(route_rows):
        y = 140 + index * 70
        label = str(row["case_id"]).replace("_", " ")
        estimate = row.get("mean_auc_delta")
        low = row.get("auc_bootstrap_95ci_low")
        high = row.get("auc_bootstrap_95ci_high")
        color = "#C05A47" if row["evidence_tier"] == "post_freeze_extension" else "#2F6B7C"
        lines.append(f'<text x="40" y="{y + 6}" font-family="Arial, sans-serif" font-size="16" fill="#30363D">{html.escape(label)}</text>')
        if estimate is not None:
            if low is not None and high is not None:
                lines.append(f'<line x1="{x(float(low)):.1f}" y1="{y}" x2="{x(float(high)):.1f}" y2="{y}" stroke="{color}" stroke-width="5"/>')
            lines.append(f'<circle cx="{x(float(estimate)):.1f}" cy="{y}" r="9" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>')
            lines.append(f'<text x="{x(float(estimate)) + 15:.1f}" y="{y + 6}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="{color}">{float(estimate):+.2f} (n={row["completed_replicates"]})</text>')
    lines.extend([
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#30363D">Best-so-far AUC delta vs same-initial target-only GP-UCB</text>',
        '</svg>',
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def aggregate(output_root: Path, suite: Mapping[str, Any]) -> dict[str, Any]:
    rows, failures = collect_rows(output_root, suite)
    protocol = suite["protocol"]
    expected_replicates = int(protocol["replicates_per_route"])
    bootstrap_samples = int(protocol["bootstrap_samples"])
    bootstrap_seed = int(protocol["bootstrap_seed"])
    by_case: dict[str, list[dict[str, Any]]] = {
        str(case["case_id"]): [] for case in suite["cases"]
    }
    for row in rows:
        by_case[str(row["case_id"])].append(row)

    route_rows: list[dict[str, Any]] = []
    route_p_values: list[float | None] = []
    auc_values_by_route: dict[str, list[float]] = {}
    for index, case in enumerate(suite["cases"]):
        case_id = str(case["case_id"])
        case_rows = sorted(by_case[case_id], key=lambda row: int(row["replicate_id"]))
        auc_values = [float(row["auc_delta_vs_same_initial_gp"]) for row in case_rows]
        final_values = [float(row["final_delta_vs_same_initial_gp"]) for row in case_rows]
        auc_values_by_route[case_id] = auc_values
        auc_summary = summarize_values(
            auc_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index + 1,
        )
        final_summary = summarize_values(
            final_values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100 + index,
        )
        route_p_values.append(auc_summary["two_sided_sign_test_p"])
        route_rows.append(
            {
                "case_id": case_id,
                "domain": case["domain"],
                "evidence_tier": case["evidence_tier"],
                "completed_replicates": len(case_rows),
                "expected_replicates": expected_replicates,
                "mean_auc_delta": auc_summary["mean"],
                "auc_standard_deviation": auc_summary["standard_deviation"],
                "auc_bootstrap_95ci_low": auc_summary["bootstrap_95ci_low"],
                "auc_bootstrap_95ci_high": auc_summary["bootstrap_95ci_high"],
                "auc_wins": auc_summary["wins"],
                "auc_ties": auc_summary["ties"],
                "auc_losses": auc_summary["losses"],
                "auc_sign_test_p": auc_summary["two_sided_sign_test_p"],
                "mean_final_delta": final_summary["mean"],
                "final_bootstrap_95ci_low": final_summary["bootstrap_95ci_low"],
                "final_bootstrap_95ci_high": final_summary["bootstrap_95ci_high"],
                "mean_total_tokens": (
                    round(mean(int(row["total_tokens"]) for row in case_rows), 2)
                    if case_rows
                    else None
                ),
                "calibration_gate_trigger_rate": (
                    round(
                        mean(
                            float(bool(row["calibration_gate_triggered"]))
                            for row in case_rows
                        ),
                        6,
                    )
                    if case_rows
                    else None
                ),
                "mean_calibration_gate_fallback_rounds": (
                    round(
                        mean(
                            int(row["calibration_gate_fallback_rounds"])
                            for row in case_rows
                        ),
                        6,
                    )
                    if case_rows
                    else None
                ),
                "mean_llm_rounds_saved_by_gate": (
                    round(
                        mean(
                            int(row["calibration_gate_llm_rounds_saved"])
                            for row in case_rows
                        ),
                        6,
                    )
                    if case_rows
                    else None
                ),
                "mean_nominal_llm_calls_avoided_by_gate": (
                    round(
                        mean(
                            int(row["calibration_gate_nominal_llm_calls_avoided"])
                            for row in case_rows
                        ),
                        6,
                    )
                    if case_rows
                    else None
                ),
            }
        )
    adjusted = benjamini_hochberg(route_p_values)
    for row, adjusted_p in zip(route_rows, adjusted):
        row["auc_sign_test_bh_adjusted_p"] = adjusted_p

    complete = all(
        int(row["completed_replicates"]) == expected_replicates
        for row in route_rows
    ) and not failures
    overall = hierarchical_bootstrap(
        auc_values_by_route,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    low = overall["bootstrap_95ci_low"]
    if not complete:
        decision = "not_evaluated_incomplete_protocol"
    elif low is not None and float(low) > 0.0:
        decision = "supports_positive_equal_route_mean_transfer"
    else:
        decision = "does_not_support_positive_equal_route_mean_transfer"
    report = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite["suite"],
        "protocol_complete": complete,
        "claim_decision": decision,
        "primary_estimand": protocol["primary_estimand"],
        "primary_success_rule": protocol["primary_success_rule"],
        "expected_case_count": len(suite["cases"]),
        "expected_replicates_per_case": expected_replicates,
        "completed_trajectory_count": len(rows),
        "failed_trajectory_count": len(failures),
        "hierarchical_bootstrap": overall,
        "routes": route_rows,
        "failures": failures,
    }
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    write_json(aggregate_dir / "repeated_confirmation.json", report)
    write_csv(aggregate_dir / "trajectory_metrics.csv", rows)
    write_csv(aggregate_dir / "route_statistics.csv", route_rows)
    write_forest_svg(
        route_rows,
        aggregate_dir / "repeated_route_effects.svg",
        complete,
    )
    lines = [
        f"# {suite['suite']}",
        "",
        f"- Protocol complete: `{str(complete).lower()}`",
        f"- Claim decision: `{decision}`",
        f"- Completed trajectories: `{len(rows)}` / `{len(suite['cases']) * expected_replicates}`",
        f"- Failed trajectories: `{len(failures)}`",
        "",
        "Partial execution is operational evidence only. No confirmatory claim is evaluated until all declared trajectories are complete.",
        "",
        "## Primary estimand",
        "",
        str(protocol["primary_estimand"]),
        "",
        "## Route statistics",
        "",
        "| Route | Tier | n | Mean AUC delta | 95% CI | W/T/L | BH-adjusted sign p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in route_rows:
        ci = (
            f"[{row['auc_bootstrap_95ci_low']:+.3f}, {row['auc_bootstrap_95ci_high']:+.3f}]"
            if row["auc_bootstrap_95ci_low"] is not None
            else "not estimable"
        )
        adjusted_p = row["auc_sign_test_bh_adjusted_p"]
        lines.append(
            f"| {row['case_id']} | {row['evidence_tier']} | {row['completed_replicates']} | "
            f"{row['mean_auc_delta'] if row['mean_auc_delta'] is not None else 'NA'} | {ci} | "
            f"{row['auc_wins']}/{row['auc_ties']}/{row['auc_losses']} | "
            f"{adjusted_p if adjusted_p is not None else 'NA'} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "The repeated protocol estimates stochastic variation of the frozen online controller. "
            "Development and post-freeze routes remain labeled separately, and completion of this "
            "protocol does not convert retrospective routes into independent external validation."
        ),
        "",
    ])
    (aggregate_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    for path in (
        aggregate_dir / "repeated_confirmation.json",
        aggregate_dir / "trajectory_metrics.csv",
        aggregate_dir / "route_statistics.csv",
        aggregate_dir / "repeated_route_effects.svg",
        aggregate_dir / "RESULTS.md",
    ):
        if path.exists():
            online.write_fingerprint(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--replicate", action="append", type=int, default=[])
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    suite_config = args.suite_config.resolve()
    suite = load_json(suite_config)
    validate_suite(suite)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = ensure_protocol_lock(args.output_root, suite_config, suite)

    selected_cases = set(args.case)
    selected_replicates = set(args.replicate)
    expected_replicates = set(replicate_ids(suite))
    if selected_replicates - expected_replicates:
        parser.error("Requested replicate is outside the frozen protocol.")
    jobs = []
    for replicate_id in replicate_ids(suite):
        if selected_replicates and replicate_id not in selected_replicates:
            continue
        for case in suite["cases"]:
            case_id = str(case["case_id"])
            if selected_cases and case_id not in selected_cases:
                continue
            output = trajectory_dir(args.output_root, case_id, replicate_id)
            if (output / "summary.json").exists():
                continue
            jobs.append((case, replicate_id, output))
    if args.max_new_runs is not None:
        if args.max_new_runs < 0:
            parser.error("--max-new-runs cannot be negative.")
        jobs = jobs[: args.max_new_runs]

    if not args.aggregate_only:
        for case, replicate_id, output in jobs:
            try:
                run_one(case, suite, output, replicate_id, lock)
                print(f"completed {case['case_id']} trajectory {replicate_id}")
            except Exception as exc:
                print(
                    f"failed {case['case_id']} trajectory {replicate_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if not args.continue_on_error:
                    raise

    report = aggregate(args.output_root, suite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
