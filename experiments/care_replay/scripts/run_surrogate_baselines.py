#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RUNS = ROOT / "outputs" / "runs"
OUTPUT_TABLES = ROOT / "outputs" / "tables"

BaselineMode = str
DEFAULT_MODES: tuple[BaselineMode, ...] = (
    "random",
    "public_incumbent",
    "mixed_kernel_gp_ucb",
    "mixed_kernel_gp_ei",
    "knn_ucb",
)

FeatureRecord = tuple[tuple[float, ...], tuple[str, ...]]


def parse_modes(raw: str) -> tuple[BaselineMode, ...]:
    modes = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not modes:
        raise ValueError("At least one mode is required.")
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}. Available modes: {', '.join(DEFAULT_MODES)}")
    return modes


def candidate_features(adapter: replay.DatasetAdapter, candidate: replay.Candidate) -> FeatureRecord:
    numeric = (float(candidate.x1), float(candidate.x2), float(candidate.x3))
    categorical = [candidate.group]
    for field_name in adapter.decision_columns:
        categorical.append(str(candidate.metadata.get(field_name, "")))
    return numeric, tuple(categorical)


def mixed_distance(a: FeatureRecord, b: FeatureRecord) -> float:
    a_num, a_cat = a
    b_num, b_cat = b
    numeric = math.sqrt(sum((x - y) ** 2 for x, y in zip(a_num, b_num)))
    categorical = sum(1.0 for x, y in zip(a_cat, b_cat) if x != y) / max(1, len(a_cat))
    return numeric + categorical


def mixed_kernel(
    a: FeatureRecord,
    b: FeatureRecord,
    numeric_length_scale: float,
    categorical_length_scale: float,
) -> float:
    a_num, a_cat = a
    b_num, b_cat = b
    numeric_sq = sum((x - y) ** 2 for x, y in zip(a_num, b_num))
    categorical_mismatch = sum(1.0 for x, y in zip(a_cat, b_cat) if x != y)
    return math.exp(
        -(numeric_sq / (2.0 * numeric_length_scale * numeric_length_scale))
        - (categorical_mismatch / categorical_length_scale)
    )


def cholesky_spd(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            subtotal = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(matrix[i][i] - subtotal, 1e-10))
            else:
                lower[i][j] = (matrix[i][j] - subtotal) / lower[j][j]
    return lower


def solve_cholesky(lower: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(lower)
    y = [0.0] * size
    for i in range(size):
        subtotal = sum(lower[i][j] * y[j] for j in range(i))
        y[i] = (rhs[i] - subtotal) / lower[i][i]
    x = [0.0] * size
    for i in range(size - 1, -1, -1):
        subtotal = sum(lower[j][i] * x[j] for j in range(i + 1, size))
        x[i] = (y[i] - subtotal) / lower[i][i]
    return x


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def gp_scores(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, FeatureRecord],
    mode: BaselineMode,
    beta: float,
    xi: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    noise: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    observed_features = [features_by_id[candidate.candidate_id] for candidate in observed]
    y_values = [candidate.objective_value / 100.0 for candidate in observed]
    y_mean = mean(y_values)
    y_scale = max(pstdev(y_values), 0.05)
    y_norm = [(value - y_mean) / y_scale for value in y_values]
    size = len(observed)
    kernel_matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            value = mixed_kernel(observed_features[i], observed_features[j], numeric_length_scale, categorical_length_scale)
            kernel_matrix[i][j] = value
            kernel_matrix[j][i] = value
    for i in range(size):
        kernel_matrix[i][i] += noise
    lower = cholesky_spd(kernel_matrix)
    alpha = solve_cholesky(lower, y_norm)
    best_seen = max(y_values)

    scores: dict[str, float] = {}
    diagnostics = {
        "posterior_mean_min": 1.0,
        "posterior_mean_max": 0.0,
        "posterior_std_mean": 0.0,
        "candidate_count": 0,
    }
    std_values: list[float] = []
    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue
        feature = features_by_id[candidate.candidate_id]
        k_vec = [mixed_kernel(feature, observed_feature, numeric_length_scale, categorical_length_scale) for observed_feature in observed_features]
        mean_norm = sum(k * a for k, a in zip(k_vec, alpha))
        solved = solve_cholesky(lower, k_vec)
        variance = max(1e-9, 1.0 - sum(k * v for k, v in zip(k_vec, solved)))
        posterior_mean = y_mean + y_scale * mean_norm
        posterior_std = y_scale * math.sqrt(variance)
        if mode == "mixed_kernel_gp_ucb":
            score = posterior_mean + beta * posterior_std
        elif mode == "mixed_kernel_gp_ei":
            if posterior_std <= 1e-9:
                score = max(0.0, posterior_mean - best_seen - xi)
            else:
                improvement = posterior_mean - best_seen - xi
                z = improvement / posterior_std
                score = improvement * normal_cdf(z) + posterior_std * normal_pdf(z)
        else:
            raise ValueError(f"Unsupported GP mode: {mode}")
        scores[candidate.candidate_id] = score
        diagnostics["posterior_mean_min"] = min(diagnostics["posterior_mean_min"], round(posterior_mean, 6))
        diagnostics["posterior_mean_max"] = max(diagnostics["posterior_mean_max"], round(posterior_mean, 6))
        std_values.append(posterior_std)
    diagnostics["candidate_count"] = len(scores)
    diagnostics["posterior_std_mean"] = round(mean(std_values), 6) if std_values else 0.0
    diagnostics["y_mean"] = round(y_mean, 6)
    diagnostics["y_scale"] = round(y_scale, 6)
    return scores, diagnostics


def knn_scores(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, FeatureRecord],
    k: int,
    beta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    observed_features = [(candidate, features_by_id[candidate.candidate_id]) for candidate in observed]
    scores: dict[str, float] = {}
    for candidate in adapter.candidates:
        if candidate.candidate_id in observed_ids:
            continue
        feature = features_by_id[candidate.candidate_id]
        neighbors = sorted(
            ((mixed_distance(feature, observed_feature), observed_candidate.objective_value / 100.0) for observed_candidate, observed_feature in observed_features),
            key=lambda item: item[0],
        )[: max(1, k)]
        weights = [1.0 / (distance + 0.05) for distance, _value in neighbors]
        weight_sum = sum(weights)
        prediction = sum(weight * value for weight, (_distance, value) in zip(weights, neighbors)) / weight_sum
        variance = sum(weight * (value - prediction) ** 2 for weight, (_distance, value) in zip(weights, neighbors)) / weight_sum
        scores[candidate.candidate_id] = prediction + beta * math.sqrt(max(0.0, variance))
    return scores, {"k": k, "candidate_count": len(scores)}


def score_candidates(
    adapter: replay.DatasetAdapter,
    observed_ids: set[str],
    observed: list[replay.Candidate],
    features_by_id: dict[str, FeatureRecord],
    mode: BaselineMode,
    gp_beta: float,
    gp_xi: float,
    knn_k: int,
    knn_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if mode == "public_incumbent":
        return replay.public_incumbent_scores(adapter, observed_ids, observed), {}
    if mode in {"mixed_kernel_gp_ucb", "mixed_kernel_gp_ei"}:
        return gp_scores(
            adapter,
            observed_ids,
            observed,
            features_by_id,
            mode,
            gp_beta,
            gp_xi,
            numeric_length_scale,
            categorical_length_scale,
            gp_noise,
        )
    if mode == "knn_ucb":
        return knn_scores(adapter, observed_ids, observed, features_by_id, knn_k, knn_beta)
    raise ValueError(f"Unsupported score mode: {mode}")


def run_policy(
    adapter: replay.DatasetAdapter,
    task: replay.TaskSpec,
    seed: int,
    mode: BaselineMode,
    gp_beta: float,
    gp_xi: float,
    knn_k: int,
    knn_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pool = adapter.candidates
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    features_by_id = {candidate.candidate_id: candidate_features(adapter, candidate) for candidate in pool}
    shuffled = list(pool)
    rng.shuffle(shuffled)
    observed = shuffled[: task.initial_observations]
    observed_ids = {candidate.candidate_id for candidate in observed}
    top10 = {candidate.candidate_id for candidate in sorted(pool, key=lambda x: x.objective_value, reverse=True)[:10]}
    selected_top10 = any(candidate.candidate_id in top10 for candidate in observed)
    best_trace: list[float] = []
    audit: list[dict[str, Any]] = []

    for round_index in range(task.reveal_budget):
        if mode == "random":
            selected = rng.choice([candidate for candidate in pool if candidate.candidate_id not in observed_ids])
            selected_score: float | None = None
            diagnostics: dict[str, Any] = {}
        else:
            scores, diagnostics = score_candidates(
                adapter,
                observed_ids,
                observed,
                features_by_id,
                mode,
                gp_beta,
                gp_xi,
                knn_k,
                knn_beta,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            selected_id = replay.top_candidate(scores)
            selected = by_id[selected_id]
            selected_score = round(scores[selected_id], 6)

        observed.append(selected)
        observed_ids.add(selected.candidate_id)
        selected_top10 = selected_top10 or selected.candidate_id in top10
        best_so_far = max(candidate.objective_value for candidate in observed)
        best_trace.append(best_so_far)
        audit.append(
            {
                "dataset_id": adapter.dataset_id,
                "seed": seed,
                "round_index": round_index,
                "mode": mode,
                "public_observed_count": len(observed) - 1,
                "selected_candidate": selected.candidate_id,
                "selected_score": selected_score,
                "revealed_value": selected.objective_value,
                "best_so_far": best_so_far,
                "diagnostics": diagnostics,
            }
        )

    final_best = max(candidate.objective_value for candidate in observed)
    metrics = {
        "dataset": adapter.dataset_id,
        "mode": mode,
        "seed": seed,
        "final_best": round(final_best, 4),
        "best_so_far_auc": round(mean(best_trace), 4),
        "simple_regret": round(task.oracle_value - final_best, 4),
        "top10_hit": int(selected_top10),
    }
    return metrics, audit


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out: dict[str, Any] = {}
    for mode, items in by_mode.items():
        out[mode] = {}
        for field_name in ("final_best", "best_so_far_auc", "simple_regret", "top10_hit"):
            values = [float(item[field_name]) for item in items]
            out[mode][field_name] = {
                "mean": round(mean(values), 4),
                "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
            }
    return out


def write_outputs(
    output_id: str,
    rows: list[dict[str, Any]],
    audits: dict[tuple[str, int], list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_TABLES / f"{output_id}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_RUNS / f"{output_id}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for (mode, seed), audit_rows in sorted(audits.items()):
        with (OUTPUT_RUNS / f"{output_id}_audit_{mode}_seed{seed}.jsonl").open("w", encoding="utf-8") as f:
            for row in audit_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_surrogate_baselines(
    dataset_id: str,
    seeds: int,
    rounds: int,
    initial: int,
    modes: tuple[BaselineMode, ...],
    gp_beta: float,
    gp_xi: float,
    knn_k: int,
    knn_beta: float,
    numeric_length_scale: float,
    categorical_length_scale: float,
    gp_noise: float,
    output_tag: str,
) -> dict[str, Any]:
    adapter = replay.DATASET_BUILDERS[dataset_id]()
    task = replay.make_task(adapter, initial, rounds)
    rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seed in range(seeds):
        for mode in modes:
            metrics, audit = run_policy(
                adapter,
                task,
                seed,
                mode,
                gp_beta,
                gp_xi,
                knn_k,
                knn_beta,
                numeric_length_scale,
                categorical_length_scale,
                gp_noise,
            )
            rows.append(metrics)
            audits[(mode, seed)] = audit

    output_id = f"surrogate_baselines_{dataset_id}" if not output_tag else f"surrogate_baselines_{dataset_id}_{output_tag}"
    summary = {
        "experiment": "care_surrogate_baseline_comparison",
        "output_id": output_id,
        "dataset": dataset_id,
        "task": asdict(task),
        "seeds": seeds,
        "rounds": rounds,
        "initial_observations": initial,
        "modes": list(modes),
        "baseline_notes": {
            "public_incumbent": "Current transparent target-only evidence baseline.",
            "mixed_kernel_gp_ucb": "Dependency-free Gaussian-process-style kernel surrogate with a UCB acquisition over public mixed categorical/numeric features.",
            "mixed_kernel_gp_ei": "Same mixed-kernel surrogate with expected-improvement acquisition.",
            "knn_ucb": "Weighted nearest-neighbor surrogate with uncertainty bonus over the same public feature space.",
            "random": "Uniform random selection from unrevealed candidates.",
        },
        "parameters": {
            "gp_beta": gp_beta,
            "gp_xi": gp_xi,
            "knn_k": knn_k,
            "knn_beta": knn_beta,
            "numeric_length_scale": numeric_length_scale,
            "categorical_length_scale": categorical_length_scale,
            "gp_noise": gp_noise,
        },
        "aggregate": aggregate(rows),
    }
    write_outputs(output_id, rows, audits, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dependency-free surrogate baselines for finite-pool replay.")
    parser.add_argument("--dataset", default="real_buchwald_hartwig", choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help=f"Comma-separated modes from: {', '.join(DEFAULT_MODES)}")
    parser.add_argument("--gp-beta", type=float, default=1.5)
    parser.add_argument("--gp-xi", type=float, default=0.005)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-beta", type=float, default=1.0)
    parser.add_argument("--numeric-length-scale", type=float, default=0.35)
    parser.add_argument("--categorical-length-scale", type=float, default=3.0)
    parser.add_argument("--gp-noise", type=float, default=0.05)
    parser.add_argument("--output-tag", default="", help="Optional suffix for output filenames.")
    args = parser.parse_args()
    summary = run_surrogate_baselines(
        dataset_id=args.dataset,
        seeds=args.seeds,
        rounds=args.rounds,
        initial=args.initial,
        modes=parse_modes(args.modes),
        gp_beta=args.gp_beta,
        gp_xi=args.gp_xi,
        knn_k=args.knn_k,
        knn_beta=args.knn_beta,
        numeric_length_scale=args.numeric_length_scale,
        categorical_length_scale=args.categorical_length_scale,
        gp_noise=args.gp_noise,
        output_tag=args.output_tag,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
