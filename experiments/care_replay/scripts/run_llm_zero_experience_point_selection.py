#!/usr/bin/env python3
"""Evaluate direct LLM point choice without retrieved or experimental evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_synthetic_suzuki as replay


SCHEMA_VERSION = "care.llm_zero_experience_point_selection/v1"
SYSTEM_MESSAGE = """You are choosing experiments before any result is observed.

Use only your pretrained scientific knowledge, the public target description,
and the candidate conditions supplied in this request. You have no source
campaign, retrieved document, RAG context, knowledge-base card, transfer skill,
GP prediction, previous target observation, or hidden outcome. Select the one
candidate in each menu that you expect to maximize the stated objective.

Return one JSON object with a `selections` list. Each list item must contain
exactly `menu_id`, `candidate_id`, `hypothesis`, `reason`, and `confidence`.
`candidate_id` must come from that menu and confidence must be between 0 and 1.
Do not claim that an outcome was measured or observed.
"""


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def rounded_public_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def public_candidate(
    adapter: replay.DatasetAdapter,
    candidate: replay.Candidate,
    public_fields: Sequence[str],
) -> dict[str, Any]:
    prohibited = {adapter.hidden_target, "source_row"}
    overlap = prohibited.intersection(public_fields)
    if overlap:
        raise ValueError(f"Public fields expose prohibited keys: {sorted(overlap)}")
    missing = [field for field in public_fields if field not in candidate.metadata]
    if missing:
        raise ValueError(
            f"Dataset {adapter.dataset_id} is missing public fields: {missing}"
        )
    return {
        "candidate_id": candidate.candidate_id,
        "conditions": {
            field: rounded_public_value(candidate.metadata[field])
            for field in public_fields
        },
    }


def build_menus(
    adapter: replay.DatasetAdapter,
    public_fields: Sequence[str],
    *,
    seed_start: int,
    menu_count: int,
    menu_size: int,
) -> list[dict[str, Any]]:
    if menu_count <= 0:
        raise ValueError("menu_count must be positive")
    if menu_size < 2 or menu_size > len(adapter.candidates):
        raise ValueError("menu_size must be between 2 and the candidate count")
    menus = []
    for offset in range(menu_count):
        seed = seed_start + offset
        rng = random.Random(seed)
        indices = rng.sample(range(len(adapter.candidates)), menu_size)
        public_candidates = []
        candidate_id_map = {}
        for option_index, index in enumerate(indices):
            public_id = f"option_{option_index:02d}"
            candidate = public_candidate(
                adapter, adapter.candidates[index], public_fields
            )
            candidate_id_map[public_id] = candidate["candidate_id"]
            candidate["candidate_id"] = public_id
            public_candidates.append(candidate)
        menus.append(
            {
                "menu_id": f"{adapter.dataset_id}-menu-{offset:03d}",
                "seed": seed,
                "candidate_indices": indices,
                "candidate_id_map": candidate_id_map,
                "public_candidates": public_candidates,
            }
        )
    return menus


def prompt_payload(
    adapter: replay.DatasetAdapter,
    public_fields: Sequence[str],
    menus: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "protocol": SCHEMA_VERSION,
        "target": {
            "dataset_id": adapter.dataset_id,
            "title": adapter.title,
            "objective": adapter.objective,
            "public_fields": list(public_fields),
        },
        "evidence_boundary": {
            "source_campaigns": False,
            "source_outcomes": False,
            "target_outcomes": False,
            "target_history": False,
            "gp_scores": False,
            "rag_or_knowledge_base": False,
            "retrieved_or_frozen_skills": False,
        },
        "menus": [
            {
                "menu_id": menu["menu_id"],
                "candidates": menu["public_candidates"],
            }
            for menu in menus
        ],
        "required_output": {
            "selections": [
                {
                    "menu_id": "one supplied menu_id",
                    "candidate_id": "one candidate_id from that menu",
                    "hypothesis": "short falsifiable scientific expectation",
                    "reason": "why this point should maximize the objective",
                    "confidence": 0.0,
                }
            ]
        },
    }


def normalize_selections(
    response: Mapping[str, Any], menus: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    raw = response.get("selections")
    if not isinstance(raw, list):
        raise ValueError("Response must contain a selections list")
    by_menu: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Every selection must be an object")
        menu_id = str(item.get("menu_id", ""))
        if menu_id in by_menu:
            raise ValueError(f"Duplicate selection for {menu_id}")
        by_menu[menu_id] = item

    normalized = []
    for menu in menus:
        menu_id = str(menu["menu_id"])
        if menu_id not in by_menu:
            raise ValueError(f"Missing selection for {menu_id}")
        item = by_menu[menu_id]
        allowed = {
            str(candidate["candidate_id"])
            for candidate in menu["public_candidates"]
        }
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in allowed:
            raise ValueError(
                f"Selection {candidate_id!r} is outside menu {menu_id}"
            )
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        normalized.append(
            {
                "menu_id": menu_id,
                "candidate_id": candidate_id,
                "hypothesis": str(item.get("hypothesis", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "confidence": confidence,
            }
        )
    return normalized


def call_selection_batch(
    llm_config: replay.LLMConfig,
    adapter: replay.DatasetAdapter,
    public_fields: Sequence[str],
    menus: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = prompt_payload(adapter, public_fields, menus)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    attempts = []
    for attempt in range(2):
        content, metadata = replay.chat_completion_text(llm_config, messages)
        record = {
            "attempt": attempt + 1,
            "raw_response": content,
            "metadata": metadata,
        }
        attempts.append(record)
        try:
            parsed = replay.extract_json_object(content)
            normalized = normalize_selections(parsed, menus)
            return normalized, {
                "prompt": prompt,
                "prompt_sha256": sha256_payload(prompt),
                "attempts": attempts,
                "normalized_selections": normalized,
            }
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if attempt == 1:
                raise
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "The response failed schema validation: "
                            f"{type(exc).__name__}: {exc}. Return the complete "
                            "corrected JSON object. Do not add evidence or outcomes."
                        ),
                    },
                ]
            )
    raise AssertionError("unreachable")


def farthest_from_centroid_index(
    adapter: replay.DatasetAdapter, indices: Sequence[int]
) -> int:
    features = []
    for index in indices:
        candidate = adapter.candidates[index]
        row = candidate.numeric_features or (candidate.x1, candidate.x2, candidate.x3)
        features.append(tuple(float(value) for value in row))
    dimension = min(len(row) for row in features)
    normalized = []
    for row in features:
        normalized_row = []
        for column in range(dimension):
            values = [item[column] for item in features]
            low, high = min(values), max(values)
            normalized_row.append(
                0.0 if high == low else (row[column] - low) / (high - low)
            )
        normalized.append(tuple(normalized_row))
    centroid = [
        sum(row[column] for row in normalized) / len(normalized)
        for column in range(dimension)
    ]
    local_index = max(
        range(len(indices)),
        key=lambda item: sum(
            (normalized[item][column] - centroid[column]) ** 2
            for column in range(dimension)
        ),
    )
    return int(indices[local_index])


def evaluate_selection(
    adapter: replay.DatasetAdapter,
    menu: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    indices = [int(index) for index in menu["candidate_indices"]]
    candidate_by_id = {
        adapter.candidates[index].candidate_id: adapter.candidates[index]
        for index in indices
    }
    public_candidate_id = str(selection["candidate_id"])
    selected_candidate_id = str(menu["candidate_id_map"][public_candidate_id])
    selected = candidate_by_id[selected_candidate_id]
    values = [float(adapter.candidates[index].objective_value) for index in indices]
    selected_value = float(selected.objective_value)
    rank = 1 + sum(value > selected_value for value in values)
    percentile = (len(values) - rank) / (len(values) - 1)
    random_expected = sum(values) / len(values)
    random_rng = random.Random(int(menu["seed"]) + 99173)
    random_index = indices[random_rng.randrange(len(indices))]
    space_index = farthest_from_centroid_index(adapter, indices)
    return {
        "dataset_id": adapter.dataset_id,
        "menu_id": menu["menu_id"],
        "seed": menu["seed"],
        "menu_size": len(indices),
        "selected_public_candidate_id": public_candidate_id,
        "selected_candidate_id": selected.candidate_id,
        "selected_value": round(selected_value, 8),
        "random_expected_value": round(random_expected, 8),
        "random_realized_candidate_id": adapter.candidates[random_index].candidate_id,
        "random_realized_value": round(
            float(adapter.candidates[random_index].objective_value), 8
        ),
        "space_filling_candidate_id": adapter.candidates[space_index].candidate_id,
        "space_filling_value": round(
            float(adapter.candidates[space_index].objective_value), 8
        ),
        "oracle_value": round(max(values), 8),
        "llm_minus_random_expected": round(selected_value - random_expected, 8),
        "llm_minus_random_realized": round(
            selected_value - float(adapter.candidates[random_index].objective_value), 8
        ),
        "llm_minus_space_filling": round(
            selected_value - float(adapter.candidates[space_index].objective_value), 8
        ),
        "rank": rank,
        "percentile": round(percentile, 8),
        "top_quartile_hit": rank <= math.ceil(len(values) / 4),
        "top_one_hit": rank == 1,
        "hypothesis": selection["hypothesis"],
        "reason": selection["reason"],
        "confidence": selection["confidence"],
    }


def bootstrap_mean_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    estimates.sort()
    low_index = max(0, int(0.025 * samples) - 1)
    high_index = min(samples - 1, int(0.975 * samples))
    return estimates[low_index], estimates[high_index]


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    deltas = [float(row["llm_minus_random_expected"]) for row in rows]
    space_deltas = [float(row["llm_minus_space_filling"]) for row in rows]
    low, high = bootstrap_mean_interval(
        deltas, samples=bootstrap_samples, seed=seed
    )
    space_low, space_high = bootstrap_mean_interval(
        space_deltas, samples=bootstrap_samples, seed=seed + 1
    )
    return {
        "menu_count": len(rows),
        "mean_selected_value": sum(float(row["selected_value"]) for row in rows)
        / len(rows),
        "mean_random_expected_value": sum(
            float(row["random_expected_value"]) for row in rows
        )
        / len(rows),
        "mean_llm_minus_random_expected": sum(deltas) / len(deltas),
        "bootstrap_95ci_llm_minus_random_expected": [low, high],
        "mean_llm_minus_space_filling": sum(space_deltas) / len(space_deltas),
        "bootstrap_95ci_llm_minus_space_filling": [space_low, space_high],
        "mean_percentile": sum(float(row["percentile"]) for row in rows)
        / len(rows),
        "top_quartile_hit_rate": sum(bool(row["top_quartile_hit"]) for row in rows)
        / len(rows),
        "top_one_hit_rate": sum(bool(row["top_one_hit"]) for row in rows)
        / len(rows),
        "above_random_expected_count": sum(delta > 0 for delta in deltas),
        "equal_random_expected_count": sum(delta == 0 for delta in deltas),
        "below_random_expected_count": sum(delta < 0 for delta in deltas),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Run one frozen dataset per flag; omit to run the full suite.",
    )
    parser.add_argument("--api-key-env", default="COMMONSTACK_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.commonstack.ai/v1"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    available_datasets = {
        str(spec["dataset_id"]): spec for spec in protocol["datasets"]
    }
    unknown_datasets = sorted(set(args.dataset) - set(available_datasets))
    if unknown_datasets:
        parser.error(f"Unknown frozen datasets: {unknown_datasets}")
    dataset_specs = [
        spec
        for spec in protocol["datasets"]
        if not args.dataset or str(spec["dataset_id"]) in set(args.dataset)
    ]
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"Environment variable {args.api_key_env} is required")
    llm_config = replay.LLMConfig(
        base_url=args.base_url,
        api_key=api_key,
        model=str(protocol["model"]),
        temperature=float(protocol["temperature"]),
        max_tokens=int(protocol["max_tokens"]),
        api_mode="chat",
        structured_mode="tool",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "llm_trace.jsonl"
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing trace: {trace_path}")

    rows: list[dict[str, Any]] = []
    usage_totals: defaultdict[str, int] = defaultdict(int)
    dataset_summaries = {}
    dataset_offsets = {
        str(spec["dataset_id"]): offset
        for offset, spec in enumerate(protocol["datasets"])
    }
    for dataset_spec in dataset_specs:
        dataset_id = str(dataset_spec["dataset_id"])
        dataset_offset = dataset_offsets[dataset_id]
        adapter = replay.DATASET_BUILDERS[dataset_id]()
        public_fields = [str(field) for field in dataset_spec["public_fields"]]
        menus = build_menus(
            adapter,
            public_fields,
            seed_start=int(protocol["seed_start"]) + dataset_offset * 1000,
            menu_count=int(protocol["menus_per_dataset"]),
            menu_size=int(protocol["menu_size"]),
        )
        batch_size = int(protocol["batch_size"])
        selections_by_menu = {}
        for start in range(0, len(menus), batch_size):
            batch = menus[start : start + batch_size]
            selections, trace = call_selection_batch(
                llm_config, adapter, public_fields, batch
            )
            for attempt in trace["attempts"]:
                usage = attempt.get("metadata", {}).get("usage", {})
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage_totals[key] += int(usage.get(key, 0) or 0)
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "dataset_id": dataset_id,
                            "batch_start": start,
                            **trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            selections_by_menu.update(
                {str(selection["menu_id"]): selection for selection in selections}
            )

        dataset_rows = [
            evaluate_selection(adapter, menu, selections_by_menu[str(menu["menu_id"])])
            for menu in menus
        ]
        rows.extend(dataset_rows)
        dataset_summaries[dataset_id] = summarize_rows(
            dataset_rows,
            bootstrap_samples=int(protocol["bootstrap_samples"]),
            seed=int(protocol["bootstrap_seed"]) + dataset_offset * 10,
        )

    overall = summarize_rows(
        rows,
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        seed=int(protocol["bootstrap_seed"]) + 999,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "protocol": protocol,
        "executed_datasets": [
            str(spec["dataset_id"]) for spec in dataset_specs
        ],
        "evidence_boundary": {
            "source_campaigns": False,
            "source_outcomes": False,
            "target_outcomes_in_prompt": False,
            "target_history": False,
            "gp_scores": False,
            "rag_or_knowledge_base": False,
            "retrieved_or_frozen_skills": False,
            "candidate_conditions_only": True,
        },
        "dataset_summaries": dataset_summaries,
        "overall_pooled_menu_summary": overall,
        "llm_usage": dict(usage_totals),
        "claim_boundary": (
            "This tests one-step outcome-blind point choice from matched random "
            "menus. It is not sequential transfer, sample-efficiency evidence, "
            "or an estimate over arbitrary LLM generations."
        ),
    }
    write_csv(args.output_dir / "decisions.csv", rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "protocol_lock.json",
        {
            "config_path": str(args.config),
            "config_sha256": summary["config_sha256"],
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": protocol["model"],
            "executed_datasets": summary["executed_datasets"],
            "schema_version": SCHEMA_VERSION,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
