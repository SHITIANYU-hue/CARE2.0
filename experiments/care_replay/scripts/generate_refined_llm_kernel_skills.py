#!/usr/bin/env python3
"""Refine an LLM kernel-skill portfolio from calibration-only diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import run_llm_kernel_skill_evolution as evolution
import run_synthetic_suzuki as replay


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_refinement_payload(
    llm_record: dict[str, Any],
    calibration_summary: dict[str, Any],
    patch_count: int,
) -> dict[str, Any]:
    selection = calibration_summary["selection"]
    diagnostics = selection["patch_diagnostics"]
    patches = []
    for patch in llm_record.get("normalized_patches", []):
        mode = f"llm_evolved_kernel_{patch['patch_id']}"
        if mode not in diagnostics:
            continue
        evidence = diagnostics[mode]
        patches.append(
            {
                "patch": patch,
                "calibration_only_diagnostics_vs_target_anchor": {
                    "final_best": evidence["final_best"],
                    "best_so_far_auc": evidence["best_so_far_auc"],
                    "composite": evidence["composite"],
                    "positive_fold_rate": evidence.get("positive_fold_rate"),
                    "fold_composite_means": evidence.get("fold_composite_means", []),
                    "risk_adjusted_composite_gain": evidence[
                        "risk_adjusted_composite_gain"
                    ],
                },
            }
        )
    prompt_payload = llm_record.get("prompt_payload", {})
    return {
        "task": (
            "Evolve the first-round source-to-target kernel skills using calibration-only "
            "performance. Produce a frozen second-round portfolio for evaluation on unseen seeds."
        ),
        "source_dataset": calibration_summary["source_dataset"],
        "target_dataset": calibration_summary["target_dataset"],
        "target_schema": prompt_payload.get("target_schema", {}),
        "target_anchor_mode": selection["target_anchor_mode"],
        "calibration_seed_count": selection["calibration_seed_count"],
        "first_round_patches": patches,
        "evidence_boundary": (
            "Only calibration diagnostics are present. Held-out metrics, candidate identities, "
            "and unrevealed target outcomes are excluded."
        ),
        "refinement_rules": [
            "Optimize paired Final Best and Best-So-Far AUC together; do not trade a large final loss for a small AUC gain.",
            "Preserve or gently mutate skills that are positive across calibration folds.",
            "Reduce the transfer scale or add scale 0 when a skill has high variance or poor non-loss rate.",
            "Use role multipliers below 0.8 for weak, uncertain, or failure-associated mappings.",
            "Retain one exploration-first beta schedule, one conservative schedule, one counter-hypothesis, and one multi-scale anchor.",
            "Do not reuse one confidence value across all patches; calibrate confidence from support, fold stability, and uncertainty.",
            "Return executable parameter patches only. Do not select target candidates or discuss held-out results.",
        ],
        "allowed_space": prompt_payload.get("allowed_space", {}),
        "output_contract": prompt_payload.get("output_contract", {}),
        "patch_count": patch_count,
    }


def refine(
    llm_record: dict[str, Any],
    calibration_summary: dict[str, Any],
    target_fields: tuple[str, ...],
    config: replay.LLMConfig,
    patch_count: int,
) -> tuple[tuple[evolution.KernelSkillPatch, ...], dict[str, Any]]:
    payload = build_refinement_payload(llm_record, calibration_summary, patch_count)
    system = (
        "You are the skill-evolution component of CARE 2.0. Refine an executable portfolio "
        "from calibration-only evidence. Favor effects that are stable across folds, preserve "
        "target-only anchors, and reduce negative-transfer risk. Return only valid JSON."
    )
    user = (
        f"Generate exactly {patch_count} refined kernel skill patches. JSON input:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    content, response_meta = replay.chat_completion_text(
        config,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    parse_error = ""
    try:
        parsed = replay.extract_json_object(content)
    except ValueError as exc:
        parsed = {}
        parse_error = str(exc)
    patches = evolution.normalize_patches(parsed, target_fields, patch_count)
    if not patches:
        raise RuntimeError(
            f"LLM returned no usable refined kernel skill patches: {parse_error or content[:300]}"
        )
    record = {
        "experiment": "care_llm_kernel_skill_refinement",
        "model": response_meta["model"],
        "usage": response_meta["usage"],
        "llm_generation_call_count": int(
            llm_record.get("llm_generation_call_count", 1)
        ) + 1,
        "api_configuration": {
            "base_url": config.base_url,
            "api_mode": config.api_mode,
            "structured_mode": config.structured_mode,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        "prompt_payload": payload,
        "raw_response": content,
        "parsed_response": parsed,
        "normalized_patches": [asdict(patch) for patch in patches],
        "parse_error": parse_error,
    }
    return patches, record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a second-round LLM skill portfolio from calibration-only evidence."
    )
    parser.add_argument("--llm-record", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--target-dataset", required=True, choices=replay.DATASET_BUILDERS.keys())
    parser.add_argument("--patch-count", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("CARE_LLM_BASE_URL", "https://api.commonstack.ai/v1"),
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("CARE_LLM_MODEL", "openai/gpt-4o-mini"),
    )
    parser.add_argument("--llm-api-key-env", default="CARE_LLM_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument("--llm-max-tokens", type=int, default=2600)
    parser.add_argument("--llm-api-mode", choices=("chat", "completion"), default="chat")
    parser.add_argument("--llm-structured-mode", choices=("tool", "json"), default="json")
    args = parser.parse_args()
    api_key = os.environ.get(args.llm_api_key_env) or os.environ.get("CARE_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(f"Set {args.llm_api_key_env} or CARE_LLM_API_KEY.")
    llm_record = json.loads(args.llm_record.read_text(encoding="utf-8"))
    calibration_summary = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    target = replay.DATASET_BUILDERS[args.target_dataset]()
    config = replay.LLMConfig(
        base_url=args.llm_base_url,
        api_key=api_key,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_mode=args.llm_api_mode,
        structured_mode=args.llm_structured_mode,
    )
    patches, record = refine(
        llm_record,
        calibration_summary,
        target.decision_columns,
        config,
        args.patch_count,
    )
    record["source_llm_record_sha256"] = file_sha256(args.llm_record)
    record["calibration_summary_sha256"] = file_sha256(args.calibration_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": record["model"],
                "usage": record["usage"],
                "patch_ids": [patch.patch_id for patch in patches],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
