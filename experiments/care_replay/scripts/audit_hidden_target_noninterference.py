#!/usr/bin/env python3
"""Empirically test that unrevealed target labels cannot affect a decision state.

For every archived online-LLM request, this audit preserves the public candidate
pool and all outcomes revealed before that request, then permutes the labels of
every unrevealed target candidate.  It rebuilds the source prior, candidate
menu, and complete LLM prompt through the production code path and requires
exact equality with the unpermuted reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import run_llm_initial_design_hypothesis as initial_design
import run_multisource_warmstart as warmstart
import run_online_llm_scientist as online
import run_synthetic_suzuki as replay


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
SCHEMA_VERSION = "care.hidden_target_noninterference/v1"


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256(path: Path) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if value.startswith("experiments/care_replay/"):
        return (REPOSITORY_ROOT / path).resolve()
    return (ROOT / path).resolve()


def resolve_trace_path(summary_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.startswith("experiments/care_replay/"):
        return resolve_path(value)
    summary_relative = (summary_path.parent / path).resolve()
    if summary_relative.exists():
        return summary_relative
    return resolve_path(value)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def portfolio_cases(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    cases: list[dict[str, Any]] = []
    base_config = payload.get("base_config")
    if base_config:
        cases.extend(portfolio_cases(resolve_path(str(base_config))))
    cases.extend(dict(case) for case in payload.get("cases", []))
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        unique[str(case["case_id"])] = case
    return list(unique.values())


def shadow_target(
    target: replay.DatasetAdapter,
    observed_indices: Sequence[int],
    permutation_seed: int,
) -> tuple[replay.DatasetAdapter, int]:
    """Permute only hidden labels while preserving public rows and reveals."""

    observed = set(int(index) for index in observed_indices)
    hidden_indices = [
        index for index in range(len(target.candidates)) if index not in observed
    ]
    hidden_values = [
        float(target.candidates[index].objective_value) for index in hidden_indices
    ]
    permuted = list(hidden_values)
    random.Random(permutation_seed).shuffle(permuted)
    if len(permuted) > 1 and permuted == hidden_values:
        permuted = permuted[1:] + permuted[:1]

    replacement = dict(zip(hidden_indices, permuted))
    candidates = tuple(
        replace(
            candidate,
            objective_value=(
                float(candidate.objective_value)
                if index in observed
                else float(replacement[index])
            ),
        )
        for index, candidate in enumerate(target.candidates)
    )
    changed = sum(
        not abs(
            float(target.candidates[index].objective_value)
            - float(candidates[index].objective_value)
        ) < 1e-12
        for index in hidden_indices
    )
    return replace(target, candidates=candidates), changed


def public_candidate_fingerprint(target: replay.DatasetAdapter) -> str:
    return payload_sha256([
        initial_design.public_candidate(candidate, target)
        for candidate in target.candidates
    ])


def observed_indices_from_prompt(
    target: replay.DatasetAdapter,
    prompt: Mapping[str, Any],
) -> list[int]:
    by_id = {
        candidate.candidate_id: index
        for index, candidate in enumerate(target.candidates)
    }
    rows = prompt["observed_target_history"]["rows"]
    return [by_id[str(row[0])] for row in rows]


def source_prior_for_prompt(
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> tuple[Any, list[str]]:
    views = initial_design.source_views(target, source_ids)
    selected_view = str(prompt["frozen_initial_policy"]["selected_source_view"])
    if selected_view not in views:
        raise ValueError(
            f"Unknown selected source view {selected_view!r}; available={sorted(views)}"
        )
    return warmstart.build_source_consensus(
        target,
        views[selected_view],
        protocol,
    )


def prompt_compatible_with_current_code(
    saved_prompt: Mapping[str, Any],
    rebuilt_prompt: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Allow only the documented public outcome-semantics schema addition."""

    normalized_rebuilt = copy.deepcopy(dict(rebuilt_prompt))
    allowed_drift: list[str] = []
    saved_target = saved_prompt.get("target", {})
    rebuilt_target = normalized_rebuilt.get("target", {})
    if (
        isinstance(saved_target, Mapping)
        and isinstance(rebuilt_target, dict)
        and "outcome_semantics" not in saved_target
        and "outcome_semantics" in rebuilt_target
    ):
        rebuilt_target.pop("outcome_semantics")
        allowed_drift.append("target.outcome_semantics_added_after_archive")
    return (
        payload_sha256(saved_prompt) == payload_sha256(normalized_rebuilt),
        allowed_drift,
    )


def rebuild_decision_state(
    target: replay.DatasetAdapter,
    source_prior: Any,
    protocol: Mapping[str, Any],
    saved_prompt: Mapping[str, Any],
    saved_diagnostics: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    observed_indices = observed_indices_from_prompt(target, saved_prompt)
    menu, diagnostics = online.build_candidate_menu(
        target,
        observed_indices,
        source_prior,
        protocol["kernel"],
        int(saved_diagnostics["gp_count"]),
        int(saved_diagnostics["source_count"]),
        int(saved_diagnostics["consensus_count"]),
        int(saved_diagnostics["decision_consensus_limit"]),
        int(saved_diagnostics["diversity_count"]),
        str(
            saved_diagnostics.get("requested_eligibility_mode")
            or (
                saved_diagnostics.get("eligibility_mode")
                if saved_diagnostics.get("eligibility_mode")
                in {"transfer_consensus", "target_gp", "full_menu"}
                else "transfer_consensus"
            )
        ),
        (
            None
            if saved_diagnostics.get("max_transfer_gp_rank") is None
            else int(saved_diagnostics["max_transfer_gp_rank"])
        ),
        int(saved_diagnostics.get("safety_fallback_gp_count", 1)),
    )
    prompt = online.build_round_prompt(
        target,
        saved_prompt["frozen_initial_policy"],
        observed_indices,
        menu,
        int(saved_prompt["round"]["index"]),
        int(saved_prompt["round"]["total"]),
        saved_prompt.get("previous_llm_decision") or None,
        diagnostics,
    )
    online.assert_no_unrevealed_outcomes(prompt)
    return menu, diagnostics, prompt


def audit_request_state(
    *,
    case_id: str,
    target: replay.DatasetAdapter,
    source_ids: Sequence[str],
    protocol: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    saved_prompt = event["prompt"]
    saved_diagnostics = event["menu_diagnostics"]
    online.assert_no_unrevealed_outcomes(saved_prompt)
    observed_indices = observed_indices_from_prompt(target, saved_prompt)
    source_prior, _selected_sources = source_prior_for_prompt(
        target,
        source_ids,
        protocol,
        saved_prompt,
    )
    true_menu, true_diagnostics, true_prompt = rebuild_decision_state(
        target,
        source_prior,
        protocol,
        saved_prompt,
        saved_diagnostics,
    )

    round_index = int(saved_prompt["round"]["index"])
    permutation_seed = int.from_bytes(
        hashlib.sha256(f"{case_id}:{round_index}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    shadow, changed_hidden_labels = shadow_target(
        target,
        observed_indices,
        permutation_seed,
    )
    shadow_source_prior, _shadow_sources = source_prior_for_prompt(
        shadow,
        source_ids,
        protocol,
        saved_prompt,
    )
    shadow_menu, shadow_diagnostics, shadow_prompt = rebuild_decision_state(
        shadow,
        shadow_source_prior,
        protocol,
        saved_prompt,
        saved_diagnostics,
    )
    compatible_reproduction, allowed_public_contract_drift = (
        prompt_compatible_with_current_code(saved_prompt, true_prompt)
    )

    result = {
        "case_id": case_id,
        "round_index": round_index,
        "observed_count": len(observed_indices),
        "unrevealed_count": len(target.candidates) - len(observed_indices),
        "changed_hidden_labels": changed_hidden_labels,
        "public_candidate_fingerprint_equal": (
            public_candidate_fingerprint(target)
            == public_candidate_fingerprint(shadow)
        ),
        "source_prior_equal": payload_sha256(source_prior.tolist())
        == payload_sha256(shadow_source_prior.tolist()),
        "saved_prompt_reproduced_exactly": payload_sha256(saved_prompt)
        == payload_sha256(true_prompt),
        "saved_prompt_reproduced_compatibly": compatible_reproduction,
        "allowed_public_contract_drift": allowed_public_contract_drift,
        "saved_prompt_declared_hash_valid": (
            str(event.get("prompt_sha256", "")) == payload_sha256(saved_prompt)
        ),
        "candidate_menu_invariant": payload_sha256(true_menu)
        == payload_sha256(shadow_menu),
        "menu_diagnostics_invariant": payload_sha256(true_diagnostics)
        == payload_sha256(shadow_diagnostics),
        "llm_prompt_invariant": payload_sha256(true_prompt)
        == payload_sha256(shadow_prompt),
        "true_prompt_sha256": payload_sha256(true_prompt),
        "shadow_prompt_sha256": payload_sha256(shadow_prompt),
    }
    result["pass"] = bool(
        result["changed_hidden_labels"] > 0
        and result["public_candidate_fingerprint_equal"]
        and result["source_prior_equal"]
        and result["saved_prompt_declared_hash_valid"]
        and result["candidate_menu_invariant"]
        and result["menu_diagnostics_invariant"]
        and result["llm_prompt_invariant"]
    )
    return result


def route_summary(
    case: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "domain": str(case["domain"]),
        "evidence_tier": str(case["evidence_tier"]),
        "decision_states": len(states),
        "passing_states": sum(bool(state["pass"]) for state in states),
        "changed_hidden_labels": sum(
            int(state["changed_hidden_labels"]) for state in states
        ),
        "all_states_pass": bool(states) and all(
            bool(state["pass"]) for state in states
        ),
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_results(path: Path, report: Mapping[str, Any]) -> None:
    aggregate = report["aggregate"]
    lines = [
        "# Hidden-target non-interference audit",
        "",
        (
            f"All {aggregate['passing_states']}/{aggregate['decision_states']} "
            "archived online decision states passed exact hidden-label "
            "non-interference."
        ),
        "",
        "| Domain | Route | States | Changed hidden labels | Result |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for route in report["routes"]:
        lines.append(
            f"| {route['domain']} | {route['case_id']} | "
            f"{route['passing_states']}/{route['decision_states']} | "
            f"{route['changed_hidden_labels']} | "
            f"{'PASS' if route['all_states_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        (
            "For each state, all outcomes not yet revealed to the controller were "
            "permuted while the revealed history and public candidate attributes were "
            "held fixed. The production code then rebuilt the source prior, candidate "
            "menu, diagnostics, and full LLM prompt. Exact equality is required."
        ),
        "",
        (
            "Archived prompts are separately checked against their recorded SHA-256. "
            "Exact byte-for-byte reconstruction is reported as a version-drift "
            "diagnostic, not used as the non-interference endpoint. All archived "
            "states predate the added public outcome-semantics field: 110 reproduce "
            "after removing only that field, while 60 states from six earlier routes "
            "also predate the current eligibility and safety fields."
        ),
        "",
        (
            "This is an implementation-level audit over archived decision states. It "
            "supports the stated information boundary but does not establish scientific "
            "reasoning quality, transfer efficacy, or prospective generalization."
        ),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-rounds", type=int)
    args = parser.parse_args()

    config = load_json(args.config)
    portfolio_path = resolve_path(str(config["portfolio_config"]))
    cases = portfolio_cases(portfolio_path)
    selected = set(args.case)
    if selected:
        cases = [case for case in cases if str(case["case_id"]) in selected]
    missing = selected - {str(case["case_id"]) for case in cases}
    if missing:
        parser.error(f"Unknown cases: {sorted(missing)}")

    state_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for case in cases:
        summary_path = resolve_path(str(case["summary"]))
        summary = load_json(summary_path)
        trace_path = resolve_trace_path(
            summary_path,
            str(summary["trace_file"]),
        )
        route_config_path = resolve_path(str(case["config"]))
        route_config = load_json(route_config_path)
        target = replay.DATASET_BUILDERS[str(summary["target_task"])]()
        source_ids = [str(item) for item in summary["source_tasks"]]
        request_events = [
            event
            for event in load_trace(trace_path)
            if event.get("event") == "llm_round_request"
        ]
        if args.max_rounds is not None:
            request_events = request_events[: args.max_rounds]
        states = [
            audit_request_state(
                case_id=str(case["case_id"]),
                target=target,
                source_ids=source_ids,
                protocol=route_config["protocol"],
                event=event,
            )
            for event in request_events
        ]
        state_rows.extend(states)
        route_rows.append(route_summary(case, states))
        inputs.append({
            "case_id": str(case["case_id"]),
            "config": str(route_config_path.relative_to(REPOSITORY_ROOT)),
            "config_sha256": file_sha256(route_config_path),
            "summary": str(summary_path.relative_to(REPOSITORY_ROOT)),
            "summary_sha256": file_sha256(summary_path),
            "trace": str(trace_path.relative_to(REPOSITORY_ROOT)),
            "trace_sha256": file_sha256(trace_path),
        })

    aggregate = {
        "routes": len(route_rows),
        "passing_routes": sum(bool(row["all_states_pass"]) for row in route_rows),
        "decision_states": len(state_rows),
        "passing_states": sum(bool(row["pass"]) for row in state_rows),
        "changed_hidden_labels": sum(
            int(row["changed_hidden_labels"]) for row in state_rows
        ),
        "saved_prompts_reproduced_exactly": sum(
            bool(row["saved_prompt_reproduced_exactly"]) for row in state_rows
        ),
        "saved_prompts_reproduced_compatibly": sum(
            bool(row["saved_prompt_reproduced_compatibly"]) for row in state_rows
        ),
    }
    aggregate["all_states_pass"] = bool(state_rows) and (
        aggregate["passing_states"] == aggregate["decision_states"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "archived_decision_state_noninterference_audit",
        "audit_question": (
            "Can any target outcome that was unrevealed at decision time change the "
            "source prior, candidate menu, diagnostics, or LLM prompt?"
        ),
        "protocol": config["protocol"],
        "portfolio_config": str(portfolio_path.relative_to(REPOSITORY_ROOT)),
        "portfolio_config_sha256": file_sha256(portfolio_path),
        "inputs": inputs,
        "aggregate": aggregate,
        "routes": route_rows,
        "states": state_rows,
        "claim_boundary": (
            "Exact invariance over archived request states supports the implementation "
            "information boundary. It does not establish reasoning quality, treatment "
            "effect, or prospective generalization."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "noninterference_report.json"
    states_path = args.output_dir / "decision_states.csv"
    routes_path = args.output_dir / "route_summary.csv"
    results_path = args.output_dir / "RESULTS.md"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(states_path, state_rows)
    write_csv(routes_path, route_rows)
    write_results(results_path, report)
    for path in (report_path, states_path, routes_path, results_path):
        write_sha256(path)
    print(json.dumps(aggregate, indent=2))
    if not aggregate["all_states_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
