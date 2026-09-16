#!/usr/bin/env python3
"""Run the repeated online-LLM protocol with auditable infrastructure retries."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Mapping

import run_online_llm_scientist as online
import run_repeated_online_llm_confirmation as base


SCHEMA_VERSION = "care.repeated_online_confirmation/v2"
SCRIPT_PATH = Path(__file__).resolve()
ONLINE_RUNNER_PATH = SCRIPT_PATH.with_name("run_online_llm_scientist.py")
BASE_ANALYSIS_PATH = SCRIPT_PATH.with_name(
    "run_repeated_online_llm_confirmation.py"
)


def validate_suite(suite: Mapping[str, Any], *, check_paths: bool = True) -> None:
    base.validate_suite(suite, check_paths=check_paths)
    policy = suite.get("retry_policy", {})
    if int(policy.get("max_infrastructure_attempts", 0)) < 1:
        raise ValueError("retry_policy.max_infrastructure_attempts must be positive.")
    if policy.get("model_validation_failure") != "terminal":
        raise ValueError("Model validation failures must remain terminal.")
    if policy.get("scientific_outcome_failure") != "not_retryable":
        raise ValueError("Scientific outcomes must never trigger retries.")


def protocol_lock(suite_config: Path, suite: Mapping[str, Any]) -> dict[str, Any]:
    lock = base.protocol_lock(suite_config, suite)
    lock["schema_version"] = SCHEMA_VERSION
    lock["artifacts"]["runner_script"] = {
        "path": str(SCRIPT_PATH),
        "sha256": base.file_sha256(SCRIPT_PATH),
    }
    lock["artifacts"]["base_analysis_script"] = {
        "path": str(BASE_ANALYSIS_PATH),
        "sha256": base.file_sha256(BASE_ANALYSIS_PATH),
    }
    lock["artifacts"]["online_scientist_runner"] = {
        "path": str(ONLINE_RUNNER_PATH),
        "sha256": base.file_sha256(ONLINE_RUNNER_PATH),
    }
    return lock


def ensure_protocol_lock(
    output_root: Path,
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_root / "protocol_lock.json"
    candidate = protocol_lock(suite_config, suite)
    if path.exists():
        existing = base.load_json(path)
        if base.lock_identity(existing) != base.lock_identity(candidate):
            raise ValueError(
                "Frozen protocol artifacts changed after the output lock was created."
            )
        return existing
    base.write_json(path, candidate)
    online.write_fingerprint(path)
    return candidate


def retryable_infrastructure_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    infrastructure_markers = (
        "http 408",
        "http 409",
        "http 425",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "status 408",
        "status 429",
        "status 500",
        "status 502",
        "status 503",
        "status 504",
        "rate limit",
        "max cost limit",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "connection aborted",
        "timed out",
        "timeout",
        "dns",
    )
    return (
        any(marker in message for marker in infrastructure_markers)
        or "timeout" in name
        or "connection" in name
    )


def attempt_dir(trajectory: Path, attempt_index: int) -> Path:
    return trajectory / "attempts" / f"attempt_{attempt_index:02d}"


def read_attempts(trajectory: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for path in sorted((trajectory / "attempts").glob("attempt_*/attempt_metadata.json")):
        payload = base.load_json(path)
        payload["metadata_path"] = str(path)
        attempts.append(payload)
    return attempts


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def trajectory_lock(trajectory: Path):
    trajectory.mkdir(parents=True, exist_ok=True)
    lock_path = trajectory / ".trajectory.lock"
    payload = {"pid": os.getpid(), "host": socket.gethostname()}
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = base.load_json(lock_path)
        same_host = existing.get("host") == socket.gethostname()
        running = same_host and _pid_is_running(int(existing.get("pid", -1)))
        if running:
            raise RuntimeError(f"Trajectory is already running: {trajectory}")
        lock_path.unlink()
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    try:
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()


def mark_interrupted_attempts(trajectory: Path) -> None:
    for attempt in read_attempts(trajectory):
        if attempt.get("status") != "running":
            continue
        metadata_path = Path(str(attempt["metadata_path"]))
        attempt.update(
            {
                "status": "infrastructure_error",
                "finished_at": base.utc_now(),
                "retryable": True,
                "error_type": "InterruptedAttempt",
                "error": "Previous process ended before the attempt completed.",
            }
        )
        attempt.pop("metadata_path", None)
        base.write_json(metadata_path, attempt)
        base.write_json(
            metadata_path.parent / "error.json",
            {
                "error_type": "InterruptedAttempt",
                "error": "Previous process ended before the attempt completed.",
                "retryable": True,
            },
        )
        online.write_fingerprint(metadata_path)
        online.write_fingerprint(metadata_path.parent / "error.json")


def promote_success(attempt: Path, trajectory: Path) -> None:
    for name in (
        "summary.json",
        "summary.json.sha256",
        "llm_trace.jsonl",
        "llm_trace.jsonl.sha256",
        "initial_record.json",
        "initial_record.json.sha256",
    ):
        source = attempt / name
        destination = trajectory / name
        if not source.exists() or destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def run_trajectory(
    case: Mapping[str, Any],
    suite: Mapping[str, Any],
    trajectory: Path,
    replicate_id: int,
    lock: Mapping[str, Any],
) -> None:
    if (trajectory / "summary.json").exists():
        return
    max_attempts = int(suite["retry_policy"]["max_infrastructure_attempts"])
    with trajectory_lock(trajectory):
        mark_interrupted_attempts(trajectory)
        prior_attempts = read_attempts(trajectory)
        if prior_attempts and prior_attempts[-1].get("status") in {
            "model_validation_error",
            "terminal_error",
        }:
            raise RuntimeError("Trajectory already has a terminal failure.")
        next_index = len(prior_attempts) + 1
        while next_index <= max_attempts:
            attempt = attempt_dir(trajectory, next_index)
            attempt.mkdir(parents=True, exist_ok=False)
            metadata_path = attempt / "attempt_metadata.json"
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "suite": suite["suite"],
                "case_id": case["case_id"],
                "replicate_id": replicate_id,
                "attempt_index": next_index,
                "status": "running",
                "started_at": base.utc_now(),
                "model": suite["runner"]["model"],
                "temperature": suite["runner"]["temperature"],
                "protocol_lock_sha256": base.file_sha256(
                    trajectory.parents[1] / "protocol_lock.json"
                ),
                "git_commit": lock.get("git_commit"),
            }
            base.write_json(metadata_path, metadata)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    online.run_online(base.run_args(case, suite, attempt))
            except Exception as exc:
                retryable = retryable_infrastructure_error(exc)
                status = (
                    "infrastructure_error" if retryable else "model_validation_error"
                )
                metadata.update(
                    {
                        "status": status,
                        "finished_at": base.utc_now(),
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                )
                base.write_json(metadata_path, metadata)
                base.write_json(
                    attempt / "error.json",
                    {
                        "case_id": case["case_id"],
                        "replicate_id": replicate_id,
                        "attempt_index": next_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "retryable": retryable,
                    },
                )
                online.write_fingerprint(metadata_path)
                online.write_fingerprint(attempt / "error.json")
                if retryable and next_index < max_attempts:
                    next_index += 1
                    continue
                base.write_json(
                    trajectory / "error.json",
                    {
                        "case_id": case["case_id"],
                        "replicate_id": replicate_id,
                        "attempts": next_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                        "retryable_attempts_exhausted": retryable,
                    },
                )
                online.write_fingerprint(trajectory / "error.json")
                raise
            metadata.update(
                {"status": "complete", "finished_at": base.utc_now()}
            )
            base.write_json(metadata_path, metadata)
            online.write_fingerprint(metadata_path)
            promote_success(attempt, trajectory)
            base.write_json(
                trajectory / "trajectory_metadata.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "suite": suite["suite"],
                    "case_id": case["case_id"],
                    "replicate_id": replicate_id,
                    "status": "complete",
                    "successful_attempt": next_index,
                    "infrastructure_failures_before_success": next_index - 1,
                    "finished_at": base.utc_now(),
                },
            )
            online.write_fingerprint(trajectory / "trajectory_metadata.json")
            return
        raise RuntimeError("Infrastructure attempts exhausted.")


def attempt_statistics(output_root: Path) -> dict[str, int]:
    counts = {
        "complete": 0,
        "infrastructure_error": 0,
        "model_validation_error": 0,
        "running": 0,
    }
    for path in output_root.glob(
        "*/trajectory_*/attempts/attempt_*/attempt_metadata.json"
    ):
        status = str(base.load_json(path).get("status", "running"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def aggregate(output_root: Path, suite: Mapping[str, Any]) -> dict[str, Any]:
    report = base.aggregate(output_root, suite)
    report["schema_version"] = SCHEMA_VERSION
    report["attempt_statistics"] = attempt_statistics(output_root)
    report["retry_policy"] = dict(suite["retry_policy"])
    path = output_root / "aggregate" / "repeated_confirmation.json"
    base.write_json(path, report)
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
    suite = base.load_json(suite_config)
    validate_suite(suite)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = ensure_protocol_lock(args.output_root, suite_config, suite)

    selected_cases = set(args.case)
    selected_replicates = set(args.replicate)
    expected_replicates = set(base.replicate_ids(suite))
    if selected_replicates - expected_replicates:
        parser.error("Requested replicate is outside the frozen protocol.")
    jobs: list[tuple[Mapping[str, Any], int, Path]] = []
    for replicate_id in base.replicate_ids(suite):
        if selected_replicates and replicate_id not in selected_replicates:
            continue
        for case in suite["cases"]:
            case_id = str(case["case_id"])
            if selected_cases and case_id not in selected_cases:
                continue
            trajectory = base.trajectory_dir(
                args.output_root, case_id, replicate_id
            )
            if (trajectory / "summary.json").exists() or (
                trajectory / "error.json"
            ).exists():
                continue
            jobs.append((case, replicate_id, trajectory))
    if args.max_new_runs is not None:
        if args.max_new_runs < 0:
            parser.error("--max-new-runs cannot be negative.")
        jobs = jobs[: args.max_new_runs]

    if not args.aggregate_only:
        for case, replicate_id, trajectory in jobs:
            try:
                run_trajectory(case, suite, trajectory, replicate_id, lock)
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
