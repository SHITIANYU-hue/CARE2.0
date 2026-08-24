#!/usr/bin/env python3
"""Run frozen online-LLM confirmation with provider-aware request pacing."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_online_llm_scientist as online
import run_repeated_online_llm_confirmation_v2 as v2


SCHEMA_VERSION = "care.repeated_online_confirmation/v3"
SCRIPT_PATH = Path(__file__).resolve()
_ORIGINAL_PROTOCOL_LOCK = v2.protocol_lock
_ORIGINAL_VALIDATE_SUITE = v2.validate_suite


def validate_suite(suite: Mapping[str, Any], *, check_paths: bool = True) -> None:
    _ORIGINAL_VALIDATE_SUITE(suite, check_paths=check_paths)
    rate = suite.get("rate_control", {})
    required = {
        "min_request_interval_seconds",
        "max_call_attempts",
        "rate_limit_wait_seconds",
        "transient_wait_seconds",
        "request_timeout_seconds",
        "reasoning_effort",
    }
    missing = required - set(rate)
    if missing:
        raise ValueError(f"Missing rate_control fields: {sorted(missing)}")
    if float(rate["min_request_interval_seconds"]) < 0:
        raise ValueError("min_request_interval_seconds cannot be negative.")
    if int(rate["max_call_attempts"]) < 1:
        raise ValueError("max_call_attempts must be positive.")
    if str(rate["reasoning_effort"]) not in {"low", "high", "max"}:
        raise ValueError("reasoning_effort must be low, high, or max.")
    if suite["runner"].get("api_mode") != "chat":
        raise ValueError("The provider-aware v3 runner currently supports chat mode only.")


def protocol_lock(
    suite_config: Path,
    suite: Mapping[str, Any],
) -> dict[str, Any]:
    lock = _ORIGINAL_PROTOCOL_LOCK(suite_config, suite)
    lock["schema_version"] = SCHEMA_VERSION
    lock["artifacts"]["provider_rate_control_runner"] = {
        "path": str(SCRIPT_PATH),
        "sha256": v2.base.file_sha256(SCRIPT_PATH),
    }
    return lock


class ProviderAwareChatClient:
    """Pace requests and retry one API call without restarting its trajectory."""

    def __init__(self, rate: Mapping[str, Any]) -> None:
        self.min_interval = float(rate["min_request_interval_seconds"])
        self.max_attempts = int(rate["max_call_attempts"])
        self.rate_limit_wait = float(rate["rate_limit_wait_seconds"])
        self.transient_wait = float(rate["transient_wait_seconds"])
        self.timeout = float(rate["request_timeout_seconds"])
        self.reasoning_effort = str(rate["reasoning_effort"])
        self._lock = threading.Lock()
        self._last_request_started = 0.0

    def _pace(self) -> float:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_started
            wait = max(0.0, self.min_interval - elapsed)
            if wait:
                time.sleep(wait)
            self._last_request_started = time.monotonic()
            return wait

    @staticmethod
    def _retry_after(headers: Any) -> float | None:
        if headers is None:
            return None
        raw = headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    def __call__(
        self,
        config: Any,
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        if config.api_mode != "chat" or config.structured_mode != "json":
            raise ValueError(
                "Provider-aware runner requires chat mode with JSON structured output."
            )
        payload = {
            "model": config.model,
            "messages": [dict(message) for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
            "reasoning_effort": self.reasoning_effort,
        }
        endpoint = config.base_url.rstrip("/") + "/chat/completions"
        call_id = f"{time.time_ns()}"
        call_started = time.monotonic()
        retryable_http = {408, 409, 425, 429, 500, 502, 503, 504}
        retryable_errors = (
            TimeoutError,
            socket.timeout,
            ConnectionError,
            urllib.error.URLError,
        )
        online.replay.trace_llm_event(
            {
                "event": "provider_call_start",
                "call_id": call_id,
                "model": config.model,
                "reasoning_effort": self.reasoning_effort,
                "max_tokens": config.max_tokens,
                "max_call_attempts": self.max_attempts,
                "min_request_interval_seconds": self.min_interval,
            }
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            paced_seconds = self._pace()
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + config.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            attempt_started = time.monotonic()
            online.replay.trace_llm_event(
                {
                    "event": "provider_attempt_start",
                    "call_id": call_id,
                    "attempt": attempt,
                    "paced_seconds": round(paced_seconds, 3),
                }
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                last_error = RuntimeError(
                    f"LLM endpoint returned HTTP {exc.code}: {body[:500]}"
                )
                retryable = exc.code in retryable_http
                if not retryable or attempt >= self.max_attempts:
                    online.replay.trace_llm_event(
                        {
                            "event": "provider_call_error",
                            "call_id": call_id,
                            "attempt": attempt,
                            "http_code": exc.code,
                            "retryable": retryable,
                        }
                    )
                    raise last_error from exc
                retry_after = self._retry_after(exc.headers)
                wait = max(
                    self.rate_limit_wait if exc.code == 429 else self.transient_wait,
                    retry_after or 0.0,
                )
                online.replay.trace_llm_event(
                    {
                        "event": "provider_attempt_retry",
                        "call_id": call_id,
                        "attempt": attempt,
                        "http_code": exc.code,
                        "wait_seconds": wait,
                    }
                )
                time.sleep(wait)
                continue
            except retryable_errors as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    raise RuntimeError(
                        f"LLM endpoint request failed after retries: {exc}"
                    ) from exc
                online.replay.trace_llm_event(
                    {
                        "event": "provider_attempt_retry",
                        "call_id": call_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "wait_seconds": self.transient_wait,
                    }
                )
                time.sleep(self.transient_wait)
                continue
            usage = data.get("usage", {})
            message = data["choices"][0]["message"]
            content = str(message.get("content", ""))
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                content = (
                    tool_calls[0].get("function", {}).get("arguments", "") or content
                )
            online.replay.trace_llm_event(
                {
                    "event": "provider_call_ok",
                    "call_id": call_id,
                    "attempt": attempt,
                    "elapsed_seconds": round(time.monotonic() - call_started, 3),
                    "response_model": data.get("model", config.model),
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "usage": usage,
                }
            )
            return content, {
                "model": data.get("model", config.model),
                "usage": usage,
            }
        raise RuntimeError(f"LLM endpoint request failed after retries: {last_error}")


def _suite_path(argv: Sequence[str]) -> Path:
    try:
        index = argv.index("--suite-config")
        return Path(argv[index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("--suite-config is required") from exc


def main() -> None:
    suite = v2.base.load_json(_suite_path(sys.argv))
    validate_suite(suite)
    client = ProviderAwareChatClient(suite["rate_control"])
    online.replay.chat_completion_text = client
    v2.validate_suite = validate_suite
    v2.protocol_lock = protocol_lock
    v2.SCHEMA_VERSION = SCHEMA_VERSION
    v2.main()


if __name__ == "__main__":
    main()
