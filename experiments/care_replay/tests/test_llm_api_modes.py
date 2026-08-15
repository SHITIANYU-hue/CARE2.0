from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_synthetic_suzuki as replay  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LlmApiModeTest(unittest.TestCase):
    def test_completion_mode_uses_prompt_endpoint_and_preserves_usage(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "model": "cheap-model",
                    "choices": [{"text": '{"patches": []}'}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }
            )

        config = replay.LLMConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="cheap-model",
            temperature=0.0,
            max_tokens=128,
            api_mode="completion",
        )
        with mock.patch.object(replay.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, metadata = replay.chat_completion_text(
                config,
                [
                    {"role": "system", "content": "Return JSON."},
                    {"role": "user", "content": "Make one patch."},
                ],
            )

        self.assertEqual(captured["url"], "https://example.test/v1/completions")
        payload = captured["payload"]
        self.assertIn("SYSTEM:\nReturn JSON.", payload["prompt"])  # type: ignore[index]
        self.assertIn("USER:\nMake one patch.", payload["prompt"])  # type: ignore[index]
        self.assertNotIn("tools", payload)  # type: ignore[operator]
        self.assertEqual(content, '{"patches": []}')
        self.assertEqual(metadata["usage"]["completion_tokens"], 4)

    def test_chat_json_mode_omits_tool_call_contract(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            return FakeResponse(
                {
                    "model": "cheap-chat-model",
                    "choices": [{"message": {"content": '{"patches": []}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }
            )

        config = replay.LLMConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="cheap-chat-model",
            temperature=0.0,
            max_tokens=128,
            structured_mode="json",
        )
        with mock.patch.object(replay.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, _metadata = replay.chat_completion_text(
                config,
                [{"role": "user", "content": "Make one patch."}],
            )

        self.assertEqual(captured["url"], "https://example.test/v1/chat/completions")
        payload = captured["payload"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})  # type: ignore[index]
        self.assertNotIn("tools", payload)  # type: ignore[operator]
        self.assertNotIn("tool_choice", payload)  # type: ignore[operator]
        self.assertEqual(content, '{"patches": []}')

    def test_anthropic_mode_uses_messages_endpoint_and_normalizes_usage(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["payload"] = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
            captured["headers"] = dict(request.header_items())  # type: ignore[attr-defined]
            return FakeResponse(
                {
                    "model": "claude-test",
                    "content": [{"type": "text", "text": '{"patches": []}'}],
                    "usage": {"input_tokens": 12, "output_tokens": 5},
                }
            )

        config = replay.LLMConfig(
            base_url="http://10.168.0.1:3000/api",
            api_key="secret",
            model="claude-test",
            temperature=0.0,
            max_tokens=128,
            api_mode="anthropic",
            structured_mode="json",
        )
        with mock.patch.object(replay.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, metadata = replay.chat_completion_text(
                config,
                [
                    {"role": "system", "content": "Return JSON."},
                    {"role": "user", "content": "Make one patch."},
                ],
            )

        self.assertEqual(
            captured["url"], "http://10.168.0.1:3000/api/v1/messages"
        )
        payload = captured["payload"]
        self.assertEqual(payload["system"], "Return JSON.")  # type: ignore[index]
        self.assertEqual(payload["messages"][0]["role"], "user")  # type: ignore[index]
        self.assertNotIn("temperature", payload)  # type: ignore[operator]
        headers = {key.lower(): value for key, value in captured["headers"].items()}  # type: ignore[union-attr]
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(content, '{"patches": []}')
        self.assertEqual(metadata["usage"]["prompt_tokens"], 12)
        self.assertEqual(metadata["usage"]["completion_tokens"], 5)
        self.assertEqual(metadata["usage"]["total_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
