"""Exercise the CLI contract without credentials or any model requests."""

import json
import sys

import pytest

from care_harness.agents.codex import CodexAgent
from care_harness.environment import ReplayTask
from care_harness.runner import run_episode


FAKE_CODEX = r'''
import json
import pathlib
import sys

args = sys.argv[1:]
view = json.loads(pathlib.Path("view.json").read_text())
observed = {row["candidate_id"] for row in view["observations"]}
candidate_id = next(row["candidate_id"] for row in view["candidates"] if row["candidate_id"] not in observed)
pathlib.Path("invocation.json").write_text(json.dumps({"args": args, "prompt": sys.stdin.read()}))
pathlib.Path("notes.txt").write_text("retained across rounds")
final_path = pathlib.Path(args[args.index("--output-last-message") + 1])
final_path.write_text(json.dumps({"candidate_id": candidate_id, "rationale": "fake CLI transport test"}))
print(json.dumps({"type": "thread.started", "thread_id": "offline-test-thread"}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "cached_input_tokens": 2, "output_tokens": 3}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "cached_input_tokens": 1, "output_tokens": 4}}))
'''


@pytest.fixture
def view():
    return {
        "task": {"task_id": "test", "objective_direction": "maximize"},
        "candidates": [
            {"candidate_id": "a", "group": "one", "numeric_features": [0.1], "metadata": {}},
            {"candidate_id": "b", "group": "one", "numeric_features": [0.2], "metadata": {}},
            {"candidate_id": "c", "group": "two", "numeric_features": [0.3], "metadata": {}},
        ],
        "observations": [{"candidate_id": "a", "outcome": 0.4}],
        "source": {"task": {"task_id": "source"}, "observations": [{"candidate_id": "s", "outcome": 0.8}]},
        "round_index": 0,
        "remaining_budget": 2,
        "seed": 3,
        "initial_observations": 1,
        "reveal_rounds": 2,
    }


def test_codex_cli_stages_public_view_and_accounts_usage(tmp_path, view):
    agent = CodexAgent({
        "command": [sys.executable, "-c", FAKE_CODEX],
        "model": "configured-model",
        "reasoning_effort": "high",
    }, tmp_path)
    selection = agent.select(view)
    assert selection["candidate_id"] == "b"
    assert selection["usage"]["input_tokens"] == 12
    assert selection["usage"]["cached_input_tokens"] == 3
    assert selection["usage"]["output_tokens"] == 7
    assert selection["usage"]["agent_invocations"] == 1
    assert selection["diagnostics"]["thread_id"] == "offline-test-thread"
    assert json.loads((tmp_path / "view.json").read_text()) == view
    args = json.loads((tmp_path / "invocation.json").read_text())["args"]
    assert args[0] == "exec"
    assert args[args.index("--sandbox") + 1] == "workspace-write"
    assert args[args.index("--model") + 1] == "configured-model"
    assert "--ignore-user-config" in args
    assert 'web_search="disabled"' in args
    assert "features.apps=false" in args
    assert "mcp_servers={}" in args
    assert "sandbox_workspace_write.network_access=false" in args
    assert not any("bypass" in arg for arg in args)
    assert (tmp_path / "traces/round-000/events.jsonl").is_file()

    view["observations"].append({"candidate_id": "b", "outcome": 0.6})
    view["round_index"] = 1
    view["remaining_budget"] = 1
    assert agent.select(view)["candidate_id"] == "c"
    assert (tmp_path / "notes.txt").read_text() == "retained across rounds"
    assert json.loads((tmp_path / "traces/round-000/view.json").read_text())["round_index"] == 0
    assert json.loads((tmp_path / "view.json").read_text())["round_index"] == 1


def test_codex_failure_is_not_replaced_by_another_policy(tmp_path, view):
    agent = CodexAgent({
        "command": [sys.executable, "-c", "import sys; print('quota exhausted', file=sys.stderr); sys.exit(7)"],
    }, tmp_path)
    with pytest.raises(RuntimeError, match="status 7"):
        agent.select(view)
    assert "quota exhausted" in (tmp_path / "traces/round-000/stderr.log").read_text()


@pytest.mark.parametrize("message", ["not JSON", '{"candidate_id": 7}', '{}'])
def test_codex_rejects_unusable_final_selection(tmp_path, view, message):
    command = (
        "import pathlib,sys; "
        "p=pathlib.Path(sys.argv[sys.argv.index('--output-last-message')+1]); "
        f"p.write_text({message!r})"
    )
    agent = CodexAgent({"command": [sys.executable, "-c", command]}, tmp_path)
    with pytest.raises(RuntimeError, match="Invalid Codex selection"):
        agent.select(view)


def test_codex_requires_final_message(tmp_path, view):
    agent = CodexAgent({"command": [sys.executable, "-c", "pass"]}, tmp_path)
    with pytest.raises(RuntimeError, match="did not produce"):
        agent.select(view)


def test_codex_timeout_leaves_trace(tmp_path, view):
    agent = CodexAgent({
        "command": [sys.executable, "-c", "import time; time.sleep(10)"],
        "timeout_seconds": 0.02,
    }, tmp_path)
    with pytest.raises(RuntimeError, match="exceeded"):
        agent.select(view)
    assert (tmp_path / "traces/round-000/command.json").is_file()


def test_codex_replay_uses_common_reveal_budget_without_an_api_key(tmp_path, view):
    task = ReplayTask(
        {"dataset_id": "offline-test", "direction": "maximize"},
        view["candidates"], {"a": 0.4, "b": 0.6, "c": 0.9}, view["source"],
    )
    workspace = tmp_path / "agent"
    agent = CodexAgent({"command": [sys.executable, "-c", FAKE_CODEX]}, workspace)
    result = run_episode(
        task, agent, initial_observations=1, reveal_rounds=2,
        initial_ids=["a"], output_dir=tmp_path / "results",
    )
    assert [event["candidate_id"] for event in result["trace"]] == ["b", "c"]
    assert result["metrics"]["adaptive_count"] == 2
    assert result["metrics"]["final_best"] == 0.9
    assert result["usage"]["agent_invocations"] == 2
    assert result["usage"]["input_tokens"] == 24
    first_view = json.loads((workspace / "traces/round-000/view.json").read_text())
    assert first_view["observations"] == [{"candidate_id": "a", "outcome": 0.4}]
    assert "outcomes" not in first_view
    assert all("outcome" not in row for row in first_view["candidates"])
