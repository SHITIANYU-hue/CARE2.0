"""Offline EvE contract checks; no evolutionary search/model calls are made."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from care_harness.agents.eve import EvEAgent
from care_harness.environment import ReplayEnvironment, load_task
from care_harness.eve import UPSTREAM_REVISION
from care_harness.eve import client, integration
from care_harness.eve.broker import DevelopmentEvaluator
from care_harness.runner import run_episode


@pytest.fixture
def development(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({
        "task": {"dataset_id": "development-only", "objective": "yield"},
        "candidates": [{"candidate_id": str(i), "group": "g", "numeric_features": [i],
                        "metadata": {}} for i in range(6)],
        "outcomes": {str(i): float(i) for i in range(6)},
        "source": {"task": {"dataset_id": "source-only"}, "observations": []},
    }))
    return {"tasks": [{"task_file": str(task_file)}], "seeds": [0, 1],
            "initial_observations": 2, "reveal_rounds": 3}


def test_frozen_solver_runs_shared_replay_from_public_view(development):
    seed = Path(integration.__file__).parent / "seed_solver.py"
    task = load_task(development["tasks"][0])
    result = run_episode(task, EvEAgent(seed), initial_observations=2, reveal_rounds=3)
    assert result["metrics"]["adaptive_count"] == 3
    assert len({event["candidate_id"] for event in result["trace"]}) == 3
    assert all("solver_sha256" in event["diagnostics"] for event in result["trace"])


def test_worker_passes_only_public_view_and_handles_solver_print(tmp_path, development):
    solver = tmp_path / "solver.py"
    solver.write_text('''def select_candidate(view):
    assert "outcomes" not in view
    assert all("outcome" not in row for row in view["candidates"])
    print("ordinary debugging output")
    seen = {row["candidate_id"] for row in view["observations"]}
    return {"candidate_id": next(row["candidate_id"] for row in view["candidates"] if row["candidate_id"] not in seen)}
''')
    task = load_task(development["tasks"][0])
    env = ReplayEnvironment(task, 0, 2, 3)
    decision = EvEAgent(solver).select(env.view())
    assert decision["candidate_id"] not in env.initial_ids


def test_development_score_matches_shared_metric(tmp_path, development):
    seed = Path(integration.__file__).parent / "seed_solver.py"
    evaluate = DevelopmentEvaluator(development, tmp_path / "evaluations")
    payload = evaluate(seed.read_text())
    scores = [run_episode(load_task(development["tasks"][0]), EvEAgent(seed),
                          seed=s, initial_observations=2, reveal_rounds=3)["metrics"]["normalized_best_so_far_auc"]
              for s in development["seeds"]]
    assert payload["score"] == pytest.approx(sum(scores) / len(scores))
    assert payload["development_reveals"] == 6
    assert all(row["task_id"] == "development-only" for row in payload["development_episodes"])
    assert "outcomes" not in payload


def test_invalid_solver_gets_official_failure_score(tmp_path, development):
    evaluate = DevelopmentEvaluator(development, tmp_path / "evaluations")
    payload = evaluate('def select_candidate(view): return "not-a-candidate"\n')
    assert payload["score"] == -1.0
    assert payload["status"] == "error"
    assert "Unknown candidate" in payload["summary"]


def test_prepare_pins_official_engine_and_never_launches_model(tmp_path, development, monkeypatch):
    monkeypatch.setattr(integration, "_revision", lambda _: UPSTREAM_REVISION)
    work = tmp_path / "prepared"
    manifest = integration.prepare(tmp_path / "eve", work, development, "explicit-model")
    config = json.loads((work / "config" / "care_replay.yaml").read_text())
    assert manifest["search_executed"] is False
    assert manifest["upstream_revision"] == UPSTREAM_REVISION
    assert manifest["development_task_ids"] == ["development-only", "source-only"]
    assert "scaling_evolve.algorithms.eve.runner" in manifest["command"]
    assert config["application"]["editable"]["files"] == ["solver.py"]
    assert config["optimizer"]["evaluation"]["_target_"].endswith("ScalarEloEvaluator")
    assert config["driver"]["model"] == "explicit-model"
    assert "outcomes" not in json.loads((work / "seed" / "example_view.json").read_text())
    assert not (work / "seed" / "task.json").exists()


def test_worker_seeds_random_for_repeatable_policy(tmp_path, development):
    solver = tmp_path / "random_solver.py"
    solver.write_text('''import random
import numpy as np

def select_candidate(view):
    observed = {row["candidate_id"] for row in view["observations"]}
    pool = [row["candidate_id"] for row in view["candidates"] if row["candidate_id"] not in observed]
    return {"candidate_id": random.choice(pool), "diagnostics": {"numpy_draw": float(np.random.random())}}
''')
    task = load_task(development["tasks"][0])
    view = ReplayEnvironment(task, 2 ** 32 + 17, 2, 3).view()
    first, second = EvEAgent(solver).select(view), EvEAgent(solver).select(view)
    assert first["candidate_id"] == second["candidate_id"]
    assert first["diagnostics"] == second["diagnostics"]


def test_launch_uses_official_engine_with_evaluator_endpoint(tmp_path, development, monkeypatch):
    monkeypatch.setattr(integration, "_revision", lambda _: UPSTREAM_REVISION)
    work = tmp_path / "prepared"
    integration.prepare(tmp_path / "eve", work, development, "explicit-model")
    events = []
    server = SimpleNamespace(server_port=19876, serve_forever=lambda: None,
                             shutdown=lambda: events.append("stop"), server_close=lambda: None)
    monkeypatch.setattr(integration, "make_server", lambda evaluator: server)

    def fake_run(command, **kwargs):
        assert "scaling_evolve.algorithms.eve.runner" in command
        assert kwargs["env"]["CARE_EVE_EVALUATOR_URL"] == "http://127.0.0.1:19876/evaluate"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(integration.subprocess, "run", fake_run)
    assert integration.launch(work) == 0
    assert events == ["stop"]
    assert json.loads((work / "manifest.json").read_text())["search_executed"] is True


def test_client_writes_official_score_yaml(tmp_path, monkeypatch):
    solver = tmp_path / "solver.py"
    solver.write_text("def select_candidate(view): return 'a'\n")
    monkeypatch.setenv("EVE_SOLVER_ROOT", str(tmp_path))
    monkeypatch.setenv("EVE_EVAL_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("CARE_EVE_EVALUATOR_URL", "http://127.0.0.1:1234/evaluate")
    monkeypatch.setattr(client.sys, "argv", ["evaluate.py"])

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self): return b'{"score": 0.75, "status": "ok", "summary": "development"}'

    def send(request, **_kwargs):
        assert json.loads(request.data)["solver"] == solver.read_text()
        return Response()

    monkeypatch.setattr(client, "urlopen", send)
    client.main()
    assert json.loads((tmp_path / "logs" / "score.yaml").read_text())["score"] == 0.75


def test_export_tracks_frozen_solver_hash(tmp_path):
    source = Path(integration.__file__).parent / "seed_solver.py"
    payload = integration.export_solver(source, tmp_path / "frozen")
    assert (tmp_path / "frozen" / "solver.py").read_text() == source.read_text()
    assert payload["solver_sha256"] == EvEAgent(source).sha256
    assert payload["evolution"] is None  # Never label an unevolved seed as a search result.
    assert EvEAgent(tmp_path / "frozen" / "solver.py").provenance == payload
    (tmp_path / "frozen" / "solver.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="changed after export"):
        EvEAgent(tmp_path / "frozen" / "solver.py")
