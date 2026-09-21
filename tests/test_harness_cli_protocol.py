"""Separate baseline jobs must retain the same configured experiment protocol."""

from copy import deepcopy
import json

import pytest

from care_harness import cli
from care_harness.agents.random import RandomAgent
from care_harness.environment import ReplayTask


@pytest.fixture
def shared_protocol(monkeypatch, tmp_path):
    config = {
        "protocol": {
            "initialization": "care", "seeds": [200],
            "initial_observations": 2, "reveal_rounds": 2,
        },
        "tasks": [{
            "id": "transfer", "source_dataset": "source", "target_dataset": "target",
            "source_seed": 0, "source_observations": 1,
        }],
        "methods": {
            "target_gp": {"type": "target_gp"},
            "care": {"type": "care", "selection_file": "test-selection.json"},
            "codex": {"type": "codex"},
            "eve": {"type": "eve"},
        },
        "reference": "target_gp",
    }
    loaded = []

    def resolve(task_config, method):
        result = {**task_config, **method}
        if method.get("selection_file"):
            result.update(
                initial_observations=2, reveal_rounds=2,
                source_seed=91, source_observations=3,
                deployment_mode="matched_target_only_llm",
                calibration_provenance={
                    "calibration_seed_start": 100, "calibration_seed_count": 3,
                    "source_dataset": "source", "target_dataset": "target",
                },
            )
        return result

    def load(spec):
        loaded.append(deepcopy(spec))
        return ReplayTask(
            {"dataset_id": "target", "direction": "maximize"},
            [{"candidate_id": f"c{i}", "group": "g", "numeric_features": [i], "metadata": {}}
             for i in range(8)],
            {f"c{i}": i / 10 for i in range(8)},
            {"task": {"dataset_id": "source"}, "observations": [
                {"candidate_id": f"s{i}", "outcome": spec["source_seed"] / 100}
                for i in range(spec["source_observations"])
            ]},
        )

    class Initializer:
        def __init__(self, options):
            self.options = options

        def initial_candidate_ids(self, view):
            if self.options.get("deployment_mode") == "matched_target_only_llm":
                return ["c2", "c3"]
            return ["c0", "c1"]

    monkeypatch.setattr(cli, "method_config", resolve)
    monkeypatch.setattr(cli, "load_task", load)
    monkeypatch.setattr(cli, "make_agent", lambda options, seed, workspace: RandomAgent(seed))
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "fake-codex-workspace"))
    monkeypatch.setattr("care_harness.agents.care.CareAgent", Initializer)
    return config, loaded


def test_independent_method_jobs_keep_frozen_source_and_initial_design(tmp_path, shared_protocol):
    config, loaded = shared_protocol
    jobs = [("care-job", ["target_gp", "care"]), ("codex-job", ["codex"]), ("eve-job", ["eve"])]
    source_hashes = []
    for directory, methods in jobs:
        output = tmp_path / directory
        result = cli.run_config(config, output, methods=methods)
        assert result["status"] == "complete"
        manifest = json.loads((output / "manifest.json").read_text())
        source_hashes.append(manifest["task_data"]["transfer"]["source_sha256"])
        for method in methods:
            result = json.loads((output / f"transfer/{method}/seed-200/result.json").read_text())
            assert result["initial_ids"] == ["c2", "c3"]
            assert result["metrics"]["adaptive_count"] == 2
    assert len(set(source_hashes)) == 1
    assert [(spec["source_seed"], spec["source_observations"]) for spec in loaded] == [(91, 3)] * 3
    assert all(spec["source_dataset"] == "source" and spec["target_dataset"] == "target" for spec in loaded)


def test_explicit_frozen_initializer_applies_without_care_method(tmp_path, shared_protocol):
    config, loaded = shared_protocol
    config["methods"] = {"codex": {"type": "codex"}}
    config["initializer"] = {"selection_file": "test-selection.json"}
    output = tmp_path / "codex-only"
    assert cli.run_config(config, output)["status"] == "complete"
    result = json.loads((output / "transfer/codex/seed-200/result.json").read_text())
    assert result["initial_ids"] == ["c2", "c3"]
    assert loaded[0]["source_seed"] == 91
    assert loaded[0]["source_observations"] == 3


@pytest.mark.parametrize("explicit_id", [False, True])
def test_duplicate_case_ids_fail_before_creating_output(tmp_path, monkeypatch, explicit_id):
    tasks = [
        {"source_dataset": "source-a", "target_dataset": "shared-target"},
        {"source_dataset": "source-b", "target_dataset": "shared-target"},
    ]
    if explicit_id:
        for task in tasks:
            task["id"] = "same-id"
    config = {"tasks": tasks, "methods": {"random": {"type": "random"}}}
    output = tmp_path / "must-not-exist"

    def unexpected_load(_):
        pytest.fail("Duplicate case IDs must be rejected before loading task data")

    monkeypatch.setattr(cli, "load_task", unexpected_load)
    with pytest.raises(ValueError, match="(?i)(case|duplicate|unique)"):
        cli.run_config(config, output)
    assert not output.exists()
