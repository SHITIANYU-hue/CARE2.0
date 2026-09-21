"""Shared experiment semantics; all tests run without model calls."""

from copy import deepcopy
import csv
import json
from types import SimpleNamespace

import pytest

from care_harness import cli
from care_harness.environment import ReplayEnvironment, ReplayTask, _public_candidate, load_task
from care_harness.runner import run_episode


@pytest.fixture
def task():
    return ReplayTask(
        {"dataset_id": "toy", "direction": "maximize", "objective": "utility"},
        [{"candidate_id": f"c{i}", "group": "g", "numeric_features": [i], "metadata": {}}
         for i in range(12)],
        {f"c{i}": i / 10 for i in range(12)},
        {"task": {"dataset_id": "source"}, "observations": [{"candidate_id": "s0", "outcome": 0.2}]},
    )


@pytest.fixture
def config(tmp_path, task):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({
        "task": task.task, "candidates": task.candidates,
        "outcomes": task.outcomes, "source": task.source,
    }))
    return {
        "tasks": [{"id": "toy-case", "task_file": str(task_file)}],
        "methods": {"left": {"type": "random"}, "right": {"type": "random"}},
        "protocol": {"seeds": [2, 7], "initial_observations": 2, "reveal_rounds": 3},
        "reference": "left",
    }


def test_hidden_target_labels_do_not_change_agent_view(task):
    changed = deepcopy(task)
    changed.outcomes.update(c5=9371.5, c11=-400)
    first = ReplayEnvironment(task, 0, 1, 2, ["c0"])
    second = ReplayEnvironment(changed, 0, 1, 2, ["c0"])
    assert first.view() == second.view()
    public = first.view()
    assert "outcomes" not in public
    assert public["source"]["observations"][0]["outcome"] == 0.2
    assert public["observations"] == [{"candidate_id": "c0", "outcome": 0.0}]
    assert all("outcome" not in row and "objective_value" not in row for row in public["candidates"])
    public["observations"][0]["outcome"] = 123
    public["candidates"][0]["numeric_features"][0] = 99
    assert first.view()["observations"][0]["outcome"] == 0
    assert first.view()["candidates"][0]["numeric_features"] == [0]
    first.reveal("c5")
    second.reveal("c5")
    assert first.view()["observations"][-1]["outcome"] != second.view()["observations"][-1]["outcome"]


def test_legacy_candidate_export_drops_raw_result_metadata():
    candidate = SimpleNamespace(
        candidate_id="c", group="g", x1=1, x2=2, x3=3,
        numeric_features=(1, 2, 3), objective_value=88,
        metadata={"ligand": "L1", "yield": 88, "raw_result": 0.88},
    )
    public = _public_candidate(candidate, {"ligand"})
    assert public["metadata"] == {"ligand": "L1"}
    assert "objective_value" not in public


def test_reveal_budget_only_counts_valid_new_experiments(task):
    env = ReplayEnvironment(task, 0, 1, 2, ["c0"])
    for candidate, message in [("c0", "already observed"), ("absent", "Unknown candidate")]:
        with pytest.raises(ValueError, match=message):
            env.reveal(candidate)
        assert env.remaining_budget == 2
        assert len(env.observations) == 1
    env.reveal("c5")
    assert env.remaining_budget == 1
    with pytest.raises(ValueError, match="already observed"):
        env.reveal("c5")
    assert env.remaining_budget == 1
    env.reveal("c8")
    assert env.remaining_budget == 0
    with pytest.raises(ValueError, match="budget exhausted"):
        env.reveal("c11")
    assert len(env.observations) == 3


def test_metrics_match_hand_computed_adaptive_trajectory(task):
    env = ReplayEnvironment(task, 0, 1, 2, ["c0"])
    env.reveal("c5")
    env.reveal("c8")
    metrics = env.metrics()
    assert metrics["final_best"] == 0.8
    assert metrics["best_so_far_auc"] == pytest.approx((0.5 + 0.8) / 2)
    assert metrics["simple_regret"] == pytest.approx(0.3)
    assert metrics["normalized_regret"] == pytest.approx(0.3 / 1.1)
    assert metrics["normalized_best_so_far_auc"] == pytest.approx(0.65 / 1.1)
    assert metrics["top10_hit"] is True
    assert metrics["top1_percent_hit"] is False
    assert metrics["initial_count"] == 1
    assert metrics["adaptive_count"] == 2


class RepeatsItsFirstChoice:
    def __init__(self):
        self.candidate_id = None

    def select(self, view):
        if self.candidate_id is None:
            observed = {row["candidate_id"] for row in view["observations"]}
            self.candidate_id = next(row["candidate_id"] for row in view["candidates"]
                                     if row["candidate_id"] not in observed)
        return {"candidate_id": self.candidate_id, "usage": {"input_tokens": 7, "agent_invocations": 1}}


def test_runner_records_partial_failure_and_incurred_decision_usage(tmp_path, task):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="already observed"):
        run_episode(task, RepeatsItsFirstChoice(), initial_observations=1,
                    reveal_rounds=2, initial_ids=["c0"], output_dir=output)
    failure = json.loads((output / "result.json").read_text())
    assert failure["status"] == "failed"
    assert failure["completed_rounds"] == 1
    # A failed selection can still incur a model bill; count both decisions.
    assert failure["usage"] == {"input_tokens": 14, "agent_invocations": 2}
    assert len((output / "trace.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "decision_001.json").read_text())["candidate_id"] == "c1"


def test_cli_accepts_task_file_and_pairs_initialization(tmp_path, config):
    task = load_task(config["tasks"][0])
    assert task.task_id == "toy"
    output = tmp_path / "results"
    summary = cli.run_config(config, output)
    assert summary["status"] == "complete"
    for seed in [2, 7]:
        left = json.loads((output / f"toy-case/left/seed-{seed}/result.json").read_text())
        right = json.loads((output / f"toy-case/right/seed-{seed}/result.json").read_text())
        assert left["initial_ids"] == right["initial_ids"]
        assert left["metrics"] == right["metrics"]
        assert len(left["initial_ids"]) == 2
        assert len(left["trace"]) == 3
    right_summary = next(row for row in summary["results"] if row["method"] == "right")
    assert right_summary["paired_count"] == 2
    assert right_summary["paired_auc_delta"] == 0
    assert right_summary["wins_ties_losses"] == [0, 2, 0]
    with (output / "metrics.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_cli_explicit_methods_and_seeds_override_config(tmp_path, config):
    config["default_methods"] = ["left"]
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    output = tmp_path / "explicit"
    assert cli.main([
        "run", "--config", str(config_file), "--methods", "right",
        "--seeds", "9", "--output", str(output),
    ]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["plan"]["methods"] == ["right"]
    assert manifest["plan"]["seeds"] == [9]
    assert manifest["plan"]["campaign_count"] == 1
    assert (output / "toy-case/right/seed-9/result.json").is_file()
    assert not (output / "toy-case/left").exists()


def test_cli_dry_run_does_not_build_agents_or_create_output(tmp_path, config, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("dry-run must not start an agent")

    monkeypatch.setattr(cli, "make_agent", must_not_run)
    output = tmp_path / "dry-run"
    plan = cli.run_config(config, output, methods=["right"], seeds=[9], dry_run=True)
    assert plan["campaign_count"] == 1
    assert not output.exists()


def test_cli_keeps_partial_failure_details_and_counts_cost(tmp_path, config, monkeypatch):
    config["methods"] = {"broken": {"type": "scripted-test"}}
    config["reference"] = "broken"
    config["protocol"].update(seeds=[0], initial_observations=1, reveal_rounds=2)
    monkeypatch.setattr(cli, "make_agent", lambda *args: RepeatsItsFirstChoice())
    output = tmp_path / "failures"
    summary = cli.run_config(config, output)
    assert summary["status"] == "partial_failure"
    failure = json.loads((output / "toy-case/broken/seed-0/result.json").read_text())
    assert failure["completed_rounds"] == 1
    assert failure["usage"]["input_tokens"] == 14
    row = summary["results"][0]
    assert row["completed"] == 0
    assert row["failed"] == 1
    assert row["mean_auc"] is None
    assert row["total_usage"]["input_tokens"] == 14


def test_cli_returns_nonzero_for_failed_campaign(tmp_path, config, monkeypatch):
    config["methods"] = {"broken": {"type": "scripted-test"}}
    config["protocol"].update(seeds=[0], initial_observations=1, reveal_rounds=2)
    monkeypatch.setattr(cli, "make_agent", lambda *args: RepeatsItsFirstChoice())
    config_file = tmp_path / "failure-config.json"
    config_file.write_text(json.dumps(config))
    assert cli.main(["run", "--config", str(config_file), "--output", str(tmp_path / "run")]) == 1
