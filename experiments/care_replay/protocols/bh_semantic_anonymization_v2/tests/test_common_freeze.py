"""All freeze mutations stay inside an isolated temporary repository."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys
import urllib.request

import pytest

PROTOCOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROTOCOL))
import common
import freeze


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("No network is allowed during freeze tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(urllib.request, "urlopen", denied)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    real_root = common.ROOT
    config = common.load(real_root / common.CONFIG_REL)
    record_bytes = (real_root / config["llm_record"]).read_bytes()
    root = tmp_path / "isolated_repo"
    root.mkdir()
    monkeypatch.setattr(common, "ROOT", root)
    def write(relative, contents="offline fixture\n"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
        return path
    for i in range(116):
        write(f"experiments/care_replay/scripts/fixture_{i:03d}.py")
    for name in ("common.py", "freeze.py", "generation.py", "evaluation.py", "tests/test_common_freeze.py"):
        write(common.PROTOCOL_REL + "/" + name)
    for name in config["data_files"]:
        write(name, b"offline binary input fixture")
    write(config["llm_record"], record_bytes)
    write("uv.lock")
    write("pyproject.toml")
    config_path = write(common.CONFIG_REL, json.dumps(config))
    previous_dir = "experiments/care_replay/outputs/v1_fixture_frozen"
    labeled = [{"role": "system", "content": "offline pinned prompt fixture"},
               {"role": "user", "content": "offline numeric evidence fixture"}]
    message_hashes = {}
    for pair in range(10):
        name = f"pair_{pair:02d}_labeled"
        path = write(previous_dir + "/messages/" + name + ".json", json.dumps(labeled))
        message_hashes[name] = common.file_hash(path)
    parent = {"protocol_id": "bh_semantic_anonymization_v1", "status": "frozen_before_generation",
              "model_api_calls_at_freeze": 0, "freeze_directory": previous_dir,
              "messages_sha256": message_hashes,
              "source_sha256": {path.relative_to(root).as_posix(): common.file_hash(path)
                                for path in (root / "experiments/care_replay/scripts").glob("*.py")},
              "data_sha256": {name: common.file_hash(root / name) for name in config["data_files"]},
              "input_sha256": {name: common.file_hash(root / name)
                               for name in (config["llm_record"], "uv.lock", "pyproject.toml")}}
    write(config["parent_v1_protocol_lock"], json.dumps(parent))
    ledger = {"attempts": [{"cost_usd": "0"}] * 10 + [{"cost_usd": "0.0125145"}],
              "known_cost_usd": "0.0125145"}
    write(config["prior_spend_ledger"], json.dumps(ledger))
    def fake_builder(pair_index, alias_seed):
        return {"pair_index": pair_index, "alias_seed": alias_seed,
                "private_mapping": {"never_send_to_model": True},
                "arms": {"labeled": {"messages": deepcopy(labeled)}, "anonymous": {"messages": [
                    {"role": "system", "content": "offline pinned prompt fixture"},
                    {"role": "user", "content": f"offline aliases pair {pair_index} seed {alias_seed}"}]}}}
    monkeypatch.setattr(freeze, "build_pair", fake_builder)
    log = write("test_validation.log", "offline isolated fixture validation\n30 passed in 0.12s\n")
    return {"root": root, "config": config, "config_path": config_path,
            "lock": root / "experiments/care_replay/preregistrations/v2_fixture.lock.json",
            "output": root / "experiments/care_replay/outputs/v2_fixture_frozen",
            "log": log, "write": write, "parent": parent}


def create(repo):
    return freeze.freeze_protocol(repo["config_path"], repo["lock"], repo["output"], repo["log"])


def verify(repo):
    return common.verify_freeze(repo["config_path"], repo["lock"], repo["output"])


def test_isolated_freeze_is_complete_verified_and_never_changes_v1(repo):
    original = dict(repo["parent"]["source_sha256"])
    lock = create(repo)
    config, verified = verify(repo)
    assert verified == lock and config == repo["config"]
    assert len(lock["messages_sha256"]) == 20 and len(lock["pair_material_sha256"]) == 10
    assert len(lock["source_sha256"]) == 121
    for group, folder in (("source_sha256", "source_snapshot"),
                          ("data_sha256", "input_snapshot"), ("input_sha256", "input_snapshot")):
        assert all(common.file_hash(repo["output"] / folder / relative) == digest
                   for relative, digest in lock[group].items())
    assert (repo["output"] / "protocol_lock.json").read_bytes() == repo["lock"].read_bytes()
    assert (repo["output"] / "config.json").read_bytes() == repo["config_path"].read_bytes()
    assert set(lock["input_sha256"]) == set(common.input_inventory(config))
    assert lock["model_api_calls_at_freeze"] == 0
    assert lock["prior_known_cost_usd"] == "0.0125145" and lock["prior_model_api_calls"] == 11
    assert lock["new_authorized_model_calls"] == 20 and lock["cumulative_max_cost_usd"] == "1"
    assert all(common.file_hash(repo["root"] / path) == digest for path, digest in original.items())
    assert not (repo["root"] / config["generation_output_directory"]).exists()


@pytest.mark.parametrize("existing", ["output", "lock"])
def test_existing_freeze_or_lock_is_never_overwritten(repo, existing):
    path = repo[existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing == "output":
        path.mkdir()
        sentinel = path / "preserve.txt"
        sentinel.write_text("preserve")
    else:
        sentinel = path
        sentinel.write_text("preserve")
    with pytest.raises(FileExistsError): create(repo)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("fault", ["new_recursive_source", "changed_v2_source", "changed_v1_source", "removed_source"])
def test_added_changed_and_removed_source_inventory_fails(repo, fault):
    create(repo)
    if fault == "new_recursive_source":
        repo["write"](common.PROTOCOL_REL + "/tests/nested/new_test.py")
    elif fault == "changed_v2_source":
        (repo["root"] / common.PROTOCOL_REL / "generation.py").write_text("changed")
    elif fault == "changed_v1_source":
        (repo["root"] / "experiments/care_replay/scripts/fixture_000.py").write_text("changed")
    else:
        (repo["root"] / common.PROTOCOL_REL / "evaluation.py").unlink()
    with pytest.raises(ValueError): verify(repo)


def test_config_change_and_unapproved_draft_are_rejected(repo):
    create(repo)
    changed = deepcopy(repo["config"])
    changed["analysis"]["scope"] = "changed description also changes frozen bytes"
    common.save(repo["config_path"], changed)
    with pytest.raises(ValueError, match="config changed"): verify(repo)
    changed = deepcopy(repo["config"])
    changed["resource_authorization"]["generation_enabled"] = False
    with pytest.raises(ValueError, match="authorization"): common.validate_config(changed)


@pytest.mark.parametrize("fault", ["crossed_ranges", "prior_overlap", "wrong_alias_seed", "unbalanced_order"])
def test_seeds_and_balanced_twenty_slot_design_cannot_change(repo, fault):
    config = deepcopy(repo["config"])
    if fault == "crossed_ranges": config["heldout_seed_start"] = config["calibration_seed_start"]
    elif fault == "prior_overlap": config["excluded_prior_seed_ranges"].append([202610205000, 202610205000])
    elif fault == "wrong_alias_seed": config["paired_design"]["alias_seed_by_pair"]["pair_00"] = 202610203000
    else: config["generation_order"][0]["arm"] = "anonymous"
    with pytest.raises(ValueError): common.validate_config(config)


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/outside.json", "a/../../x", "a\\..\\x"])
def test_bad_repository_paths_are_rejected(repo, name):
    with pytest.raises(ValueError): common.repo_path(name, must_exist=False)


def test_symlink_path_and_external_cli_destination_are_rejected(repo, tmp_path):
    external = tmp_path / "outside.txt"
    external.write_text("outside")
    (repo["root"] / "linked.txt").symlink_to(external)
    with pytest.raises(ValueError): common.repo_path("linked.txt")
    with pytest.raises(ValueError):
        freeze.freeze_protocol(repo["config_path"], repo["lock"], tmp_path / "outside_freeze", repo["log"])


@pytest.mark.parametrize("fault", ["message", "material", "extra_message", "prior_ledger", "data", "validation_log", "config_snapshot", "source_snapshot", "input_snapshot", "lock_snapshot"])
def test_frozen_data_messages_material_and_billing_hashes_are_closed(repo, fault):
    create(repo)
    paths = {"message": repo["output"] / "messages/pair_00_labeled.json",
             "material": repo["output"] / "private_pairs/pair_00.json",
             "prior_ledger": repo["root"] / repo["config"]["prior_spend_ledger"],
             "data": repo["root"] / repo["config"]["data_files"][0],
             "validation_log": repo["output"] / "offline_validation.log",
             "config_snapshot": repo["output"] / "config.json",
             "source_snapshot": repo["output"] / "source_snapshot" / common.PROTOCOL_REL / "common.py",
             "input_snapshot": repo["output"] / "input_snapshot" / repo["config"]["prior_spend_ledger"],
             "lock_snapshot": repo["output"] / "protocol_lock.json"}
    if fault == "extra_message":
        (repo["output"] / "messages/extra.json").write_text("{}")
    elif fault == "prior_ledger":
        ledger = common.load(paths[fault]); ledger["note"] = "a change with identical accounting"
        common.save(paths[fault], ledger)
    else:
        with paths[fault].open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError): verify(repo)


@pytest.mark.parametrize("case", ["failed", "stale"])
def test_freeze_requires_current_successful_test_log(repo, case):
    if case == "failed": repo["log"].write_text("29 passed, 1 failed in 1s")
    else:
        source = repo["root"] / common.PROTOCOL_REL / "common.py"
        when = repo["log"].stat().st_mtime_ns + 1_000_000_000
        os.utime(source, ns=(when, when))
    with pytest.raises(ValueError): create(repo)
    assert not repo["output"].exists() and not repo["lock"].exists()


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path):
    for raw in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
        path = tmp_path / "strict.json"
        path.write_text(raw)
        with pytest.raises(ValueError): common.load(path)


def test_prior_spend_is_not_double_counted(repo):
    path = repo["root"] / repo["config"]["prior_spend_ledger"]
    ledger = common.load(path)
    ledger["attempts"].append({"cost_usd": "0.0012393"})
    common.save(path, ledger)
    with pytest.raises(ValueError, match="eleven"): common.prior_context(repo["config"])


def test_existing_generation_output_blocks_another_freeze(repo):
    path = repo["root"] / repo["config"]["generation_output_directory"]
    path.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="generation output"): create(repo)
