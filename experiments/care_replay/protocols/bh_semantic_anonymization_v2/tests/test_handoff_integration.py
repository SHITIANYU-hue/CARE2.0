"""Production producer/consumer handoff using twenty fake responses, no replay or API."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_generation import (FakeAPI, run, run_fixture, library, materials,
                             no_network_or_credentials)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation as e
import generation as g


@pytest.mark.parametrize("all_invalid", [False, True])
def test_real_generation_artifacts_pass_evaluation_handoff(run_fixture, library, materials, all_invalid):
    def mutate(index, response, api):
        if all_invalid or index in (0, 5, 9):
            response["choices"][0]["message"]["content"] = "malformed fixture JSON"
    api = FakeAPI(run_fixture, library, materials, mutate)
    result = run(run_fixture, api)
    config, lock, cfg, lockpath, freeze, output = run_fixture
    outcomes, payloads, hashes = e.verify_generation(cfg, lockpath, output, config, lock)
    assert len(api.posts) == 20 and len(outcomes) == 20
    assert len(payloads) == (0 if all_invalid else 17)
    assert set(payloads) == set(result["libraries_sha256"])
    assert len({outcome["response_id"] for outcome in outcomes.values()}) == 20
    assert hashes["ledger.json"] == result["ledger_sha256"]
    # Recorded fallback is a terminal outcome, never a manufactured library.
    assert all(outcomes[name]["status"] == g.VALID for name in payloads)
    for name, outcome in outcomes.items():
        if outcome["status"] == g.INVALID:
            assert name not in payloads
            assert not (output / "libraries" / (name + ".json")).exists()
