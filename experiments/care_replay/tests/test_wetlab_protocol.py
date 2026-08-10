from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_synthetic_suzuki as replay  # noqa: E402
import wetlab_protocol  # noqa: E402


class WetLabProtocolTests(unittest.TestCase):
    def test_public_queue_strips_target_and_derived_outcomes(self) -> None:
        adapter = replay.real_reizman_suzuki_case_4_adapter()
        queue = wetlab_protocol.WetLabQueue.from_adapter(adapter)
        state = queue.public_state()
        serialized = repr(state)
        self.assertNotIn("yield_value", serialized)
        self.assertNotIn("turnover_number", serialized)
        self.assertNotIn("objective_value", serialized)
        self.assertIn("temperature_celsius", serialized)

    def test_ask_tell_requires_one_completed_experiment_at_a_time(self) -> None:
        adapter = replay.real_reizman_suzuki_case_4_adapter()
        queue = wetlab_protocol.WetLabQueue.from_adapter(adapter)
        oracle = wetlab_protocol.ReplayOracle(adapter)
        scores = {
            candidate["candidate_id"]: float(index)
            for index, candidate in enumerate(queue.public_state()["candidates"])
        }
        selected = queue.ask(scores)
        with self.assertRaisesRegex(RuntimeError, "pending experiment"):
            queue.ask(scores)
        observation = queue.tell(selected.candidate_id, oracle.reveal(selected.candidate_id))
        self.assertEqual(observation.candidate_id, selected.candidate_id)
        self.assertIsNone(queue.pending_candidate_id)

    def test_tell_rejects_an_unrequested_candidate(self) -> None:
        adapter = replay.real_reizman_suzuki_case_4_adapter()
        queue = wetlab_protocol.WetLabQueue.from_adapter(adapter)
        ids = [item["candidate_id"] for item in queue.public_state()["candidates"]]
        scores = {candidate_id: float(index) for index, candidate_id in enumerate(ids)}
        selected = queue.ask(scores)
        wrong = next(candidate_id for candidate_id in ids if candidate_id != selected.candidate_id)
        with self.assertRaisesRegex(ValueError, "expected pending candidate"):
            queue.tell(wrong, 1.0)


if __name__ == "__main__":
    unittest.main()
