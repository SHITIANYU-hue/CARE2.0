from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cross_task_router as router  # noqa: E402


class CrossTaskRouterTest(unittest.TestCase):
    def test_materials_route_uses_shared_descriptor_candidate(self) -> None:
        proposal = router.propose_route(
            "real_matbench_dielectric",
            "real_matbench_expt_gap",
        )
        self.assertEqual(proposal.source_family, "materials")
        self.assertEqual(proposal.shared_field_count, 7)
        self.assertEqual(proposal.recommended_candidate, "shared_descriptor_source_outcome")
        self.assertTrue(proposal.as_dict()["outcome_gate_required"])

    def test_reaction_route_uses_component_roles(self) -> None:
        proposal = router.propose_route(
            "real_chemlex_acidamine",
            "real_buchwald_hartwig",
        )
        self.assertGreaterEqual(proposal.shared_role_count, 2)
        self.assertEqual(proposal.recommended_candidate, "reaction_role_source_outcome")

    def test_unaligned_pair_is_target_only(self) -> None:
        proposal = router.propose_route(
            "real_chemlex_acidamine",
            "real_matbench_phonons",
        )
        self.assertEqual(proposal.candidate_route_families, ("target_only",))


if __name__ == "__main__":
    unittest.main()
