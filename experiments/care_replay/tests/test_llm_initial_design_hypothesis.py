from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_llm_initial_design_hypothesis as llm_design


def shortlist() -> list[dict[str, object]]:
    return [
        {"candidate_id": "a"},
        {"candidate_id": "b"},
        {"candidate_id": "c"},
        {"candidate_id": "d"},
    ]


class LLMInitialDesignHypothesisTest(unittest.TestCase):
    def test_public_candidate_exposes_generic_descriptors_without_target(self) -> None:
        adapter = llm_design.replay.real_moleculenet_freesolv_continuous_adapter()
        public = llm_design.public_candidate(adapter.candidates[0], adapter)
        self.assertIn("smiles", public["conditions"])
        self.assertIn("hetero_atom_bin", public["conditions"])
        self.assertNotIn(adapter.hidden_target, public["conditions"])

    def test_normalize_hypothesis_accepts_three_catalog_candidates(self) -> None:
        result = llm_design.normalize_hypothesis(
            {
                "hypothesis": "Source-supported quality anchors transfer.",
                "selected_candidate_ids": ["a", "b", "c"],
                "selected_source_view": "all_sources",
                "confidence": 0.72,
                "failure_conditions": ["source rankings disagree"],
                "abstain": False,
            },
            shortlist(),
            {"all_sources": ["source"]},
        )
        self.assertEqual(result["selected_candidate_ids"], ["a", "b", "c"])
        self.assertAlmostEqual(result["confidence"], 0.72)

    def test_normalize_hypothesis_rejects_invalid_selection(self) -> None:
        for candidate_ids in (
            ["a", "b"],
            ["a", "a", "b"],
            ["a", "b", "unknown"],
        ):
            with self.subTest(candidate_ids=candidate_ids), self.assertRaises(ValueError):
                llm_design.normalize_hypothesis(
                    {
                        "selected_candidate_ids": candidate_ids,
                        "selected_source_view": "all_sources",
                        "confidence": 0.5,
                        "abstain": False,
                    },
                    shortlist(),
                    {"all_sources": ["source"]},
                )

    def test_outcome_blind_prompt_rejects_target_value_fields(self) -> None:
        llm_design.assert_outcome_blind_prompt(
            {
                "target": {
                    "candidate_id": "a",
                    "target_outcomes_available_to_llm": False,
                }
            }
        )
        with self.assertRaises(ValueError):
            llm_design.assert_outcome_blind_prompt(
                {"target": {"candidate_id": "a", "yield_value": 99.0}}
            )

    def test_abstention_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            llm_design.normalize_hypothesis(
                {
                    "selected_source_view": "all_sources",
                    "confidence": 0.2,
                    "abstain": True,
                    "abstain_reason": "",
                },
                shortlist(),
                {"all_sources": ["source"]},
            )

    def test_selection_must_cover_multiple_available_groups(self) -> None:
        grouped = [
            {"candidate_id": "a", "group": "g1"},
            {"candidate_id": "b", "group": "g1"},
            {"candidate_id": "c", "group": "g1"},
            {"candidate_id": "d", "group": "g2"},
        ]
        with self.assertRaises(ValueError):
            llm_design.normalize_hypothesis(
                {
                    "selected_candidate_ids": ["a", "b", "c"],
                    "selected_source_view": "all_sources",
                    "confidence": 0.5,
                    "abstain": False,
                },
                grouped,
                {"all_sources": ["source"]},
            )

    def test_compiler_combines_source_semantic_and_geometry_anchors(self) -> None:
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "baumgartner_multisource_warmstart_v2.json").read_text()
        )
        target = llm_design.replay.DATASET_BUILDERS[
            "real_baumgartner_suzuki_minlp2"
        ]()
        selected, sources, record = llm_design.compiled_llm_initial(
            target,
            ["real_baumgartner_suzuki_minlp1"],
            config["protocol"],
            {
                "selected_source_view": "same_substrate",
                "selected_candidate_ids": [
                    "baumgartner_suzuki_minlp2_024",
                    "baumgartner_suzuki_minlp2_059",
                    "baumgartner_suzuki_minlp2_057",
                ],
                "selection_roles": [
                    {
                        "candidate_id": "baumgartner_suzuki_minlp2_024",
                        "role": "quality_anchor",
                    }
                ],
                "hypothesis": "high-temperature source evidence transfers",
                "failure_conditions": [],
            },
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertEqual(sources, ["real_baumgartner_suzuki_minlp1"])
        self.assertEqual(
            record["llm_semantic_anchor"], "baumgartner_suzuki_minlp2_024"
        )

    def test_reflection_normalizer_accepts_revision_and_stop(self) -> None:
        result = llm_design.normalize_reflection(
            {
                "hypothesis_status": "mixed",
                "evidence_interpretation": "The anchor worked but the mechanism is narrow.",
                "revised_hypothesis": "The effect holds only at high temperature.",
                "selected_candidate_ids": ["c", "d"],
                "confidence": 0.61,
                "stop_transfer": False,
            },
            shortlist(),
            2,
        )
        self.assertEqual(result["selected_candidate_ids"], ["c", "d"])
        self.assertEqual(result["hypothesis_status"], "mixed")
        stopped = llm_design.normalize_reflection(
            {
                "hypothesis_status": "falsified",
                "selected_candidate_ids": ["c", "d"],
                "confidence": 0.2,
                "stop_transfer": True,
                "stop_reason": "The revealed evidence contradicts the mechanism.",
            },
            shortlist(),
            2,
        )
        self.assertEqual(stopped["selected_candidate_ids"], [])

    def test_reflection_boundary_rejects_unrevealed_outcome(self) -> None:
        llm_design.assert_reflection_prompt_boundary(
            {
                "scientific_context": {
                    "remaining_candidate_shortlist": [
                        {"candidate_id": "a", "temperature_celsius": 100}
                    ]
                }
            }
        )
        with self.assertRaises(ValueError):
            llm_design.assert_reflection_prompt_boundary(
                {
                    "scientific_context": {
                        "remaining_candidate_shortlist": [
                            {"candidate_id": "a", "target_outcome": 99.0}
                        ]
                    }
                }
            )

    def test_reflective_policy_preserves_total_reveal_budget(self) -> None:
        config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "baumgartner_multisource_warmstart_v2.json"
            ).read_text()
        )
        target = llm_design.replay.DATASET_BUILDERS[
            "real_baumgartner_suzuki_minlp2"
        ]()
        metrics, audit = llm_design.run_reflective_target_gp(
            target,
            [0, 1, 2],
            [3, 4],
            4,
            config["protocol"]["kernel"],
            ["real_baumgartner_suzuki_minlp1"],
            {"hypothesis_status": "mixed"},
        )
        self.assertEqual(metrics["initial_observations"], 3)
        self.assertEqual(metrics["reveal_rounds"], 4)
        self.assertEqual(
            [event["event"] for event in audit].count("llm_reflection_reveal"),
            2,
        )
        self.assertEqual(
            [event["event"] for event in audit].count("target_gp_reveal"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
