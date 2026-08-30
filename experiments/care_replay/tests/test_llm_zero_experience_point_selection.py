from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_llm_zero_experience_point_selection as ablation
import run_synthetic_suzuki as replay


def test_prompt_contains_no_outcomes_or_transfer_evidence() -> None:
    adapter = replay.DATASET_BUILDERS["real_baumgartner_suzuki_minlp2"]()
    fields = [
        "substrate",
        "precatalyst",
        "temperature_celsius",
        "residence_time_seconds",
        "precatalyst_fraction",
    ]
    menus = ablation.build_menus(
        adapter,
        fields,
        seed_start=81000,
        menu_count=2,
        menu_size=6,
    )
    prompt = ablation.prompt_payload(adapter, fields, menus)
    serialized = json.dumps(prompt)

    assert adapter.hidden_target not in serialized
    assert "source_row" not in serialized
    assert prompt["evidence_boundary"] == {
        "source_campaigns": False,
        "source_outcomes": False,
        "target_outcomes": False,
        "target_history": False,
        "gp_scores": False,
        "rag_or_knowledge_base": False,
        "retrieved_or_frozen_skills": False,
    }


def test_all_frozen_datasets_expose_only_allowlisted_candidate_fields() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "llm_zero_experience_point_selection_v1.json"
    )
    protocol = json.loads(config_path.read_text(encoding="utf-8"))["protocol"]
    for offset, spec in enumerate(protocol["datasets"]):
        adapter = replay.DATASET_BUILDERS[spec["dataset_id"]]()
        menus = ablation.build_menus(
            adapter,
            spec["public_fields"],
            seed_start=int(protocol["seed_start"]) + offset * 1000,
            menu_count=int(protocol["menus_per_dataset"]),
            menu_size=int(protocol["menu_size"]),
        )
        prompt = ablation.prompt_payload(adapter, spec["public_fields"], menus)
        serialized = json.dumps(prompt)

        assert "source_row" not in serialized
        for hidden_menu, public_menu in zip(menus, prompt["menus"]):
            assert not any(
                candidate_id in serialized
                for candidate_id in hidden_menu["candidate_id_map"].values()
            )
            for index, candidate in enumerate(public_menu["candidates"]):
                assert candidate["candidate_id"] == f"option_{index:02d}"
                assert set(candidate["conditions"]) == set(spec["public_fields"])
                assert adapter.hidden_target not in candidate["conditions"]


def test_normalize_selections_rejects_out_of_menu_candidate() -> None:
    menus = [
        {
            "menu_id": "m0",
            "public_candidates": [
                {"candidate_id": "a", "conditions": {}},
                {"candidate_id": "b", "conditions": {}},
            ],
        }
    ]
    good = ablation.normalize_selections(
        {
            "selections": [
                {
                    "menu_id": "m0",
                    "candidate_id": "a",
                    "hypothesis": "h",
                    "reason": "r",
                    "confidence": 0.7,
                }
            ]
        },
        menus,
    )
    assert good[0]["candidate_id"] == "a"

    try:
        ablation.normalize_selections(
            {
                "selections": [
                    {
                        "menu_id": "m0",
                        "candidate_id": "outside",
                        "hypothesis": "h",
                        "reason": "r",
                        "confidence": 0.7,
                    }
                ]
            },
            menus,
        )
    except ValueError as exc:
        assert "outside menu" in str(exc)
    else:
        raise AssertionError("Expected an out-of-menu selection to fail")


def test_summary_reports_exact_random_expectation_delta() -> None:
    rows = [
        {
            "selected_value": 8.0,
            "random_expected_value": 5.0,
            "llm_minus_random_expected": 3.0,
            "llm_minus_space_filling": 2.0,
            "percentile": 1.0,
            "top_quartile_hit": True,
            "top_one_hit": True,
        },
        {
            "selected_value": 2.0,
            "random_expected_value": 4.0,
            "llm_minus_random_expected": -2.0,
            "llm_minus_space_filling": -1.0,
            "percentile": 0.0,
            "top_quartile_hit": False,
            "top_one_hit": False,
        },
    ]
    summary = ablation.summarize_rows(rows, bootstrap_samples=1000, seed=7)

    assert summary["mean_llm_minus_random_expected"] == 0.5
    assert summary["top_quartile_hit_rate"] == 0.5
    assert summary["above_random_expected_count"] == 1
    assert summary["below_random_expected_count"] == 1
