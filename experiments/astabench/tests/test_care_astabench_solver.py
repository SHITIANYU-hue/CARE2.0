import json
from pathlib import Path

import care_astabench_solver as solver_module


def test_frozen_skill_map_contains_no_benchmark_outcomes():
    skill_map = solver_module.load_skill_map()
    assert skill_map["benchmark_outcomes_used"] is False
    serialized = json.dumps(skill_map).casefold()
    assert "gold_output" not in serialized
    assert "answer_key" not in serialized


def test_selection_keeps_start_terminal_and_relevant_temporal_skill():
    skill_map = solver_module.load_skill_map()
    nodes = solver_module.select_skill_nodes(
        "Analyze a longitudinal dataset with year, treatment group, and numeric outcome",
        skill_map,
        max_skills=9,
    )
    ids = {node["id"] for node in nodes}
    assert "schema_audit" in ids
    assert "workflow_report" in ids
    assert "temporal_structure" in ids
    assert "group_contrast" in ids


def test_rendered_map_exposes_outputs_and_barriers():
    nodes = solver_module.select_skill_nodes(
        "Estimate a continuous association",
        solver_module.load_skill_map(),
        max_skills=8,
    )
    rendered = solver_module.render_skill_map(nodes)
    assert "Output:" in rendered
    assert "Barrier:" in rendered
    assert "Allowed next nodes:" in rendered


def test_protocol_is_frozen_and_matched():
    protocol_path = (
        Path(__file__).parents[1] / "configs/discoverybench_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["protocol_status"] == "frozen_before_official_benchmark_access"
    assert protocol["fairness_constraints"]["same_model_within_pair"] is True
    assert (
        protocol["fairness_constraints"]["target_or_gold_outcomes_in_skills"] is False
    )
