import asyncio
import json
from pathlib import Path

import care_astabench_solver as solver_module
import pytest
from inspect_ai.tool import ToolDef
from inspect_ai.util import store


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


def test_system_message_bounds_tool_calls_and_output_volume():
    message = solver_module.BASE_SYSTEM_MESSAGE
    assert "enforces a maximum of three" in message
    assert "at most 1,500 characters per call" in message
    assert "do not start another analysis path" in message


def test_current_protocol_is_frozen_and_matched():
    protocol_path = (
        Path(__file__).parents[1] / "configs/discoverybench_protocol_v4.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert (
        protocol["protocol_status"]
        == "frozen_after_validation_development_before_test"
    )
    assert protocol["fairness_constraints"]["same_model_within_pair"] is True
    assert (
        protocol["fairness_constraints"]["target_or_gold_outcomes_in_skills"] is False
    )
    assert protocol["development_history"]["outcome_or_gold_answers_used"] is False
    assert protocol["development_history"]["test_trajectory_used"] is False


def test_bounded_python_tool_enforces_call_limit_and_records_count():
    async def original(code: str) -> str:
        """Execute code.

        Args:
            code: Code to execute.
        """

        return f"result:{code}"

    current_store = store()
    current_store.set(solver_module.PYTHON_CALL_COUNT_KEY, 0)
    wrapped = solver_module.bounded_python_tool(
        ToolDef(original, name="python_session").as_tool(),
        max_calls=2,
        max_output_bytes=2000,
    )

    first = asyncio.run(wrapped(code="one"))
    second = asyncio.run(wrapped(code="two"))
    blocked = asyncio.run(wrapped(code="three"))

    assert first == "result:one"
    assert "final allowed Python call" in second
    assert len(second.encode("utf-8")) <= 2000
    assert "budget exhausted" in blocked
    assert current_store.get(solver_module.PYTHON_CALL_COUNT_KEY) == 2


def test_bounded_python_tool_rejects_nonpositive_limit():
    async def original(code: str) -> str:
        """Execute code.

        Args:
            code: Code to execute.
        """

        return code

    with pytest.raises(ValueError, match="positive"):
        solver_module.bounded_python_tool(
            ToolDef(original, name="python_session").as_tool(), max_calls=0
        )


def test_python_output_is_utf8_safe_and_bounded():
    bounded = solver_module.truncate_text_to_bytes("data-" + "x" * 5000, 2000)
    assert len(bounded.encode("utf-8")) <= 2000
    assert bounded.endswith("[CARE controller: Python output truncated]")
