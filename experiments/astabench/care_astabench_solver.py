"""AstaBench solver for CARE 2.0 scientific data analysis.

The solver adds no information-retrieval or execution capability beyond the
tools supplied by the benchmark. Its contribution is a frozen, auditable
scientific workflow: sample competing hypotheses, execute tests, falsify the
leader, reroute after failed assumptions, and submit the narrowest supported
claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, basic_agent, solver
from inspect_ai.tool import Tool, ToolDef
from inspect_ai.util import store


DEFAULT_SKILL_MAP = Path(__file__).with_name("frozen_skill_map.json")
PYTHON_CALL_COUNT_KEY = "care2.python_call_count"
PYTHON_CALL_LIMIT_KEY = "care2.python_call_limit"


BASE_SYSTEM_MESSAGE = """You are the CARE 2.0 scientific analysis controller.

Your job is to derive the highest-specificity scientific hypothesis supported
by the supplied datasets. Use only the task input and the tools already
provided by the benchmark. Never use benchmark answers, hidden labels, or
unstated domain facts as evidence.

Follow this observable workflow:
1. Load every relevant dataset. Record schema, units, missingness, duplicates,
   sample size, and possible leakage before testing a hypothesis.
2. Sample two to four competing analysis paths. For each path state the
   variables, expected direction, executable test, and a condition that would
   falsify it. These are candidates, not conclusions.
3. Use the provided Python tool to execute the tests. Report concise decision
   records before tool calls; do not substitute narrative reasoning for data.
4. Challenge the current leader with at least one assumption, robustness, null,
   subset, or multiple-testing check. If a path fails, preserve that negative
   evidence and reroute to another candidate instead of repairing it post hoc.
5. Select the narrowest claim supported by the executed evidence. A negative,
   mixed, or unresolved result is preferable to an unsupported positive claim.
6. Submit exactly one valid JSON object with two string keys:
   {"hypothesis": "...", "workflow": "..."}

Execution budget:
- Normally use two Python calls. The controller enforces a maximum of three, so
  each call should run several related checks rather than one statistic.
- In the first call, combine a compact schema audit with the primary analyses
  needed to compare candidate hypotheses. In the second, run the decisive
  robustness or falsification checks. Use a third call only to repair failed
  code or execute a justified evidence reroute.
- Print compact decision statistics only, at most 1,500 characters per call.
  Never print a full dataframe, a full column list, or broad descriptive tables.
- After the second successful call, submit if the evidence is sufficient. After
  a third call, submit the narrowest supported or explicitly unresolved result;
  do not start another analysis path.

The workflow string must name files, variables, tests, sample sizes, effect
estimates, uncertainty or significance where applicable, robustness checks,
and rejected alternatives. All numeric claims must come from tool output.
"""


def load_skill_map(path: str | Path | None = None) -> dict[str, Any]:
    skill_path = Path(path) if path else DEFAULT_SKILL_MAP
    with skill_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("benchmark_outcomes_used") is not False:
        raise ValueError("The AstaBench skill map must not contain benchmark outcomes")
    if not isinstance(payload.get("nodes"), list) or not payload["nodes"]:
        raise ValueError("The AstaBench skill map must contain at least one node")
    return payload


def select_skill_nodes(
    task_text: str,
    skill_map: dict[str, Any],
    max_skills: int = 9,
) -> list[dict[str, Any]]:
    """Select frozen procedural skills from public task text only."""

    normalized = task_text.casefold()
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, node in enumerate(skill_map["nodes"]):
        tags = [str(tag).casefold() for tag in node.get("tags", [])]
        score = sum(1 for tag in tags if tag and tag in normalized)
        if node.get("always_include"):
            score += 100
        ranked.append((score, -index, node))

    selected = [
        node for score, _index, node in sorted(ranked, reverse=True) if score > 0
    ][:max_skills]
    selected_ids = {str(node["id"]) for node in selected}

    # The start and terminal nodes are invariants even under a very small limit.
    for required_id in ("schema_audit", "workflow_report"):
        if required_id not in selected_ids:
            required = next(
                node for node in skill_map["nodes"] if node["id"] == required_id
            )
            if len(selected) >= max_skills:
                selected[-1] = required
            else:
                selected.append(required)
            selected_ids.add(required_id)
    return selected


def render_skill_map(nodes: list[dict[str, Any]]) -> str:
    lines = [
        "Frozen task-conditioned skill map (procedures only; no benchmark outcomes):"
    ]
    for node in nodes:
        next_nodes = ", ".join(node.get("next", [])) or "submit"
        lines.extend(
            [
                f"- {node['id']}: {node['purpose']}",
                f"  Output: {node['output']}",
                f"  Barrier: {node['failure_condition']}",
                f"  Allowed next nodes: {next_nodes}",
            ]
        )
    return "\n".join(lines)


def bounded_python_tool(original_tool: Tool, max_calls: int) -> Tool:
    """Wrap AstaBench's Python tool with an auditable per-sample call limit."""

    if max_calls <= 0:
        raise ValueError("max_calls must be positive")

    original = ToolDef(original_tool)

    async def execute(code: str) -> Any:
        current_store = store()
        call_count = int(current_store.get(PYTHON_CALL_COUNT_KEY, 0))
        if call_count >= max_calls:
            return (
                f"CARE execution budget exhausted after {max_calls} Python calls. "
                "Do not request another analysis. Submit the narrowest supported "
                "or explicitly unresolved JSON result now."
            )

        call_count += 1
        current_store.set(PYTHON_CALL_COUNT_KEY, call_count)
        result = await original.tool(code=code)
        if call_count == max_calls:
            return (
                f"{result}\n\n[CARE controller: this was the final allowed Python "
                "call. Submit the strict JSON result now; no further analysis is "
                "available.]"
            )
        return result

    return ToolDef(
        execute,
        name=original.name,
        description=original.description,
        parameters=original.parameters,
        parallel=False,
        viewer=original.viewer,
        model_input=original.model_input,
        options=original.options,
    ).as_tool()


@solver
def inject_care_skill_map(
    skill_map_path: str | None = None,
    max_skills: int = 9,
    enable_rerouting: bool = True,
    max_python_calls: int = 3,
) -> Solver:
    """Add a task-conditioned frozen map before the first model call."""

    skill_map = load_skill_map(skill_map_path)
    if max_python_calls <= 0:
        raise ValueError("max_python_calls must be positive")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nodes = select_skill_nodes(state.input_text, skill_map, max_skills=max_skills)
        if not enable_rerouting:
            nodes = [node for node in nodes if node["id"] != "evidence_reroute"]
        selected_ids = [str(node["id"]) for node in nodes]
        state.store.set("care2.skill_map_schema", skill_map["schema_version"])
        state.store.set("care2.selected_skill_ids", selected_ids)
        state.store.set("care2.rerouting_enabled", enable_rerouting)
        state.store.set(PYTHON_CALL_COUNT_KEY, 0)
        state.store.set(PYTHON_CALL_LIMIT_KEY, max_python_calls)
        state.tools = [
            bounded_python_tool(tool, max_python_calls)
            if ToolDef(tool).name == "python_session"
            else tool
            for tool in state.tools
        ]
        state.messages.insert(
            0,
            ChatMessageSystem(
                content=BASE_SYSTEM_MESSAGE + "\n\n" + render_skill_map(nodes)
            ),
        )
        return state

    return solve


@solver
def care_discovery_agent(
    skill_map_path: str | None = None,
    max_skills: int = 9,
    enable_rerouting: bool = True,
    message_limit: int = 36,
    token_limit: int = 60000,
    max_python_calls: int = 3,
    max_tool_output: int = 2000,
) -> Solver:
    """Run CARE's evidence-bounded loop with the benchmark's original tools."""

    return basic_agent(
        init=inject_care_skill_map(
            skill_map_path=skill_map_path,
            max_skills=max_skills,
            enable_rerouting=enable_rerouting,
            max_python_calls=max_python_calls,
        ),
        message_limit=message_limit,
        token_limit=token_limit,
        max_tool_output=max_tool_output,
        continue_message=(
            "Continue with the next executable CARE stage. Use a provided tool if "
            "evidence is still missing; otherwise submit the strict JSON result."
        ),
        submit_description=(
            "Submit the final strict JSON object with hypothesis and workflow keys."
        ),
    )


__all__ = [
    "care_discovery_agent",
    "bounded_python_tool",
    "inject_care_skill_map",
    "load_skill_map",
    "render_skill_map",
    "select_skill_nodes",
]
