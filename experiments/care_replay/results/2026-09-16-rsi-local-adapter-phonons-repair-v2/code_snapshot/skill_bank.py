#!/usr/bin/env python3
"""Persistent, agent-readable CARE 2.0 skill bank.

Unlike a pair-local TransferSkill, this artifact consolidates lessons from
multiple completed tasks.  It stores evidence separately from concise agent
instructions and enforces a development/evaluation task boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "care.skill_bank/v1"
FORBIDDEN_QUERY_KEYS = frozenset(
    {
        "hidden_target_outcomes",
        "target_outcomes",
        "oracle_values",
        "heldout_outcomes",
        "yield_value",
    }
)
VALID_STATUSES = frozenset({"candidate", "validated", "rejected"})


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_QUERY_KEYS:
                return child_path
            violation = _forbidden_key(child, child_path)
            if violation:
                return violation
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            violation = _forbidden_key(child, f"{path}[{index}]")
            if violation:
                return violation
    return None


@dataclass(frozen=True)
class SkillEvidence:
    evidence_id: str
    source_tasks: tuple[str, ...]
    task_family: str
    status: str
    lesson: str
    applicability: tuple[str, ...]
    failure_modes: tuple[str, ...]
    trace_references: tuple[str, ...]
    observation_count: int
    provenance: dict[str, Any]

    def validate(self) -> None:
        if not self.evidence_id or not self.source_tasks or not self.lesson:
            raise ValueError("Evidence requires an id, source tasks, and a lesson.")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Unknown evidence status: {self.status}")
        if self.observation_count <= 0:
            raise ValueError("Evidence observation_count must be positive.")


@dataclass(frozen=True)
class ReusableSkill:
    skill_id: str
    title: str
    task_families: tuple[str, ...]
    instructions: tuple[str, ...]
    abstain_when: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    version: str = "1"

    def validate(self) -> None:
        if not self.skill_id or not self.instructions:
            raise ValueError("ReusableSkill requires an id and instructions.")
        if not self.abstain_when:
            raise ValueError("ReusableSkill must declare abstention conditions.")


@dataclass(frozen=True)
class SkillBank:
    schema_version: str
    bank_id: str
    development_task_ids: tuple[str, ...]
    evaluation_task_ids: tuple[str, ...]
    skills: tuple[ReusableSkill, ...]
    evidence: tuple[SkillEvidence, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported skill-bank schema: {self.schema_version}")
        overlap = set(self.development_task_ids) & set(self.evaluation_task_ids)
        if overlap:
            raise ValueError(f"Development and evaluation tasks overlap: {sorted(overlap)}")
        if not self.skills or not self.evidence:
            raise ValueError("SkillBank requires skills and evidence.")
        evidence_ids = set()
        for item in self.evidence:
            item.validate()
            if item.evidence_id in evidence_ids:
                raise ValueError(f"Duplicate evidence id: {item.evidence_id}")
            evidence_ids.add(item.evidence_id)
            unknown_sources = set(item.source_tasks) - set(self.development_task_ids)
            if unknown_sources:
                raise ValueError(
                    f"Evidence uses undeclared development tasks: {sorted(unknown_sources)}"
                )
        for skill in self.skills:
            skill.validate()
            unknown = set(skill.evidence_ids) - evidence_ids
            if unknown:
                raise ValueError(f"Skill references unknown evidence: {sorted(unknown)}")

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.as_dict(include_fingerprint=False))

    def as_dict(self, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "bank_id": self.bank_id,
            "development_task_ids": list(self.development_task_ids),
            "evaluation_task_ids": list(self.evaluation_task_ids),
            "skills": [asdict(skill) for skill in self.skills],
            "evidence": [asdict(item) for item in self.evidence],
            "provenance": _json_clone(self.provenance),
        }
        if include_fingerprint:
            payload["fingerprint"] = _sha256_json(payload)
        return payload

    def render_skill_md(self) -> str:
        lines = [
            "---",
            f"name: {self.bank_id}",
            "description: Evidence-bounded multi-source transfer for sequential scientific experiments.",
            "---",
            "",
            "# CARE 2.0 reusable transfer",
            "",
            "Use this skill when completed experiments may help rank candidates for a new",
            "sequential optimization task. Treat every source as a separate expert and",
            "preserve an exact target-only fallback.",
            "",
            "## Required inputs",
            "",
            "- Public source schemas and measured source outcomes.",
            "- Public target schema, bounds, constraints, and observations revealed so far.",
            "- A declared development/evaluation task split.",
            "- A fixed target-only acquisition policy for fallback.",
            "",
            "## Procedure",
            "",
        ]
        step = 1
        for skill in self.skills:
            lines.append(f"### {skill.title}")
            lines.append("")
            for instruction in skill.instructions:
                lines.append(f"{step}. {instruction}")
                step += 1
            lines.append("")
            lines.append("Abstain from this skill when:")
            for condition in skill.abstain_when:
                lines.append(f"- {condition}")
            lines.append("")
            lines.append(
                "Evidence: " + ", ".join(f"`{item}`" for item in skill.evidence_ids)
            )
            lines.append("")
        lines.extend(
            [
                "## Non-negotiable boundaries",
                "",
                "- Never read unrevealed target outcomes to choose a route or tune a threshold.",
                "- Never merge incompatible source and target spaces without a declared mapping.",
                "- Do not call a target-only fallback a successful source-transfer result.",
                "- Keep rejected and negative-transfer evidence; it defines when to abstain.",
                "- Evaluate an evolved skill on task-disjoint data before promoting it.",
                "",
                "Detailed evidence and provenance are in `references/evidence.json`.",
                "",
            ]
        )
        return "\n".join(lines)

    def build_agent_prompt(self, target_context: Mapping[str, Any]) -> str:
        violation = _forbidden_key(target_context)
        if violation:
            raise ValueError(f"Agent query contains hidden target evidence: {violation}")
        return (
            self.render_skill_md()
            + "\n## Current target context\n\n```json\n"
            + json.dumps(target_context, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n```\n\nReturn a route proposal, its evidence ids, and an explicit abstain decision."
        )

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        references = directory / "references"
        references.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(self.render_skill_md(), encoding="utf-8")
        (references / "evidence.json").write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "bank_id": self.bank_id,
                    "evidence": [asdict(item) for item in self.evidence],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def compile_skill_bank(
    *,
    bank_id: str,
    development_task_ids: Sequence[str],
    evaluation_task_ids: Sequence[str],
    evidence: Iterable[SkillEvidence],
    skills: Iterable[ReusableSkill],
    provenance: Mapping[str, Any],
) -> SkillBank:
    return SkillBank(
        schema_version=SCHEMA_VERSION,
        bank_id=bank_id,
        development_task_ids=tuple(development_task_ids),
        evaluation_task_ids=tuple(evaluation_task_ids),
        skills=tuple(skills),
        evidence=tuple(evidence),
        provenance=_json_clone(provenance),
    )
