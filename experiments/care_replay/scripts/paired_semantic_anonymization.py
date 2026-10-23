"""Offline BH paired semantic-label intervention and fail-closed output contract.

No API, credentials, retrieval or replay are used here. ``build_pair`` is zero-based
for the ten prespecified pairs. Save its entire result privately; send only
``material["arms"][arm]["messages"]`` to the model. The private mapping must never
be part of an API request. The two arms differ only in exact semantic labels.

The [2, 1, 1, 1, 1] rule-slot contract is a new paired-generation control, not a
reproduction of the historical request for eight skills with 3--10 rules each.
Statistical structure remains visible: this is semantic-label anonymization,
not information-theoretic concealment of dataset identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RECORD = ROOT / "experiments/care_replay/skill_banks/frozen_records/bh_semantic_headline_20260721.json"
EXPECTED_RECORD_SHA256 = "a3b4c29d4a55ec24c9b4305f8309ed401e9248ef66c92a4268b5a4a6d049fbdc"
SCHEMA_VERSION = "bh_paired_semantic_labels_v1"
PAIR_COUNT = 10
RULE_COUNTS = (2, 1, 1, 1, 1)
SYSTEM = (
    "You design optimization policies over a finite candidate pool. "
    "Return one valid JSON object only. Never invent field names or values "
    "outside the supplied catalog."
)
PARAMETER_RANGES = {
    "ridge": (0.05, 20), "prior_scale": (0, 1.5),
    "semantic_mass_start": (0, 0.85), "semantic_mass_end": (0, 0.85),
    "ucb_weight": (0, 1), "gp_beta_start": (0.2, 4),
    "gp_beta_end": (0.2, 4), "gp_xi": (0, 0.2), "confidence": (0.05, 1),
}
EXPLICIT_LABEL_DENYLIST = (
    "buchwald", "hartwig", "dreher", "doyle", "suzuki", "miyaura",
    "ligand", "reagent", "aryl", "additive", "phosph", "amidine",
    "guanidine", "bromide", "chloride", "iodide", "tpsa", "mw_",
    "smiles", "chemical", "scientific", "scientist", "yield_value", "maximize_yield",
)


class ValidationError(ValueError):
    """The whole response or its private pair material failed validation."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(errors))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def input_hashes() -> dict[str, str]:
    """All on-disk inputs to this builder: the frozen record and this module.

    The caller must additionally freeze its runner, replay inputs/config, and the
    exact serialized files it writes. No data or code is loaded dynamically.
    """
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (RECORD, Path(__file__).resolve())
    }


def numeric_leaves(value: Any) -> list[int | float]:
    if isinstance(value, dict):
        return [n for child in value.values() for n in numeric_leaves(child)]
    if isinstance(value, list):
        return [n for child in value for n in numeric_leaves(child)]
    return [value] if type(value) in (int, float) else []


def map_tree(value: Any, aliases: dict[str, str]) -> Any:
    """Replace exact strings and object keys, preserving order and all numbers."""
    if isinstance(value, dict):
        mapped = {aliases.get(k, k): map_tree(v, aliases) for k, v in value.items()}
        if len(mapped) != len(value):
            raise ValueError("Alias collision in an object.")
        return mapped
    if isinstance(value, list):
        return [map_tree(child, aliases) for child in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def _combined(sections: dict[str, dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in sections.values():
        for label, alias in section.items():
            if label in result or alias in result.values():
                raise ValueError("Alias mapping is not bijective.")
            result[label] = alias
    return result


def _aliases(payload: dict[str, Any], pair_index: int, alias_seed: int) -> dict[str, dict[str, str]]:
    catalog = payload["target"]["public_semantic_fields"]
    fields = set(catalog)
    values = {value for counts in catalog.values() for value in counts}
    evidence = payload["source_transfer_evidence"]
    for row in evidence["roles"] + evidence["shared_value_priors"]:
        fields.update((row["source_field"], row["target_field"]))
        if "value" in row:
            values.add(row["value"])
    fields.add(payload["target"]["hidden_target"])
    tasks = {payload["target"]["dataset"], payload["target"]["title"], evidence["source_dataset"]}
    sections = {}
    for family, labels, prefix in (("fields", fields, "F"), ("values", values, "V"), ("tasks", tasks, "T")):
        # Sorting fixes the starting order only. Independently shuffle code
        # assignments for each family, pair and seed; never ordinal-code labels.
        codes = [f"{prefix}{i:02d}" for i in range(1, len(labels) + 1)]
        seed_text = canonical_json([SCHEMA_VERSION, alias_seed, pair_index, family])
        rng = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16))
        rng.shuffle(codes)
        sections[family] = dict(zip(sorted(labels), codes))
    # Preserve the maximization direction; there is only one objective label.
    sections["objectives"] = {"maximize_yield": "maximize_O01"}
    combined = _combined(sections)
    if set(combined) & set(combined.values()):
        raise ValueError("Aliases collide with original labels.")
    return sections


def common_payload(historical):
    """Common text changes applied to both arms BEFORE the label intervention."""
    payload = copy.deepcopy(historical)
    payload["task"] = (
        "Design a diverse library of executable optimization skills for finite-pool search. "
        "Each skill defines interpretable rule features plus a conservative Bayesian linear "
        "surrogate blended with target-only GP-UCB and GP-EI anchors."
    )
    payload["design_requirements"] = [
        "Return exactly five executable skills in the skills array.",
        "Use only exact field/value pairs listed in target.public_semantic_fields.",
        "Each conditions object maps actual catalog field names directly to legal values for those fields; placeholder keys are not valid output.",
        "The first skill must have exactly two rules. Each of the second through fifth skills must have exactly one rule. This matches the number of rules in each historical executable library slot.",
        "Each rule is a conjunction of one or more legal catalog conditions, with no field repeated within one rule. This retains the existing executor's condition grammar.",
        "Keep all execution parameters within allowed_ranges and supply every parameter shown in output_contract.",
        "Include at least one conservative low-semantic-mass skill.",
        "Include at least one proposal grounded in the supplied source evidence and one independent proposal.",
        "Include a counter-hypothesis that reverses or downweights an uncertain source relation.",
        "Prefer interpretable relationships supported by the supplied representation over arbitrary coverage rules.",
        "Use positive weights for conditions expected to improve the target objective and negative weights for risks.",
        "Do not claim access to target outcomes and do not output candidate identifiers.",
        "Use unique lowercase ASCII skill_id values; rules within each skill must have unique lowercase ASCII rule_id values. IDs may contain digits and underscores.",
        "Supply exactly the keys shown in output_contract. Keep each rationale at most 400 characters and each hypothesis at most 800 characters, the executor's existing text limits.",
        "Return one JSON object only. All numeric values must be finite JSON numbers, not strings or booleans.",
        "The output is checked before any evaluation; invalid output will not be silently shortened, repaired, clamped, or expanded.",
    ]
    payload["source_transfer_evidence"]["boundary"] = (
        "Only source outcomes and public source/target fields are summarized. "
        "No target outcomes are present."
    )
    example = payload["output_contract"]["skills"][0]
    example["skill_id"] = "unique_skill_identifier"
    example["rules"][0]["rule_id"] = "unique_rule_identifier"
    example["rules"][0]["conditions"] = {"catalog_field": "legal_value_for_that_field"}
    example["rules"][0]["rationale"] = "short rationale using the supplied representation"
    example["hypothesis"] = "what this skill transfers and when it may fail"
    if "care_knowledge_context" in payload:
        raise ValueError("Knowledge context is forbidden in the paired prompt.")
    return payload


def _messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]


def build_pair(pair_index: int, alias_seed: int) -> dict[str, Any]:
    """Build one deterministic private material bundle for pair 0 through 9.

    The seed controls label permutations, not model sampling or replay seeds.
    Only ``arms.labeled.messages`` or ``arms.anonymous.messages`` is outbound.
    Hashes of JSON objects use ``canonical_json``; file-byte hashes are separate.
    """
    if type(pair_index) is not int or not 0 <= pair_index < PAIR_COUNT:
        raise ValueError("pair_index must be an integer in [0, 9].")
    if type(alias_seed) is not int or alias_seed < 0:
        raise ValueError("alias_seed must be a nonnegative integer.")
    hashes = input_hashes()
    if hashes[str(RECORD.relative_to(ROOT))] != EXPECTED_RECORD_SHA256:
        raise ValueError("Frozen historical record hash changed.")
    historical = parse_strict_json(RECORD.read_text(encoding="utf-8"))["prompt_payload"]
    labeled = common_payload(historical)
    sections = _aliases(labeled, pair_index, alias_seed)
    aliases = _combined(sections)
    anonymous = map_tree(labeled, aliases)
    inverse = {family: {v: k for k, v in mapping.items()} for family, mapping in sections.items()}
    if map_tree(anonymous, _combined(inverse)) != labeled:
        raise ValueError("Payload inverse mapping is not exact.")
    numbers = numeric_leaves(historical)
    if numbers != numeric_leaves(labeled) or numbers != numeric_leaves(anonymous) or len(numbers) != 204:
        raise ValueError("Historical numeric statistics changed.")
    arms = {}
    for arm, payload in (("labeled", labeled), ("anonymous", anonymous)):
        messages = _messages(payload)
        arms[arm] = {"payload": payload, "messages": messages,
                     "payload_sha256": json_sha256(payload), "messages_sha256": json_sha256(messages)}
    wire = canonical_json(arms["anonymous"]["messages"]).lower()
    leaks = [term for term in EXPLICIT_LABEL_DENYLIST if term in wire]
    if leaks:
        raise ValueError("Unmapped explicit labels in anonymous messages: " + ", ".join(leaks))
    return {
        "schema_version": SCHEMA_VERSION, "pair_index": pair_index, "alias_seed": alias_seed,
        "input_hashes": hashes, "arms": arms,
        "private_mapping": {"never_send_to_model": True, "forward": sections, "inverse": inverse},
        "invariants": {"numeric_leaf_count": len(numbers), "numeric_leaves_equal": True,
                       "inverse_mapping_exact": True, "explicit_label_denylist_hits": leaks},
    }


def _bounded_number(value: Any, low: float, high: float) -> bool:
    # Bounds first avoid overflow in math.isfinite for enormous JSON integers.
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


def strict_validate_library(payload, catalog):
    """Validate the whole library without normalizing, trimming or repairing it."""
    errors = []
    skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(skills, list) or len(skills) != 5:
        return ["Expected exactly five skills; nothing is trimmed or filled."]
    if set(payload) != {"skills"}:
        errors.append("Only the skills top-level key is permitted.")
    seen = set()
    for i, skill in enumerate(skills):
        path = f"skills[{i}]"
        if not isinstance(skill, dict):
            errors.append(path + ": not an object")
            continue
        if set(skill) != {"skill_id", "rules", "hypothesis", *PARAMETER_RANGES}:
            errors.append(path + ": unexpected or missing skill keys")
        identifier = skill.get("skill_id")
        if not isinstance(identifier, str) or re.fullmatch("[a-z0-9_]{1,64}", identifier) is None:
            errors.append(path + ": invalid skill_id")
        if isinstance(identifier, str) and identifier in seen:
            errors.append(path + ": repeated skill_id")
        if isinstance(identifier, str):
            seen.add(identifier)
        for name, (low, high) in PARAMETER_RANGES.items():
            value = skill.get(name)
            if not _bounded_number(value, low, high):
                errors.append(path + ": invalid " + name)
        if not isinstance(skill.get("hypothesis"), str) or len(skill["hypothesis"]) > 800:
            errors.append(path + ": hypothesis must be a string no longer than 800 characters")
        rules = skill.get("rules")
        if not isinstance(rules, list) or len(rules) != RULE_COUNTS[i]:
            errors.append(path + ": expected exactly " + str(RULE_COUNTS[i]) + " rules in this slot")
            continue
        rule_ids = set()
        for j, rule in enumerate(rules):
            rpath = f"{path}.rules[{j}]"
            if not isinstance(rule, dict):
                errors.append(rpath + ": not an object")
                continue
            if set(rule) != {"rule_id", "conditions", "weight", "rationale"}:
                errors.append(rpath + ": unexpected or missing rule keys")
            rule_id = rule.get("rule_id")
            if not isinstance(rule_id, str) or re.fullmatch("[a-z0-9_]{1,64}", rule_id) is None:
                errors.append(rpath + ": invalid rule_id")
            if isinstance(rule_id, str):
                if rule_id in rule_ids:
                    errors.append(rpath + ": repeated rule_id")
                rule_ids.add(rule_id)
            conditions = rule.get("conditions")
            if not isinstance(conditions, dict) or not conditions:
                errors.append(rpath + ": nonempty conditions object required")
            else:
                for field, value in conditions.items():
                    if field not in catalog or not isinstance(value, str) or value not in catalog[field]:
                        errors.append(rpath + ": unknown field/value")
            weight = rule.get("weight")
            if not _bounded_number(weight, -1, 1):
                errors.append(rpath + ": invalid weight")
            if not isinstance(rule.get("rationale"), str) or len(rule["rationale"]) > 400:
                errors.append(rpath + ": rationale must be a string no longer than 400 characters")
    return errors


def parse_strict_json(text: str) -> Any:
    """Reject duplicate keys, nonfinite constants/overflow and non-JSON wrappers."""
    if not isinstance(text, str):
        raise ValidationError(["Response content must be a string containing one JSON object."])

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant: " + value)

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number: " + value)
        return result

    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    try:
        return json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                          object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValidationError(["Invalid JSON: " + str(exc)]) from exc


def validate_and_decode(content: str, arm: str, pair_material: dict[str, Any]) -> dict[str, Any]:
    """Return validated canonical ``{skills: [...]}``, or raise ValidationError.

    ``arm`` is labeled/anonymous (coded is an API-name alias for anonymous).
    The private material is rechecked against deterministic construction before
    use. Rule conditions are decoded only after validation in their own catalog;
    IDs/prose are preserved verbatim and are not interpreted as instructions.
    This method never drops, normalizes, clamps, fills, retries or repairs output.
    """
    if arm == "coded":
        arm = "anonymous"
    if arm not in ("labeled", "anonymous"):
        raise ValidationError(["Unknown arm; expected labeled or anonymous."])
    try:
        expected = build_pair(pair_material["pair_index"], pair_material["alias_seed"])
        if canonical_json(pair_material) != canonical_json(expected):
            raise ValueError("Private pair material does not match deterministic construction.")
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValidationError(["Invalid pair material: " + str(exc)]) from exc
    payload = parse_strict_json(content)
    catalog = pair_material["arms"][arm]["payload"]["target"]["public_semantic_fields"]
    errors = strict_validate_library(payload, catalog)
    if errors:
        raise ValidationError(errors)
    decoded = copy.deepcopy(payload)
    if arm == "anonymous":
        inverse = pair_material["private_mapping"]["inverse"]
        for skill in decoded["skills"]:
            for rule in skill["rules"]:
                rule["conditions"] = {inverse["fields"][field]: inverse["values"][value]
                                      for field, value in rule["conditions"].items()}
    canonical_catalog = pair_material["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"]
    errors = strict_validate_library(decoded, canonical_catalog)
    if errors:
        raise ValidationError(errors)
    return decoded
