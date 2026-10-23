"""Exact finite null sampling conditional on public metadata coverage.

Pool construction never reads outcomes. R1 fixes fields; R2 enumerates uniform
type-matched injective field maps. R2 requires equal value cardinality within
every declared type, making all enumerated condition configurations equally
probable under the original null. Independent rule signs are sampled at draw
time because the coverage event does not depend on signs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, ROUND_FLOOR
import hashlib
from itertools import permutations, product
import json
import math
import random
from typing import Any

import llm_semantic_skills as semantic
import run_random_rule_control as nulls


Conditions = tuple[tuple[str, str], ...]
Configuration = tuple[Conditions, ...]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CoverageMatchedPool:
    template: semantic.SemanticSkill
    family: str
    configurations: tuple[Configuration, ...]
    diagnostics: dict[str, Any]


def _validate_options(tolerance: float, max_configurations: int) -> None:
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or not 0 <= tolerance <= 1):
        raise ValueError("tolerance must be a finite number between zero and one")
    if type(max_configurations) is not int or max_configurations < 1:
        raise ValueError("max_configurations must be a positive integer")


def _public_metadata(candidates, fields):
    """Snapshot only selected metadata columns; do not inspect candidate labels."""
    rows = []
    for candidate in candidates:
        metadata = candidate if isinstance(candidate, Mapping) else candidate.metadata
        if not isinstance(metadata, Mapping):
            raise ValueError("Each candidate must provide a public metadata mapping")
        rows.append(tuple(str(metadata.get(field, "")) for field in fields))
    if not rows:
        raise ValueError("Coverage matching requires at least one public candidate")
    # Coverage is invariant to candidate ordering; hash the public row multiset.
    profile = sorted((row, count) for row, count in Counter(rows).items())
    metadata_hash = _digest({"fields": fields, "rows": profile})
    masks = {}
    for index, row in enumerate(rows):
        for field, value in zip(fields, row):
            masks[(field, value)] = masks.get((field, value), 0) | (1 << index)
    return len(rows), masks, metadata_hash


def _rule_masks(configuration, value_masks, universe):
    result = []
    for conditions in configuration:
        mask = universe
        for field, value in conditions:
            mask &= value_masks.get((field, value), 0)
        result.append(mask)
    return tuple(result)


def _joint_counts(rule_masks, universe):
    # Sparse cells include every observed activation pattern; omitted cells are
    # zero. Bit i indicates ordered rule i. Splitting bitsets avoids reading any
    # candidate property during enumeration.
    cells = {0: universe}
    for index, mask in enumerate(rule_masks):
        updated = {}
        for pattern, members in cells.items():
            active, inactive = members & mask, members & ~mask
            if active:
                updated[pattern | (1 << index)] = active
            if inactive:
                updated[pattern] = inactive
        cells = updated
    return {pattern: members.bit_count() for pattern, members in cells.items()}


def _mapping_groups(skill, catalog, field_types, family):
    source_fields = sorted({field for rule in skill.rules for field, _ in rule.conditions})
    if family == "r1":
        return source_fields, []
    for kind in set(field_types.values()):
        cardinalities = {len(catalog[field]) for field in catalog if field_types[field] == kind}
        if len(cardinalities) != 1:
            raise ValueError(f"R2 requires equal value cardinality within field type {kind!r}")
    groups = []
    for kind in sorted({field_types[field] for field in source_fields}):
        sources = tuple(field for field in source_fields if field_types[field] == kind)
        targets = tuple(sorted(field for field in catalog if field_types[field] == kind))
        groups.append((sources, targets))
    return source_fields, groups


def _field_maps(source_fields, groups):
    if not groups:
        yield {field: field for field in source_fields}
        return
    for group_targets in product(*(permutations(targets, len(sources)) for sources, targets in groups)):
        yield {source: target for (sources, _), targets in zip(groups, group_targets)
               for source, target in zip(sources, targets)}


def _build_pool(skill, catalog, field_types, family, tolerance, max_configurations,
                candidate_count, value_masks, metadata_hash, cache):
    nulls._validate_random_rule_inputs(skill, catalog, family, field_types)
    source_fields, groups = _mapping_groups(skill, catalog, field_types, family)
    mapping_count = math.prod(math.perm(len(targets), len(sources)) for sources, targets in groups)
    values_per_mapping = math.prod(len(catalog[field])
                                   for rule in skill.rules for field, _ in rule.conditions)
    total_count = mapping_count * values_per_mapping
    if total_count > max_configurations:
        raise ValueError(f"{family}/{skill.skill_id}: exact pool has {total_count} condition configurations; "
                         f"limit is {max_configurations}. No approximate or unmatched fallback is used.")
    cache_key = _digest({"skill": asdict(skill), "catalog": catalog, "field_types": field_types,
                         "family": family, "tolerance": tolerance, "metadata_sha256": metadata_hash})
    if cache_key in cache:
        return cache[cache_key]
    universe = (1 << candidate_count) - 1
    original = tuple(rule.conditions for rule in skill.rules)
    original_masks = _rule_masks(original, value_masks, universe)
    original_support = tuple(mask.bit_count() for mask in original_masks)
    original_joint = _joint_counts(original_masks, universe)
    # Integer counts enforce the inclusive fraction threshold without floating
    # point subtraction changing whether a boundary cell is accepted.
    allowed_delta = int((Decimal(str(tolerance)) * candidate_count).to_integral_value(rounding=ROUND_FLOOR))
    accepted = []
    enumerated = maximum_support_delta = maximum_joint_delta = 0
    for field_map in _field_maps(source_fields, groups):
        field_positions = tuple(field_map[field] for rule in skill.rules for field, _ in rule.conditions)
        value_options = tuple(tuple(sorted(catalog[field])) for field in field_positions)
        for values in product(*value_options):
            enumerated += 1
            pairs = iter(zip(field_positions, values))
            configuration = tuple(tuple(next(pairs) for _ in rule.conditions) for rule in skill.rules)
            masks = _rule_masks(configuration, value_masks, universe)
            support_delta = max(abs(mask.bit_count() - count) for mask, count in zip(masks, original_support))
            if support_delta > allowed_delta:
                continue
            joint = _joint_counts(masks, universe)
            joint_delta = max(abs(joint.get(pattern, 0) - original_joint.get(pattern, 0))
                              for pattern in joint.keys() | original_joint.keys())
            if joint_delta > allowed_delta:
                continue
            accepted.append(configuration)
            maximum_support_delta = max(maximum_support_delta, support_delta)
            maximum_joint_delta = max(maximum_joint_delta, joint_delta)
    if enumerated != total_count:
        raise RuntimeError("Exact enumeration did not cover the original null condition space")
    if not accepted:
        # The identity field map and original legal conditions must be in the
        # finite null. Reaching this branch indicates an implementation error.
        raise RuntimeError("Coverage pool is empty despite the legal identity configuration")
    sign_count = 1 << len(skill.rules)
    diagnostics = {
        "family": family, "template_skill_id": skill.skill_id, "candidate_count": candidate_count,
        "absolute_fraction_tolerance": tolerance, "allowed_count_deviation": allowed_delta,
        "original_rule_support_counts": list(original_support),
        "original_rule_support_fractions": [count / candidate_count for count in original_support],
        "original_joint_activation_counts": {str(pattern): count for pattern, count in sorted(original_joint.items())},
        "original_joint_activation_fractions": {str(pattern): count / candidate_count
                                                 for pattern, count in sorted(original_joint.items())},
        "joint_histogram_encoding": "Sparse integer bitmask keys; bit i denotes ordered rule i; omitted cells are zero.",
        "field_mapping_count": mapping_count, "condition_configurations_per_mapping": values_per_mapping,
        "total_condition_configurations": total_count, "accepted_condition_configurations": len(accepted),
        "total_signed_configurations": total_count * sign_count,
        "accepted_signed_configurations": len(accepted) * sign_count,
        "acceptance_probability": len(accepted) / total_count,
        "max_accepted_rule_support_fraction_deviation": maximum_support_delta / candidate_count,
        "max_accepted_joint_activation_fraction_deviation": maximum_joint_delta / candidate_count,
        "public_metadata_sha256": metadata_hash, "accepted_pool_sha256": _digest(accepted),
        "same_type_cardinality_validated": family == "r2",
        "sampling": "original_null_conditioned_on_public_coverage",
        "sampler": "uniform accepted condition configuration, then independent equiprobable rule signs",
        "identity_and_duplicates": "allowed",
    }
    pool = CoverageMatchedPool(skill, family, tuple(accepted), diagnostics)
    cache[cache_key] = pool
    return pool


def build_pools(skills, catalog, field_types, candidates, *, tolerance=0.005,
                max_configurations=1_000_000, families=("r1", "r2"), cache=None):
    """Build reusable exact pools as ``pools[family][template.skill_id]``.

    ``candidates`` may contain public metadata mappings or objects exposing a
    ``metadata`` mapping. No candidate IDs, outcomes, rewards, or other properties
    are accessed. ``cache`` is an optional caller-owned dictionary; pool identity
    includes the template, catalog, types, tolerance and public metadata multiset.
    The limit applies separately to each template/family before enumeration.
    """
    _validate_options(tolerance, max_configurations)
    skills, families = tuple(skills), tuple(families)
    if not skills or len({skill.skill_id for skill in skills}) != len(skills):
        raise ValueError("skills must be nonempty with unique skill IDs")
    if not families or len(set(families)) != len(families) or any(family not in {"r1", "r2"} for family in families):
        raise ValueError("families must contain unique entries chosen from r1 and r2")
    # Validate before reading candidates or preparing a cache entry.
    for skill in skills:
        for family in families:
            nulls._validate_random_rule_inputs(skill, catalog, family, field_types)
            _mapping_groups(skill, catalog, field_types, family)
    candidate_count, value_masks, metadata_hash = _public_metadata(candidates, tuple(sorted(catalog)))
    cache = {} if cache is None else cache
    return {family: {skill.skill_id: _build_pool(
        skill, catalog, field_types, family, tolerance, max_configurations,
        candidate_count, value_masks, metadata_hash, cache) for skill in skills} for family in families}


def draw_skill(pool: CoverageMatchedPool, rng: random.Random, skill_id: str) -> semantic.SemanticSkill:
    """Draw with replacement from the exact conditional null, without fallback."""
    configuration = rng.choice(pool.configurations)
    rules = tuple(semantic.SemanticRule(
        rule_id=f"coverage_{pool.family}_{index + 1}", conditions=conditions,
        weight=(-1.0 if rng.random() < 0.5 else 1.0) * abs(float(rule.weight)),
        rationale="Public-coverage-conditioned random null: matched ordered rule supports and joint activation histogram.",
    ) for index, (rule, conditions) in enumerate(zip(pool.template.rules, configuration)))
    return replace(pool.template, skill_id=skill_id, rules=rules,
                   hypothesis="Exact original random-rule null conditioned only on public metadata coverage.")
