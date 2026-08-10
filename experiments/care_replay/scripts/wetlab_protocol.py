#!/usr/bin/env python3
"""Leakage-resistant ask/tell boundary for real wet-lab execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import run_synthetic_suzuki as replay


PRIVATE_METADATA_FIELDS = frozenset(
    {
        "yield_value",
        "conversion_value",
        "turnover_number",
        "objective_value",
        "oracle_value",
    }
)


@dataclass(frozen=True)
class PublicCandidate:
    candidate_id: str
    categorical: dict[str, str]
    continuous: dict[str, float]


@dataclass(frozen=True)
class Observation:
    candidate_id: str
    outcome: float
    metadata: dict[str, Any]


class WetLabQueue:
    """Stateful queue that never stores unrevealed outcomes."""

    def __init__(self, candidates: Iterable[PublicCandidate]) -> None:
        items = tuple(candidates)
        if not items:
            raise ValueError("WetLabQueue requires at least one candidate.")
        self._candidates = {candidate.candidate_id: candidate for candidate in items}
        if len(self._candidates) != len(items):
            raise ValueError("Candidate ids must be unique.")
        self._observations: dict[str, Observation] = {}
        self._pending: str | None = None

    @classmethod
    def from_adapter(cls, adapter: replay.DatasetAdapter) -> "WetLabQueue":
        candidates = []
        for candidate in adapter.candidates:
            public_metadata = {
                key: value
                for key, value in candidate.metadata.items()
                if key not in PRIVATE_METADATA_FIELDS and key != adapter.hidden_target
            }
            categorical = {
                field: str(public_metadata[field])
                for field in adapter.decision_columns
                if field in public_metadata
            }
            continuous = {
                "normalized_x1": float(candidate.numeric_features[0]),
                "normalized_x2": float(candidate.numeric_features[1]),
                "normalized_x3": float(candidate.numeric_features[2]),
            }
            for field in (
                "residence_time_seconds",
                "temperature_celsius",
                "catalyst_loading_mol_percent",
            ):
                if field in public_metadata:
                    continuous[field] = float(public_metadata[field])
            candidates.append(
                PublicCandidate(
                    candidate_id=candidate.candidate_id,
                    categorical=categorical,
                    continuous=continuous,
                )
            )
        return cls(candidates)

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations.values())

    @property
    def pending_candidate_id(self) -> str | None:
        return self._pending

    def public_state(self) -> dict[str, Any]:
        return {
            "candidates": [asdict(candidate) for candidate in self._candidates.values()],
            "observations": [asdict(item) for item in self.observations],
            "pending_candidate_id": self._pending,
        }

    def ask(self, scores: Mapping[str, float]) -> PublicCandidate:
        if self._pending is not None:
            raise RuntimeError("The pending experiment must be completed before ask().")
        available = [
            candidate
            for candidate in self._candidates.values()
            if candidate.candidate_id not in self._observations
        ]
        if not available:
            raise RuntimeError("No unrevealed candidates remain.")
        missing = [
            candidate.candidate_id
            for candidate in available
            if candidate.candidate_id not in scores
        ]
        if missing:
            raise ValueError(f"Scores are missing {len(missing)} available candidates.")
        selected = max(
            available,
            key=lambda candidate: (float(scores[candidate.candidate_id]), candidate.candidate_id),
        )
        self._pending = selected.candidate_id
        return selected

    def tell(
        self,
        candidate_id: str,
        outcome: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Observation:
        if self._pending != candidate_id:
            raise ValueError(
                f"tell() expected pending candidate {self._pending!r}, got {candidate_id!r}."
            )
        observation = Observation(
            candidate_id=candidate_id,
            outcome=float(outcome),
            metadata=dict(metadata or {}),
        )
        self._observations[candidate_id] = observation
        self._pending = None
        return observation


class ReplayOracle:
    """Offline-only reveal service kept outside the policy state."""

    def __init__(self, adapter: replay.DatasetAdapter) -> None:
        self._outcomes = {
            candidate.candidate_id: candidate.objective_value
            for candidate in adapter.candidates
        }

    def reveal(self, candidate_id: str) -> float:
        return float(self._outcomes[candidate_id])
