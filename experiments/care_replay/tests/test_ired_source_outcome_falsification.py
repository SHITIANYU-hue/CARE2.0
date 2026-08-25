from __future__ import annotations

import run_ired_source_outcome_falsification as falsification
import run_synthetic_suzuki as replay


def candidates() -> list[replay.Candidate]:
    return [
        replay.Candidate(
            candidate_id=f"candidate_{index}",
            group="g",
            x1=0.0,
            x2=0.0,
            x3=0.0,
            objective_value=float(index),
            metadata={"mutation_tokens": [f"A{index}V"]},
        )
        for index in range(8)
    ]


def test_permutation_preserves_candidates_and_outcome_marginal() -> None:
    original = candidates()
    permuted = falsification.permuted_source_observations(original, 91)
    assert [row.candidate_id for row in permuted] == [
        row.candidate_id for row in original
    ]
    assert sorted(row.objective_value for row in permuted) == sorted(
        row.objective_value for row in original
    )
    assert [row.objective_value for row in permuted] != [
        row.objective_value for row in original
    ]


def test_randomization_p_value_uses_plus_one_correction() -> None:
    assert falsification.randomization_p_value(1.0, [0.2, 0.4, 0.8]) == 0.25
    assert falsification.randomization_p_value(0.5, [0.2, 0.6, 0.8]) == 0.75


def test_normal_interval_reports_sample_size() -> None:
    summary = falsification.normal_interval([1.0, 2.0, 3.0])
    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["ci95_low"] < 2.0 < summary["ci95_high"]
