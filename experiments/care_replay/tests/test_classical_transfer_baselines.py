from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_classical_transfer_baselines as classical  # noqa: E402
import run_synthetic_suzuki as replay  # noqa: E402


def adapter(dataset_id: str, decision_columns: tuple[str, ...]) -> replay.DatasetAdapter:
    candidates = tuple(
        replay.Candidate(
            candidate_id=f"c{index}",
            group="g0" if index % 2 == 0 else "g1",
            x1=index / 10.0,
            x2=(index % 3) / 3.0,
            x3=(index % 4) / 4.0,
            objective_value=float(index * 10),
            metadata={decision_columns[0]: "a" if index % 2 == 0 else "b"},
        )
        for index in range(10)
    )
    return replay.DatasetAdapter(
        dataset_id=dataset_id,
        title=dataset_id,
        objective="maximize",
        decision_columns=decision_columns,
        hidden_target="score",
        group_column=decision_columns[0],
        preferred_groups=(),
        failure_note="",
        candidates=candidates,
    )


def test_rejects_incompatible_feature_spaces() -> None:
    with pytest.raises(ValueError, match="identical ordered decision columns"):
        classical.validate_compatible_spaces(adapter("source", ("a",)), adapter("target", ("b",)))


def test_mixed_kernel_is_symmetric_with_unit_diagonal() -> None:
    dataset = adapter("task", ("factor",))
    features = classical.feature_arrays(dataset, dataset.candidates)
    kernel = classical.mixed_kernel_matrix(features, features, 0.35, 3.0)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)


def test_inducing_selection_is_deterministic_and_bounded() -> None:
    dataset = adapter("task", ("factor",))
    features = classical.feature_arrays(dataset, dataset.candidates)
    first = classical.select_inducing_indices(features, 4)
    second = classical.select_inducing_indices(features, 4)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 4


def test_multitask_rho_zero_matches_target_gp() -> None:
    source = adapter("source", ("factor",))
    target = adapter("target", ("factor",))
    source_posterior = classical.build_source_posterior(
        source,
        target,
        list(source.candidates),
        6,
        0.35,
        3.0,
        0.05,
    )
    features = classical.feature_arrays(target, target.candidates)
    observed_indices = [0, 3, 5]
    observed_y, _center, _scale = classical.normalized_outcomes(
        [target.candidates[index] for index in observed_indices]
    )
    target_mean, target_variance, _diagnostics = classical.target_gp_posterior(
        features,
        observed_indices,
        observed_y,
        0.35,
        3.0,
        0.05,
    )
    multitask_mean, multitask_variance, diagnostics = classical.multitask_gp_posterior(
        features,
        source_posterior,
        observed_indices,
        observed_y,
        (0.0,),
        0.35,
        3.0,
        0.05,
    )
    assert diagnostics["selected_rho"] == 0.0
    assert np.allclose(multitask_mean, target_mean)
    assert np.allclose(multitask_variance, target_variance)


def test_all_modes_share_identical_initial_observations() -> None:
    source = adapter("source", ("factor",))
    target = adapter("target", ("factor",))
    source_posterior = classical.build_source_posterior(
        source,
        target,
        list(source.candidates),
        6,
        0.35,
        3.0,
        0.05,
    )
    initial_ids = []
    for mode in classical.MODES:
        _metrics, audit = classical.run_seed(
            source,
            target,
            source_posterior,
            seed=17,
            initial=2,
            rounds=2,
            mode=mode,
            gp_beta=1.5,
            numeric_length_scale=0.35,
            categorical_length_scale=3.0,
            gp_noise=0.05,
            rgpe_draws=16,
            rho_grid=(0.0, 0.5),
        )
        initial_ids.append(audit[0]["initial_candidate_ids"])
    assert initial_ids[0] == initial_ids[1] == initial_ids[2]


def test_classical_and_care_confirmation_configs_are_matched() -> None:
    root = Path(__file__).resolve().parents[1]
    classical_config = json.loads(
        (root / "configs" / "classical_transfer_benchmark_v1.json").read_text()
    )
    care_config = json.loads(
        (root / "configs" / "source_outcome_classical_confirmation_v1.json").read_text()
    )
    assert classical_config["protocol"]["version"] == care_config["protocol"][
        "protocol_version"
    ]
    assert classical_config["protocol"]["calibration_seed_count"] == care_config[
        "protocol"
    ]["calibration_seed_count"]
    assert classical_config["protocol"]["heldout_seed_count"] == care_config[
        "protocol"
    ]["heldout_seed_count"]
    classical_pairs = {pair["pair_id"]: pair for pair in classical_config["pairs"]}
    care_pairs = {pair["pair_id"]: pair for pair in care_config["pairs"]}
    assert classical_pairs.keys() == care_pairs.keys()
    for pair_id, classical_pair in classical_pairs.items():
        care_pair = care_pairs[pair_id]
        assert classical_pair["source_dataset"] == care_pair["source_dataset"]
        assert classical_pair["target_dataset"] == care_pair["target_dataset"]
        assert classical_pair["source_observations"] == care_pair["source_observations"]
        assert classical_pair["initial_observations"] == care_pair["frozen_policy"][
            "initial_observations"
        ]
        assert classical_pair["reveal_rounds"] == care_pair["frozen_policy"][
            "reveal_rounds"
        ]
        assert care_pair["frozen_policy"]["source_initial_strategy"] == "matched"
