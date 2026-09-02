from configurable_discoverybench_scorer import (
    OFFICIAL_JUDGE_MODEL,
    score_discoverybench_configurable,
)


def test_configurable_scorer_defaults_to_official_model() -> None:
    configured_scorer = score_discoverybench_configurable()
    assert configured_scorer is not None
    assert OFFICIAL_JUDGE_MODEL == "gpt-4o-2024-08-06"


def test_configurable_scorer_accepts_compatible_model() -> None:
    configured_scorer = score_discoverybench_configurable(
        judge_model="openai/gpt-4.1"
    )
    assert configured_scorer is not None
