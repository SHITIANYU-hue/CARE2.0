from pathlib import Path


def test_scored_summary_script_exists() -> None:
    script = Path(__file__).parents[1] / "scripts/summarize_scored_paired_logs.py"
    text = script.read_text(encoding="utf-8")
    assert "paired_bootstrap_95_ci" in text
    assert "quality_superiority_supported" in text
    assert "judge_model" in text
