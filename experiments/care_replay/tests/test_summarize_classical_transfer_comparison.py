from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_classical_transfer_comparison as summary  # noqa: E402


def row(mode: str, seed: int, split: str, final: float, auc: float) -> dict[str, object]:
    return {
        "mode": mode,
        "seed": seed,
        "split": split,
        "final_best": final,
        "best_so_far_auc": auc,
        "top10_hit": 0,
    }


def test_classical_selection_uses_calibration_only() -> None:
    rows = []
    for mode, calibration, heldout in (
        ("target_gp_ucb", 1.0, 100.0),
        ("rgpe", 3.0, 0.0),
        ("multitask_gp_icm", 2.0, 200.0),
    ):
        rows.append(row(mode, 1, "calibration", calibration, calibration))
        rows.append(row(mode, 2, "heldout", heldout, heldout))
    selected, _scores = summary.select_classical_mode(rows)
    assert selected == "rgpe"


def test_paired_comparison_matches_seeds() -> None:
    care = [
        row(summary.CARE_MODE, 10, "heldout", 5.0, 4.0),
        row(summary.CARE_MODE, 11, "heldout", 7.0, 8.0),
    ]
    baseline = [
        row("rgpe", 10, "heldout", 3.0, 3.0),
        row("rgpe", 11, "heldout", 6.0, 5.0),
    ]
    result = summary.paired_comparison(care, summary.CARE_MODE, baseline, "rgpe")
    assert result["seed_count"] == 2
    assert result["final_best"]["mean_delta"] == 1.5
    assert result["best_so_far_auc"]["mean_delta"] == 2.0
