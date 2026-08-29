import csv
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_internal_quality_cost_audit.py"
SPEC = importlib.util.spec_from_file_location("quality_cost_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_read_and_summarize_real_shape(tmp_path):
    source = tmp_path / "route_statistics.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "domain",
                "evidence_tier",
                "completed_replicates",
                "expected_replicates",
                "mean_auc_delta",
                "mean_total_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "case_id": "positive",
                    "domain": "materials",
                    "evidence_tier": "pilot",
                    "completed_replicates": 1,
                    "expected_replicates": 30,
                    "mean_auc_delta": 2.0,
                    "mean_total_tokens": 100000,
                },
                {
                    "case_id": "negative",
                    "domain": "molecular",
                    "evidence_tier": "pilot",
                    "completed_replicates": 1,
                    "expected_replicates": 30,
                    "mean_auc_delta": -1.0,
                    "mean_total_tokens": 50000,
                },
            ]
        )

    rows = MODULE.read_routes(source)
    summary = MODULE.summarize(rows)
    assert summary["route_count"] == 2
    assert summary["mean_auc_delta_vs_same_initial_gp"] == 0.5
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["completion_fraction"] == 2 / 60
