from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_reasoning_trace_index as trace_index  # noqa: E402
import build_transfer_visualizations as visualizations  # noqa: E402


class TransferVisualizationTest(unittest.TestCase):
    def test_load_records_marks_positive_negative_and_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw_summaries").mkdir()
            rows = [
                {
                    "source": "source_a", "target": "target_a",
                    "selected_skill": "source_outcome_router",
                    "delta_final_best": "2", "final_ci_low": "1", "final_ci_high": "3",
                    "delta_auc": "4", "auc_ci_low": "2", "auc_ci_high": "6",
                    "rounds_saved_top10": "1",
                },
                {
                    "source": "source_b", "target": "target_b",
                    "selected_skill": "exact_target_only_fallback",
                    "delta_final_best": "0", "final_ci_low": "0", "final_ci_high": "0",
                    "delta_auc": "0", "auc_ci_low": "0", "auc_ci_high": "0",
                    "rounds_saved_top10": "0",
                },
                {
                    "source": "source_c", "target": "target_c",
                    "selected_skill": "exact_target_only_fallback",
                    "delta_final_best": "0", "final_ci_low": "0", "final_ci_high": "0",
                    "delta_auc": "0", "auc_ci_low": "0", "auc_ci_high": "0",
                    "rounds_saved_top10": "0",
                },
            ]
            with (root / "headline_results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                (root / "raw_summaries" / f"{row['source']}.json").write_text(json.dumps({
                    "source_dataset": row["source"],
                    "target_dataset": row["target"],
                    "patches": [{"role_multipliers": {"role": 1.2}}],
                }), encoding="utf-8")
            (root / "run_manifest.json").write_text(json.dumps({
                "negative_results": [{
                    "source": "source_b", "target": "target_b",
                    "case": "negative-route",
                    "result": "raw route had composite delta -2.0 with 95% CI [-3.0, -1.0]",
                }, {
                    "source": "source_c", "target": "target_c",
                    "case": "unstable-route",
                    "result": "raw route had composite delta +0.5 with 95% CI [-1.0, +2.0]",
                }],
            }), encoding="utf-8")
            records = visualizations.load_records(root)
            self.assertEqual([record["status"] for record in records], [
                "deployed_positive", "rejected_negative", "fallback_uncertain",
            ])
            self.assertEqual(records[0]["role_weights"]["role"], 1.2)

    def test_trace_index_counts_nested_replay_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audits.jsonl"
            path.write_text(json.dumps({
                "seed": 1,
                "events": [
                    {"seed": 1, "round_index": 0, "mode": "a"},
                    {"seed": 1, "round_index": 1, "mode": "a"},
                ],
            }) + "\n", encoding="utf-8")
            count, metadata = trace_index.count_jsonl(path)
            self.assertEqual(count, 2)
            self.assertEqual(metadata["first_round"], 0)
            self.assertEqual(metadata["last_round"], 1)


if __name__ == "__main__":
    unittest.main()
