from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import llm_semantic_skills as semantic  # noqa: E402
import run_matched_random_rule_controls as study  # noqa: E402


class PublicCandidate:
    """Explodes if rule diagnostics use an outcome rather than public metadata."""

    def __init__(self, candidate_id, metadata):
        self.candidate_id = candidate_id
        self.metadata = metadata

    @property
    def objective(self):
        raise AssertionError("Diagnostics must not read objectives")

    @property
    def value(self):
        raise AssertionError("Diagnostics must not read objective values")


class MatchedRandomRuleControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "a": {"one": 2, "two": 4}, "b": {"three": 3, "four": 3},
            "x": {"yes": 4, "no": 2}, "y": {"true": 3, "false": 3},
        }
        first = semantic.SemanticSkill(
            skill_id="first",
            rules=(semantic.SemanticRule("rule_a", (("a", "one"),), 0.8, "test"),),
            ridge=2.0, prior_scale=0.3, semantic_mass_start=0.3,
            semantic_mass_end=0.1, ucb_weight=0.5, gp_beta_start=1.5,
            gp_beta_end=1.0, gp_xi=0.01, confidence=0.7, hypothesis="test",
        )
        second = replace(first, skill_id="second", ridge=3.0, rules=(
            semantic.SemanticRule("rule_b", (("b", "four"), ("x", "yes")), -0.2, "test"),
            semantic.SemanticRule("rule_c", (("a", "two"),), 0.4, "test"),
        ))
        self.skills = (first, second)
        self.config = {
            "calibration_seed_start": 100, "calibration_seed_count": 5,
            "heldout_seed_start": 200, "heldout_seed_count": 2,
            "null_replicates": 2, "random_seed": 71,
            "initial_observations": 5, "reveal_rounds": 10,
            "target_dataset": "real_buchwald_hartwig",
            "field_types": {"a": "category", "b": "category", "x": "binary", "y": "binary"},
            "min_risk_adjusted_gain": 0.25, "min_positive_fold_rate": 0.8,
        }
        self.calibration_seeds = set(range(100, 105))
        self.heldout_seeds = {200, 201}
        self.libraries = study.make_libraries(self.skills, self.catalog, self.config)

    def rows(self, split="calibration", winning_index=0, harmful=False):
        seeds = self.calibration_seeds if split == "calibration" else self.heldout_seeds
        rows = []
        for seed in sorted(seeds):
            for index, mode in enumerate(study.BASELINES):
                rows.append({"mode": mode, "seed": seed, "split": split,
                             "final_best": 10.0 + index, "best_so_far_auc": 10.0 + index,
                             "top10_hit": 0.0})
            for bank in self.libraries.values():
                for index, skill in enumerate(bank):
                    # The strongest baseline is 12. All banks share the same
                    # calibration evidence, with a uniquely best template.
                    value = 8.0 if harmful else (16.0 if index == winning_index else 13.0)
                    rows.append({"mode": semantic.skill_mode(skill), "seed": seed, "split": split,
                                 "final_best": value, "best_so_far_auc": value, "top10_hit": 1.0})
        return rows

    def test_each_null_library_has_same_template_budget(self):
        self.assertIs(self.libraries["llm"], self.skills)
        self.assertEqual(set(self.libraries), {"llm", "r1_001", "r1_002", "r2_001", "r2_002"})
        modes = []
        for bank in self.libraries.values():
            self.assertEqual(len(bank), len(self.skills))
            for template, generated in zip(self.skills, bank):
                self.assertEqual([len(rule.conditions) for rule in generated.rules],
                                 [len(rule.conditions) for rule in template.rules])
                self.assertEqual([abs(rule.weight) for rule in generated.rules],
                                 [abs(rule.weight) for rule in template.rules])
                self.assertEqual(generated.ridge, template.ridge)
                modes.append(semantic.skill_mode(generated))
        self.assertEqual(len(modes), len(set(modes)))

    def test_library_generation_reproducible_and_replication_count_does_not_change_existing_banks(self):
        repeated = study.make_libraries(self.skills, self.catalog, dict(self.config))
        self.assertEqual(repeated, self.libraries)
        extended = study.make_libraries(self.skills, self.catalog,
                                        {**self.config, "null_replicates": 4, "workers": 8})
        for name, bank in self.libraries.items():
            self.assertEqual(extended[name], bank)

    def test_r2_stream_is_independent_of_r1_random_consumption(self):
        original = study.nulls.matched_random_skill

        def perturb_r1(skill, catalog, rng, skill_id, **kwargs):
            if kwargs["null_type"] == "r1":
                for _ in range(37):
                    rng.random()
            return original(skill, catalog, rng, skill_id, **kwargs)

        with patch.object(study.nulls, "matched_random_skill", side_effect=perturb_r1):
            perturbed = study.make_libraries(self.skills, self.catalog, self.config)
        for name in ("llm", "r2_001", "r2_002"):
            self.assertEqual(perturbed[name], self.libraries[name])
        self.assertNotEqual(perturbed["r1_001"], self.libraries["r1_001"])

    def test_changing_random_seed_changes_nulls_but_not_llm_bank(self):
        changed = study.make_libraries(self.skills, self.catalog, {**self.config, "random_seed": 72})
        self.assertEqual(changed["llm"], self.libraries["llm"])
        for family in ("r1", "r2"):
            self.assertNotEqual([bank for name, bank in changed.items() if name.startswith(family)],
                                [bank for name, bank in self.libraries.items() if name.startswith(family)])

    def test_freeze_rejects_heldout_rows_before_selector(self):
        calibration = self.rows()
        cases = [
            [dict(row, split="heldout") for row in calibration],
            calibration + self.rows("heldout"),
            self.rows("heldout"),
        ]
        for rows in cases:
            with self.subTest(row_count=len(rows)), patch.object(study.calibrated, "select_skill") as selector:
                with self.assertRaises(ValueError):
                    study.freeze_selection(rows, self.libraries, self.calibration_seeds, self.config)
                selector.assert_not_called()

    def test_freeze_rejects_missing_and_duplicate_cells_before_selector(self):
        rows = self.rows()
        cases = [rows[:-1], rows + [dict(rows[0])], [dict(rows[0]), *rows[:-1]]]
        for invalid in cases:
            with self.subTest(row_count=len(invalid)), patch.object(study.calibrated, "select_skill") as selector:
                with self.assertRaises(ValueError):
                    study.freeze_selection(invalid, self.libraries, self.calibration_seeds, self.config)
                selector.assert_not_called()

    def test_freeze_rejects_noninteger_seeds_without_coercion(self):
        for seed in (100.0, 100.5, "100", True):
            rows = self.rows()
            rows[0]["seed"] = seed
            with self.subTest(seed=seed), patch.object(study.calibrated, "select_skill") as selector:
                with self.assertRaisesRegex(ValueError, "Noninteger seed"):
                    study.freeze_selection(rows, self.libraries, self.calibration_seeds, self.config)
                selector.assert_not_called()

    def test_freeze_rejects_nonfinite_metrics_before_selector(self):
        for metric in ("final_best", "best_so_far_auc", "top10_hit"):
            for invalid in (float("nan"), float("inf"), float("-inf")):
                rows = self.rows()
                rows[0][metric] = invalid
                with self.subTest(metric=metric, invalid=invalid), patch.object(
                    study.calibrated, "select_skill"
                ) as selector:
                    with self.assertRaisesRegex(ValueError, "Nonfinite metric"):
                        study.freeze_selection(rows, self.libraries, self.calibration_seeds, self.config)
                    selector.assert_not_called()

    def test_evaluate_seed_fills_missing_portfolio_count_and_preserves_reported_count(self):
        config = {**self.config, "gp_beta": 1.5, "gp_xi": 0.01,
                  "numeric_length_scale": 0.35, "categorical_length_scale": 3.0,
                  "gp_noise": 0.05}
        portfolio_event = {
            "mode": "target_acquisition_portfolio", "seed": 100, "round_index": 2,
            "selected_candidate": "c1", "revealed_value": 8.0, "best_so_far": 10.0,
        }
        semantic_event = {
            "mode": semantic.skill_mode(self.skills[0]), "seed": 100, "round_index": 3,
            "selected_candidate": "c2", "revealed_value": 9.0, "best_so_far": 10.0,
            # Intentionally distinguish an explicit count from the fallback.
            "public_observed_count": 19,
        }
        audits = {
            (portfolio_event["mode"], 100): [portfolio_event],
            (semantic_event["mode"], 100): [semantic_event],
        }
        before = deepcopy(audits)
        expected_rows = [{"mode": "sentinel"}]
        with patch.object(study.calibrated, "evaluate_seed", return_value=(expected_rows, audits)) as evaluator:
            rows, traces, duration = study.evaluate_seed(100, "calibration", config, self.skills)
        self.assertIs(rows, expected_rows)
        self.assertEqual(traces[0], {**portfolio_event, "public_observed_count": 7})
        self.assertEqual(traces[1], semantic_event)
        self.assertEqual(audits, before)
        self.assertGreaterEqual(duration, 0.0)
        self.assertFalse(evaluator.call_args.kwargs["include_llm_baselines"])

    def test_heldout_winner_cannot_change_frozen_calibration_choice(self):
        selection = study.freeze_selection(self.rows(), self.libraries, self.calibration_seeds, self.config)
        before = deepcopy(selection)
        heldout = self.rows("heldout", winning_index=1)
        # Make calibration's winner harmful on held-out data; the other skill
        # remains excellent. Reports must retain the frozen, now poor choice.
        first_modes = {semantic.skill_mode(bank[0]) for bank in self.libraries.values()}
        for row in heldout:
            if row["mode"] in first_modes:
                row.update(final_best=7.0, best_so_far_auc=7.0)
        summary = study.summarize(heldout, self.libraries, selection, self.heldout_seeds)
        self.assertEqual(selection, before)
        for name, bank in self.libraries.items():
            expected = semantic.skill_mode(bank[0])
            self.assertEqual(selection[name]["selected_mode"], expected)
            self.assertEqual(summary["libraries"][name]["selected_mode"], expected)
            self.assertLess(summary["libraries"][name]["deployed_vs_anchor"]["best_so_far_auc"]["mean"], 0)
            better = semantic.skill_mode(bank[1])
            self.assertGreater(summary["all_skill_effects"][better]["best_so_far_auc"]["mean"], 0)

    def test_canonical_selector_and_fallback_apply_to_every_library(self):
        rows = self.rows(harmful=True)
        original = study.calibrated.select_skill
        with patch.object(study.calibrated, "select_skill", wraps=original) as selector:
            choices = study.freeze_selection(rows, self.libraries, self.calibration_seeds, self.config)
        self.assertEqual(selector.call_count, len(self.libraries))
        for (name, bank), call in zip(self.libraries.items(), selector.call_args_list):
            self.assertIs(call.args[0], rows)
            self.assertEqual(call.args[1], self.calibration_seeds)
            self.assertIs(call.args[2], bank)
            self.assertEqual(call.args[3:], (0.25, 0.8))
            self.assertEqual(choices[name]["selected_mode"], study.BASELINES[-1])
            self.assertEqual(choices[name]["selection"]["eligible_skill_modes"], [])
            self.assertIn(choices[name]["ungated_calibration_winner"],
                          {semantic.skill_mode(skill) for skill in bank})

    def test_row_order_does_not_change_frozen_choice(self):
        rows = self.rows()
        self.assertEqual(study.freeze_selection(rows, self.libraries, self.calibration_seeds, self.config),
                         study.freeze_selection(list(reversed(rows)), self.libraries,
                                                self.calibration_seeds, self.config))

    def test_offline_guard_blocks_all_socket_entrypoints_and_restores(self):
        originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
        with socket.socket() as sock:
            with study.offline_guard():
                for operation in (
                    lambda: sock.connect(("127.0.0.1", 9)),
                    lambda: sock.connect_ex(("127.0.0.1", 9)),
                    lambda: socket.create_connection(("127.0.0.1", 9)),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
                        operation()
        self.assertEqual((socket.socket.connect, socket.socket.connect_ex, socket.create_connection), originals)

    def test_offline_guard_restores_on_exception_and_nested_use(self):
        originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
        with self.assertRaisesRegex(ValueError, "sentinel"):
            with study.offline_guard():
                outer_connect = socket.socket.connect
                with study.offline_guard():
                    with self.assertRaises(RuntimeError):
                        socket.create_connection(("127.0.0.1", 9))
                self.assertIs(socket.socket.connect, outer_connect)
                raise ValueError("sentinel")
        self.assertEqual((socket.socket.connect, socket.socket.connect_ex, socket.create_connection), originals)

    def test_public_diagnostics_uses_only_metadata_and_counts_support(self):
        base = self.skills[0]
        skill = replace(base, rules=(
            base.rules[0],
            replace(base.rules[0], rule_id="duplicate_activation"),
            semantic.SemanticRule("never", (("a", "absent"),), 0.2, "test"),
        ))
        candidates = [PublicCandidate("c1", {"a": "one"}), PublicCandidate("c2", {"a": "two"}),
                      PublicCandidate("c3", {"a": "one"})]
        result = study.public_rule_diagnostics((skill,), candidates)[semantic.skill_mode(skill)]
        self.assertEqual([rule["support_count"] for rule in result["rules"]], [2, 2, 0])
        self.assertEqual(result["rules"][0]["support_fraction"], 2 / 3)
        self.assertEqual(result["unique_activation_count"], 2)
        self.assertEqual(result["zero_support_count"], 1)
        reordered = study.public_rule_diagnostics((skill,), list(reversed(candidates)))
        self.assertEqual(reordered[semantic.skill_mode(skill)], result)

    def test_config_returns_disjoint_declared_seed_sets(self):
        self.assertEqual(study.validate_config(self.config), (self.calibration_seeds, self.heldout_seeds))

    def test_config_rejects_overlapping_seed_ranges(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            study.validate_config({**self.config, "heldout_seed_start": 104})

    def test_config_rejects_nonpositive_or_noninteger_counts(self):
        for key in ("calibration_seed_count", "heldout_seed_count", "null_replicates",
                    "initial_observations", "reveal_rounds"):
            for invalid in (0, -1, 1.5, True, "5", None):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    study.validate_config({**self.config, key: invalid})

    def test_config_requires_enough_calibration_and_heldout_seeds(self):
        for key, invalid in (("calibration_seed_count", 4), ("heldout_seed_count", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                study.validate_config({**self.config, key: invalid})

    def test_config_requires_integer_seeds_and_known_dataset(self):
        for key in ("calibration_seed_start", "heldout_seed_start", "random_seed"):
            for invalid in (True, 1.5, "3", None):
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    study.validate_config({**self.config, key: invalid})
        with self.assertRaisesRegex(ValueError, "Unknown target dataset"):
            study.validate_config({**self.config, "target_dataset": "not_a_dataset"})


if __name__ == "__main__":
    unittest.main()
