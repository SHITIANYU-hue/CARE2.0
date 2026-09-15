from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compare_rsi_posttraining as report


class MatchedTrainingTest(unittest.TestCase):
    def fixture(self):
        config = {'tasks': [{'id': 'task'}], 'model_replicates': [0],
                  'evaluation_seed_start': 10, 'evaluation_seed_count': 2,
                  'reveal_rounds': 10}
        adapter_config = {**config, 'adapter_path': '/adapter'}
        base = {('task', 0, seed, arm): {'best_so_far_auc': value}
                for seed in (10, 11) for arm, value in
                [('fixed_initial', 9), ('true_feedback', 10), ('no_feedback', 8),
                 ('shuffled_feedback', 8), ('gp_ucb', 7)]}
        adapter = {key: dict(value) for key, value in base.items()}
        for key in adapter:
            adapter[key]['best_so_far_auc'] += 0 if key[-1] == 'gp_ucb' else (2 if key[-1] == 'true_feedback' else 1)
        return config, adapter_config, base, adapter

    def test_separates_initial_policy_improvement_from_update_improvement(self):
        c, a, b, t = self.fixture()
        with patch.object(report, 'read_complete', side_effect=[(c, b), (a, t)]):
            result = report.compare(Path('/base'), Path('/trained'))['tasks'][0]
        self.assertEqual(result['adapter_minus_base']['true_feedback']['mean'], 2)
        self.assertEqual(result['change_in_recursive_update_gain']['mean'], 1)

    def test_different_reveal_budget_rejected(self):
        c, a, b, t = self.fixture(); a['reveal_rounds'] = 11
        with patch.object(report, 'read_complete', side_effect=[(c, b), (a, t)]):
            with self.assertRaisesRegex(ValueError, 'reveal_rounds'):
                report.compare(Path('/base'), Path('/trained'))

    def test_changed_gp_control_rejected(self):
        c, a, b, t = self.fixture(); t['task', 0, 10, 'gp_ucb']['best_so_far_auc'] = 8
        with patch.object(report, 'read_complete', side_effect=[(c, b), (a, t)]):
            with self.assertRaisesRegex(ValueError, 'GP control'):
                report.compare(Path('/base'), Path('/trained'))


if __name__ == '__main__':
    unittest.main()
