from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_rsi_memory_component as rsi


class MemoryComponentTest(unittest.TestCase):
    def config(self):
        return json.loads((rsi.ROOT / 'configs/rsi_memory_component_v1.json').read_text())

    def test_split_overlap_rejected(self):
        config = self.config()
        config['evaluation_seed_start'] = config['development_seed_start'] + 1
        with self.assertRaisesRegex(ValueError, 'overlap'):
            rsi.validate_config(config)

    def test_negative_feedback_can_replace_and_later_restore_skill(self):
        scores = {'fixed': [6., 5.], 'alternative': [8., 9.]}
        chosen = rsi.select_skill(scores, 'fixed')
        self.assertEqual(chosen, 'alternative')
        scores['fixed'].extend([50., 50.])
        scores['alternative'].extend([1., 1.])
        self.assertEqual(rsi.select_skill(scores, chosen), 'fixed')

    def test_ties_retain_incumbent(self):
        self.assertEqual(rsi.select_skill({'a': [1.], 'b': [1.]}, 'b'), 'b')

    def test_nonfinite_feedback_rejected(self):
        with self.assertRaises(ValueError):
            rsi.select_skill({'a': [float('nan')]}, 'a')

    def test_seed_lists_are_disjoint_and_complete(self):
        dev, evaluation = rsi.validate_config(self.config())
        self.assertEqual(len(dev), 30)
        self.assertEqual(len(evaluation), 40)
        self.assertFalse(dev & evaluation)

    def test_constant_paired_difference_has_exact_interval(self):
        stats = rsi.paired_stats([-3.] * 40, np.random.default_rng(1), 100, 3)
        self.assertEqual(stats['bootstrap_ci95'], [-3., -3.])
        self.assertEqual(stats['losses'], 40)

    def test_real_seed_replays_identically_and_keeps_budget(self):
        config = self.config()
        spec = config['tasks'][0]
        record = json.loads((rsi.ROOT / spec['record']).read_text())
        job = (spec['id'], record['normalized_skills'][:1], 209090, config, 'development')
        a, b = rsi.run_job(job), rsi.run_job(job)
        adapter = rsi.replay.DATASET_BUILDERS[spec['id']]()
        skills = rsi.semantic.normalize_skills({'skills': record['normalized_skills'][:1]}, rsi.semantic.semantic_field_catalog(adapter))
        marshaled = json.loads(json.dumps([rsi.asdict(s) for s in skills]))
        c = rsi.run_job((spec['id'], marshaled, 209090, config, 'development'))
        self.assertEqual(a[0]['metrics'], c[0]['metrics'])
        self.assertEqual(a[0]['metrics'], b[0]['metrics'])
        self.assertEqual(a[0]['events'], b[0]['events'])
        self.assertEqual(len(a[0]['events']), 10)
        self.assertEqual(len(a[0]['initial']['candidate_ids']), 5)


if __name__ == '__main__':
    unittest.main()
