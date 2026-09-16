import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import llm_semantic_skills as semantic
import run_rsi_evidence_gate as gated
import run_synthetic_suzuki as replay


class EvidenceGateTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((
            gated.ROOT / 'configs/rsi_evidence_gate_adapter_phonons_v1.json'
        ).read_text())

    def test_three_seed_splits_are_pairwise_disjoint(self):
        splits = gated.validate_config(self.config)
        self.assertFalse(splits['development'] & splits['promotion'])
        self.assertFalse(splits['development'] & splits['evaluation'])
        self.assertFalse(splits['promotion'] & splits['evaluation'])

    def test_overlap_is_rejected(self):
        config = json.loads(json.dumps(self.config))
        config['promotion_seed_start'] = config['development_seed_start']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            gated.validate_config(config)

    def _row(self, skill_id, phase='development', base=10.0):
        adapter = replay.DATASET_BUILDERS['real_matbench_phonons']()
        candidate_ids = [candidate.candidate_id for candidate in adapter.candidates[:2]]
        return {
            'phase': phase,
            'seed': 7,
            'skill_id': skill_id,
            'initial': {'candidate_ids': [], 'revealed_values': [base]},
            'metrics': {'seed': 7, 'best_so_far_auc': base + 2, 'final_best': base + 3},
            'events': [
                {
                    'round_index': index,
                    'selected_candidate': candidate_id,
                    'revealed_value': base + index,
                    'best_so_far': base + index,
                    'hypothesis_snapshot': {'matched_rules': ['r'] if index == 0 else []},
                }
                for index, candidate_id in enumerate(candidate_ids)
            ],
        }

    def test_rich_feedback_has_only_revealed_values_and_public_features(self):
        adapter = replay.DATASET_BUILDERS['real_matbench_phonons']()
        catalog = semantic.semantic_field_catalog(adapter)
        decision = {'skills': [{'skill_id': 'a'}, {'skill_id': 'b'}]}
        result = gated.rich_feedback(
            [self._row('a'), self._row('b', base=20)], decision, adapter, catalog, 1)
        encoded = json.dumps(result)
        self.assertEqual(result['evidence_scope']['unobserved_candidate_labels_included'], False)
        self.assertNotIn('objective_value', encoded)
        self.assertNotIn('source_row', encoded)
        self.assertNotIn('composition', encoded)
        self.assertEqual(result['skill_results'][0]['selected_candidate_count'], 2)

    def test_rich_feedback_rejects_final_evaluation_rows(self):
        adapter = replay.DATASET_BUILDERS['real_matbench_phonons']()
        catalog = semantic.semantic_field_catalog(adapter)
        decision = {'skills': [{'skill_id': 'a'}]}
        with self.assertRaisesRegex(ValueError, 'Evaluation'):
            gated.rich_feedback([self._row('a', phase='evaluation')], decision,
                                adapter, catalog)

    def _gate_rows(self, values, skill_id):
        return [
            {'seed': seed, 'skill_id': skill_id,
             'metrics': {'best_so_far_auc': value, 'final_best': value}}
            for seed, value in enumerate(values)
        ]

    def test_promotion_requires_effect_wins_and_no_final_regression(self):
        incumbent = self._gate_rows([10] * 8, 'old')
        challenger = self._gate_rows([12, 12, 12, 12, 12, 10, 10, 10], 'new')
        accepted = gated.promotion_gate(incumbent, challenger, self.config, 1)
        self.assertTrue(accepted['accepted'])
        weak = self._gate_rows([12, 12, 12, 12, 10, 10, 10, 10], 'new')
        rejected = gated.promotion_gate(incumbent, weak, self.config, 1)
        self.assertFalse(rejected['accepted'])

    def test_selected_policy_resolves_uniquely(self):
        decision = {'skills': [{'skill_id': 'a'}, {'skill_id': 'b'}],
                    'selected_skill_id': 'b'}
        self.assertEqual(gated.selected_only(decision)['skills'], [{'skill_id': 'b'}])


if __name__ == '__main__':
    unittest.main()
