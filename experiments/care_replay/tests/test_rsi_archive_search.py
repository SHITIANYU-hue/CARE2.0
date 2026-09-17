import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import run_rsi_archive_search as archive


class ArchiveSearchTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((archive.ROOT / 'configs/rsi_archive_search_esol_v1.json').read_text())

    def test_split_disjointness_and_overlap_rejection(self):
        splits = archive.validate_config(self.config)
        self.assertFalse(splits['development'] & splits['promotion'])
        self.assertFalse(splits['development'] & splits['evaluation'])
        self.assertFalse(splits['promotion'] & splits['evaluation'])
        bad = json.loads(json.dumps(self.config))
        bad['evaluation_seed_start'] = bad['promotion_seed_start']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            archive.validate_config(bad)

    def test_two_candidates_are_measured_not_just_model_choice(self):
        incumbent = self._rows([10] * 8, 'old')
        a = self._rows([12, 12, 12, 12, 12, 12, 10, 10], 'a')
        b = self._rows([13, 13, 13, 13, 13, 13, 10, 10], 'b')
        chosen, scored = archive.choose_archive_candidate(
            incumbent, [({'skill_id': 'a'}, a), ({'skill_id': 'b'}, b)],
            self.config, 1)
        self.assertEqual(chosen, 'b')
        self.assertEqual({item['skill_id'] for item in scored}, {'a', 'b'})

    def test_no_candidate_promoted_when_gate_fails(self):
        incumbent = self._rows([10] * 8, 'old')
        a = self._rows([12, 12, 12, 12, 12, 10, 10, 10], 'a')
        b = self._rows([9] * 8, 'b')
        chosen, scored = archive.choose_archive_candidate(
            incumbent, [({'skill_id': 'a'}, a), ({'skill_id': 'b'}, b)],
            self.config, 1)
        self.assertIsNone(chosen)
        self.assertFalse(any(item['gate']['accepted'] for item in scored))

    def test_prompt_contains_bounded_verified_archive(self):
        incumbent = {'skills': [{'skill_id': 'old'}], 'selected_skill_id': 'old'}
        cards = [{'generation': index, 'status': 'accepted_on_promotion_only'}
                 for index in (1, 2, 3)]
        prompt = archive.archive_prompt({'source': 'fixture'}, incumbent,
                                        {'development_only': True}, cards)
        self.assertEqual([item['generation'] for item in prompt['validated_policy_archive']], [2, 3])
        self.assertNotIn('evaluation', prompt.get('current_development_evidence', {}))

    def test_initial_prompt_is_target_only(self):
        task = self.config['task']
        payload = archive.generator.build_prompt_payload(
            task['source'], task['id'], self.config['source_observations'],
            self.config['discount'], self.config['skills_per_generation'],
            self.config['evidence_mode'], knowledge_context=None,
            proposal_mode='parametric')
        source = payload['source_transfer_evidence']
        self.assertIsNone(source['source_dataset'])
        self.assertIsNone(source['source_objective'])
        self.assertEqual(source['shared_value_priors'], [])

    def test_pinned_phonons_initial_hash(self):
        config = json.loads((archive.ROOT / 'configs/rsi_archive_search_phonons_v1.json').read_text())
        self.assertEqual(config['evidence_mode'], 'full')
        self.assertEqual(
            archive.sha(archive.REPO / config['initial_decision_path']),
            config['initial_decision_sha256'],
        )

    @staticmethod
    def _rows(values, skill_id):
        return [
            {'seed': index, 'skill_id': skill_id,
             'metrics': {'best_so_far_auc': value, 'final_best': value}}
            for index, value in enumerate(values)
        ]


if __name__ == '__main__':
    unittest.main()
