import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_rsi_live_feedback as live
import local_rsi_backend
import export_rsi_full_dataset as dataset


class LocalBackendTest(unittest.TestCase):
    def response(self):
        skill = lambda k: {'skill_id': k, 'rules': [
            {'rule_id': 'r', 'conditions': {'x': 'good'}, 'weight': .5}]}
        return {'choices': [{'message': {'content': json.dumps({
            'skills': [skill('a'), skill('b')], 'selected_skill_id': 'b',
            'lesson': 'l', 'revision_summary': 'r'})}}],
            'local_provenance': {'weights_frozen_during_search': True}}

    def test_local_generation_needs_no_external_key_and_uses_same_compiler(self):
        config = {'generation_backend': 'local_qlora', 'base_model_path': '/model',
                  'model': 'base', 'temperature': 0, 'max_tokens': 100,
                  'max_attempts_per_generation': 1, 'skills_per_generation': 2}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch.object(local_rsi_backend, 'complete', return_value=self.response()) as complete, \
             patch.object(live.urllib.request, 'urlopen') as external:
            decision = live.generate(config, {'target': {}}, {'x': {'good': 1}}, Path(tmp))
            self.assertEqual(decision['selected_skill_id'], 'b')
            external.assert_not_called()
            self.assertEqual(complete.call_args.args[1]['messages'][1]['content'], '{"target": {}}')
            record = json.loads((Path(tmp) / 'attempt_0_response.json').read_text())
            self.assertTrue(record['response']['local_provenance']['weights_frozen_during_search'])

    def test_invalid_local_rule_is_retained_and_rejected(self):
        response = self.response()
        response['choices'][0]['message']['content'] = response['choices'][0]['message']['content'].replace('good', 'unknown')
        config = {'generation_backend': 'local_qlora', 'model': 'adapter', 'temperature': 0,
                  'max_tokens': 100, 'max_attempts_per_generation': 1, 'skills_per_generation': 2}
        with tempfile.TemporaryDirectory() as tmp, patch.object(local_rsi_backend, 'complete', return_value=response):
            with self.assertRaises(RuntimeError):
                live.generate(config, {}, {'x': {'good': 1}}, Path(tmp))
            record = json.loads((Path(tmp) / 'attempt_0_response.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('unsupported', record['validation_error'])

    def test_prompt_group_split_keeps_duplicate_group_together(self):
        content = self.response()['choices'][0]['message']['content']
        def row(tag):
            return {'messages': [
                {'role': 'user', 'content': json.dumps({'target': {'public_semantic_fields': {'x': {'good': 1}}}, 'tag': tag})},
                {'role': 'assistant', 'content': content}], 'metadata': {'task': 'test'}}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'; source.mkdir()
            (source / 'train.jsonl').write_text(json.dumps(row('shared')) + '\n')
            (source / 'validation.jsonl').write_text(json.dumps(row('shared')) + '\n' + json.dumps(row('unique')) + '\n')
            output = Path(tmp) / 'output'
            manifest = dataset.build(source, output)
            self.assertEqual(manifest['overlapping_validation_rows_moved'], 1)
            train = [json.loads(x) for x in (output / 'train.jsonl').read_text().splitlines()]
            validation = [json.loads(x) for x in (output / 'validation.jsonl').read_text().splitlines()]
            self.assertEqual(len(train), 2)
            self.assertFalse({dataset.prompt_hash(r) for r in train} & {dataset.prompt_hash(r) for r in validation})


if __name__ == '__main__':
    unittest.main()
