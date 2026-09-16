from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_rsi_prompt_exposure as exposure


class PromptExposureTest(unittest.TestCase):
    def test_json_whitespace_does_not_hide_semantic_prompt_reuse(self):
        a = [{'role': 'user', 'content': '{"target":1,"source":2}'}]
        b = [{'role': 'user', 'content': '{ "source": 2, "target": 1 }'}]
        self.assertNotEqual(exposure.fingerprint(a), exposure.fingerprint(b))
        self.assertEqual(exposure.fingerprint(a, True), exposure.fingerprint(b, True))

    def test_extra_schema_correction_is_distinct_prompt(self):
        a = [{'role': 'user', 'content': '{"target":1}'}]
        b = a + [{'role': 'user', 'content': 'Fix invalid schema.'}]
        self.assertNotEqual(exposure.fingerprint(a, True), exposure.fingerprint(b, True))

    def test_training_source_exposure_is_counted_without_target_exposure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'run'; sft = root / 'sft'
            run.mkdir(); sft.mkdir()
            (run / 'run_status.json').write_text('{"complete":true}')
            (run / 'config.json').write_text(json.dumps({'tasks': [{'id': 'new_task', 'source': 'reference'}]}))
            def messages(target):
                return [{'role': 'system', 'content': 'Use supplied evidence.'},
                        {'role': 'user', 'content': json.dumps({'target': {'dataset': target}, 'source_transfer_evidence': {'source_dataset': 'reference'}})}]
            teacher = {'messages': messages('old_task') + [{'role': 'assistant', 'content': '{}'}]}
            (sft / 'train.jsonl').write_text(json.dumps(teacher) + '\n')
            (sft / 'validation.jsonl').write_text('')
            (run / 'attempt_0_request.json').write_text(json.dumps({'request': {'messages': messages('new_task')}}))
            result = exposure.audit(run, sft)
            self.assertEqual(result['targets']['new_task']['sft_target_rows'], 0)
            self.assertEqual(result['targets']['new_task']['source_as_sft_source_rows'], 1)
            self.assertEqual(result['exact_matching_requests'], 0)
            (run / 'attempt_1_request.json').write_text(json.dumps({'request': {'messages': messages('old_task')}}))
            result = exposure.audit(run, sft)
            self.assertEqual(result['exact_matching_requests'], 1)
            self.assertEqual(result['requests'][1]['exact_sft_rows'][0]['split'], 'train')
            (run / 'run_status.json').write_text('{"complete":false}')
            with self.assertRaisesRegex(ValueError, 'complete'):
                exposure.audit(run, sft)


if __name__ == '__main__':
    unittest.main()
