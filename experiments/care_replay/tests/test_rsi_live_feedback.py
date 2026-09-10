import json
import os
from pathlib import Path
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_rsi_live_feedback as live


class LiveRSITest(unittest.TestCase):
    def test_credentials_are_redacted_from_serialized_text(self):
        with patch.dict(os.environ, {'COMMONSTACK_API_KEY':'fixture_sensitive_value'}):
            self.assertEqual(live.clean_text('prefix fixture_sensitive_value suffix'), 'prefix [REDACTED_CREDENTIAL] suffix')

    def test_overlap_rejected(self):
        c=json.loads((live.ROOT/'configs/rsi_live_feedback_v1.json').read_text())
        c['evaluation_seed_start']=c['development_seed_start']
        with self.assertRaises(ValueError): live.validate_config(c)

    def rows(self):
        return [{'phase':'development','skill_id':k,'metrics':{'seed':s,'best_so_far_auc':v,'final_best':v+1}}
                for k,v in [('a',10),('b',90)] for s in (1,2)]

    def test_shuffling_breaks_skill_binding_preserves_scores(self):
        decision={'skills':[{'skill_id':'a'},{'skill_id':'b'}]}
        true=live.feedback(self.rows(),decision,'true_feedback',1)['skill_results']
        wrong=live.feedback(self.rows(),decision,'shuffled_feedback',1)['skill_results']
        self.assertEqual(true[0]['mean_auc'],wrong[1]['mean_auc'])
        self.assertNotEqual(true[0]['mean_auc'],wrong[0]['mean_auc'])

    def test_no_feedback_never_exposes_metrics(self):
        payload=live.build_revision_prompt({'target':{}},{'skills':[]},[{'secret_evaluation':99}],'no_feedback')
        self.assertNotIn('secret_evaluation',json.dumps(payload))
        self.assertIsNone(live.feedback(self.rows(),{},'no_feedback',1))

    def test_evaluation_rows_rejected(self):
        rows=self.rows();rows[0]['phase']='evaluation'
        with self.assertRaisesRegex(ValueError,'Evaluation'):
            live.feedback(rows,{'skills':[{'skill_id':'a'},{'skill_id':'b'}]},'true_feedback',1)

    def test_compiler_rejects_invalid_public_value(self):
        skill=lambda k:{'skill_id':k,'rules':[{'rule_id':'r','conditions':{'x':'unknown'},'weight':1}]}
        with self.assertRaisesRegex(ValueError,'unsupported'):
            live.normalize_response(json.dumps({'skills':[skill('a'),skill('b')],'selected_skill_id':'a'}),{'x':{'good':1}},2)

    def test_base_prompt_invariant_to_hidden_target_labels(self):
        config=json.loads((live.ROOT/'configs/rsi_live_feedback_v1.json').read_text())
        for spec in config['tasks']:
            target_id=spec['id']
            target=live.replay.DATASET_BUILDERS[target_id]()
            original=live.generator.build_prompt_payload(spec['source'],target_id,32,.65,2)
            shadow=replace(target,candidates=tuple(replace(c,objective_value=100.-c.objective_value) for c in target.candidates))
            with patch.dict(live.replay.DATASET_BUILDERS,{target_id:lambda:shadow}):
                changed=live.generator.build_prompt_payload(spec['source'],target_id,32,.65,2)
            self.assertEqual(original,changed,target_id)

    def test_valid_response_round_trips_to_executor(self):
        skill=lambda k:{'skill_id':k,'rules':[{'rule_id':'r','conditions':{'x':'good'},'weight':1}]}
        result=live.normalize_response(json.dumps({'skills':[skill('a'),skill('b')],'selected_skill_id':'b'}),{'x':{'good':1}},2)
        self.assertEqual(result['selected_skill_id'],'b')
        self.assertEqual(len(result['skills']),2)


if __name__=='__main__': unittest.main()
