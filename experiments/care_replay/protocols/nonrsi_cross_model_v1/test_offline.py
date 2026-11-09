import unittest, tempfile, itertools, copy, socket
from pathlib import Path
from dataclasses import replace
import numpy as np
from .common import *
from . import inputs,generation,evaluation,analysis

class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load(BASE/'configs/nonrsi_cross_model_v1.json')
        cls.slot=slots(cls.config)[0]
        cls.cat={'x':{'a':10,'b':10},'z':{'c':10,'d':10}}
        cls.raw={'skills':[]}
        example=load(contract.RECORD)['prompt_payload']['output_contract']['skills'][0]
        for i,n in enumerate([2,1,1,1,1]):
            s=copy.deepcopy(example);s.update(skill_id=f'skill_{i}',hypothesis='synthetic only')
            s['rules']=[{'rule_id':f'rule_{j}','conditions':{'x':'a' if j==0 else 'b'},'weight':.5,'rationale':'synthetic'} for j in range(n)]
            cls.raw['skills'].append(s)
        assert not contract.strict_validate_library(cls.raw,cls.cat)
    def response(self,model=None,raw=None):
        model=model or next(m for m in self.config['models'] if m['id']==self.slot['model'])
        return {'object':'chat.completion','id':'synthetic_response','model':model['openrouter_id'],'provider':model['provider_candidate'],'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':__import__('json').dumps(self.raw if raw is None else raw)}}],'usage':{'is_byok':False,'cost':.001,'prompt_tokens':500,'completion_tokens':400,'completion_tokens_details':{'reasoning_tokens':0},'prompt_tokens_details':{'cache_write_tokens':0}}}
    def classify(self,r):
        m=next(m for m in self.config['models'] if m['id']==self.slot['model'])
        return generation.classify(r,m,self.cat,'.037')
    def test_slot_inventory(self):
        a=slots(self.config);self.assertEqual(len(a),660)
        self.assertEqual(a,slots(self.config));self.assertEqual(len({x['slot_id'] for x in a}),660)
    def test_all_models_same_contract(self):
        for m in self.config['models']:
            self.assertEqual(generation.classify(self.response(m),m,self.cat,'.037')['library'],self.raw)
    def test_whole_invalid_no_partial(self):
        raw=copy.deepcopy(self.raw);raw['skills'][4]['rules'][0]['weight']=3
        r=self.classify(self.response(raw=raw));self.assertEqual(r['status'],generation.INVALID);self.assertIsNone(r['library'])
    def test_duplicate_json_key(self):
        r=self.response();r['choices'][0]['message']['content']='{"skills":[],"skills":[]}'
        self.assertEqual(self.classify(r)['status'],generation.INVALID)
    def test_refusal_is_invalid(self):
        r=self.response();r['choices'][0]['message']['refusal']='cannot'
        self.assertEqual(self.classify(r)['status'],generation.INVALID)
    def test_truncation_is_invalid(self):
        r=self.response();r['choices'][0]['finish_reason']='length'
        self.assertEqual(self.classify(r)['status'],generation.INVALID)
    def test_unknown_finish_halts(self):
        r=self.response();r['choices'][0]['finish_reason']='unexpected'
        with self.assertRaises(Halt):self.classify(r)
    def test_wrong_identity_halts(self):
        r=self.response();r['provider']='other'
        with self.assertRaises(Halt):self.classify(r)
    def test_missing_cost_halts(self):
        r=self.response();r['usage'].pop('cost')
        with self.assertRaises(Halt):self.classify(r)
    def test_bad_usage_halts(self):
        for key,value in [('cost',float('nan')),('cost',-.1),('cost',True),('prompt_tokens',12001),('completion_tokens',5001),('is_byok',True)]:
            with self.subTest(key=key,value=value):
                r=self.response();r['usage'][key]=value
                with self.assertRaises(Halt):self.classify(r)
    def test_reasoning_halts(self):
        r=self.response();r['usage']['completion_tokens_details']['reasoning_tokens']=1
        with self.assertRaises(Halt):self.classify(r)
    def test_luna_cache_write_halts(self):
        m=self.config['models'][0];r=self.response(m);r['usage']['prompt_tokens_details']['cache_write_tokens']=1
        with self.assertRaises(Halt):generation.classify(r,m,self.cat,'.037')
    def test_ledger_no_network_or_retry(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=generation.Ledger(d,self.config)
            with self.assertRaises(Halt):ledger.reserve(self.slot,{'test':1})
            i=ledger.reserve(self.slot,{'test':1},test_only=True)
            with self.assertRaises(Halt):ledger.reserve(self.slot,{'test':1},test_only=True)
            ledger.settle(i,self.response(),self.cat)
            with self.assertRaises(Halt):ledger.settle(i,self.response(),self.cat)
    def test_ledger_unknown_charge_stays_reserved(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=generation.Ledger(d,self.config);i=ledger.reserve(self.slot,{},test_only=True)
            r=self.response();r['usage'].pop('cost')
            with self.assertRaises(Halt):ledger.settle(i,r,self.cat)
            state=load(Path(d)/'ledger.json')['attempts'][0]
            self.assertEqual(state['status'],'HALTED');self.assertIsNone(state['cost_usd'])
            with self.assertRaises(Halt):ledger.reserve(slots(self.config)[1],{},test_only=True)
    def test_ledger_budget(self):
        cfg=copy.deepcopy(self.config);cfg['proposed_model_inference_spend_cap_usd']='0.00001'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Halt):generation.Ledger(d,cfg).reserve(slots(cfg)[0],{},test_only=True)
    def test_compilation_no_loss(self):
        a=evaluation.compile_library(self.raw,self.cat,'a');b=evaluation.compile_library(self.raw,self.cat,'b')
        self.assertEqual([matched.skill_execution_signature(s) for s in a],[matched.skill_execution_signature(s) for s in b])
    def test_selector_invalid_fallback_and_no_heldout(self):
        rows=[];seeds=set(range(10));bank=evaluation.compile_library(self.raw,self.cat,'test')
        for seed in seeds:
            for mode in [*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in bank)]:
                rows.append({'mode':mode,'seed':seed,'split':'calibration','final_best':10.,'best_so_far_auc':10.,'top10_hit':0.,'rounds_to_top10':11.})
        result=evaluation.select(rows,{'test':bank},{'test':'VALID','bad':'INVALID_ANCHOR_FALLBACK'},seeds)
        self.assertEqual(result['choices']['bad']['primary_mode'],result['anchor'])
        self.assertEqual(result['choices']['test']['selected_mode'],result['anchor'])
        rows[0]['split']='heldout'
        with self.assertRaises(ValueError):evaluation.select(rows,{'test':bank},{'test':'VALID'},seeds)
    def test_selector_positive_and_negative_gate(self):
        bank=evaluation.compile_library(self.raw,self.cat,'gate');seeds=set(range(50))
        for delta in [2.,-2.]:
            rows=[]
            for seed in seeds:
                for mode in [*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in bank)]:
                    v=10.+(delta if mode not in matched.BASELINES else 0.)
                    rows.append({'mode':mode,'seed':seed,'split':'calibration','final_best':v,'best_so_far_auc':v,'top10_hit':0.,'rounds_to_top10':11.})
            selected=evaluation.select(rows,{'test':bank},{'test':'VALID'},seeds)
            c=selected['choices']['test']
            self.assertNotEqual(c['primary_mode'],selected['anchor'])
            self.assertEqual(c['selected_mode']==selected['anchor'],delta<0)
    def test_synthetic_replay_action_budget_and_identifier_invariance(self):
        from unittest.mock import patch
        a=matched.replay.DATASET_BUILDERS['synthetic_suzuki_i']()
        a=replace(a,candidates=tuple(replace(c,metadata={**c.metadata,'x':'a' if i%2 else 'b','z':'c' if i%2 else 'd'}) for i,c in enumerate(a.candidates[:24])))
        left=evaluation.compile_library(self.raw,self.cat,'left')
        right=evaluation.compile_library(self.raw,self.cat,'right')
        with patch.object(inputs,'adapter',return_value=a),offline():
            rows,traces,_=evaluation.replay_seed('synthetic_suzuki_i',731,'calibration',left+right)
        self.assertEqual(len(rows),13)
        bymode={m:[t for t in traces if t['mode']==m] for m in {r['mode'] for r in rows}}
        for events in bymode.values():
            self.assertEqual(len(events),10)
            self.assertEqual(len({t['selected_candidate'] for t in events}),10)
            self.assertEqual([t['public_observed_count'] for t in events],list(range(5,15)))
        for l,r in zip(left,right):
            self.assertEqual([e['selected_candidate'] for e in bymode[matched.semantic.skill_mode(l)]],[e['selected_candidate'] for e in bymode[matched.semantic.skill_mode(r)]])
    def test_random_reference_exhaustive(self):
        values=[-2.,1.,1.,6.];ref=analysis.random_reference(values,1,2)
        expected=np.mean([(max(p[:2])+max(p[:3]))/2 for p in itertools.permutations(values)])
        self.assertAlmostEqual(ref['random_auc'],expected)
    def test_zero_and_nearzero_reference(self):
        ref=analysis.random_reference([5.]*20)
        self.assertIsNone(analysis.normalize_delta([[0]],ref))
        ref=analysis.random_reference([0.]+[1.]*19)
        self.assertEqual(ref['zero_denominator'],True)
    def test_bootstrap_constant_and_sign(self):
        rng=np.random.default_rng(42);ld=rng.integers(0,10,(1000,10));sd=rng.integers(0,20,(1000,20))
        r=analysis.bootstrap(np.ones((10,20))*3,ld,sd)
        self.assertEqual(r['low'],3);self.assertEqual(r['high'],3)
        x=np.arange(200).reshape(10,20);r=analysis.bootstrap(x,ld,sd);s=analysis.bootstrap(-x,ld,sd)
        self.assertAlmostEqual(r['low'],-s['high'])
    def test_random_controls_public_only(self):
        bank=evaluation.compile_library(self.raw,self.cat,'template')
        children,diag=evaluation.random_children(bank,self.cat,{'x':'categorical_2','z':'categorical_2'},[{'x':'a','z':'c'}]*10+[{'x':'b','z':'d'}]*10,'parent',3)
        self.assertEqual(len(children),20);self.assertTrue(all(len(b)==5 for b in children.values()))
    def test_network_guard(self):
        with offline():
            with self.assertRaises(RuntimeError):socket.create_connection(('127.0.0.1',1))
    def test_target_outcome_invariance_all_prompts(self):
        for cell in {c['cell']:c for c in self.config['generation_cells']}.values():
            with self.subTest(cell=cell['cell']),offline():
                a=inputs.adapter(cell['target']);cat=inputs.catalog(a)
                protected=set(inputs.catalog(a))
                outcome_keys={a.hidden_target,'yield_value','experimental_hydration_free_energy','calculated_hydration_free_energy','experimental_lipophilicity','hydration_affinity_score','normalized_lipophilicity_score'}-protected
                altered=replace(a,candidates=tuple(replace(c,objective_value=98765.,metadata={k:98765. if k in outcome_keys else v for k,v in c.metadata.items()}) for c in a.candidates))
                self.assertEqual(inputs.payload(self.config,cell),inputs.payload(self.config,cell,target=altered))
    def test_source_outcome_visibility(self):
        for r in self.config['routes']:
            with self.subTest(route=r['route_id']),offline():
                a=inputs.adapter(r['source']);b=replace(a,candidates=tuple(replace(c,objective_value=0.) for c in a.candidates))
                self.assertEqual(inputs.evidence(r,'source_schema_only',a),inputs.evidence(r,'source_schema_only',b))
                self.assertNotEqual(inputs.evidence(r,'full',a),inputs.evidence(r,'full',b))
    def test_T_S_F_shared_prompt_shell(self):
        d=BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'
        for r in self.config['routes']:
            t=load(d/(r['shared_target_cell']+'.json'))['payload']
            s=load(d/('S_'+r['route_id']+'.json'))['payload'];f=load(d/('F_'+r['route_id']+'.json'))['payload']
            for x in [t,s,f]:x.pop('source_transfer_evidence')
            self.assertEqual(t,s);self.assertEqual(s,f)

if __name__=='__main__':unittest.main()
