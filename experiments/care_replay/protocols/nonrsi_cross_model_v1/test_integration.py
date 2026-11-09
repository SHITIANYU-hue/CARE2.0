import unittest,tempfile,copy,json,os
from pathlib import Path
from dataclasses import replace,asdict
from unittest.mock import patch
import numpy as np
from .common import *
from . import generation,request_contract,inputs,evaluation,pipeline,analysis,reporting,transport,freeze
from . import test_offline as fixtures

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OfflineTests.setUpClass();cls.config=fixtures.OfflineTests.config;cls.raw=fixtures.OfflineTests.raw;cls.cat=fixtures.OfflineTests.cat
    def synthetic_adapter(self):
        a=matched.replay.DATASET_BUILDERS['synthetic_suzuki_i']()
        return replace(a,candidates=tuple(replace(c,metadata={**c.metadata,'x':'a' if i%2 else 'b','z':'c' if i%2 else 'd'}) for i,c in enumerate(a.candidates[:24])))
    def test_all_660_requests_obey_approved_parameters(self):
        for slot in slots(self.config):
            prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(slot['cell']+'.json'))
            body=request_contract.build(self.config,slot,prompt)
            self.assertEqual(body['max_tokens'],5000);self.assertEqual(body['plugins'],[]);self.assertEqual(body['transforms'],[])
            self.assertTrue(body['provider']['require_parameters']);self.assertFalse(body['provider']['allow_fallbacks'])
            self.assertEqual(len(body['provider']['only']),1)
            if slot['model']=='gpt6luna_public_id':self.assertNotIn('temperature',body);self.assertEqual(body['reasoning'],{'effort':'none'})
            else:self.assertEqual(body['temperature'],.45)
    def test_current_endpoints_and_drift_halt(self):
        evidence=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/endpoint_page_evidence.json')
        for m in self.config['models']:
            r=next(r for rows in evidence.values() for r in rows if r['id']==m['observed_endpoint_id'])
            self.assertTrue(request_contract.check_endpoint(m,r))
            r=copy.deepcopy(r);r['pricing_version_id']='changed'
            with self.assertRaises(Halt):request_contract.check_endpoint(m,r)
    def test_freeze_integrity_and_no_implicit_paid_authorization(self):
        import platform
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=copy.deepcopy(self.config);c['freeze_complete']=True
            save(root/'config.json',c);save(root/'payload.json',{'immutable':True})
            lock={'kind':'nonrsi_cross_model_v1_formal_freeze','offline_only':True,'config_path':'config.json','config_sha256':digest(c),'files':{name:freeze.file_hash(root/name) for name in ['config.json','payload.json']},'environment':{'python':platform.python_version(),'packages':{}}}
            save(root/'lock.json',lock)
            with patch.object(freeze,'ROOT',root):
                self.assertEqual(freeze.verify(root/'lock.json'),c)
                with self.assertRaises(FileNotFoundError):freeze.authorize(root/'lock.json',root/'missing_authorization.json',root/'missing_attestation.json')
                save(root/'payload.json',{'immutable':False})
                with self.assertRaises(Halt):freeze.verify(root/'lock.json')
    def test_actual_cost_releases_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=generation.Ledger(d,self.config);a,b=slots(self.config)[:2]
            i=ledger.reserve(a,{},test_only=True);m=ledger.models[a['model']]
            fixture=fixtures.OfflineTests();r=fixture.response(m);r['usage']['cost']=.05
            ledger.settle(i,r,self.cat)
            j=ledger.reserve(b,{},test_only=True)
            state=load(ledger.path);self.assertEqual(j,1);self.assertEqual(state['attempts'][0]['cost_usd'],'0.05');self.assertEqual(state['attempts'][1]['reservation_usd'],'1')
    def test_remaining_below_one_dollar_stops(self):
        c=copy.deepcopy(self.config);c['proposed_model_inference_spend_cap_usd']='1.04'
        with tempfile.TemporaryDirectory() as d:
            ledger=generation.Ledger(d,c);a,b=slots(c)[:2];i=ledger.reserve(a,{},test_only=True)
            r=fixtures.OfflineTests().response(ledger.models[a['model']]);r['usage']['cost']=.05;ledger.settle(i,r,self.cat)
            with self.assertRaises(Halt):ledger.reserve(b,{},test_only=True)
    def test_no_authorization_stops_before_transport_or_key_access(self):
        with patch.object(os.environ,'get',side_effect=AssertionError('must not read key')),patch.object(transport.urllib.request,'build_opener',side_effect=AssertionError('must not create transport')):
            with self.assertRaises(FileNotFoundError):transport.run('/nonexistent-freeze','/no-auth','/no-attestation','/unused')
    def test_gap_numerically_stable_and_conditional_NA(self):
        x=[1e12+i*.0001220703125 for i in range(20)]
        self.assertGreater(analysis.random_reference(x)['headroom'],0)
        ref=analysis.random_reference([0,1,2,3],1,2,[3]);self.assertTrue(ref['zero_denominator'])
        ld=np.tile(np.arange(10),(100,1));sd=np.tile(np.arange(4),(100,1))
        r=reporting.summarize_delta(np.ones((10,4)),ld,sd,analysis.random_reference(range(20)),conditional=[ref]*4)
        self.assertIsNone(r['same_initial_secondary']['summary']);self.assertEqual(r['same_initial_secondary']['zero_denominator_seeds'],4)
    def test_library_only_sensitivity_equals_full_crossed_bootstrap(self):
        rng=np.random.default_rng(11);library=rng.normal(size=10);x=np.tile(library[:,None],(1,100))
        ld=rng.integers(0,10,(2000,10));sd=rng.integers(0,100,(2000,100))
        full=reporting.samples(x,ld,sd);self.assertTrue(np.allclose(full,library[ld].mean(axis=1),rtol=0,atol=1e-14))
    def test_completed_ledger_requires_all_slots_and_raw_provenance(self):
        c=self.config
        with tempfile.TemporaryDirectory() as d:
            ledger=generation.Ledger(d,c)
            # Fill a synthetic ledger using exactly the validator's input/output contract.
            attempts=[]
            for i,slot in enumerate(slots(c)):
                m=ledger.models[slot['model']];r=fixtures.OfflineTests().response(m,raw={'skills':[]});r['id']=f'synthetic_{i}'
                decision=generation.classify(r,m,self.cat,'1')
                request={};save(Path(d)/f'{i:04d}.request.json',request);save(Path(d)/f'{i:04d}.response.json',r)
                attempts.append(dict(slot=slot,request_sha256=digest(request),response_sha256=digest(r),reservation_usd='1',test_only=True,**decision))
            save(ledger.path,{'config_sha256':digest(c),'attempts':attempts})
            cats={r['target']:self.cat for r in c['generation_cells']}
            self.assertEqual(len(ledger.complete(allow_test=True,catalogs=cats)),660)
            with self.assertRaises(Halt):ledger.complete()
            state=load(ledger.path);state['attempts'][0]['cost_usd']='0';save(ledger.path,state)
            with self.assertRaises(Halt):ledger.complete(allow_test=True,catalogs=cats)
    def test_nested_children_preserve_parent_cluster(self):
        parent=np.full((3,8),5.);kids=np.full((3,10,8),2.);sd=np.tile(np.arange(8),(300,1))
        r=reporting.nested_random_delta(parent,kids,sd,np.random.default_rng(1));self.assertTrue(np.all(r==3))
        kids=np.array([np.full((10,8),v) for v in [1.,3.,5.]])
        r=reporting.nested_random_delta(parent,kids,sd,np.random.default_rng(1));self.assertGreater(r.std(),0)
    def test_resumable_pipeline_and_global_selection_lock(self):
        target='synthetic_suzuki_i';a=self.synthetic_adapter();cfg=copy.deepcopy(self.config)
        cfg['replay_proposed']['target_seed_ranges']=[{'target':target,'calibration_start':700,'heldout_start':800}];cfg['replay_proposed']['calibration_seeds']=10;cfg['replay_proposed']['heldout_seeds']=3
        bank=evaluation.compile_library(self.raw,self.cat,'fixture')
        definition={'config_sha256':digest(cfg),'generation_sha256':'synthetic','targets':{target:{'banks':{'fixture':[asdict(s) for s in bank]},'outcomes':{'fixture':'VALID','invalid':'INVALID_ANCHOR_FALLBACK'},'pool_sha256':None}}}
        with tempfile.TemporaryDirectory() as d,patch.object(inputs,'adapter',return_value=a):
            definition['targets'][target]['pool_sha256']=inputs.certify(a)['complete_candidate_records_sha256']
            with self.assertRaises(FileNotFoundError):pipeline.heldout(cfg,definition,d)
            lock=pipeline.calibrate(cfg,definition,d)
            self.assertEqual(lock,pipeline.calibrate(cfg,definition,d))
            rows=pipeline.heldout(cfg,definition,d)
            self.assertEqual(rows,pipeline.heldout(cfg,definition,d))
            choice=load(Path(d)/target/'selection.json');choice['anchor']='tampered';save(Path(d)/target/'selection.json',choice)
            with self.assertRaises(Halt):pipeline.heldout(cfg,definition,d)
    def test_all_invalid_660_slots_end_to_end(self):
        cfg=copy.deepcopy(self.config);cfg['replay_proposed']['calibration_seeds']=10;cfg['replay_proposed']['heldout_seeds']=3;cfg['statistics_proposed']['bootstrap_replicates']=100
        attempts=[{'slot':s,'status':'INVALID_ANCHOR_FALLBACK','cost_usd':'0.001','library':None} for s in slots(cfg)]
        a=self.synthetic_adapter()
        with tempfile.TemporaryDirectory() as d,patch.object(inputs,'adapter',return_value=a):
            definition=pipeline.prepare(cfg,attempts,d)
            self.assertEqual(len(definition['slots']),660);self.assertEqual(definition['random_children'],{})
            lock=pipeline.calibrate(cfg,definition,d)
            held=pipeline.heldout(cfg,definition,d)
            report=reporting.analyze(cfg,definition,lock,held,d)
            self.assertTrue(all(x['raw_auc']['mean']==0 for x in report['primary']))
            self.assertEqual(sum(c['libraries'] for c in report['cells'].values()),660)
            self.assertTrue(all(c['valid_libraries']==0 for c in report['cells'].values()))
    def test_36_primary_comparisons_and_shared_T(self):
        cfg=copy.deepcopy(self.config);cfg['statistics_proposed']['bootstrap_replicates']=100;cfg['replay_proposed']['heldout_seeds']=4
        definition={'targets':{},'random_children':{}};lock={'selections':{}};rows={};a=self.synthetic_adapter()
        for item in cfg['replay_proposed']['target_seed_ranges']:
            target=item['target'];seeds=pipeline.seed_set(cfg,target,'heldout');choices={};outcomes={};tr=[]
            for mode in matched.BASELINES:
                tr.extend(dict(mode=mode,seed=s,best_so_far_auc=10.,final_best=10.,top10_hit=0.) for s in seeds)
            for cell in [c for c in cfg['generation_cells'] if c['target']==target]:
                for i in range(10):
                    name=cell['model']+'__'+cell['cell']+f'__{i:02d}'
                    choices[name]={'primary_mode':name,'selected_mode':name};outcomes[name]='INVALID_ANCHOR_FALLBACK'
                    value=12. if cell['evidence_mode']=='full' else 11. if cell['evidence_mode']=='source_schema_only' else 10.
                    tr.extend(dict(mode=name,seed=s,best_so_far_auc=value,final_best=value,top10_hit=0.) for s in seeds)
            definition['targets'][target]={'outcomes':outcomes};lock['selections'][target]={'anchor':matched.BASELINES[0],'choices':choices};rows[target]=tr
        with tempfile.TemporaryDirectory() as d,patch.object(inputs,'adapter',return_value=a):
            r=reporting.analyze(cfg,definition,lock,rows,d)
        self.assertEqual(len(r['primary']),36);self.assertEqual(len(r['cells']),66)
        ft=[x for x in r['primary'] if 'F minus T' in x['contrast']]
        self.assertEqual(len(ft),24);self.assertTrue(all(x['raw_auc']['mean']==2 for x in ft))
        self.assertTrue(all(x['raw_auc']['nominal_998611']['nominal_mass']==1-.05/36 for x in ft))
        self.assertEqual(len(r['random_controls']),12)
        self.assertEqual(r['statistical_scope'],'descriptive_cross_model_replication')
        self.assertEqual(sum(len(v['per_library']) for v in r['cells'].values()),660)
        self.assertNotIn('bonferroni_36',json.dumps(r))
        self.assertTrue(all(len(v['per_library_mean_delta_auc'])==10 for v in r['primary']))

if __name__=='__main__':unittest.main(verbosity=2)
