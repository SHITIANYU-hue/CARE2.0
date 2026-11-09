import unittest,tempfile,json,io,copy
from pathlib import Path
from unittest.mock import patch
import urllib.error
from . import runner as r
from protocols.nonrsi_cross_model_v1 import freeze,generation,inputs,request_contract
from protocols.nonrsi_cross_model_v1.common import *

BASE_LOCK=BASE/'outputs/nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json'
ORIGINAL=BASE/'outputs/nonrsi_cross_model_resume_v1/ledger'

class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=freeze.verify(BASE_LOCK);cls.order=slots(cls.config)
        cls.catalogs={s['target']:inputs.catalog(inputs.adapter(s['target'])) for s in cls.order}
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.directory=Path(self.tmp.name)/'ledger'
        r.initialize(self.directory,ORIGINAL,self.config)
        self.ledger=r.ResumeLedger(self.directory,self.config,production_authorized=True)
    def tearDown(self):self.tmp.cleanup()
    def request(self,i):
        s=self.order[i];p=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(s['cell']+'.json'))
        return request_contract.build(self.config,s,p)
    def response(self,i,cost='0.001'):
        m=next(m for m in self.config['models'] if m['id']==self.order[i]['model'])
        return {'object':'chat.completion','id':f'synthetic-resume-{i}','provider':m['provider_candidate'],'model':m['allowed_response_models'][0],'usage':{'is_byok':False,'cost':float(cost),'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'cache_write_tokens':0},'completion_tokens_details':{'reasoning_tokens':0}},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'intentionally invalid JSON'}}]}
    def reserve(self,i=25):return self.ledger.reserve(self.order[i],self.request(i),test_only=True)
    def test_import_exactly_preserves_25(self):
        self.assertEqual(load(self.ledger.path)['attempts'],load(ORIGINAL/'ledger.json')['attempts'][:25]);r.check_prefix(self.directory,ORIGINAL,self.config)
        self.assertFalse((self.directory/'0025.request.json').exists())
    def test_only_failed_slot_can_be_next(self):
        with self.assertRaises(Halt):self.ledger.reserve(self.order[26],self.request(26),test_only=True)
        self.assertEqual(self.reserve(),25)
    def test_reserved_attempt_never_replayed(self):
        self.reserve()
        with self.assertRaises(Halt):self.reserve()
        self.assertEqual(len(load(self.ledger.path)['attempts']),26)
    def test_halted_response_never_replayed(self):
        self.reserve();response=self.response(25);response['provider']='wrong'
        with self.assertRaises(Halt):self.ledger.settle(25,response,self.catalogs[self.order[25]['target']])
        self.assertEqual(load(self.ledger.path)['attempts'][-1]['status'],'HALTED')
        with self.assertRaises(Halt):self.ledger.reserve(self.order[26],self.request(26),test_only=True)
    def test_original_unknown_hold_limits_budget(self):
        s=load(self.ledger.path)
        for x in s['attempts']:x['cost_usd']='0'
        s['attempts'][0]['cost_usd']='12.000001';save(self.ledger.path,s)
        with self.assertRaisesRegex(Halt,'including unknown'):self.reserve()
        s['attempts'][0]['cost_usd']='12';save(self.ledger.path,s);self.assertEqual(self.reserve(),25)
    def test_import_tampering_rejected(self):
        (self.directory/'0000.response.txt').write_text('changed')
        with self.assertRaises(Halt):r.check_prefix(self.directory,ORIGINAL,self.config)
    def test_complete_rejects_incomplete_inventory(self):
        with self.assertRaises(Halt):self.ledger.complete(allow_test=True,catalogs=self.catalogs)
    def test_invalid_library_still_falls_back(self):
        self.reserve();d=self.ledger.settle(25,self.response(25),self.catalogs[self.order[25]['target']]);self.assertEqual(d['status'],generation.INVALID);self.assertEqual(d['reasons'][0],'invalid_json')
    def test_duplicate_response_id_stops(self):
        self.reserve();response=self.response(25);response['id']=load(self.ledger.path)['attempts'][0]['response_id']
        with self.assertRaises(Halt):self.ledger.settle(25,response,self.catalogs[self.order[25]['target']])
    def test_http_error_evidence_redacts_key(self):
        key='sk-test-secret';body=json.dumps({'error':{'code':429,'message':'Bearer '+key+' busy','metadata':{'secret':key}}}).encode()
        e=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',429,'ignored',{'x-request-id':'safe'},io.BytesIO(body))
        result=r.error_evidence(e,key);self.assertEqual(result['http_status'],429);self.assertNotIn(key,json.dumps(result));self.assertEqual(result['provider_metadata'],{});self.assertIn('REDACTED',json.dumps(result))
    def test_non_json_http_error_still_records_status(self):
        e=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',502,'ignored',{},io.BytesIO(b'<bad gateway>'))
        result=r.error_evidence(e,'secret');self.assertEqual(result['http_status'],502);self.assertIn('error_body_parse_exception',result)
    def test_non_http_failure_records_type_only(self):
        result=r.error_evidence(RuntimeError('secret-key'),'secret-key');self.assertNotIn('secret-key',json.dumps(result));self.assertEqual(result['exception_type'],'RuntimeError')
    def authorization_fixture(self):
        lockfile=Path(self.tmp.name)/'freeze.json';authfile=Path(self.tmp.name)/'auth.json'
        code=BASE/'protocols/nonrsi_cross_model_resume_v2/runner.py'
        lock={'kind':'nonrsi_cross_model_second_replacement_freeze','files':{str(code.relative_to(ROOT)):freeze.file_hash(code)},'base_lock':str(BASE_LOCK.relative_to(ROOT)),'base_freeze_sha256':freeze.file_hash(BASE_LOCK),'previous_lock':'unused-in-mocked-test','previous_authorization':'unused-in-mocked-test','original_authorization':'unused-in-mocked-test','original_ledger':str(ORIGINAL.relative_to(ROOT)),'replacement_slot_id':self.order[25]['slot_id']}
        save(lockfile,lock)
        auth={'kind':'explicit_second_replacement_authorization','resume_freeze_sha256':freeze.file_hash(lockfile),'base_freeze_sha256':freeze.file_hash(BASE_LOCK),'model_calls':662,'model_spend_cap_usd':'15','unknown_charge_reservation_usd':'2','replacement_slot_id':self.order[25]['slot_id'],'allow_execution':True,'qwen_minimum_interval_seconds':60,'user_message_reference':'synthetic test only'}
        save(authfile,auth);return lockfile,authfile
    def test_resume_authorization_binds_exact_count_slot_cost_and_hold(self):
        lockfile,authfile=self.authorization_fixture();original=load(authfile)
        with patch.object(freeze,'authorize',return_value=self.config), patch.object(r.previous,'verify',return_value=(self.config,{})):
            r.verify(lockfile,authfile,None)
            for key,value in [('model_calls',663),('qwen_minimum_interval_seconds',59),('replacement_slot_id',self.order[26]['slot_id']),('model_spend_cap_usd','25'),('unknown_charge_reservation_usd','0'),('allow_execution',False),('user_message_reference','')]:
                save(authfile,{**original,key:value})
                with self.assertRaises(Halt):r.verify(lockfile,authfile,None)
    def test_resume_code_tamper_hash_rejected(self):
        lockfile,authfile=self.authorization_fixture();lock=load(lockfile);name=next(iter(lock['files']));lock['files'][name]='0'*64;save(lockfile,lock)
        with self.assertRaises(Halt):r.verify(lockfile,authfile,None)
    def test_unapproved_destination_stops_before_key_or_network(self):
        with patch.object(r,'verify',return_value=(self.config,{'resume_ledger':'approved-ledger-only'})):
            with patch.object(r.urllib.request,'build_opener',side_effect=AssertionError('network attempted')):
                with self.assertRaisesRegex(Halt,'destination'):r.run(None,None,None,self.directory)
    def test_transport_failure_stops_and_cannot_be_reissued(self):
        lock={'resume_ledger':str(self.directory),'original_ledger':str(ORIGINAL)}
        certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
        calls=[];clock=[0.0]
        def sleep(dt):clock[0]+=dt
        class Opener:
            def open(inner,request,timeout):
                calls.append(request.data)
                raise urllib.error.HTTPError(request.full_url,503,'ignored',{},io.BytesIO(b'{"error":{"code":503,"message":"temporary unavailable"}}'))
        with patch.object(r.time,'monotonic',side_effect=lambda:clock[0]), patch.object(r.time,'sleep',side_effect=sleep), patch.object(r,'verify',return_value=(self.config,lock)), patch.dict(r.os.environ,{'OPENROUTER_API_KEY':'synthetic-secret'}), patch.object(r.urllib.request,'build_opener',return_value=Opener()), patch.object(inputs,'certify',side_effect=list(certs.values())*2):
            with self.assertRaisesRegex(Halt,'no additional replacement'):r.run(None,None,None,self.directory)
            self.assertEqual(len(calls),1)
            evidence=load(self.directory/'0025.interruption.json');self.assertEqual(evidence['http_status'],503)
            with self.assertRaisesRegex(Halt,'Unresolved attempt'):r.run(None,None,None,self.directory)
            self.assertEqual(len(calls),1)
        self.assertEqual(load(self.ledger.path)['attempts'][-1]['status'],'RESERVED')
    def test_first_qwen_wait_and_chunk_bound(self):
        clock=[0.0];sleeps=[]
        def sleep(dt):sleeps.append(dt);clock[0]+=dt
        throttle=r.QwenThrottle(clock=lambda:clock[0],sleeper=sleep)
        throttle.wait({'model':'qwen3_instruct_2507'})
        self.assertEqual(clock[0],60);self.assertEqual(sleeps,[30,30])
    def test_qwen_cooldown_after_finish_and_other_models_unaffected(self):
        clock=[0.0]
        def sleep(dt):clock[0]+=dt
        throttle=r.QwenThrottle(clock=lambda:clock[0],sleeper=sleep)
        throttle.wait({'model':'haiku45_20251001'});self.assertEqual(clock[0],0)
        throttle.wait({'model':'qwen3_instruct_2507'});clock[0]+=20
        throttle.finished({'model':'qwen3_instruct_2507'})
        throttle.wait({'model':'gpt6luna_public_id'});self.assertEqual(clock[0],80)
        throttle.wait({'model':'qwen3_instruct_2507'});self.assertEqual(clock[0],140)
    def test_restart_still_waits_full_interval(self):
        clock=[500.0]
        def sleep(dt):clock[0]+=dt
        r.QwenThrottle(clock=lambda:clock[0],sleeper=sleep).wait({'model':'qwen3_instruct_2507'})
        self.assertEqual(clock[0],560)
    def test_request_timing_proves_total_attempt_mapping(self):
        throttle=r.QwenThrottle(clock=lambda:123.0,sleeper=lambda _:None)
        throttle.started(self.directory,25,self.order[25]);v=load(self.directory/'0025.request_timing.json')
        self.assertEqual(v['total_attempt_index'],27);self.assertEqual(v['minimum_qwen_interval_seconds'],60)
    def test_retry_header_metadata_allowlist_and_redaction(self):
        key='sk-synthetic-secret';body=json.dumps({'error':{'code':429,'message':'limited','metadata':{'provider_name':'Parasail','limit_source':'upstream_provider_shared_pool','remedy_hint':'Bearer '+key,'raw':'excluded','authorization':key}}}).encode()
        e=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',429,'ignored',{'retry-after':'120','x-request-id':key},io.BytesIO(body))
        d=r.error_evidence(e,key);self.assertEqual(d['headers']['retry-after'],'120');self.assertEqual(d['provider_metadata']['provider_name'],'Parasail')
        self.assertNotIn(key,json.dumps(d));self.assertNotIn('raw',d['provider_metadata']);self.assertNotIn('authorization',d['provider_metadata'])
    def test_non_json_error_headers_still_redacted(self):
        e=urllib.error.HTTPError('https://openrouter.ai/api/v1/chat/completions',502,'ignored',{'x-request-id':'sk-hidden'},io.BytesIO(b'bad gateway'))
        self.assertNotIn('sk-hidden',json.dumps(r.error_evidence(e,'sk-hidden')))
    def test_total_662_attempts_full_inventory_and_production_test_guard(self):
        for i in range(25,660):
            self.reserve(i);response=self.response(i)
            (self.directory/f'{i:04d}.response.txt').write_text(json.dumps(response))
            self.ledger.settle(i,response,self.catalogs[self.order[i]['target']])
        a=self.ledger.complete(allow_test=True,catalogs=self.catalogs)
        self.assertEqual(len(a),660);r.check_prefix(self.directory,ORIGINAL,self.config)
        r.write_costs(self.directory,a);cost=load(self.directory/'TOTAL_ACCOUNTING.json')
        self.assertEqual(cost['total_http_attempts'],662);self.assertEqual(cost['unknown_charge_reservation_usd'],'2');self.assertEqual(cost['conservative_accounted_usd'],'2.73836070')
        with self.assertRaises(Halt):self.reserve()
        with self.assertRaises(Halt):self.ledger.complete()
        with patch('protocols.nonrsi_cross_model_v1.generation.Ledger.complete',return_value=[dict(x,cost_usd='1') for x in a]):
            with self.assertRaises(Halt):self.ledger.complete()

if __name__=='__main__':
    with offline():unittest.main()
