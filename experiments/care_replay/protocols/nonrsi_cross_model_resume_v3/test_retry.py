import unittest,tempfile,io,copy,json
from unittest.mock import patch
from . import runner as r
from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze,generation,inputs,request_contract
from datetime import datetime,timezone,timedelta
from email.utils import format_datetime
ORIGINAL=BASE/'outputs/nonrsi_cross_model_resume_v2/ledger'
BASE_LOCK=BASE/'outputs/nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json'

def evidence():return {'exception_type':'HTTPError','http_status':429,'error':{'code':429,'message':'Provider returned error'},'provider_metadata':{'provider_name':'Parasail','limit_source':'upstream_provider_shared_pool'},'headers':{}}

class RetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=freeze.verify(BASE_LOCK);cls.order=slots(cls.config)
        cls.catalogs={x['target']:inputs.catalog(inputs.adapter(x['target'])) for x in cls.order}
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.directory=Path(self.tmp.name)/'journal';self.j=r.Journal(self.directory,ORIGINAL,self.config,production_authorized=True)
    def tearDown(self):self.tmp.cleanup()
    def reserve(self,wait=None):
        p=self.j.plan();body=request_contract.build(self.config,p['slot'],load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(p['slot']['cell']+'.json')))
        return self.j.reserve(p,body,{'waited_seconds':p['required_backoff_seconds'] if wait is None else wait},test_only=True)
    def response(self,n):
        slot=self.j.state()['attempts'][n]['slot'];m=self.j.models[slot['model']]
        return {'object':'chat.completion','id':f'synthetic-{n}','provider':m['provider_candidate'],'model':m['allowed_response_models'][0],'usage':{'is_byok':False,'cost':0.001,'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'cache_write_tokens':0},'completion_tokens_details':{'reasoning_tokens':0}},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'invalid JSON for deterministic fallback'}}]}
    def settle(self,n):
        raw=self.response(n);(self.directory/f'{n:04d}.response.txt').write_text(json.dumps(raw));self.j.settle(n,raw,self.catalogs[self.j.state()['attempts'][n]['slot']['target']])
    def succeed(self):n=self.reserve();self.settle(n);return n
    def fail(self):n=self.reserve();self.assertTrue(self.j.fail(n,evidence()));return n
    def test_exact_47_result_import(self):
        self.assertEqual(load(self.j.science/'ledger.json')['attempts'],load(ORIGINAL/'ledger.json')['attempts'][:47]);self.assertEqual(self.j.plan()['logical_index'],47);self.assertTrue(self.j.plan()['is_replacement']);self.assertEqual(self.j.plan()['total_attempt_index'],50)
    def test_only_exact_error_qualifies(self):
        slot=self.order[47];self.assertTrue(r.qualifying(slot,evidence()))
        variants=[]
        for key,value in [('http_status',503),('exception_type','TimeoutError')]:x=evidence();x[key]=value;variants.append(x)
        for key,value in [('provider_name','Other'),('limit_source','unknown')]:x=evidence();x['provider_metadata'][key]=value;variants.append(x)
        x=evidence();x.pop('provider_metadata');variants.append(x)
        for x in variants:self.assertFalse(r.qualifying(slot,x))
        self.assertFalse(r.qualifying(dict(slot,model='gpt6luna_public_id'),evidence()))
    def test_per_slot_two_replacements_then_stop(self):
        self.assertEqual(self.j.plan()['required_backoff_seconds'],60);self.fail()
        self.assertEqual(self.j.plan()['required_backoff_seconds'],180);self.assertEqual(self.j.plan()['slot_replacement_ordinal'],2);self.fail()
        with self.assertRaisesRegex(Halt,'Per-slot'):self.j.plan()
        self.assertEqual(len(self.j.state()['attempts']),2)
    def test_global_six_replacements_limit(self):
        self.succeed()
        while sum(a['is_replacement'] for a in self.j.state()['attempts'])<6:
            p=self.j.plan()
            if p['slot']['model']=='qwen3_instruct_2507':self.fail()
            self.succeed()
        while self.j.plan()['slot']['model']!='qwen3_instruct_2507':self.succeed()
        self.fail()
        with self.assertRaisesRegex(Halt,'Global'):self.j.plan()
    def test_unknown_charges_accumulate(self):
        self.fail();known,holds=self.j.totals(self.j.state()['attempts']);self.assertEqual(holds,r.Decimal('4'));self.assertEqual(known,r.Decimal('0.19575682'))
    def test_budget_before_reservation(self):
        self.j.base_cost=r.Decimal('11.000001')
        with self.assertRaisesRegex(Halt,'Budget'):self.j.plan()
        self.j.base_cost=r.Decimal('11');self.fail()
        with self.assertRaisesRegex(Halt,'Budget'):self.j.plan()
    def test_total_http_cap_prevents_post(self):
        with patch.object(r,'MAX_ATTEMPTS',50):
            with self.assertRaisesRegex(Halt,'Total HTTP'):self.j.plan()
    def test_insufficient_wait_rejected(self):
        with self.assertRaisesRegex(Halt,'Backoff'):self.reserve(wait=59.9)
        self.assertEqual(self.j.state()['attempts'],[])
    def test_unknown_interrupted_reservation_cannot_retry(self):
        self.reserve();other=r.Journal(self.directory,ORIGINAL,self.config)
        with self.assertRaisesRegex(Halt,'Unresolved'):other.plan()
    def test_nonqualifying_http_error_halts(self):
        n=self.reserve();e=evidence();e['http_status']=503;self.assertFalse(self.j.fail(n,e))
        with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
    def test_unknown_cost_response_halts(self):
        n=self.reserve();raw=self.response(n);raw['usage'].pop('cost')
        with self.assertRaises(Halt):self.j.settle(n,raw,self.catalogs[self.order[47]['target']])
        with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
    def test_duplicate_response_halts(self):
        n=self.reserve();raw=self.response(n);raw['id']=self.j.prefix[0]['response_id']
        with self.assertRaisesRegex(Halt,'Duplicate'):self.j.settle(n,raw,self.catalogs[self.order[47]['target']])
    def test_invalid_library_is_never_resampled(self):
        n=self.succeed();self.assertEqual(self.j.state()['attempts'][n]['status'],generation.INVALID);self.assertEqual(self.j.plan()['logical_index'],48);self.assertFalse(self.j.plan()['is_replacement'])
        with self.assertRaises(Halt):self.j.fail(n,evidence())
    def test_projection_recovery_after_journal_commit(self):
        original=self.j.sync;self.j.sync=lambda:None;n=self.succeed();self.assertEqual(len(load(self.j.science/'ledger.json')['attempts']),47)
        self.j.sync=original;other=r.Journal(self.directory,ORIGINAL,self.config);self.assertEqual(len(load(other.science/'ledger.json')['attempts']),48);self.assertEqual(other.plan()['logical_index'],48)
    def test_imported_evidence_tampering_rejected(self):
        (self.j.science/'0000.response.txt').write_text('tampered')
        with self.assertRaises(Halt):self.j.sync()
    def test_retry_header_numeric_date_and_invalid(self):
        now=datetime(2026,10,21,tzinfo=timezone.utc);e=evidence();e['headers']['retry-after']='240';self.assertEqual(r.retry_hint(e,now),240)
        e['headers']['retry-after']=format_datetime(now+timedelta(seconds=300));self.assertEqual(r.retry_hint(e,now),300)
        for v in ['nonsense','-1','NaN','Infinity']:
            e['headers']['retry-after']=v;self.assertEqual(r.retry_hint(e,now),0)
    def test_longer_retry_after_overrides_backoff(self):
        n=self.reserve();e=evidence();e['headers']['retry-after']='500';self.j.fail(n,e);self.assertEqual(self.j.plan()['required_backoff_seconds'],500)
    def test_scheduler_cooldown_and_bounded_wait_chunks(self):
        clock=[0.0];sleeps=[]
        def sleep(dt):sleeps.append(dt);clock[0]+=dt
        s=r.Scheduler(clock=lambda:clock[0],sleeper=sleep);plan=self.j.plan();record=s.wait(plan);self.assertEqual(record['waited_seconds'],60)
        clock[0]+=10;s.finished(plan['slot']);s.wait({**plan,'required_backoff_seconds':180});self.assertEqual(clock[0],250)
        self.assertTrue(all(v<=30 for v in sleeps))
        start=clock[0];s.wait({'slot':{'model':'haiku45_20251001'},'required_backoff_seconds':0});self.assertEqual(clock[0],start)
    def test_synthetic_data_rejected_from_production(self):
        self.succeed()
        with self.assertRaisesRegex(Halt,'Synthetic'):self.j.audit()
    def test_frozen_body_required(self):
        p=self.j.plan()
        with self.assertRaisesRegex(Halt,'Request differs'):self.j.reserve(p,{'model':'wrong'},{'waited_seconds':60},test_only=True)
    def test_retry_plan_fields_tamper_rejected(self):
        self.succeed();s=self.j.state();s['attempts'][0]['required_backoff_seconds']=0;save(self.j.path,s)
        with self.assertRaisesRegex(Halt,'backoff'):self.j.plan()
    def test_full_660_slots_668_attempts_projection_and_cost(self):
        failures=set(x['slot_id'] for x in [v for v in self.order[48:] if v['model']=='qwen3_instruct_2507'][:5]);failed=set()
        while True:
            plan=self.j.plan()
            if plan is None:break
            sid=plan['slot']['slot_id']
            if sid in failures and sid not in failed:self.fail();failed.add(sid)
            else:self.succeed()
        rows=self.j.audit(allow_test=True,catalogs=self.catalogs,require_complete=True)
        totals=load(self.directory/'TOTAL_ACCOUNTING.json');self.assertEqual(len(rows),660);self.assertEqual(totals['total_http_attempts'],668);self.assertEqual(totals['new_replacement_attempts'],6);self.assertEqual(totals['unknown_charge_holds_usd'],'8');self.assertEqual(totals['conservative_accounted_usd'],'8.80875682')
        self.assertEqual(rows[:47],self.j.prefix)
        self.assertIsNone(self.j.plan())
    def test_transport_retries_only_twice_and_preserves_current_slot(self):
        clock=[0.0];calls=[];certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
        def sleep(dt):clock[0]+=dt
        class Opener:
            def open(inner,req,timeout):
                calls.append((clock[0],json.loads(req.data)))
                body=json.dumps({'error':{'code':429,'message':'Provider returned error','metadata':evidence()['provider_metadata']}}).encode()
                raise r.urllib.error.HTTPError(req.full_url,429,'ignored',{},io.BytesIO(body))
        lock={'journal_directory':str(self.directory),'original_ledger':str(ORIGINAL)}
        with patch.object(r,'verify',return_value=(self.config,lock)),patch.dict(r.os.environ,{'OPENROUTER_API_KEY':'synthetic-secret'}),patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),patch.object(r.time,'sleep',side_effect=sleep),patch.object(r.urllib.request,'build_opener',return_value=Opener()),patch.object(inputs,'certify',side_effect=list(certs.values())):
            with self.assertRaisesRegex(Halt,'Per-slot'):r.run(None,None,None,self.directory)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0][1],calls[1][1]);self.assertGreaterEqual(calls[0][0],60);self.assertGreaterEqual(calls[1][0]-calls[0][0],180)
    def test_unapproved_path_stops_before_network(self):
        with patch.object(r,'verify',return_value=(self.config,{'journal_directory':'unapproved'})):
            with self.assertRaisesRegex(Halt,'destination'):r.run(None,None,None,self.directory)
    def test_authorization_limits_and_scope_are_bound(self):
        code=BASE/'protocols/nonrsi_cross_model_resume_v3/runner.py';lockfile=Path(self.tmp.name)/'freeze.json';authfile=Path(self.tmp.name)/'auth.json'
        lock={'kind':'nonrsi_cross_model_bounded_429_freeze','files':{str(code.relative_to(ROOT)):freeze.file_hash(code)},'base_lock':str(BASE_LOCK.relative_to(ROOT)),'original_authorization':'mock','previous_lock':'mock','previous_authorization':'mock','original_ledger':str(ORIGINAL.relative_to(ROOT))};save(lockfile,lock)
        auth={'kind':'explicit_bounded_429_authorization','resume_freeze_sha256':freeze.file_hash(lockfile),'base_freeze_sha256':freeze.file_hash(BASE_LOCK),'allow_execution':True,'model_calls':668,'model_spend_cap_usd':'15','historical_unknown_charge_hold_usd':'3','additional_replacements_cap':6,'per_slot_replacements_cap':2,'backoff_seconds':[60,180],'qwen_minimum_interval_seconds':60,'only_qwen_parasail_shared_pool_429':True,'honor_longer_valid_retry_after':True,'user_message_reference':'test only'};save(authfile,auth)
        with patch.object(freeze,'authorize',return_value=self.config),patch.object(r.previous,'verify',return_value=(self.config,{})):
            r.verify(lockfile,authfile,None)
            for k,v in [('model_calls',669),('model_spend_cap_usd','16'),('additional_replacements_cap',7),('per_slot_replacements_cap',3),('backoff_seconds',[0,0]),('historical_unknown_charge_hold_usd','0'),('only_qwen_parasail_shared_pool_429',False),('allow_execution',False)]:
                save(authfile,{**auth,k:v})
                with self.assertRaises(Halt):r.verify(lockfile,authfile,None)

if __name__=='__main__':
    with offline():unittest.main()
