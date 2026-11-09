import unittest,tempfile,json,copy,io
from unittest.mock import patch
from . import runner as r
from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze,inputs,generation,request_contract
ORIGINAL=BASE/'outputs/nonrsi_cross_model_resume_v2/ledger'
BASE_LOCK=BASE/'outputs/nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json'
def network():return {'exception_type':'URLError','transport_phase':'open_or_read','response_received':False,'unknown_charge_reservation_retained':True}
def shared():return {'exception_type':'HTTPError','http_status':429,'error':{'code':429},'provider_metadata':{'provider_name':'Parasail','limit_source':'upstream_provider_shared_pool'}}
class TransportTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.config=freeze.verify(BASE_LOCK);cls.order=slots(cls.config);cls.catalogs={t:inputs.catalog(inputs.adapter(t)) for t in {x['target'] for x in cls.order}}
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.directory=Path(self.tmp.name)/'journal';self.j=r.Journal(self.directory,ORIGINAL,self.config,production_authorized=True)
 def tearDown(self):self.tmp.cleanup()
 def reserve(self,wait=None):
  p=self.j.plan();body=request_contract.build(self.config,p['slot'],load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(p['slot']['cell']+'.json')))
  return self.j.reserve(p,body,{'waited_seconds':p['required_backoff_seconds'] if wait is None else wait},test_only=True)
 def response(self,n):
  a=self.j.state()['attempts'][n];m=self.j.models[a['slot']['model']]
  return {'object':'chat.completion','id':f'v4-synthetic-{n}','provider':m['provider_candidate'],'model':m['allowed_response_models'][0],'usage':{'is_byok':False,'cost':0.001,'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'cache_write_tokens':0},'completion_tokens_details':{'reasoning_tokens':0}},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'invalid JSON for fixed fallback'}}]}
 def succeed(self):
  n=self.reserve();raw=self.response(n);(self.directory/f'{n:04d}.response.txt').write_text(json.dumps(raw));self.j.settle(n,raw,self.catalogs[self.j.state()['attempts'][n]['slot']['target']]);return n
 def fail(self):
  n=self.reserve();self.assertTrue(self.j.fail(n,network()));return n
 def test_import_and_current_plan(self):
  self.assertEqual(self.j.state()['attempts'],load(r.SOURCE_JOURNAL/'ATTEMPTS.json')['attempts']);self.assertEqual(load(self.j.science/'ledger.json'),load(r.SOURCE_JOURNAL/'science_ledger/ledger.json'))
  p=self.j.plan();self.assertEqual((p['logical_index'],p['total_attempt_index'],p['slot_replacement_ordinal'],p['required_backoff_seconds']),(242,248,1,60));self.assertTrue(p['is_replacement'])
  self.assertEqual(self.j.totals(self.j.state()['attempts']),(r.Decimal('1.00369190'),r.Decimal('6')))
 def test_qualifiers_and_phase(self):
  for model in ['gpt6luna_public_id','haiku45_20251001','qwen3_instruct_2507']:
   for kind in r.TRANSPORT_EXCEPTIONS:self.assertTrue(r.qualifying({'model':model},{**network(),'exception_type':kind}))
  for e in [{'exception_type':'URLError'},dict(network(),transport_phase='response_received'),dict(network(),response_received=True),dict(network(),http_status=503),dict(network(),exception_type='ValueError'),dict(network(),exception_type='HTTPError')]:self.assertFalse(r.qualifying(self.order[242],e))
  self.assertTrue(r.qualifying(self.order[242],shared()));self.assertFalse(r.qualifying({'model':'gpt6luna_public_id'},shared()))
  self.assertFalse(r.qualifying(self.order[242],{**shared(),'provider_metadata':{}}))
 def test_imported_stop_is_one_specific_failure(self):
  rows=self.j.state()['attempts'];a=rows[197];self.assertTrue(r.permitted_imported_stop(197,a,rows))
  for index,item in [(196,a),(197,dict(a,logical_index=243)),(197,dict(a,request_sha256='other'))]:self.assertFalse(r.permitted_imported_stop(index,item,rows))
 def test_imported_prefix_cannot_change(self):
  s=self.j.state();s['attempts'][-1]['status']='RETRYABLE_TRANSPORT';save(self.j.path,s)
  with self.assertRaisesRegex(Halt,'prefix'):self.j.plan()
 def test_invalid_library_advances_without_retry(self):
  n=self.succeed();self.assertEqual(n,198);self.assertEqual(self.j.state()['attempts'][n]['status'],generation.INVALID);self.assertEqual(self.j.plan()['logical_index'],243);self.assertFalse(self.j.plan()['is_replacement'])
  with self.assertRaises(Halt):self.j.fail(n,network())
 def test_per_slot_remaining_two(self):
  self.fail();self.assertEqual(self.j.plan()['required_backoff_seconds'],180);self.fail()
  with self.assertRaisesRegex(Halt,'Per-slot'):self.j.plan()
  self.assertEqual(self.j.totals(self.j.state()['attempts'])[1],8)
 def test_global_remaining_three(self):
  self.succeed();self.fail();self.succeed();self.fail();self.succeed();self.fail()
  with self.assertRaisesRegex(Halt,'Global'):self.j.plan()
  self.assertEqual(sum(a['is_replacement'] for a in self.j.state()['attempts']),6)
 def test_total_cap_before_request(self):
  with patch.object(r,'MAX_ATTEMPTS',248):
   with self.assertRaisesRegex(Halt,'Total HTTP'):self.j.plan()
 def test_budget_accounts_all_six_holds(self):
  self.j.base_cost+=r.Decimal('7.1')
  with self.assertRaisesRegex(Halt,'Budget'):self.j.plan()
 def test_wait_required_and_larger_hint(self):
  with self.assertRaisesRegex(Halt,'Backoff'):self.reserve(wait=59)
  n=self.reserve();e=shared();e['headers']={'retry-after':'300'};self.j.fail(n,e);self.assertEqual(self.j.plan()['required_backoff_seconds'],300)
 def test_unknown_reservation_stops(self):
  self.reserve();other=r.Journal(self.directory,ORIGINAL,self.config)
  with self.assertRaisesRegex(Halt,'Unresolved'):other.plan()
 def test_envelope_and_cost_errors_never_retry(self):
  n=self.reserve();raw=self.response(n);raw['usage'].pop('cost')
  with self.assertRaises(Halt):self.j.settle(n,raw,self.catalogs[self.order[242]['target']])
  self.assertFalse(self.j.fail(n,network()))
  with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
 def test_nonqualifying_http_stops(self):
  n=self.reserve();self.assertFalse(self.j.fail(n,dict(shared(),http_status=503)))
  with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
 def test_duplicate_response_id_stops(self):
  n=self.reserve();raw=self.response(n);raw['id']=self.j.prefix[0]['response_id']
  with self.assertRaisesRegex(Halt,'Duplicate'):self.j.settle(n,raw,self.catalogs[self.order[242]['target']])
 def test_projection_recovery_without_resend(self):
  original=self.j.sync;self.j.sync=lambda:None;self.succeed();self.assertEqual(len(load(self.j.science/'ledger.json')['attempts']),242);self.j.sync=original
  other=r.Journal(self.directory,ORIGINAL,self.config);self.assertEqual(len(load(other.science/'ledger.json')['attempts']),243);self.assertEqual(other.plan()['logical_index'],243)
 def test_imported_raw_tampering_stops(self):
  (self.directory/'0000.request.json').write_text('{}')
  with self.assertRaisesRegex(Halt,'Derived evidence changed|Recorded request'):self.j.audit()
 def test_full_660_at_668_calls_and_original_prefix(self):
  failed=set()
  while True:
   p=self.j.plan()
   if p is None:break
   i=p['logical_index']
   if i in {243,244} and i not in failed:self.fail();failed.add(i)
   else:self.succeed()
  rows=self.j.audit(allow_test=True,catalogs=self.catalogs,require_complete=True);cost=load(self.directory/'TOTAL_ACCOUNTING.json')
  self.assertEqual(len(rows),660);self.assertEqual(rows[:242],load(r.SOURCE_JOURNAL/'science_ledger/ledger.json')['attempts']);self.assertEqual(cost['total_http_attempts'],668);self.assertEqual(cost['new_replacement_attempts'],6);self.assertEqual(cost['unknown_charge_holds_usd'],'8');self.assertEqual(cost['known_model_cost_usd'],'1.42169190');self.assertEqual(cost['conservative_accounted_usd'],'9.42169190')
 def test_real_transport_boundary_and_finite_retry(self):
  clock=[0.0];calls=[];certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
  def sleep(dt):clock[0]+=dt
  class Opener:
   def open(inner,req,timeout):calls.append((clock[0],json.loads(req.data)));raise r.urllib.error.URLError('synthetic connection failure')
  lock={'journal_directory':str(self.directory),'original_ledger':str(ORIGINAL)}
  with patch.object(r,'verify',return_value=(self.config,lock)),patch.dict(r.os.environ,{'OPENROUTER_API_KEY':'test-not-real'}),patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),patch.object(r.time,'sleep',side_effect=sleep),patch.object(r.urllib.request,'build_opener',return_value=Opener()),patch.object(inputs,'certify',side_effect=list(certs.values())):
   with self.assertRaisesRegex(Halt,'Per-slot'):r.run(None,None,None,self.directory)
  self.assertEqual(len(calls),2);self.assertEqual(calls[0][1],calls[1][1]);self.assertGreaterEqual(calls[0][0],60);self.assertGreaterEqual(calls[1][0]-calls[0][0],180)
 def test_frozen_request_guard(self):
  with self.assertRaisesRegex(Halt,'Request differs'):self.j.reserve(self.j.plan(),{'model':'changed'},{'waited_seconds':60},test_only=True)
 def test_synthetic_cannot_enter_real_replay(self):
  self.succeed()
  with self.assertRaisesRegex(Halt,'Synthetic'):self.j.audit()
 def test_current_production_inventory_audit(self):
  report=self.j.audit();self.assertEqual(report['logical_slots'],242);self.assertEqual(report['total_http_attempts'],248);self.assertEqual(report['unknown_charge_holds_usd'],'6')
if __name__=='__main__':
 with offline():unittest.main()
