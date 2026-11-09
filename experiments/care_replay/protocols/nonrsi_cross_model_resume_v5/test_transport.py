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
  return {'object':'chat.completion','id':f'v5-synthetic-{n}','provider':m['provider_candidate'],'model':m['allowed_response_models'][0],'usage':{'is_byok':False,'cost':0.001,'prompt_tokens':1,'completion_tokens':1,'prompt_tokens_details':{'cache_write_tokens':0},'completion_tokens_details':{'reasoning_tokens':0}},'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'invalid JSON for fixed fallback'}}]}
 def succeed(self):
  n=self.reserve();raw=self.response(n);(self.directory/f'{n:04d}.response.txt').write_text(json.dumps(raw));self.j.settle(n,raw,self.catalogs[self.j.state()['attempts'][n]['slot']['target']]);return n
 def fail(self):
  n=self.reserve();self.assertTrue(self.j.fail(n,network()));return n
 def test_import_and_current_plan(self):
  self.assertEqual(self.j.state()['attempts'],load(r.SOURCE_JOURNAL/'ATTEMPTS.json')['attempts']);self.assertEqual(load(self.j.science/'ledger.json'),load(r.SOURCE_JOURNAL/'science_ledger/ledger.json'))
  p=self.j.plan();self.assertEqual((p['logical_index'],p['total_attempt_index'],p['slot_replacement_ordinal'],p['required_backoff_seconds']),(314,323,1,180));self.assertTrue(p['is_replacement'])
  self.assertEqual(self.j.totals(self.j.state()['attempts']),(r.Decimal('1.31418116'),r.Decimal('9')))
 def test_qualifiers_and_phase(self):
  for model in ['gpt6luna_public_id','haiku45_20251001','qwen3_instruct_2507']:
   for kind in r.TRANSPORT_EXCEPTIONS:self.assertTrue(r.qualifying({'model':model},{**network(),'exception_type':kind}))
  for e in [{'exception_type':'URLError'},dict(network(),transport_phase='response_received'),dict(network(),response_received=True),dict(network(),http_status=503),dict(network(),exception_type='ValueError'),dict(network(),exception_type='HTTPError')]:self.assertFalse(r.qualifying(self.order[314],e))
  self.assertTrue(r.qualifying(self.order[314],shared()));self.assertFalse(r.qualifying({'model':'gpt6luna_public_id'},shared()))
  self.assertFalse(r.qualifying(self.order[314],{**shared(),'provider_metadata':{}}))
 def test_imported_stop_is_one_specific_failure(self):
  rows=self.j.state()['attempts'];a=rows[197];self.assertTrue(r.permitted_imported_stop(197,a,rows))
  for index,item in [(196,a),(197,dict(a,logical_index=243)),(197,dict(a,request_sha256='other'))]:self.assertFalse(r.permitted_imported_stop(index,item,rows))
 def test_imported_prefix_cannot_change(self):
  s=self.j.state();s['attempts'][-1]['status']='RETRYABLE_TRANSPORT';save(self.j.path,s)
  with self.assertRaisesRegex(Halt,'prefix'):self.j.plan()
 def test_invalid_library_advances_without_retry(self):
  n=self.succeed();self.assertEqual(n,273);self.assertEqual(self.j.state()['attempts'][n]['status'],generation.INVALID);self.assertEqual(self.j.plan()['logical_index'],315);self.assertFalse(self.j.plan()['is_replacement'])
  with self.assertRaises(Halt):self.j.fail(n,network())
 def test_per_slot_remaining_two(self):
  self.fail();self.assertEqual(self.j.plan()['required_backoff_seconds'],600);self.fail()
  with self.assertRaisesRegex(Halt,'Per-slot'):self.j.plan()
  self.assertEqual(self.j.totals(self.j.state()['attempts'])[1],11)
 def test_global_remaining_seven_independent_of_cost_cap(self):
  # High synthetic CAP isolates the call/replacement bound; production remains $15.
  with patch.object(r,'CAP',r.Decimal('100')):
   self.succeed()
   for _ in range(6):self.fail();self.succeed()
   self.fail()
   with self.assertRaisesRegex(Halt,'Global'):self.j.plan()
   self.assertEqual(sum(a['is_replacement'] for a in self.j.state()['attempts']),13)
 def test_total_cap_before_request(self):
  with patch.object(r,'MAX_ATTEMPTS',323):
   with self.assertRaisesRegex(Halt,'Total HTTP'):self.j.plan()
 def test_budget_accounts_all_nine_holds(self):
  self.j.base_cost+=r.Decimal('3.7')
  with self.assertRaisesRegex(Halt,'Budget'):self.j.plan()
 def test_wait_required_and_larger_hint(self):
  with self.assertRaisesRegex(Halt,'Backoff'):self.reserve(wait=179)
  n=self.reserve();e=shared();e['headers']={'retry-after':'900'};self.j.fail(n,e);self.assertEqual(self.j.plan()['required_backoff_seconds'],900)
 def test_unknown_reservation_stops(self):
  self.reserve();other=r.Journal(self.directory,ORIGINAL,self.config)
  with self.assertRaisesRegex(Halt,'Unresolved'):other.plan()
 def test_envelope_and_cost_errors_never_retry(self):
  n=self.reserve();raw=self.response(n);raw['usage'].pop('cost')
  with self.assertRaises(Halt):self.j.settle(n,raw,self.catalogs[self.order[314]['target']])
  self.assertFalse(self.j.fail(n,network()))
  with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
 def test_nonqualifying_http_stops(self):
  n=self.reserve();self.assertFalse(self.j.fail(n,dict(shared(),http_status=503)))
  with self.assertRaisesRegex(Halt,'Unresolved'):self.j.plan()
 def test_duplicate_response_id_stops(self):
  n=self.reserve();raw=self.response(n);raw['id']=self.j.prefix[0]['response_id']
  with self.assertRaisesRegex(Halt,'Duplicate'):self.j.settle(n,raw,self.catalogs[self.order[314]['target']])
 def test_projection_recovery_without_resend(self):
  original=self.j.sync;self.j.sync=lambda:None;self.succeed();self.assertEqual(len(load(self.j.science/'ledger.json')['attempts']),314);self.j.sync=original
  other=r.Journal(self.directory,ORIGINAL,self.config);self.assertEqual(len(load(other.science/'ledger.json')['attempts']),315);self.assertEqual(other.plan()['logical_index'],315)
 def test_imported_raw_tampering_stops(self):
  (self.directory/'0000.request.json').write_text('{}')
  with self.assertRaisesRegex(Halt,'Derived evidence changed|Recorded request'):self.j.audit()
 def test_full_660_at_669_calls_without_more_failures(self):
  while self.j.plan() is not None:self.succeed()
  rows=self.j.audit(allow_test=True,catalogs=self.catalogs,require_complete=True);cost=load(self.directory/'TOTAL_ACCOUNTING.json')
  self.assertEqual(len(rows),660);self.assertEqual(rows[:314],load(r.SOURCE_JOURNAL/'science_ledger/ledger.json')['attempts']);self.assertEqual(cost['total_http_attempts'],669);self.assertEqual(cost['new_replacement_attempts'],7);self.assertEqual(cost['unknown_charge_holds_usd'],'9');self.assertEqual(cost['known_model_cost_usd'],'1.66018116');self.assertEqual(cost['conservative_accounted_usd'],'10.66018116')
 def test_budget_stops_before_all_replacement_allowance_used(self):
  self.succeed()
  for _ in range(3):self.fail();self.succeed()
  self.fail()
  with self.assertRaisesRegex(Halt,'Budget'):self.j.plan()
  known,holds=self.j.totals(self.j.state()['attempts']);self.assertEqual(holds,13);self.assertLess(known+holds,15);self.assertGreater(known+holds+1,15)
  self.assertLess(sum(a['is_replacement'] for a in self.j.state()['attempts']),13)
 def test_new_backoff_cannot_revert_to_old_schedule(self):
  self.succeed();s=self.j.state();s['attempts'][-1]['required_backoff_seconds']=60;save(self.j.path,s)
  with self.assertRaisesRegex(Halt,'backoff'):self.j.plan()
 def test_scheduler_new_cooldown_and_wait_chunks(self):
  clock=[0.0];chunks=[]
  def sleep(dt):chunks.append(dt);clock[0]+=dt
  scheduler=r.Scheduler(clock=lambda:clock[0],sleeper=sleep)
  p=self.j.plan();self.assertEqual(scheduler.wait(p)['waited_seconds'],180)
  scheduler.finished(p['slot']);self.assertEqual(scheduler.wait({**p,'required_backoff_seconds':600})['waited_seconds'],600)
  scheduler.finished(p['slot']);self.assertEqual(scheduler.wait({**p,'required_backoff_seconds':0})['waited_seconds'],180)
  self.assertTrue(all(dt<=30 for dt in chunks))
 def test_real_transport_boundary_and_finite_retry(self):
  clock=[0.0];calls=[];certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
  def sleep(dt):clock[0]+=dt
  class Opener:
   def open(inner,req,timeout):calls.append((clock[0],json.loads(req.data)));raise r.urllib.error.URLError('synthetic connection failure')
  lock={'journal_directory':str(self.directory),'original_ledger':str(ORIGINAL)}
  with patch.object(r,'verify',return_value=(self.config,lock)),patch.dict(r.os.environ,{'OPENROUTER_API_KEY':'test-not-real'}),patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),patch.object(r.time,'sleep',side_effect=sleep),patch.object(r.urllib.request,'build_opener',return_value=Opener()),patch.object(inputs,'certify',side_effect=list(certs.values())):
   with self.assertRaisesRegex(Halt,'Per-slot'):r.run(None,None,None,self.directory)
  self.assertEqual(len(calls),2);self.assertEqual(calls[0][1],calls[1][1]);self.assertGreaterEqual(calls[0][0],180);self.assertGreaterEqual(calls[1][0]-calls[0][0],600)
 def test_frozen_request_guard(self):
  with self.assertRaisesRegex(Halt,'Request differs'):self.j.reserve(self.j.plan(),{'model':'changed'},{'waited_seconds':180},test_only=True)
 def test_synthetic_cannot_enter_real_replay(self):
  self.succeed()
  with self.assertRaisesRegex(Halt,'Synthetic'):self.j.audit()
 def test_current_production_inventory_audit(self):
  report=self.j.audit();self.assertEqual(report['logical_slots'],314);self.assertEqual(report['total_http_attempts'],323);self.assertEqual(report['unknown_charge_holds_usd'],'9')
if __name__=='__main__':
 with offline():unittest.main()
