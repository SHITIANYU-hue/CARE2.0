import unittest,copy,json,tempfile
from unittest.mock import patch
from . import runner as r,response_transport as e
from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze,inputs,generation
class ResponseNetworkTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.config=freeze.verify(BASE/'outputs/nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json')
  cls.model=next(m for m in cls.config['models'] if m['id']=='qwen3_instruct_2507')
  cls.raw=load(r.SOURCE_JOURNAL/'0456.response.json')
 def test_exact_observed_response_kept_unchanged(self):
  raw=copy.deepcopy(self.raw);ev=e.evidence(raw,self.model)
  self.assertEqual(raw,self.raw);self.assertTrue(e.qualifies({'model':self.model['id']},ev));self.assertEqual(ev['reported_cost_usd'],'0')
  with self.assertRaisesRegex(Halt,'Invalid message'):generation.classify(raw,self.model,[],1)
 def test_non_network_errors_not_retried(self):
  for mutate in [lambda d:d['choices'][0].update(finish_reason='length'),lambda d:d['choices'][0]['error'].update(code=500),lambda d:d['choices'][0]['error'].update(message='generic error'),lambda d:d['choices'][0]['error'].update(metadata={}),lambda d:d['choices'][0].pop('error')]:
   raw=copy.deepcopy(self.raw);mutate(raw);self.assertIsNone(e.evidence(raw,self.model))
 def test_identity_billing_and_usage_still_halt(self):
  for mutate in [lambda d:d.update(provider='Other'),lambda d:d.update(model='other'),lambda d:d.update(id=''),lambda d:d['usage'].pop('cost'),lambda d:d['usage'].update(cost=2),lambda d:d['usage'].update(is_byok=True),lambda d:d['usage'].update(prompt_tokens=12001),lambda d:d['usage']['completion_tokens_details'].update(reasoning_tokens=1),lambda d:d.update(system_fingerprint=123),lambda d:d['choices'][0]['message'].update(role='tool')]:
   raw=copy.deepcopy(self.raw);mutate(raw)
   with self.assertRaises(Halt):e.evidence(raw,self.model)
 def test_captured_stop_is_one_exact_record(self):
  rows=load(r.SOURCE_JOURNAL/'ATTEMPTS.json')['attempts'];a=rows[456]
  self.assertTrue(r.permitted_imported_stop(456,a,rows))
  for key in ['logical_index','total_attempt_index','response_sha256','request_sha256','halt_reason']:
   self.assertFalse(r.permitted_imported_stop(456,{**a,key:'changed'},rows))
 def test_response_error_recovery_keeps_full_hold_and_source(self):
  with tempfile.TemporaryDirectory() as tmp:
   j=r.Journal(Path(tmp)/'journal',BASE/'outputs/nonrsi_cross_model_resume_v2/ledger',self.config,True);p=j.plan()
   raw=copy.deepcopy(self.raw);raw['id']='new-network-failure'
   body=r.request_contract.build(self.config,p['slot'],load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(p['slot']['cell']+'.json')))
   n=j.reserve(p,body,{'waited_seconds':180},test_only=True);save(j.directory/f'{n:04d}.response.json',raw);save(j.directory/f'{n:04d}.response.txt',raw)
   self.assertTrue(j.fail(n,e.evidence(raw,self.model)));self.assertEqual(j.state()['attempts'][n]['status'],'RETRYABLE_RESPONSE_TRANSPORT')
   self.assertEqual(j.totals(j.state()['attempts'])[1],19);self.assertEqual(j.plan()['logical_index'],489);self.assertEqual(j.plan()['required_backoff_seconds'],600)
   self.assertEqual(len(load(j.science/'ledger.json')['attempts']),489)
   j.audit(allow_test=True)
   changed=copy.deepcopy(raw);changed['usage']['cost']=0.5;save(j.directory/f'{n:04d}.response.json',changed)
   with self.assertRaisesRegex(Halt,'evidence'):j.audit(allow_test=True)
 def test_duplicate_failed_response_id_not_reused(self):
  with tempfile.TemporaryDirectory() as tmp:
   j=r.Journal(Path(tmp)/'journal',BASE/'outputs/nonrsi_cross_model_resume_v2/ledger',self.config,True);p=j.plan()
   body=r.request_contract.build(self.config,p['slot'],load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(p['slot']['cell']+'.json')))
   n=j.reserve(p,body,{'waited_seconds':180},test_only=True)
   with self.assertRaisesRegex(Halt,'Duplicate'):j.response_identity(n,self.raw)
 def test_mocked_http200_error_retries_only_twice(self):
  with tempfile.TemporaryDirectory() as tmp:
   directory=Path(tmp)/'journal';clock=[0.0];calls=[];raw=self.raw;certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
   class Response:
    status=200;headers={'content-type':'application/json'}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):
     x=copy.deepcopy(raw);x['id']='mock-network-'+str(len(calls));return json.dumps(x).encode()
   class Opener:
    def open(self,req,timeout):calls.append((clock[0],json.loads(req.data)));return Response()
   def sleep(dt):clock[0]+=dt
   lock={'journal_directory':str(directory),'original_ledger':str(BASE/'outputs/nonrsi_cross_model_resume_v2/ledger')}
   with patch.object(r,'verify',return_value=(self.config,lock)),patch.dict(r.os.environ,{'OPENROUTER_API_KEY':'mock-not-real'}),patch.object(r.time,'monotonic',side_effect=lambda:clock[0]),patch.object(r.time,'sleep',side_effect=sleep),patch.object(r.urllib.request,'build_opener',return_value=Opener()),patch.object(inputs,'certify',side_effect=list(certs.values())):
    with self.assertRaisesRegex(Halt,'Per-slot'):r.run(None,None,None,directory)
   self.assertEqual(len(calls),2);self.assertEqual(calls[0][1],calls[1][1]);self.assertGreaterEqual(calls[1][0]-calls[0][0],600)
   rows=load(directory/'ATTEMPTS.json')['attempts']
   self.assertTrue(all(a['status']=='RETRYABLE_RESPONSE_TRANSPORT' and a['cost_usd'] is None for a in rows[457:]))

 def test_failed_response_fingerprint_drift_stops_audit(self):
  with tempfile.TemporaryDirectory() as tmp:
   j=r.Journal(Path(tmp)/'journal',BASE/'outputs/nonrsi_cross_model_resume_v2/ledger',self.config,True)
   for i,fp in enumerate(['first-network-fp','changed-network-fp']):
    p=j.plan();body=r.request_contract.build(self.config,p['slot'],load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(p['slot']['cell']+'.json')))
    n=j.reserve(p,body,{'waited_seconds':p['required_backoff_seconds']},test_only=True)
    raw=copy.deepcopy(self.raw);raw.update(id='network-drift-'+str(i),system_fingerprint=fp)
    save(j.directory/f'{n:04d}.response.json',raw);save(j.directory/f'{n:04d}.response.txt',raw)
    j.fail(n,e.evidence(raw,self.model))
   with self.assertRaisesRegex(Halt,'fingerprint drift'):j.audit(allow_test=True)
