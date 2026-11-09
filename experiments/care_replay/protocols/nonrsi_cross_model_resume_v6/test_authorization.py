import tempfile,unittest
from unittest.mock import patch
from . import runner as r
from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze
BASE_LOCK=BASE/'outputs/nonrsi_cross_model_freeze_v1/FROZEN_PROTOCOL.json'
class AuthorizationTests(unittest.TestCase):
 def test_exact_limits_and_scope(self):
  config=freeze.verify(BASE_LOCK)
  with tempfile.TemporaryDirectory() as tmp:
   lockfile=Path(tmp)/'lock.json';authfile=Path(tmp)/'auth.json';code=BASE/'protocols/nonrsi_cross_model_resume_v6/runner.py'
   lock={'kind':'nonrsi_cross_model_remaining_recovery_freeze','files':{str(code.relative_to(ROOT)):freeze.file_hash(code)},'base_lock':str(BASE_LOCK.relative_to(ROOT)),'original_authorization':'mock','previous_lock':'mock','previous_authorization':'mock'};save(lockfile,lock)
   auth={'kind':'explicit_remaining_recovery_authorization','resume_freeze_sha256':freeze.file_hash(lockfile),'base_freeze_sha256':freeze.file_hash(BASE_LOCK),'allow_execution':True,'model_calls':700,'model_spend_cap_usd':'45','historical_unknown_charge_hold_usd':'13','additional_replacements_cap':38,'replacements_already_used':10,'new_replacements_cap':28,'per_slot_replacements_cap':2,'backoff_seconds':[180,600],'qwen_minimum_interval_seconds':180,'transport_scope':'connection_errors_and_existing_exact_shared_pool_429','honor_longer_valid_retry_after':True,'user_message_reference':'synthetic'};save(authfile,auth)
   with patch.object(freeze,'authorize',return_value=config),patch.object(r.previous,'verify',return_value=(config,{})):
    r.verify(lockfile,authfile,None)
    for key,val in [('model_calls',701),('model_spend_cap_usd','46'),('historical_unknown_charge_hold_usd','0'),('additional_replacements_cap',39),('new_replacements_cap',29),('replacements_already_used',0),('per_slot_replacements_cap',3),('backoff_seconds',[0,0]),('qwen_minimum_interval_seconds',0),('transport_scope','anything'),('allow_execution',False)]:
     save(authfile,{**auth,key:val})
     with self.assertRaises(Halt):r.verify(lockfile,authfile,None)
 def test_unapproved_path_stops_before_key_or_network(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(r,'verify',return_value=({}, {'journal_directory':'unapproved'})):
   with self.assertRaisesRegex(Halt,'destination'):r.run(None,None,None,Path(tmp)/'journal')
if __name__=='__main__':
 with offline():unittest.main()
