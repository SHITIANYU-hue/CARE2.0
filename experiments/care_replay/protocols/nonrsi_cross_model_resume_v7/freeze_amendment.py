"""Create one immutable revision only after offline validation succeeds."""
from .runner import *
from datetime import datetime,timezone
def main():
 out=BASE/'outputs/nonrsi_cross_model_resume_v7';old=BASE/'outputs/nonrsi_cross_model_resume_v6'
 val=load(out/'VALIDATION.json')
 if not val['passed'] or val['tests_run']!=36 or not val['network_guard_enabled'] or val['inference_calls']!=0:raise Halt('Validation incomplete')
 if (out/'FROZEN_RECOVERY.json').exists() or (out/'RECOVERY_AUTHORIZATION.json').exists():raise Halt('Freeze already exists')
 with offline():
  config,prior=previous.verify(old/'FROZEN_RECOVERY.json',old/'RECOVERY_AUTHORIZATION.json',out/'preflight/ATTESTATION.json')
  j=Journal(out/'journal',ROOT/prior['original_ledger'],config);audit=j.audit()
  if audit['logical_slots']!=489 or audit['total_http_attempts']!=507 or audit['unknown_charge_holds_usd']!='18':raise Halt('Inventory moved before freezing')
  files=dict(prior['files'])
  for directory in [old,BASE/'protocols/nonrsi_cross_model_resume_v7',out/'preflight']:
   for f in directory.rglob('*'):
    if f.is_file() and f.suffix!='.lock' and '__pycache__' not in f.parts:files[str(f.relative_to(ROOT))]=freeze.file_hash(f)
  for name in ['VALIDATION.json','FROZEN_RECOVERY_PLAN.json','INITIAL_AUDIT.json','guarded_tests.log','runner_changes.diff']:
   f=out/name;files[str(f.relative_to(ROOT))]=freeze.file_hash(f)
  lock={**{k:v for k,v in prior.items() if k!='files'},'created_at_utc':datetime.now(timezone.utc).isoformat(),'previous_lock':str((old/'FROZEN_RECOVERY.json').relative_to(ROOT)),'previous_authorization':str((old/'RECOVERY_AUTHORIZATION.json').relative_to(ROOT)),'source_journal':str((old/'journal').relative_to(ROOT)),'journal_directory':str((out/'journal').relative_to(ROOT)),'evaluation_output':str((out/'evaluation').relative_to(ROOT)),'files':dict(sorted(files.items()))}
  save(out/'FROZEN_RECOVERY.json',lock)
  auth=load(old/'RECOVERY_AUTHORIZATION.json')
  auth.update(resume_freeze_sha256=freeze.file_hash(out/'FROZEN_RECOVERY.json'),model_calls=700,model_spend_cap_usd='45',historical_unknown_charge_hold_usd='18',additional_replacements_cap=38,replacements_already_used=15,new_replacements_cap=23,recorded_at_utc=datetime.now(timezone.utc).isoformat(),transport_scope='connection_errors_exact_shared_pool_429_and_explicit_qwen_response_network_loss',user_message_reference=load(out/'FROZEN_RECOVERY_PLAN.json')['authorization_basis'])
  save(out/'RECOVERY_AUTHORIZATION.json',auth)
  verify(out/'FROZEN_RECOVERY.json',out/'RECOVERY_AUTHORIZATION.json',out/'preflight/ATTESTATION.json')
  print(json.dumps({'freeze_sha256':freeze.file_hash(out/'FROZEN_RECOVERY.json'),'pinned_files':len(files),'model_calls':0,'cap_calls':700,'cap_usd':'45'}))
if __name__=='__main__':main()
