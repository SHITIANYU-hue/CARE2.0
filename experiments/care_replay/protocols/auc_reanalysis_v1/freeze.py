"""Verify certified inputs and seal this preparation package. Never runs normalization."""
import offline
from pathlib import Path
import hashlib,json,os,subprocess,datetime,platform
P=Path(__file__).resolve().parent;B=P.parents[1];ROOT=B.parents[1];OUT=B/'outputs/auc_reanalysis_v1'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def verify_inputs():
 inputs={}
 for name in ['certification_report.json','prepared_tables_report.json']:
  d=json.loads((OUT/name).read_text())
  for path,expected in d['inputs'].items():
   if path in inputs and inputs[path]!=expected:raise ValueError('Conflicting input digests')
   inputs[path]=expected
 for path,expected in inputs.items():
  if path.startswith('git:'):
   actual=hashlib.sha256(subprocess.run(['git','show',path[4:]],cwd=ROOT,capture_output=True,check=True).stdout).hexdigest()
  else:
   f=Path(path);mode=f.stat().st_mode&0o777
   try:
    if mode==0:os.chmod(f,0o400)
    actual=digest(f)
   finally:
    if mode==0:os.chmod(f,mode)
  if actual!=expected:raise ValueError('Certification input changed: '+path)
 return inputs

def main():
 tests=subprocess.run([__import__('sys').executable,'-B','-m','unittest','discover','-s',str(P),'-p','test_*.py','-v'],capture_output=True,text=True)
 (OUT/'offline_tests.txt').write_text(tests.stdout+tests.stderr)
 if tests.returncode:raise RuntimeError('Offline tests failed; no freeze')
 inputs=verify_inputs();cert=json.loads((OUT/'certification_report.json').read_text())
 if len(cert['certified'])!=32 or cert['gaps']:raise ValueError('Incomplete certification')
 artifacts={}
 for directory in [P,OUT]:
  for file in directory.rglob('*'):
   if not file.is_file() or '__pycache__' in file.parts or file.name in ['FREEZE.json','FREEZE.sha256','freeze_verification.json']:continue
   artifacts[str(file.relative_to(B))]=digest(file)
 lock={'protocol_id':'auc_reanalysis_v1','status':'offline_preparation_frozen_before_new_normalized_effects','scientific_decisions':'D1/D2/D3 user approved 2026-10-21','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifacts':artifacts,'verified_source_inputs':inputs,'pool_certificates':32,'python':platform.python_version(),'model_api_calls':0,'new_policy_runs':0,'new_normalized_effects_computed':False,'execution_boundary':'Current approved phase ends at implementation, certification, tests and freeze. No new normalized effect report is produced in this phase. Historical budget-conflict studies remain outside the 32-row scope.'}
 path=P/'FREEZE.json'
 if path.exists():raise ValueError('Already frozen; verify existing lock, do not silently replace')
 path.write_bytes(canonical(lock));h=digest(path);(P/'FREEZE.sha256').write_text(h+'\n')
 for rel,expected in artifacts.items():
  if digest(B/rel)!=expected:raise ValueError('Post-freeze artifact changed')
 (OUT/'freeze_verification.json').write_bytes(canonical({'freeze_sha256':h,'artifact_count':len(artifacts),'source_input_count':len(inputs),'all_artifact_hashes_match':True,'offline_tests_exit_code':0,'protein_raw_mode':oct((B/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip').stat().st_mode&0o777),'model_api_calls':0}))
 print('Freeze verified:',h,'artifacts',len(artifacts),'source inputs',len(inputs))
if __name__=='__main__':main()
