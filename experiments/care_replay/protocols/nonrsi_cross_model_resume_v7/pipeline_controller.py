"""Sequential approved pipeline. A failed phase prevents all later phases."""
from pathlib import Path
import argparse,os,subprocess,sys,fcntl
from datetime import datetime,timezone
from protocols.nonrsi_cross_model_v1.common import BASE,save,load
from protocols.nonrsi_cross_model_resume_v3.preflight import read_key
from .runner import verify
OUT=BASE/'outputs/nonrsi_cross_model_resume_v7'
PHASES=('generate','prepare','calibrate','heldout','analyze')
def commands():
 common=['--lock',str(OUT/'FROZEN_RECOVERY.json'),'--authorization',str(OUT/'RECOVERY_AUTHORIZATION.json'),'--attestation',str(OUT/'preflight/ATTESTATION.json'),'--journal',str(OUT/'journal')]
 return [(phase,[sys.executable,'-B','-m','protocols.nonrsi_cross_model_resume_v7.runner',phase,*common]+([] if phase=='generate' else ['--output',str(OUT/'evaluation'),'--workers','4'])) for phase in PHASES]
def execute(run_phase,write_status):
 for phase,command in commands():
  write_status({'state':'running','phase':phase,'started_at_utc':datetime.now(timezone.utc).isoformat()})
  code=run_phase(phase,command)
  if code:
   write_status({'state':'stopped','phase':phase,'exit_code':code,'finished_at_utc':datetime.now(timezone.utc).isoformat()});return code
 write_status({'state':'complete','phase':'analyze','exit_code':0,'finished_at_utc':datetime.now(timezone.utc).isoformat()});return 0
def main():
 with (OUT/'pipeline.process.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=OUT/'PIPELINE_STATUS.json'
  if status.exists():raise RuntimeError('Existing controller status: inspect before restarting')
  verify(OUT/'FROZEN_RECOVERY.json',OUT/'RECOVERY_AUTHORIZATION.json',OUT/'preflight/ATTESTATION.json')
  key=read_key()
  def run_phase(phase,command):
   env=os.environ.copy();env.pop('OPENROUTER_API_KEY',None)
   env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   if phase=='generate':env['OPENROUTER_API_KEY']=key
   with (OUT/(phase+'.log')).open('w') as log:
    return subprocess.run(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  sys.exit(execute(run_phase,lambda x:save(status,x)))
if __name__=='__main__':main()
