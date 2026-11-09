"""No-network validation for the recovery amendment."""
import unittest,sys,time
from datetime import datetime,timezone
from .runner import BASE,offline,save
MODULES=['protocols.nonrsi_cross_model_resume_v6.test_transport','protocols.nonrsi_cross_model_resume_v6.test_authorization','protocols.nonrsi_cross_model_resume_v6.test_pipeline']
def main():
 out=BASE/'outputs/nonrsi_cross_model_resume_v6';started=time.monotonic()
 with offline(),(out/'guarded_tests.log').open('w') as log:
  suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
  result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'network_guard_enabled':True,'inference_calls':0,'elapsed_seconds':time.monotonic()-started}
 save(out/'VALIDATION.json',report);print(report);sys.exit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
