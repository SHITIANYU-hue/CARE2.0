import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ssl,socket,sys,json,unittest
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from score_units import SourceScoreUnits
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
def blocked(*a,**k):raise RuntimeError('No network during synthetic preflight')
socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
sys.path.insert(0,str(ROOT/'scripts'))
import run_classical_transfer_baselines as legacy

def action(values):
    f=legacy.FeatureArrays(numeric=np.linspace(0,1,41).reshape(-1,1),categorical=np.array([['same']]*41));obs=[0,10,20]
    y,_,_=legacy.normalized_outcomes([SimpleNamespace(objective_value=v) for v in values]);mu,var,_=legacy.target_gp_posterior(f,obs,y,.45,4,.05)
    return legacy.top_unobserved(mu+1.5*np.sqrt(var),set(obs))

class ScoreUnitsTests(unittest.TestCase):
    def setUp(self):self.source=np.linspace(-1.,1.,1024)
    def test_source_only_quantiles(self):
        u=SourceScoreUnits.fit(self.source);self.assertAlmostEqual(u.source_median,0.);self.assertAlmostEqual(u.source_iqr,1.)
    def test_affine_unit_invariance(self):
        y=np.array([-.1,.2,3.]);original=SourceScoreUnits.fit(self.source).revealed_to_executor(y)
        for a,b in [(.01,0.),(100.,300.),(10000.,-7.)]:
            np.testing.assert_allclose(SourceScoreUnits.fit(a*self.source+b).revealed_to_executor(a*y+b),original,rtol=1e-10,atol=1e-10)
    def test_no_clipping(self):
        u=SourceScoreUnits.fit(self.source);np.testing.assert_allclose(u.revealed_to_executor([-2.,3.]),[-200.,300.])
    def test_fixed_source_statistics_for_future_reveals(self):
        u=SourceScoreUnits.fit(self.source);first=u.revealed_to_executor([.2]);u.revealed_to_executor([-100.,999.]);np.testing.assert_equal(first,u.revealed_to_executor([.2]));self.assertAlmostEqual(u.source_iqr,1.)
    def test_excluded_zero_iqr_not_rescaled(self):
        with self.assertRaises(ValueError):SourceScoreUnits.fit(np.ones(1024))
    def test_invalid_source_rejected(self):
        for v in [np.ones(1023),np.append(np.ones(1023),np.nan),np.append(np.ones(1023),np.inf)]:
            with self.assertRaises(ValueError):SourceScoreUnits.fit(v)
    def test_invalid_reveal_rejected(self):
        with self.assertRaises(ValueError):SourceScoreUnits.fit(self.source).revealed_to_executor([np.nan])
    def test_margin_unchanged(self):
        self.assertAlmostEqual(float(SourceScoreUnits.fit(self.source).executor_auc_delta_to_source_iqr(2.)),.02)
    def test_legacy_changes_only_from_units(self):
        y=np.array([0.,.4,.8]);self.assertEqual(action(y),40);self.assertEqual(action(100*y),36)
    def test_proposal_keeps_same_actions_across_units(self):
        y=np.array([0.,.4,.8]);selected=[]
        for scale in [.01,1.,100.,10000.]:selected.append(action(SourceScoreUnits.fit(self.source*scale).revealed_to_executor(y*scale)))
        self.assertEqual(len(set(selected)),1)
    def test_auc_differences_have_same_source_iqr_units(self):
        u=SourceScoreUnits.fit(self.source);a=np.array([.1,.2,.6]);b=np.array([.1,.3,.5])
        expected=(a.mean()-b.mean())/u.source_iqr
        got=u.executor_auc_delta_to_source_iqr(u.revealed_to_executor(a).mean()-u.revealed_to_executor(b).mean())
        self.assertAlmostEqual(float(got),float(expected))

if __name__=='__main__':unittest.main(verbosity=2)
