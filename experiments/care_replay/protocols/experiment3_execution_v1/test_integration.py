from core import *
import core,run as runner,analyze
from test_execution import synthetic
import unittest,tempfile
from unittest.mock import patch
from contextlib import ExitStack,redirect_stdout
import io

class IntegrationTests(unittest.TestCase):
    def test_real_worker_stage_lock_and_resume_with_synthetic_outcomes(self):
        source,target,targetraw,_=synthetic(328)
        fit=[c.candidate_id for c in source];cal=[c.candidate_id for c in target[:128]];held=[c.candidate_id for c in target[128:]]
        public={c.candidate_id:replace(c,objective_value=float('nan')) for c in source+target}
        raw={**{c.candidate_id:c.objective_value for c in source},**targetraw}
        r={**routes()[0],'route_id':'synthetic','G0_selected_ids':cal,'G1':{'status':'matched','selected_ids':cal}}
        m={'fit':fit,'calibration':cal,'target':held};cfg=config();cfg['seeds']={'calibration':[17,18,19],'heldout':[31,32,33,34]}
        stages=[]
        def loader(archive,filename,wanted):
            stages.append(set(wanted));return {i:raw[i] for i in wanted}
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            for mod in [core,runner]:
                stack.enter_context(patch.object(mod,'OUT',Path(d)))
                stack.enter_context(patch.object(mod,'routes',lambda:[r]))
            stack.enter_context(patch.object(runner,'membership',lambda route:m))
            stack.enter_context(patch.object(runner,'build_public',lambda assay,ids:{i:public[i] for i in ids}))
            stack.enter_context(patch.object(runner,'load_values',loader))
            stack.enter_context(patch.object(runner,'config',lambda:cfg))
            with self.assertRaises(RuntimeError):runner.worker(r,'heldout','synthetic_freeze')
            self.assertEqual(stages,[])
            runner.worker(r,'calibration','synthetic_freeze')
            self.assertFalse(stages[0]&set(held));self.assertEqual(stages[0],set(fit+cal))
            runner.freeze_gates('synthetic_freeze')
            locked=digest(Path(d)/'GATES_LOCK.json')
            runner.worker(r,'heldout','synthetic_freeze')
            self.assertEqual(stages[1],set(fit+held))
            self.assertEqual(locked,digest(Path(d)/'GATES_LOCK.json'))
            before={str(p.relative_to(d)):digest(p) for p in Path(d).rglob('*.json')}
            with patch.object(runner,'paired_episodes',side_effect=AssertionError('Must reuse frozen checkpoints')):
                runner.worker(r,'calibration','synthetic_freeze')
                runner.freeze_gates('synthetic_freeze')
                runner.worker(r,'heldout','synthetic_freeze')
            self.assertEqual(before,{str(p.relative_to(d)):digest(p) for p in Path(d).rglob('*.json')})
            cfg['bootstrap_replicates']=20
            for name,value in [('OUT',Path(d)),('routes',lambda:[r]),('membership',lambda route:m),('config',lambda:cfg),('verify_freeze',lambda:'synthetic_freeze')]:
                stack.enter_context(patch.object(analyze,name,value))
            with redirect_stdout(io.StringIO()):
                analyze.main();analyze.main()
            summary=json.loads((Path(d)/'analysis.json').read_text())
            self.assertEqual(summary['primary']['routes'],1)
            self.assertEqual(summary['api_calls'],0)
    def test_analysis_absent_denominator_and_cluster_pairing(self):
        cfg=config();cfg['bootstrap_replicates']=100
        rows=[]
        for group in ['A','B','C']:
            for j in range(2):
                rows.append({'group':group,'truth':'beneficial','G0':False,'G1':True,
                    'Always_minus_Never':1.,'G0_minus_Never':0.,'G1_minus_Never':1.,
                    'G0_minus_Always':-1.,'G1_minus_Always':0.,'G1_minus_G0':1.})
        self.assertIsNone(analyze.rate(rows,'harmful','G1')['rate'])
        out=analyze.bootstrap(rows,cfg)
        self.assertEqual(out['group_count'],3)
        self.assertEqual(out['G1_minus_G0']['estimate'],1.)
        self.assertEqual(out['G1_minus_G0']['low'],1.)
        self.assertEqual(out['false_rejection_G1_minus_G0']['estimate'],-1.)
        self.assertIsNone(out['false_acceptance_G1_minus_G0']['estimate'])

if __name__=='__main__':unittest.main(verbosity=2)
