from core import *
import unittest, tempfile, zipfile, io, csv, time
from unittest.mock import patch
import run as runner


def synthetic(n=64):
    rng=random.Random(280911)
    reference='ACDEFGHIKLMNPQRSTVWY'*3
    def candidate(i):
        seq=list(reference)
        for position in rng.sample(range(len(seq)),rng.randint(1,6)):
            seq[position]=rng.choice([a for a in 'ACDEFGHIKLMNPQRSTVWY' if a!=seq[position]])
        metadata,num=replay.flip2_ired_public_features(''.join(seq),reference)
        return replay.Candidate(str(i),metadata['mutation_count_bin'],*num[:3],float('nan'),metadata,num)
    source_public=[candidate(i) for i in range(1024)]
    target=[candidate(i+1024) for i in range(n)]
    fit_raw={c.candidate_id:rng.gauss(1,2) for c in source_public}
    units,source=fit_source({c.candidate_id:c for c in source_public},list(fit_raw),fit_raw)
    raw={c.candidate_id:rng.gauss(1,2) for c in target}
    return source,target,raw,units

class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source,cls.pool,cls.raw,cls.units=synthetic();cls.cfg=config()
    def test_full_legacy_trajectory_parity(self):
        for seed in [17,cls_seed:=self.cfg['seeds']['heldout'][0]]:
            new=paired_episodes(self.pool,self.source,self.raw,self.units,seed,self.cfg)
            valued=[replace(c,objective_value=float(self.units.revealed_to_executor(self.raw[c.candidate_id]))) for c in self.pool]
            dummy=classic.SourcePosterior(np.zeros(len(valued)),np.ones(len(valued)),np.zeros((len(valued),1)),0,1024)
            for mode,oldmode in [('Always',multi.ADDITIVE_MUTATION_POLICY_MODE),('Never','target_gp_ucb')]:
                row,audit=multi.run_seed([adapter('src',self.source)],adapter('target',valued),[dummy],seed,3,12,oldmode,1.5,.45,4.,.05,0,(),1.,.5,.1,source_observed_sets=[self.source])
                self.assertEqual([self.pool[i].candidate_id for i in new[mode]['observed_indices'][3:]],[a['selected_candidate'] for a in audit])
                self.assertEqual(new[mode]['best_so_far_auc'],row['best_so_far_auc'])
                self.assertEqual(new[mode]['final_best'],row['final_best'])
    def test_paired_initial_and_budget(self):
        pair=paired_episodes(self.pool,self.source,self.raw,self.units,127,self.cfg)
        for x in pair.values():
            self.assertEqual(len(set(x['observed_indices'])),15);self.assertEqual(len(x['best_trace']),12)
        self.assertEqual(pair['Always']['observed_indices'][:3],pair['Never']['observed_indices'][:3])
    def test_deterministic_replay(self):
        self.assertEqual(paired_episodes(self.pool,self.source,self.raw,self.units,127,self.cfg),paired_episodes(self.pool,self.source,self.raw,self.units,127,self.cfg))
    def test_no_public_objective_access(self):
        class Poison:
            def __float__(self):raise AssertionError('Unrevealed outcome accessed')
        poisoned=[replace(c,objective_value=Poison()) for c in self.pool]
        a=paired_episodes(self.pool,self.source,self.raw,self.units,127,self.cfg)
        b=paired_episodes(poisoned,self.source,self.raw,self.units,127,self.cfg)
        self.assertEqual(a,b)
    def test_target_values_do_not_change_calibration_gate(self):
        a={**self.raw,'outside_target':-1e9};b={**self.raw,'outside_target':1e9}
        ga=gate([paired_episodes(self.pool,self.source,a,self.units,s,self.cfg) for s in range(3)])
        gb=gate([paired_episodes(self.pool,self.source,b,self.units,s,self.cfg) for s in range(3)])
        self.assertEqual(ga,gb)
    def test_unrevealed_perturbation_cannot_change_next_action(self):
        features=classic.feature_arrays(adapter('target',self.pool),self.pool)
        prior=multi.additive_mutation_prior([self.source],self.pool)[0]
        vals=[0.,10.,20.]
        for mode in ('Always','Never'):
            a=choose(features,[0,1,2],vals,prior,mode,0,self.cfg)
            # Outcome map is absent from the selector's API entirely.
            self.assertEqual(a,choose(features,[0,1,2],list(vals),prior,mode,0,self.cfg))
    def test_reveal_guard(self):
        r=Reveals([c.candidate_id for c in self.pool],self.raw,self.units,1)
        r(0)
        with self.assertRaises(RuntimeError):r(0)
        with self.assertRaises(RuntimeError):r(1)
        with self.assertRaises(IndexError):r(-1)
    def test_checkpoint_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';ctx={'freeze':'a'};x={'value':[1,2]}
            save_checkpoint(p,ctx,x);save_checkpoint(p,ctx,x)
            self.assertEqual(read_checkpoint(p,ctx),x)
            with self.assertRaises(RuntimeError):read_checkpoint(p,{'freeze':'b'})
            with self.assertRaises(RuntimeError):save_checkpoint(p,ctx,{'value':[3]})
            record=json.loads(p.read_text());record['payload']['result']['value'][0]=9;p.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError):read_checkpoint(p,ctx)
    def test_gate_matches_legacy(self):
        pairs=[paired_episodes(self.pool,self.source,self.raw,self.units,s,self.cfg) for s in [1,2,3]]
        expected=classic.paired_comparison([{**pair[mode],'top10_hit':0} for pair in pairs for mode in ('Always','Never')],'Always','Never')['best_so_far_auc']
        self.assertEqual(gate(pairs)['auc_executor_units'],expected)
    def test_gate_zero_not_accepted(self):
        pairs=[{'Always':{'seed':s,'mode':'Always','final_best':5,'best_so_far_auc':5},'Never':{'seed':s,'mode':'Never','final_best':5,'best_so_far_auc':5}} for s in range(100)]
        self.assertFalse(gate(pairs)['accept'])
    def test_four_truth_classes_and_boundaries(self):
        for lo,hi,expected in [(.03,.04,'beneficial'),(-.04,-.03,'harmful'),(-.01,.01,'near_zero'),(-.03,.03,'uncertain'),(.02,.03,'uncertain')]:
            self.assertEqual(truth({'low':lo,'high':hi},.02),expected)
    def test_stages_are_disjoint_and_fresh(self):
        self.assertFalse(set(self.cfg['seeds']['calibration'])&set(self.cfg['seeds']['heldout']))
        self.assertEqual(len(set(self.cfg['seeds']['calibration'])),100)
        self.assertEqual(len(set(self.cfg['seeds']['heldout'])),200)
    def test_all_real_route_memberships_and_structural_fallbacks(self):
        self.assertEqual(len(routes()),23);matched=unsupported=0
        for r in routes():
            m=membership(r)
            self.assertFalse(set(runner.stage_ids(r,m,'calibration'))&set(m['target']))
            if r['G1']['status']=='matched':matched+=1
            else:unsupported+=1
        self.assertEqual((matched,unsupported),(13,10))
    def test_numeric_projection_only_parses_requested_ids(self):
        seq1='ACD';seq2='EFG';sid=hashlib.sha256(seq1.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('data/A.csv','mutated_sequence,DMS_score\nACD,1.5\nEFG,SECRET_NOT_NUMERIC\n')
            self.assertEqual(runner.load_values(p,'A.csv',[sid]),{sid:1.5})
            with self.assertRaises(ValueError):runner.load_values(p,'A.csv',[hashlib.sha256(seq2.encode()).hexdigest()])
    def test_barrier_blocks_missing_and_incomplete_gates(self):
        import core
        with tempfile.TemporaryDirectory() as d,patch.object(core,'OUT',Path(d)):
            with self.assertRaises(RuntimeError):verify_gate_barrier('fake')
            p=Path(d)/'GATES_LOCK.json';write_once(p,{'freeze_sha256':'fake','routes':[],'artifacts':{}})
            (Path(d)/'GATES_LOCK.sha256').write_text(digest(p))
            with self.assertRaises(RuntimeError):verify_gate_barrier('fake')
    def test_network_blocked(self):
        with self.assertRaises(RuntimeError):socket.create_connection(('example.com',443))
    def test_approved_scale_invariance_entire_episode(self):
        source,pool,raw,units=synthetic()
        a=paired_episodes(pool,source,raw,units,123,self.cfg)
        scaled={k:1000*v+100 for k,v in raw.items()}
        u2=SourceScoreUnits(1000*units.source_median+100,1000*units.source_iqr,1024)
        b=paired_episodes(pool,source,scaled,u2,123,self.cfg)
        for mode in ('Always','Never'):
            self.assertEqual(a[mode]['observed_indices'],b[mode]['observed_indices'])
            self.assertEqual(a[mode]['best_so_far_auc'],b[mode]['best_so_far_auc'])

if __name__=='__main__':unittest.main(verbosity=2)
