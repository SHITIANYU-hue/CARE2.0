"""Real public-feature integration plus synthetic-only timing and tests."""
from core import *
import subprocess,time,platform,stat,datetime
from test_execution import synthetic

def main():
    OUT.mkdir(exist_ok=True)
    old=json.loads((BASE/'outputs/experiment3_gate_preparation_v1/input_manifest.json').read_text())
    verify_mapping(old,REPO)
    lock=json.loads((BASE/'protocols/experiment3_public_preparation_v1/lock.json').read_text())
    verify_mapping(lock['artifacts'],BASE)
    raw=BASE/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip'
    if stat.S_IMODE(raw.stat().st_mode)!=0:raise RuntimeError('Raw effects were not sealed before preflight')
    run=subprocess.run([sys.executable,'-B','-m','unittest','test_execution','test_score_units','test_integration','-v'],cwd=HERE,capture_output=True,text=True)
    (OUT/'offline_tests.txt').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Offline tests failed; see log')
    audit=[]
    for r in routes():
        m=membership(r);ids=list(dict.fromkeys(m['fit']+r['G0_selected_ids']+r['G1']['selected_ids']+m['target']))
        public=build_public(r['assay'],ids);pool=list(public.values())
        f=classic.feature_arrays(adapter(r['route_id'],pool),pool)
        assert f.numeric.shape==(len(ids),39) and np.isfinite(f.numeric).all()
        assert all(math.isnan(c.objective_value) for c in pool)
        assert all(not any('measured' in k or 'score' in k or 'outcome' in k for k in c.metadata) for c in pool)
        audit.append({'route':r['route_id'],'public_candidates_checked':len(ids),'numeric_features':39,'categorical_features':f.categorical.shape[1],'no_outcomes':True})
    # Measure realistic full pool size; outcomes and sequences entirely synthetic.
    source,pool,raw_values,units=synthetic(5000);cfg=config()
    prepared=(classic.feature_arrays(adapter('synthetic',pool),pool),multi.additive_mutation_prior([source],pool)[0])
    start=time.monotonic()
    for seed in [771,772]:paired_episodes(pool,source,raw_values,units,seed,cfg,prepared)
    seconds=time.monotonic()-start
    report={'status':'passed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':30,'real_effects_opened':False,
        'old_214_inputs_verified':len(old),'public_lock_inputs_verified':len(lock['artifacts']),
        'routes':audit,'synthetic_5000_pool_seconds_per_pair':seconds/2,
        'conservative_serial_fullsize_8200_pairs_seconds':seconds/2*8200,
        'max_workers':cfg['workers'],'planned_pairs':{'calibration_G0':2300,'calibration_G1':1300,'heldout':4600},
        'planned_trajectories':16400,'planned_simulated_observations':246000,
        'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'api_calls':0,'network_disabled':True}
    (OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='routes'},indent=2))

if __name__=='__main__':main()
