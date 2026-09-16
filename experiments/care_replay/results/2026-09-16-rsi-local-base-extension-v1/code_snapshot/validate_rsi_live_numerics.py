#!/usr/bin/env python3
"""Sample exact replay and compare numerical scores with Python reference."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import run_rsi_live_feedback as live
import run_surrogate_baselines as surrogate


def validate(root):
    config=json.loads((root/'config.json').read_text())
    original_vector=surrogate._gp_anchor_scores_vectorized
    reference=surrogate._gp_anchor_scores_python
    maximum=0.;n=0
    def checked(*args,**kwargs):
        nonlocal maximum,n
        actual,diag=original_vector(*args,**kwargs);expected,_=reference(*args,**kwargs)
        for name in actual:
            keys=list(actual[name]);a=np.array([actual[name][k] for k in keys]);b=np.array([expected[name][k] for k in keys])
            assert np.isfinite(a).all() and np.isfinite(b).all()
            maximum=max(maximum,float(np.max(abs(a-b))));n+=len(a)
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        return actual,diag
    surrogate._gp_anchor_scores_vectorized=checked
    comparisons=[]
    for spec in config['tasks']:
        for rep in config['model_replicates']:
            folder=root/spec['id']/f'replicate_{rep}'
            lock=json.loads((folder/'deployment_lock.json').read_text())
            for arm in ['fixed_initial','true_feedback']:
                decision=lock['states'][arm];chosen=decision['selected_skill_id']
                skill=[s for s in decision['skills'] if s['skill_id']==chosen]
                seed=config['evaluation_seed_start']
                actual=live.offline.run_job((spec['id'],skill,seed,config,'development'))[0]
                archived=next(x for x in json.loads((folder/arm/'evaluation'/f'seed_{seed}.json').read_text()) if x['skill_id']==chosen)
                assert actual['metrics']==archived['metrics']
                assert [e['selected_candidate'] for e in actual['events']]==[e['selected_candidate'] for e in archived['events']]
                comparisons.append({'task':spec['id'],'replicate':rep,'arm':arm,'seed':seed,'replay_match':True})
    result={'status':'passed','sampled_trajectories':comparisons,'candidate_scores_checked':n,
            'max_absolute_numpy_python_error':maximum,
            'scope':'Fixed and final true-feedback skills on first held-out seed for all model replicates; not an exhaustive backend audit'}
    live.save(root/'numerical_validation.json',result)
    print(json.dumps({'status':'passed','trajectories':len(comparisons),'scores':n,'max_error':maximum}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    validate(p.parse_args().root)
