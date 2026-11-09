"""Shared selector and replay wrapper. All network access is disabled."""
from .common import *
from . import inputs
from dataclasses import replace
import run_paired_anonymization_evaluation as legacy
import coverage_matched_rule_controls as coverage
import random

def compile_library(raw,cat,identifier):
    errors=contract.strict_validate_library(raw,cat)
    if errors:raise Halt('Unvalidated library: '+str(errors))
    skills,provenance=legacy.normalize_without_loss(raw,cat,0,'crossmodel')
    result=tuple(replace(s,skill_id=f'{identifier}_slot_{i:02d}',rules=tuple(replace(r,rule_id=f'{identifier}_slot_{i:02d}_rule_{j:02d}') for j,r in enumerate(s.rules))) for i,s in enumerate(skills))
    if any(matched.skill_execution_signature(a)!=matched.skill_execution_signature(b) for a,b in zip(skills,result)):raise Halt('Renaming changed execution')
    return result

def select(rows,libraries,outcomes,seeds):
    config={'min_risk_adjusted_gain':.25,'min_positive_fold_rate':.8}
    modes=(*matched.BASELINES,*(matched.semantic.skill_mode(s) for bank in libraries.values() for s in bank))
    matched.validate_rows(rows,modes,seeds,'calibration')
    means=matched.statistics.mode_means(rows,matched.BASELINES,seeds)
    anchor=max(matched.BASELINES,key=lambda m:(means[m]['composite'],m))
    choices=matched.freeze_selection(rows,libraries,seeds,config) if libraries else {}
    if set(libraries)!={k for k,v in outcomes.items() if v=='VALID'}:raise Halt('Valid-library inventory mismatch')
    for name,state in outcomes.items():
        if state=='INVALID_ANCHOR_FALLBACK':choices[name]={'primary_mode':anchor,'selected_mode':anchor,'status':state}
        elif state=='VALID':choices[name].update(primary_mode=choices[name]['ungated_calibration_winner'],status=state)
        else:raise Halt('Incomplete generation cannot enter selection')
    return {'choices':choices,'anchor':anchor,'calibration_rows_sha256':digest(rows),'calibration_seeds':sorted(seeds)}

def replay_seed(dataset,seed,split,skills):
    config={'target_dataset':dataset,'initial_observations':5,'reveal_rounds':10,'gp_beta':1.5,'gp_xi':.01,'numeric_length_scale':.35,'categorical_length_scale':3.,'gp_noise':.05}
    original=matched.replay.DATASET_BUILDERS.get(dataset)
    try:
        matched.replay.DATASET_BUILDERS[dataset]=lambda:inputs.adapter(dataset)
        return matched.evaluate_seed(seed,split,config,skills)
    finally:
        if original is None:matched.replay.DATASET_BUILDERS.pop(dataset,None)
        else:matched.replay.DATASET_BUILDERS[dataset]=original

def random_children(skills,catalog,types,public_rows,parent_id,seed):
    pools=coverage.build_pools(skills,catalog,types,public_rows,tolerance=.005,max_configurations=1000000)
    result={}
    for family in ['r1','r2']:
        for j in range(10):
            name=f'{family}_{parent_id}_{j:02d}';rng=random.Random(int(digest([seed,parent_id,family,j]),16))
            result[name]=tuple(coverage.draw_skill(pools[family][s.skill_id],rng,f'{name}_slot_{i:02d}') for i,s in enumerate(skills))
    return result,{family:{s.skill_id:pools[family][s.skill_id].diagnostics for s in skills} for family in ['r1','r2']}
