"""Offline, resumable calibration -> global choice lock -> heldout replay."""
from .common import *
from . import evaluation,inputs,generation
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random, math, gzip

def once(path,value):
    path=Path(path)
    if path.exists():
        if digest(load(path))!=digest(value):raise Halt('Immutable artifact changed: '+str(path))
    else:save(path,value)

def unpack_skill(x):
    x=dict(x);x['rules']=tuple(matched.semantic.SemanticRule(**dict(r,conditions=tuple(tuple(v) for v in r['conditions']))) for r in x['rules'])
    return matched.semantic.SemanticSkill(**x)

def prepare(config,attempts,output):
    output=Path(output)
    if len(attempts)!=660 or [a['slot'] for a in attempts]!=slots(config):raise Halt('Full ordered generation inventory required')
    if any(a['status'] not in [generation.VALID,generation.INVALID] or a['cost_usd'] is None for a in attempts):raise Halt('Unsettled generation')
    definition={'config_sha256':digest(config),'generation_sha256':digest(attempts),'targets':{},'slots':{},'random_children':{}}
    for target in [r['target'] for r in config['replay_proposed']['target_seed_ranges']]:
        adapter=inputs.adapter(target);cat=inputs.catalog(adapter);banks={};outcomes={};diagnostics={}
        for a in attempts:
            slot=a['slot'];name=slot['slot_id']
            if slot['target']!=target:continue
            definition['slots'][name]=slot
            outcomes[name]=a['status']
            if a['status']!=generation.VALID:continue
            bank=evaluation.compile_library(a['library'],cat,name);banks[name]=bank
            if slot['evidence_mode']=='target_only' and target in ['real_buchwald_hartwig','real_suzuki_miyaura']:
                types=inputs.suzuki.field_types(cat) if target==inputs.suzuki.TARGET else config['random_controls_proposed']['field_types_bh']
                kids,diag=evaluation.random_children(bank,cat,types,[{f:str(c.metadata.get(f,'')) for f in cat} for c in adapter.candidates],name,config['random_control_seed'])
                diagnostics[name]=diag;banks.update(kids)
                for k in kids:
                    outcomes[k]=generation.VALID
                    definition['random_children'][k]={'parent':name,'family':k.split('_')[0],'target':target}
        definition['targets'][target]={'banks':{k:[asdict(s) for s in bank] for k,bank in banks.items()},'outcomes':outcomes,'coverage_diagnostics':diagnostics,'pool_sha256':inputs.certify(adapter)['complete_candidate_records_sha256']}
    if len(definition['slots'])!=660:raise Halt('Slot/target inventory incomplete')
    once(output/'definition.json',definition)
    return definition

def banks_for(definition,target):
    return {k:tuple(unpack_skill(s) for s in bank) for k,bank in definition['targets'][target]['banks'].items()}

def seed_set(config,target,split):
    item=next(r for r in config['replay_proposed']['target_seed_ranges'] if r['target']==target)
    start=item[split+'_start'];n=config['replay_proposed'][split+'_seeds']
    return list(range(start,start+n))

def initial_indices(adapter,seed):
    ids=list(range(len(adapter.candidates)));random.Random(seed).shuffle(ids);return ids[:5]

def verify_trace(adapter,seed,rows,traces,modes,split):
    matched.validate_rows(rows,modes,{seed},split)
    byid={c.candidate_id:c for c in adapter.candidates}
    initial=[adapter.candidates[i] for i in initial_indices(adapter,seed)]
    for row in rows:
        events=[e for e in traces if e['mode']==row['mode']]
        if len(events)!=10 or [e['round_index'] for e in events]!=list(range(10)):raise Halt('Trace round count')
        observed={c.candidate_id for c in initial};best=max(c.objective_value for c in initial);curve=[]
        for t,e in enumerate(events):
            cid=e['selected_candidate']
            if cid not in byid or cid in observed or e['seed']!=seed or e['public_observed_count']!=5+t:raise Halt('Trace action budget or reveal boundary')
            value=byid[cid].objective_value
            if not math.isclose(e['revealed_value'],value,abs_tol=1.01e-4,rel_tol=0.):raise Halt('Trace oracle mismatch')
            observed.add(cid);best=max(best,value);curve.append(best)
            if not math.isclose(e['best_so_far'],best,abs_tol=1.01e-4,rel_tol=0.):raise Halt('Best-so-far mismatch')
        if not math.isclose(row['best_so_far_auc'],sum(curve)/10,abs_tol=1.01e-4,rel_tol=0.) or not math.isclose(row['final_best'],best,abs_tol=1.01e-4,rel_tol=0.):raise Halt('Metric/trace mismatch')

def phase_worker(seed,*,target,split,skills,output,binding):
    path=Path(output)/target/split/f'{seed}.json'
    reps,aliases=matched.execution_plan(skills,enabled=True)
    modes=(*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in skills))
    if path.exists():
        cached=load(path)
        if cached['binding']!=binding:raise Halt('Checkpoint binding changed')
        tracepath=path.with_suffix('.traces.json.gz')
        traces=json.loads(gzip.decompress(tracepath.read_bytes()))
        if digest(traces)!=cached['traces_sha256'] or digest(cached['rows'])!=cached['rows_sha256']:raise Halt('Checkpoint changed')
        verify_trace(inputs.adapter(target),seed,cached['rows'],traces,modes,split)
        return cached['rows']
    rows,traces,duration=evaluation.replay_seed(target,seed,split,reps)
    rows=matched.expand_metric_aliases(rows,aliases)
    expanded=[dict(e,mode=alias) for alias,representative in aliases.items() for e in traces if e['mode']==representative]
    expanded.extend(e for e in traces if e['mode'] in matched.BASELINES)
    verify_trace(inputs.adapter(target),seed,rows,expanded,modes,split)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Atomic checkpoint is published last; a crash leaves only replaceable orphan trace data.
    raw=json.dumps(expanded,sort_keys=True,allow_nan=False).encode();path.with_suffix('.traces.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    save(path,{'binding':binding,'rows':rows,'rows_sha256':digest(rows),'traces_sha256':digest(expanded),'elapsed_seconds':duration,'executed_unique_skills':len(reps),'logical_skills':len(skills)})
    return rows

def phase(config,definition,target,split,skills,output,workers):
    if type(workers) is not int or workers<1:raise Halt('Invalid worker count')
    if inputs.certify(inputs.adapter(target))['complete_candidate_records_sha256']!=definition['targets'][target]['pool_sha256']:raise Halt('Pool changed before replay')
    binding=digest([digest(config),digest(definition),target,split,[asdict(s) for s in skills]])
    worker=partial(phase_worker,target=target,split=split,skills=skills,output=str(output),binding=binding)
    seeds=seed_set(config,target,split)
    with offline():
        if workers==1:rows=[r for seed in seeds for r in worker(seed)]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:rows=[r for batch in pool.map(worker,seeds) for r in batch]
    matched.validate_rows(rows,(*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in skills)),set(seeds),split)
    return rows

def calibrate(config,definition,output,workers=1):
    output=Path(output)
    if definition['config_sha256']!=digest(config):raise Halt('Definition/config mismatch')
    selected={};checkpoint_hashes={}
    for target in definition['targets']:
        banks=banks_for(definition,target);skills=tuple(s for bank in banks.values() for s in bank)
        rows=phase(config,definition,target,'calibration',skills,output,workers)
        choice=evaluation.select(rows,banks,definition['targets'][target]['outcomes'],set(seed_set(config,target,'calibration')))
        once(output/target/'selection.json',choice);selected[target]=choice
        for seed in seed_set(config,target,'calibration'):
            p=output/target/'calibration'/f'{seed}.json';checkpoint_hashes[str(p.relative_to(output))]=digest(load(p))
    lock={'kind':'all_targets_calibration_lock','config_sha256':digest(config),'definition_sha256':digest(definition),'selections':selected,'checkpoints':checkpoint_hashes}
    once(output/'CALIBRATION_LOCK.json',lock)
    return lock

def heldout(config,definition,output,workers=1):
    output=Path(output);lock=load(output/'CALIBRATION_LOCK.json')
    if lock['config_sha256']!=digest(config) or lock['definition_sha256']!=digest(definition) or set(lock['selections'])!=set(definition['targets']):raise Halt('All target selections must be frozen first')
    for name,h in lock['checkpoints'].items():
        if digest(load(output/name))!=h:raise Halt('Calibration checkpoint changed')
    rows_by_target={}
    for target in definition['targets']:
        if digest(load(output/target/'selection.json'))!=digest(lock['selections'][target]):raise Halt('Selection changed after lock')
        banks=banks_for(definition,target);choices=lock['selections'][target]['choices']
        wanted={v[k] for v in choices.values() for k in ['primary_mode','selected_mode']}
        skills=tuple(s for bank in banks.values() for s in bank if matched.semantic.skill_mode(s) in wanted)
        rows=phase(config,definition,target,'heldout',skills,output,workers);rows_by_target[target]=rows
        once(output/target/'heldout_metrics.json',rows)
    once(output/'HELDOUT_COMPLETE.json',{'config_sha256':digest(config),'definition_sha256':digest(definition),'calibration_lock_sha256':digest(lock),'metrics':{t:digest(r) for t,r in rows_by_target.items()}})
    return rows_by_target
