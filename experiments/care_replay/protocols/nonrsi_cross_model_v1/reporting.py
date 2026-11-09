"""Pre-specified per-task contrasts, with shared T libraries and seed draws."""
from .common import *
from . import analysis,pipeline,inputs
import numpy as np

METRICS=('best_so_far_auc','final_best','top10_hit')

def intervals(x,samples,primary=False):
    result={'mean':float(np.mean(x)),'units':'library generation x heldout seed; conditional on fixed calibration','bootstrap_replicates':len(samples),'interpretation':'Nominal bootstrap sensitivity interval; no calibrated coverage, significance or family-wise error control claim'}
    for name,confidence in [('nominal_95',.95)]+([('nominal_998611',1-.05/36)] if primary else []):
        tail=(1-confidence)/2;lo,hi=np.quantile(samples,[tail,1-tail]);result[name]={'nominal_mass':confidence,'low':float(lo),'high':float(hi)}
    return result

def samples(x,ld,sd):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():raise Halt('Invalid contrast array')
    if ld.shape[1]!=len(x) or sd.shape[1]!=x.shape[1] or len(ld)!=len(sd):raise Halt('Contrast resampling mismatch')
    out=np.empty(len(ld))
    for start in range(0,len(ld),256):
        a,b=ld[start:start+256],sd[start:start+256]
        out[start:start+len(a)]=x[a[:,:,None],b[:,None,:]].mean(axis=(1,2))
    return out

def summarize_delta(x,ld,sd,ref,primary=False,conditional=None):
    x=np.asarray(x,dtype=float);draw=samples(x,ld,sd);raw=intervals(x,draw,primary)
    normalized=None if ref['headroom']==0 else intervals(x/ref['headroom'],draw/ref['headroom'],primary)
    supplement=None
    if conditional is not None:
        gaps=np.asarray([r['headroom'] for r in conditional])
        zeros=int(np.sum(gaps==0))
        supplement={'zero_denominator_seeds':zeros,'total_seeds':len(gaps),'summary':None,'policy':'Any zero denominator propagates NA; no seed dropping'}
        if zeros==0:
            scaled=x/gaps[None,:];supplement['summary']=intervals(scaled,samples(scaled,ld,sd))
    return {'per_library_mean_delta_auc':[float(v) for v in x.mean(axis=1)],'per_library_mean_delta_score':None if ref['headroom']==0 else [float(v) for v in (x/ref['headroom']).mean(axis=1)],'raw_auc':raw,'delta_score':normalized,'same_initial_secondary':supplement,'near_zero_warning':ref['near_zero_warning']}

def nested_random_delta(parent,children,seed_draws,rng):
    # P valid parent libraries x C=10 random child libraries x shared heldout seeds.
    p,c,s=children.shape
    if parent.shape!=(p,s) or not np.isfinite(children).all() or not np.isfinite(parent).all():raise Halt('Nested array mismatch')
    out=np.empty(len(seed_draws))
    for start in range(0,len(out),128):
        sd=seed_draws[start:start+128];n=len(sd)
        pd=rng.integers(0,p,(n,p));cd=rng.integers(0,c,(n,p,c))
        random_means=children[pd[:,:,None],cd].mean(axis=2)
        delta=parent[pd]-random_means
        out[start:start+n]=np.take_along_axis(delta,sd[:,None,:],axis=2).mean(axis=(1,2))
    return out

def analyze(config,definition,lock,rows_by_target,output):
    if set(rows_by_target)!=set(definition['targets']) or set(lock['selections'])!=set(definition['targets']):raise Halt('Incomplete targets')
    if config['statistics_proposed'].get('scope')!='descriptive_cross_model_replication' or config['statistics_proposed'].get('familywise_error_control_claim') is not False:raise Halt('Descriptive scope must be explicitly frozen')
    n=config['statistics_proposed']['bootstrap_replicates'];count=config['replay_proposed']['heldout_seeds'];rng=np.random.default_rng(config['bootstrap_seed'])
    ld=rng.integers(0,10,(n,10));sd=rng.integers(0,count,(n,count))
    report={'inference':'Descriptive per-task/route replication; no pooled total win rate. Primary denotes preplanned reporting priority only. All generation slots include invalid-library fallback. Calibration and source fit fixed. No significance, equivalence, family-wise error control or statistically confirmed stability claims.', 'statistical_scope':'descriptive_cross_model_replication','primary':[],'secondary':[],'cells':{},'targets':{},'random_controls':[],'draws_sha256':hashlib.sha256(ld.astype('<i4').tobytes()+sd.astype('<i4').tobytes()).hexdigest()}
    matrices={};gated={};baseline={};references={};conditional={}
    for target,rows in rows_by_target.items():
        seeds=pipeline.seed_set(config,target,'heldout');by={(r['mode'],int(r['seed'])):r for r in rows}
        if len(by)!=len(rows):raise Halt('Duplicate heldout row')
        adapter=inputs.adapter(target);values=[c.objective_value for c in adapter.candidates]
        ref=analysis.random_reference(values);references[target]=ref
        conditional[target]=[analysis.random_reference(values,initial_indices=pipeline.initial_indices(adapter,seed)) for seed in seeds]
        anchor=lock['selections'][target]['anchor'];baseline[target]=np.array([float(by[(anchor,s)]['best_so_far_auc']) for s in seeds])
        report['targets'][target]={'reference':ref,'anchor':anchor,'fixed_BO_means':{m:{k:float(np.mean([by[(m,s)][k] for s in seeds])) for k in METRICS} for m in matched.BASELINES},'conditional_zero_seeds':sum(r['headroom']==0 for r in conditional[target])}
        for cell in [c for c in config['generation_cells'] if c['target']==target]:
            key=cell['model']+'__'+cell['cell'];names=[key+f'__{i:02d}' for i in range(10)]
            choice=lock['selections'][target]['choices'];primary_modes=[choice[name]['primary_mode'] for name in names];gate_modes=[choice[name]['selected_mode'] for name in names]
            x=np.array([[float(by[(m,s)]['best_so_far_auc']) for s in seeds] for m in primary_modes]);gx=np.array([[float(by[(m,s)]['best_so_far_auc']) for s in seeds] for m in gate_modes])
            matrices[key]=x;gated[key]=gx
            scores=None if ref['headroom']==0 else ((x-ref['oracle'])+ref['headroom'])/ref['headroom']
            report['cells'][key]={'target':target,'libraries':10,'valid_libraries':sum(definition['targets'][target]['outcomes'][name]=='VALID' for name in names),'gate_accepts':sum(m!=anchor for m in gate_modes),'mean_score':None if scores is None else float(scores.mean()),'primary_modes':primary_modes,'gated_modes':gate_modes,'raw_metric_means':{metric:float(np.mean([[by[(m,s)][metric] for s in seeds] for m in primary_modes])) for metric in METRICS}}
            report['cells'][key]['per_library']=[{
                'slot_id':name,'replicate':i,'status':definition['targets'][target]['outcomes'][name],
                'primary_mode':primary_modes[i],'gated_mode':gate_modes[i],
                'raw_metric_means':{metric:float(np.mean([by[(primary_modes[i],s)][metric] for s in seeds])) for metric in METRICS},
                'gated_raw_metric_means':{metric:float(np.mean([by[(gate_modes[i],s)][metric] for s in seeds])) for metric in METRICS},
                'mean_score':None if scores is None else float(scores[i].mean())
            } for i,name in enumerate(names)]
    def add(label,model,target,a,b,primary=False,collection='secondary'):
        delta=a-b
        result={'contrast':label,'model':model,'target':target,**summarize_delta(delta,ld,sd,references[target],primary,conditional[target] if primary else None)}
        report[collection].append(result)
    for cell in config['generation_cells']:
        if not cell['core_fixed_task']:continue
        k=cell['model']+'__'+cell['cell'];t=cell['target'];base=baseline[t][None,:]
        add(cell['cell']+' minus common BO',cell['model'],t,matrices[k],base,True,'primary')
        add('gated '+cell['cell']+' minus common BO',cell['model'],t,gated[k],base)
    for model in config['models']:
        m=model['id']
        for route in config['routes']:
            t=route['target'];tk=m+'__'+route['shared_target_cell'];sk=m+'__S_'+route['route_id'];fk=m+'__F_'+route['route_id']
            add(route['route_id']+' F minus T',m,t,matrices[fk],matrices[tk],True,'primary')
            add(route['route_id']+' S minus T',m,t,matrices[sk],matrices[tk])
            add(route['route_id']+' F minus S',m,t,matrices[fk],matrices[sk])
            add(route['route_id']+' gated F minus T',m,t,gated[fk],gated[tk])
    if len(report['primary'])!=36:raise Halt('Primary comparison family must contain exactly 36 contrasts')
    from itertools import combinations
    for left,right in combinations([m['id'] for m in config['models']],2):
        for cell in [c for c in config['generation_cells'] if c['model']==left and c['core_fixed_task']]:
            add('model difference '+cell['cell'],left+' minus '+right,cell['target'],matrices[left+'__'+cell['cell']],matrices[right+'__'+cell['cell']])
        for route in config['routes']:
            t=route['target'];name=route['route_id'];tc=route['shared_target_cell']
            add('model difference F-T '+name,left+' minus '+right,t,matrices[left+'__F_'+name]-matrices[left+'__'+tc],matrices[right+'__F_'+name]-matrices[right+'__'+tc])
    for cell in [c for c in config['generation_cells'] if c['cell'] in ['T_BH','T_Suzuki']]:
        target=cell['target'];by={(r['mode'],int(r['seed'])):r for r in rows_by_target[target]};seeds=pipeline.seed_set(config,target,'heldout');choices=lock['selections'][target]['choices']
        parents=[cell['model']+'__'+cell['cell']+f'__{i:02d}' for i in range(10)];parents=[p for p in parents if definition['targets'][target]['outcomes'][p]=='VALID']
        for family in ['r1','r2']:
            row={'model':cell['model'],'target':target,'family':family,'valid_parents':len(parents),'children_per_parent':10,'conditional_on_valid_parents':True,'summary':None}
            if parents:
                parent=np.array([[by[(choices[p]['primary_mode'],s)]['best_so_far_auc'] for s in seeds] for p in parents])
                children=[]
                for p in parents:
                    kids=sorted(k for k,v in definition['random_children'].items() if v['parent']==p and v['family']==family)
                    if len(kids)!=10:raise Halt('Missing random child libraries')
                    children.append([[by[(choices[k]['primary_mode'],s)]['best_so_far_auc'] for s in seeds] for k in kids])
                children=np.asarray(children);delta=parent-children.mean(axis=1)
                stream=np.random.default_rng(int(digest([config['bootstrap_seed'],cell['model'],target,family]),16))
                draw=nested_random_delta(parent,children,sd,stream)
                row['per_parent']=[{'parent':p,'parent_mean_auc':float(parent[i].mean()),'child_mean_aucs':[float(v) for v in children[i].mean(axis=1)],'mean_delta_auc':float(delta[i].mean())} for i,p in enumerate(parents)]
                row['summary']={'raw_auc':intervals(delta,draw),'delta_score':None if references[target]['headroom']==0 else intervals(delta/references[target]['headroom'],draw/references[target]['headroom']),'parent_cluster_warning':len(parents)<10}
            report['random_controls'].append(row)
    pipeline.once(Path(output)/'analysis.json',report)
    return report
