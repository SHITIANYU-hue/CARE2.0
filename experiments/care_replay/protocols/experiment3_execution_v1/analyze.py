"""Prespecified all-route analysis, paired seeds and dependency-group bootstrap."""
from core import *
from collections import Counter,defaultdict
import csv,io

def rate(rows, label, gate_name):
    selected=[r for r in rows if r['truth']==label]
    if not selected:return {'numerator':0,'denominator':0,'rate':None}
    wrong=sum((not r[gate_name]) if label=='beneficial' else r[gate_name] for r in selected)
    return {'numerator':wrong,'denominator':len(selected),'rate':wrong/len(selected)}

def bootstrap(rows,cfg):
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['group']].append(i)
    groups=sorted(grouped);rng=np.random.default_rng(cfg['bootstrap_seed'])
    keys=['G0_minus_Never','G1_minus_Never','Always_minus_Never','G0_minus_Always','G1_minus_Always','G1_minus_G0']
    draws={k:[] for k in keys};draws.update({'false_rejection_G1_minus_G0':[],'false_acceptance_G1_minus_G0':[]})
    equal={k:[] for k in keys}
    for _ in range(cfg['bootstrap_replicates']):
        sampled=rng.integers(0,len(groups),size=len(groups))
        indices=[i for g in sampled for i in grouped[groups[g]]]
        resampled=[rows[i] for i in indices]
        for key in keys:
            draws[key].append(float(np.mean([r[key] for r in resampled])))
            equal[key].append(float(np.mean([np.mean([rows[i][key] for i in grouped[groups[g]]]) for g in sampled])))
        for label,key in [('beneficial','false_rejection_G1_minus_G0'),('harmful','false_acceptance_G1_minus_G0')]:
            a,b=rate(resampled,label,'G0')['rate'],rate(resampled,label,'G1')['rate']
            if a is not None:draws[key].append(b-a)
    def summarize(values, estimate):
        if not values:return {'estimate':estimate,'low':None,'high':None,'valid_bootstrap_replicates':0}
        lo,hi=np.quantile(values,[.025,.975],method='linear')
        return {'estimate':estimate,'low':float(lo),'high':float(hi),'valid_bootstrap_replicates':len(values)}
    out={key:summarize(draws[key],float(np.mean([r[key] for r in rows]))) for key in keys}
    for label,key in [('beneficial','false_rejection_G1_minus_G0'),('harmful','false_acceptance_G1_minus_G0')]:
        a,b=rate(rows,label,'G0')['rate'],rate(rows,label,'G1')['rate']
        out[key]=summarize(draws[key],None if a is None else b-a)
    out['equal_group_weight_sensitivity']={key:summarize(equal[key],float(np.mean([np.mean([rows[i][key] for i in grouped[g]]) for g in groups]))) for key in keys}
    out['group_count']=len(groups);out['replicates']=cfg['bootstrap_replicates'];out['method']='whole dependency-group percentile bootstrap, conditional on frozen panel and estimated route means'
    return out

def analyze_rows(records,cfg):
    counts=dict(Counter(r['truth'] for r in records))
    return {'routes':len(records),'truth_counts':counts,
        'G0':{'accepted':sum(r['G0'] for r in records),'false_rejection':rate(records,'beneficial','G0'),'false_acceptance':rate(records,'harmful','G0')},
        'G1':{'accepted':sum(r['G1'] for r in records),'false_rejection':rate(records,'beneficial','G1'),'false_acceptance':rate(records,'harmful','G1')},
        'paired_cluster_analysis':bootstrap(records,cfg)}

def main():
    freeze_sha=verify_freeze();verify_gate_barrier(freeze_sha);cfg=config();rows=[];excluded=[];allfiles={}
    for r in routes():
        rid=r['route_id'];gp=OUT/'gates'/f'{rid}.json';hp=OUT/'heldout'/f'{rid}.json'
        g=json.loads(gp.read_text());h=json.loads(hp.read_text())
        if h['freeze_sha256']!=freeze_sha:raise RuntimeError('Heldout provenance mismatch')
        allfiles[str(hp.relative_to(OUT))]=digest(hp)
        if h['status']=='excluded_zero_source_iqr':excluded.append(rid);continue
        if h['gate_record_sha256']!=digest(gp):raise RuntimeError('Gate changed during heldout')
        verify_mapping(h['episode_artifacts'],OUT);allfiles.update(h['episode_artifacts'])
        m=membership(r);pairs=[]
        for seed in cfg['seeds']['heldout']:
            context={'freeze_sha256':freeze_sha,'route':rid,'pool':'heldout','seed':seed,'pool_ids_sha256':object_hash(m['target'])}
            pair=read_checkpoint(OUT/'episodes'/rid/'heldout'/f'{seed}.json',context)
            if pair is None:raise RuntimeError('Missing heldout episode')
            for mode in ('Always','Never'):
                x=pair[mode];vals=x['revealed_executor_values'];indices=x['observed_indices']
                if len(vals)!=15 or len(set(indices))!=15:raise RuntimeError('Observation audit failed')
                expected=[max(vals[:3+t+1]) for t in range(12)]
                if x['best_trace']!=expected or x['auc_full_precision']!=mean(expected):raise RuntimeError('AUC audit failed')
            if pair['Always']['observed_indices'][:3]!=pair['Never']['observed_indices'][:3]:raise RuntimeError('Pairing audit failed')
            pairs.append(pair)
        delta=np.asarray([(p['Always']['auc_full_precision']-p['Never']['auc_full_precision'])/100. for p in pairs]);ci=mean_ci(delta)
        a0=bool(g['G0']['accept']);a1=bool(g['G1']['accept'])
        row={'route':rid,'assay':r['assay'],'kind':r['kind'],'group':r['candidate_dependency_group'],'target_n':r['target_n'],
            'G1_support':r['G1']['status'],'G0':a0,'G1':a1,'source_iqr':g['units']['source_iqr'],'delta_mean':ci['mean'],'delta_low':ci['low'],'delta_high':ci['high'],
            'truth':truth(ci,cfg['delta']),'Always_minus_Never':ci['mean'],'G0_minus_Never':int(a0)*ci['mean'],'G1_minus_Never':int(a1)*ci['mean'],
            'G0_minus_Always':(int(a0)-1)*ci['mean'],'G1_minus_Always':(int(a1)-1)*ci['mean'],'G1_minus_G0':(int(a1)-int(a0))*ci['mean'],
            'Never_auc_source_iqr':float(np.mean([p['Never']['auc_full_precision']/100 for p in pairs])),
            'Always_auc_source_iqr':float(np.mean([p['Always']['auc_full_precision']/100 for p in pairs])),
            'G0_descriptive_oracle_loss':max(0.,ci['mean'])-int(a0)*ci['mean'],
            'G1_descriptive_oracle_loss':max(0.,ci['mean'])-int(a1)*ci['mean']}
        rows.append(row)
    result={'status':'complete','freeze_sha256':freeze_sha,'gate_lock_sha256':digest(OUT/'GATES_LOCK.json'),'api_calls':0,'api_cost_usd':0,
        'excluded_zero_iqr':excluded,'primary':analyze_rows(rows,cfg),'routes':rows,'sensitivity':{},
        'strata':{},'limitations':['within-assay transfers; not cross-protein','23 routes are not 23 independent families; 13 candidate dependency groups',
            'finite frozen pools and simulator seeds; not new wet lab observations','G1 unsupported routes fall back to Never and stay in analysis',
            'classification rates condition on noisy heldout truth categories; near-zero and uncertain remain visible',
            'bootstrap intervals conditional on candidate grouping; remote homology independence not certified','same-holdout oracle loss is descriptive and selection-biased',
            'no LLM used; cannot support LLM knowledge superiority','pointwise intervals, no multiplicity-adjusted route significance claims']}
    for margin in cfg['sensitivity_delta']:
        altered=[{**r,'truth':truth({'low':r['delta_low'],'high':r['delta_high']},margin)} for r in rows]
        result['sensitivity'][str(margin)]={'truth_counts':dict(Counter(r['truth'] for r in altered)),
            'G0':{'false_rejection':rate(altered,'beneficial','G0'),'false_acceptance':rate(altered,'harmful','G0')},
            'G1':{'false_rejection':rate(altered,'beneficial','G1'),'false_acceptance':rate(altered,'harmful','G1')}}
    for kind in sorted(set(r['kind'] for r in rows)):
        sub=[r for r in rows if r['kind']==kind]
        result['strata'][kind]={'n':len(sub),'truth_counts':dict(Counter(r['truth'] for r in sub)),
            'G0_accepted':sum(r['G0'] for r in sub),'G1_accepted':sum(r['G1'] for r in sub),
            'G1_minus_G0_mean':float(np.mean([r['G1_minus_G0'] for r in sub]))}
    write_once(OUT/'analysis.json',result)
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table=OUT/'route_results.csv';content=buf.getvalue()
    if table.exists() and table.read_text()!=content.replace('\r\n','\n'):raise RuntimeError('Table changed')
    if not table.exists():table.write_text(content)
    allfiles['analysis.json']=digest(OUT/'analysis.json');allfiles['route_results.csv']=digest(table)
    write_once(OUT/'RESULTS_MANIFEST.json',{'freeze_sha256':freeze_sha,'gate_lock_sha256':digest(OUT/'GATES_LOCK.json'),'artifacts':allfiles,'completed_routes':len(rows),'excluded_routes':excluded,'api_calls':0})
    print(json.dumps({k:result[k] for k in ('status','primary','strata','api_calls')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
