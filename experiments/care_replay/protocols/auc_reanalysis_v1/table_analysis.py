"""Offline table-level analysis primitives; preserve original selection and contrasts.
No optimizer execution and no new selection. Functions do not read model credentials.
"""
import offline
import math,random,hashlib,json
from statistics import mean
from normalization import reference,score,paired_delta,bind_reference

def pool_binding(ids,values):
    if len(ids)!=len(values) or len(ids)!=len(set(ids)):raise ValueError('Pool ID uniqueness/length')
    raw=(json.dumps(list(zip(ids,values)),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return hashlib.sha256(raw).hexdigest()

def transform_rows(pool,rows,initials,initial,rounds):
    ids=pool['ordered_ids'];values=pool['values'];positions={x:i for i,x in enumerate(ids)}
    bound=pool_binding(ids,values)
    if bound!=pool['certified_binding']:raise ValueError('Uncertified/mutated pool')
    full=bind_reference(reference(values,initial,rounds),bound);conditional={};out=[];keys=set()
    for row in rows:
        seed=str(row['seed']);mode=row['mode'];key=(seed,mode)
        if key in keys:raise ValueError('Duplicate seed/mode; aliases must be separately named')
        keys.add(key)
        if row['initial']!=initial or row['rounds']!=rounds:raise ValueError('Row budget differs')
        if row['pool_binding']!=bound:raise ValueError('Row belongs to another pool')
        start=initials[seed]
        if len(start)!=initial or len(set(start))!=initial:raise ValueError('Invalid exact initial design')
        if seed not in conditional:
            cond=reference(values,initial,rounds,[positions[x] for x in start])
            conditional[seed]=bind_reference(cond,bound)
        auc=float(row['auc']);out.append({**row,'raw_auc':auc,'full_process_score':score(auc,full),
            'conditional_same_initial_score':score(auc,conditional[seed]),'initial_IDs':start,
            'full_process_headroom':full['headroom'],'conditional_headroom':conditional[seed]['headroom'],
            'near_saturation':full['near_saturation']})
    return {'full_reference':full,'conditional_references':conditional,'rows':out}

def transform_existing_interval(mean_delta,low,high,ref):
    """Scale an existing interval, without changing its resampling or coverage level."""
    if not all(math.isfinite(float(x)) for x in [mean_delta,low,high]) or low>high:raise ValueError('Invalid interval')
    if ref['zero_headroom']:return {'mean':None,'low':None,'high':None,'reason':'zero_headroom'}
    if ref['headroom']<=0:raise ValueError('Invalid headroom')
    return dict(mean=mean_delta/ref['headroom'],low=low/ref['headroom'],high=high/ref['headroom'])

def equal_group_summary(route_effects,seed=202610218101,replicates=10000):
    """13-group descriptive bootstrap: sample entire group means, equal group weight.
    Route effects are already paired heldout means. No iid pooling of trajectories.
    Missing denominators propagate to summary; they are never silently dropped.
    """
    by={};seen=set()
    for row in route_effects:
        if row['route'] in seen:raise ValueError('Duplicate route')
        seen.add(row['route']);by.setdefault(row['group'],[]).append(row['delta_score'])
    if not by:raise ValueError('No groups')
    if any(v is None for group in by.values() for v in group):
        return {'mean':None,'low':None,'high':None,'reason':'contains_unidentified_zero_headroom_route','groups':len(by),'routes':len(seen)}
    if any(not math.isfinite(v) for group in by.values() for v in group):raise ValueError('Nonfinite effect')
    vals=[mean(by[g]) for g in sorted(by)];rng=random.Random(seed)
    if type(replicates) is not int or replicates<2:raise ValueError('Invalid bootstrap count')
    samples=sorted(mean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(replicates))
    def percentile(p):
        index=(len(samples)-1)*p;i=int(index);frac=index-i
        return samples[i]*(1-frac)+samples[min(i+1,len(samples)-1)]*frac
    return {'mean':mean(vals),'low':percentile(.025),'high':percentile(.975),'groups':len(by),'routes':len(seen),'bootstrap_seed':seed,'bootstrap_replicates':replicates,'interval':'descriptive percentile 95%; group means only'}
