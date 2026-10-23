"""Finite-pool normalization. No model, optimizer, file I/O or network imports."""
import math
from statistics import mean

def values_checked(values):
    x=tuple(float(v) for v in values)
    if not x or not all(math.isfinite(v) for v in x): raise ValueError('Empty or nonfinite pool')
    if not math.isfinite(max(x)-min(x)): raise ValueError('Nonfinite pool range')
    return x

def positive_int(k,name):
    if type(k) is not int or k<1: raise ValueError(name+' must be positive integer')

def expected_max_gap(values,k):
    """Return max(values)-E[max of k uniform draws without replacement].
    Accumulate nonnegative gaps, avoiding cancellation against a large offset.
    Descending recurrence: C(j,k)/C(N,k); C(N-1,k)/C(N,k)=(N-k)/N.
    """
    x=sorted(values_checked(values));positive_int(k,'k');n=len(x)
    if k>n: raise ValueError('Draw count exceeds pool')
    nonmax=sum(v<x[-1] for v in x)
    if k>nonmax: return 0.0  # Every draw contains a pool maximum.
    terms=[];p=(n-k)/n
    for j in range(n-1,k-1,-1):
        terms.append((x[j]-x[j-1])*p)
        p *= (j-k)/j
    gap=math.fsum(terms)
    if not math.isfinite(gap) or gap<=0: raise ArithmeticError('Positive headroom lost to numerical underflow')
    return gap

def reference(values,initial,rounds,initial_indices=None):
    x=values_checked(values);positive_int(initial,'initial');positive_int(rounds,'rounds')
    if initial+rounds>len(x): raise ValueError('Budget exceeds pool')
    oracle=max(x);span=oracle-min(x)
    if initial_indices is None:
        gaps=[expected_max_gap(x,k) for k in range(initial+1,initial+rounds+1)]
        kind='full_random_process'
    else:
        ids=tuple(initial_indices)
        if len(ids)!=initial or len(set(ids))!=initial: raise ValueError('Initial indices count/uniqueness')
        if any(type(i) is not int or not 0<=i<len(x) for i in ids): raise ValueError('Invalid initial index')
        seen=set(ids);floor=max(x[i] for i in ids)
        remaining=[max(floor,v) for i,v in enumerate(x) if i not in seen]
        # If the maximum is in the initial design, max(remaining)==floor==oracle.
        gaps=[expected_max_gap(remaining,k) for k in range(1,rounds+1)]
        kind='conditional_same_initial'
    headroom=math.fsum(gaps)/rounds
    return {'reference_kind':kind,'n':len(x),'initial':initial,'rounds':rounds,'oracle':oracle,
            'minimum':min(x),'random_auc':oracle-headroom,'headroom':headroom,
            'headroom_fraction':headroom/span if span>0 else None,
            'zero_headroom':all(g==0 for g in gaps),
            'near_saturation':span>0 and headroom>0 and headroom/span<0.01,
            'per_round_headroom':gaps}

def score(auc,ref):
    auc=float(auc)
    if not math.isfinite(auc): raise ValueError('Nonfinite AUC')
    if ref['zero_headroom']: return None
    if ref['headroom']<=0: raise ValueError('Invalid headroom')
    result=((auc-ref['oracle'])+ref['headroom'])/ref['headroom']
    if not math.isfinite(result): raise ArithmeticError('Nonfinite normalized score')
    return result

def paired_delta(a,b,ref_a,ref_b):
    if ref_a!=ref_b: raise ValueError('Comparisons require a common pool/budget/reference')
    if not all(math.isfinite(float(v)) for v in (a,b)): raise ValueError('Nonfinite AUC')
    return None if ref_a['zero_headroom'] else (a-b)/ref_a['headroom']

def trace_auc(values,initial,rounds):
    x=values_checked(values);positive_int(initial,'initial');positive_int(rounds,'rounds')
    if len(x)!=initial+rounds: raise ValueError('Observed trajectory length mismatch')
    return mean(max(x[:initial+t]) for t in range(1,rounds+1))

def bind_reference(ref,pool_hash):
    if not isinstance(pool_hash,str) or len(pool_hash)!=64: raise ValueError('Expected SHA256 pool binding')
    return dict(ref,pool_sha256=pool_hash)
