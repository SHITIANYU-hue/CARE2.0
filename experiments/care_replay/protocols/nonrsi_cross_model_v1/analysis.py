"""Exact random reference and conditional paired library x seed bootstrap."""
import numpy as np
from .common import Halt

from protocols.auc_reanalysis_v1.normalization import reference as stable_reference,score

def random_reference(values,initial=5,rounds=10,initial_indices=None):
    r=stable_reference(values,initial,rounds,initial_indices)
    return dict(r,zero_denominator=r['zero_headroom'],near_zero_warning=r['near_saturation'])

def normalize_delta(delta,reference):
    return None if reference['headroom']==0 else np.asarray(delta)/reference['headroom']

def bootstrap(delta,library_draws,seed_draws,confidence=1-.05/36):
    x=np.asarray(delta,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():raise Halt('Expected finite library-by-seed array')
    ld=np.asarray(library_draws);sd=np.asarray(seed_draws)
    if ld.ndim!=2 or sd.ndim!=2 or len(ld)!=len(sd) or ld.shape[1]!=x.shape[0] or sd.shape[1]!=x.shape[1] or not 0<confidence<1:raise Halt('Bootstrap draws mismatch')
    if not np.issubdtype(ld.dtype,np.integer) or not np.issubdtype(sd.dtype,np.integer) or ld.min()<0 or ld.max()>=x.shape[0] or sd.min()<0 or sd.max()>=x.shape[1]:raise Halt('Invalid resampling indices')
    samples=np.empty(len(ld))
    for start in range(0,len(ld),256):
        a,b=ld[start:start+256],sd[start:start+256]
        samples[start:start+len(a)]=x[a[:,:,None],b[:,None,:]].mean(axis=(1,2))
    tail=(1-confidence)/2;lo,hi=np.quantile(samples,[tail,1-tail])
    return {'mean':float(x.mean()),'low':float(lo),'high':float(hi),'confidence':confidence,'replicates':len(ld)}
