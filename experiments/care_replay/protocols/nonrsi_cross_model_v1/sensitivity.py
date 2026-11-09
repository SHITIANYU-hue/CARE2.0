"""Synthetic coverage/power checks. No real outcomes or model responses are used."""
from .common import *
import numpy as np
from scipy.stats import beta

def binomial_interval(k,n,confidence=.95):
    q=(1-confidence)/2
    return [0. if k==0 else float(beta.ppf(q,k,n-k+1)),1. if k==n else float(beta.ppf(1-q,k+1,n-k))]

def run(config,output,trials=1000):
    rng=np.random.default_rng(config['sensitivity_seed']);B=config['statistics_proposed']['bootstrap_replicates'];alpha=.05/36
    records=[]
    # Library-dominated random effects with identical trajectories over the 100 seeds.
    # This is an exact reduction of the proposed crossed bootstrap, not a shortcut
    # that drops a random component present in these synthetic scenarios.
    for size in [10,20]:
        draws=rng.integers(0,size,(B,size));weights=np.zeros((B,size))
        for i in range(size):weights[:,i]=(draws==i).sum(axis=1)/size
        for scenario in ['normal','centered_exponential','normal_with_20pct_invalid']:
            if scenario=='normal':data=rng.normal(size=(trials,size))
            elif scenario=='centered_exponential':data=rng.exponential(size=(trials,size))-1
            else:data=rng.normal(size=(trials,size))*(rng.random((trials,size))>=.2)
            lows=[];highs=[];lows95=[];highs95=[]
            for start in range(0,trials,8):
                boot=weights@data[start:start+8].T
                q=np.quantile(boot,[alpha/2,1-alpha/2,.025,.975],axis=0)
                lows.extend(q[0]);highs.extend(q[1]);lows95.extend(q[2]);highs95.extend(q[3])
            lo=np.array(lows);hi=np.array(highs);k=int(np.sum((lo<=0)&(hi>=0)))
            k95=int(np.sum((np.array(lows95)<=0)&(np.array(highs95)>=0)))
            records.append({'libraries':size,'scenario':scenario,'trials':trials,'bootstrap_replicates_each':B,'nominal_adjusted_coverage':1-alpha,'coverage':k/trials,'coverage_mc95':binomial_interval(k,trials),'nominal_unadjusted_coverage':.95,'coverage95':k95/trials,'positive_detection_under_constant_shift':{str(delta):float(np.mean(lo+delta>0)) for delta in [.25,.5,1.]},'shift_unit':'unit-scale library effect, not a claimed real task effect; invalid-mixture shifts are sensitivity only','median_interval_width':float(np.median(hi-lo))})
            print(json.dumps(records[-1]),flush=True)
    report={'kind':'synthetic_pre_freeze_diagnostic','model_api_calls':0,'real_effects_used':False,'seed':config['sensitivity_seed'],'records':records,'interpretation':'Coverage is assessed under chosen synthetic scenarios; does not establish coverage for all real distributions. 20-library rows are diagnostic only and do not change approved sample size.'}
    save(output,report);return report
