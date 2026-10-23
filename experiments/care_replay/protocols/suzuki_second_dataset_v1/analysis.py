"""Prespecified conditional random controls and all-pair label contrasts."""
import numpy as np

def summarize_tensor(matrix,config,seed_offset=0,conditional=False):
    # [parent, randomized bank, shared heldout seed]; label/anchor comparisons have r=1.
    m=np.asarray(matrix,dtype=float)
    if m.ndim!=3 or not np.isfinite(m).all(): raise ValueError("Finite parent x replicate x seed tensor required")
    n,r,s=m.shape
    if n==0: return {"status":"unavailable_no_valid_labeled_parent","stable_advantage":False}
    if s!=100: raise ValueError("Exactly 100 heldout columns required")
    out={"status":"estimated","parent_count":n,"replicates_per_parent":r,"seed_count":s,
         "mean_delta":float(m.mean()),"parent_means":m.mean(axis=(1,2)).tolist(),
         "heldout_block_means":[float(b.mean()) for b in np.split(m,5,axis=2)],
         "conditional_on_valid_labeled_parents":conditional,"conditional_on_frozen_calibration":True}
    if n==1 and conditional:
        return {**out,"status":"descriptive_single_valid_parent","interval_adjusted":None,"stable_advantage":False}
    rng=np.random.default_rng(config["bootstrap_seed"]+seed_offset)
    values=np.empty(config["bootstrap_replicates"])
    for b in range(len(values)):
        parents=rng.integers(0,n,n);seeds=rng.integers(0,s,s)
        reps=rng.integers(0,r,size=(n,r))
        sampled=m[parents[:,None,None],reps[:,:,None],seeds[None,None,:]]
        values[b]=sampled.mean()
    alpha=1-config["analysis"]["confidence_level"]
    out["interval_adjusted"]=np.quantile(values,[alpha/2,1-alpha/2]).tolist()
    out["interval_95_descriptive"]=np.quantile(values,[.025,.975]).tolist()
    out["confidence_level"]=config["analysis"]["confidence_level"]
    out["stable_advantage"]=out["interval_adjusted"][0]>0 and all(x>0 for x in out["heldout_block_means"])
    return out

def analyze(rows,choices,outcomes,anchor,config):
    seeds=list(range(config["heldout_seed_start"],config["heldout_seed_start"]+100))
    table={(row["mode"],row["seed"]):row for row in rows}
    if len(table)!=len(rows): raise ValueError("Duplicate heldout cells")
    def vector(bank,key="primary_mode",metric="best_so_far_auc"):
        return np.array([table[(choices[bank][key],seed)][metric] for seed in seeds])
    valid=[i for i in range(10) if outcomes[f"pair_{i:02d}_labeled"]["status"]=="VALID"]
    label=np.stack([vector(f"pair_{i:02d}_labeled")-vector(f"pair_{i:02d}_anonymous") for i in range(10)])[:,None,:]
    primary={"labeled_minus_anonymous_all_pairs":summarize_tensor(label,config,0)}
    secondary={}
    for fi,family in enumerate(("r1","r2")):
        differences=np.stack([np.stack([vector(f"pair_{i:02d}_labeled")-vector(f"{family}_pair_{i:02d}_rep_{j:02d}") for j in range(10)]) for i in valid]) if valid else np.empty((0,10,100))
        primary["labeled_minus_"+family+"_valid_parents"]=summarize_tensor(differences,config,fi+1,True)
        secondary["gated_labeled_minus_"+family]={"mean_delta":float(np.mean([vector(f"pair_{i:02d}_labeled","selected_mode")-vector(f"{family}_pair_{i:02d}_rep_{j:02d}","selected_mode") for i in valid for j in range(10)])) if valid else None}
    anchorvec=np.array([table[(anchor,seed)]["best_so_far_auc"] for seed in seeds])
    for arm in ("labeled","anonymous"):
        for key in ("primary_mode","selected_mode"):
            v=np.stack([vector(f"pair_{i:02d}_{arm}",key) for i in range(10)])
            secondary[arm+"_"+key]={"mean_auc":float(v.mean()),"mean_delta_anchor":float((v-anchorvec).mean())}
    secondary["gated_labeled_minus_anonymous"]={"mean_delta":float(np.mean([vector(f"pair_{i:02d}_labeled","selected_mode")-vector(f"pair_{i:02d}_anonymous","selected_mode") for i in range(10)]))}
    for metric in ("final_best","top10_hit"):
        secondary["label_"+metric]={"mean_delta":float(np.mean([vector(f"pair_{i:02d}_labeled",metric=metric)-vector(f"pair_{i:02d}_anonymous",metric=metric) for i in range(10)]))}
    return {"primary":primary,"secondary_descriptive":secondary,"valid_labeled_parent_indices":valid,
        "valid_counts":{arm:sum(outcomes[f"pair_{i:02d}_{arm}"]["status"]=="VALID" for i in range(10)) for arm in ("labeled","anonymous")},
        "gate_acceptance":{family:sum(v["selected_mode"]!=anchor for k,v in choices.items() if v["family"]==family) for family in ("labeled","anonymous","r1","r2")},
        "anchor":anchor,"anchor_mean_auc":float(anchorvec.mean()),"auc_units":"mean best observed yield percentage points over reveal rounds",
        "interpretation":config["analysis"]["scope"],"multiplicity":"three Bonferroni-adjusted two-sided bootstrap intervals; secondary descriptive only"}
