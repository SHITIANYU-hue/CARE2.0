"""Offline public adapter, restricted ask/tell policy and immutable checkpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import ssl, socket

def block_network(*args, **kwargs):
    raise RuntimeError('Experiment 3 forbids network and model API access')
socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network
socket.socket.sendto = block_network

import sys, json, hashlib, gzip, random, math
from pathlib import Path
from dataclasses import replace
from statistics import mean
import numpy as np
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = BASE.parents[1]
PANEL = BASE / 'outputs/experiment3_panel_amendment_v1'
PUBLIC = BASE / 'outputs/experiment3_recruitment_v1/public'
OUT = BASE / 'outputs/experiment3_execution_v1'
sys.path.insert(0, str(BASE/'scripts'))
import run_synthetic_suzuki as replay
import run_classical_transfer_baselines as classic
import run_multisource_transfer_baselines as multi
from score_units import SourceScoreUnits

def canonical(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def object_hash(x): return hashlib.sha256(canonical(x)).hexdigest()

def write_once(path, payload):
    """Atomic idempotent record; refuse to overwrite a different result."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=canonical(payload)
    if path.exists():
        if path.read_bytes()!=content: raise RuntimeError(f'Immutable record differs: {path}')
        return
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temp.open('xb') as f:
        f.write(content);f.flush();os.fsync(f.fileno())
    os.chmod(temp,0o600)
    try: os.link(temp,path)
    except FileExistsError:
        if path.read_bytes()!=content: raise RuntimeError(f'Concurrent record differs: {path}')
    finally: temp.unlink()

def save_checkpoint(path, context, result):
    record={'context':context,'result':result}
    write_once(path, {'payload':record,'sha256':object_hash(record)})

def read_checkpoint(path, context):
    if not Path(path).exists(): return None
    record=json.loads(Path(path).read_text())
    if object_hash(record['payload'])!=record['sha256']: raise RuntimeError('Checkpoint corruption')
    if record['payload']['context']!=context: raise RuntimeError('Checkpoint context mismatch')
    return record['payload']['result']

def config(): return json.loads((HERE/'config.json').read_text())
def routes(): return json.loads((PANEL/'public/routes.json').read_text())
def membership(route):
    p=PANEL/route['memberships_file']
    if digest(p)!=route['memberships_sha256']: raise RuntimeError('Membership changed')
    with gzip.open(p,'rt') as f: m=json.load(f)
    fit,cal,target=map(lambda k:set(m[k]),('fit','calibration','target'))
    assert len(m['fit'])==len(fit)==1024 and len(m['target'])==len(target)==route['target_n']
    assert not (fit&cal or fit&target or cal&target)
    assert set(route['G0_selected_ids'])<=cal and len(route['G0_selected_ids'])==128
    if route['G1']['status']=='matched':
        assert len(route['G1']['selected_ids'])==128 and set(route['G1']['selected_ids'])<=cal
    else: assert route['G1']['selected_ids']==[]
    return m

def adapter(name, pool):
    return replay.DatasetAdapter(name,name,'maximize processed DMS_score',replay.FLIP2_IRED_DECISION_COLUMNS,
         'DMS_score','mutation_count_bin',(),'',tuple(pool))

def build_public(assay, ids):
    """Projection contains no outcome fields; verify reconstructed WT variants."""
    meta=next(x for x in json.loads((PUBLIC/'assay_metadata.json').read_text()) if x['DMS_id']==assay)
    reference=meta['target_seq']; wanted=set(ids);found={}
    with gzip.open(PUBLIC/'sequence_projection'/f'{assay}.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);sid=row['sequence_sha256']
            if sid not in wanted: continue
            sequence=list(reference)
            tokens=row['mutant'].split(':') if row['mutant'] else []
            for token in tokens:
                pos=int(token[1:-1])-1
                if reference[pos]!=token[0]: raise ValueError('WT token mismatch')
                sequence[pos]=token[-1]
            sequence=''.join(sequence)
            if hashlib.sha256(sequence.encode()).hexdigest()!=sid: raise ValueError('Sequence digest mismatch')
            metadata,numeric=replay.flip2_ired_public_features(sequence,reference)
            if metadata['mutation_tokens']!=tokens: raise ValueError('Mutation adapter mismatch')
            if len(tokens)!=row['mutation_order']: raise ValueError('Order mismatch')
            if sid in found: raise ValueError('Duplicate sequence')
            found[sid]=replay.Candidate(sid,metadata['mutation_count_bin'],*numeric[:3],float('nan'),metadata,numeric)
    if set(found)!=wanted: raise ValueError('Missing public candidates')
    return found

class Reveals:
    """Policy receives only a callable for one selected index, not oracle arrays."""
    def __init__(self, ids, values, units, budget):
        self.ids=tuple(ids); self._values=values; self._units=units
        self.budget=budget;self.calls=[]
    def __call__(self,index):
        if not isinstance(index,int) or not 0<=index<len(self.ids): raise IndexError(index)
        if index in self.calls or len(self.calls)>=self.budget: raise RuntimeError('Reveal budget or uniqueness violated')
        self.calls.append(index)
        return float(self._units.revealed_to_executor(self._values[self.ids[index]]))

def fit_source(public, fit_ids, raw_values):
    units=SourceScoreUnits.fit([raw_values[i] for i in fit_ids])
    transformed=units.revealed_to_executor([raw_values[i] for i in fit_ids])
    source=[replace(public[i],objective_value=float(y)) for i,y in zip(fit_ids,transformed)]
    return units,source

def choose(features, observed, revealed, prior, mode, round_index, cfg):
    # Same normalization and exact pure GP implementation as legacy replay.
    values=np.asarray(revealed,dtype=np.float64)/100.
    normalized=(values-float(np.mean(values)))/max(float(np.std(values)),.05)
    mu,var,_=classic.target_gp_posterior(features,observed,normalized,
        cfg['numeric_length_scale'],cfg['categorical_length_scale'],cfg['gp_noise'])
    scores=mu+cfg['gp_beta']*np.sqrt(var)
    if mode=='Always':
        rank=np.asarray(list(multi.classical_rank_normalized(scores).values()),dtype=np.float64)
        mass=multi.scheduled_mass(cfg['mass_start'],cfg['mass_end'],round_index,cfg['rounds'])
        scores=(1-mass)*rank+mass*prior
    elif mode!='Never': raise ValueError(mode)
    return classic.top_unobserved(scores,set(observed))

def episode(pool, features, prior, reveal, seed, mode, cfg):
    # No access to any candidate objective_value, including initial observations.
    indices=list(range(len(pool)));random.Random(seed).shuffle(indices)
    observed=indices[:cfg['initial']];values=[reveal(i) for i in observed]
    best=[]
    for t in range(cfg['rounds']):
        selected=choose(features,observed,values,prior,mode,t,cfg)
        observed.append(selected);values.append(reveal(selected));best.append(max(values))
    return {'seed':seed,'mode':mode,'observed_indices':observed,'revealed_executor_values':values,
        'best_trace':best,'best_so_far_auc':round(mean(best),6),'auc_full_precision':mean(best),
        'final_best':round(max(values),6),'observations':len(observed)}

def paired_episodes(pool, source, raw_values, units, seed, cfg, prepared=None):
    if prepared is None:
        features=classic.feature_arrays(adapter('pool',pool),pool)
        prior,_=multi.additive_mutation_prior([source],pool)
    else: features,prior=prepared
    ids=[c.candidate_id for c in pool];out={}
    for mode in ('Always','Never'):
        reveal=Reveals(ids,raw_values,units,cfg['initial']+cfg['rounds'])
        out[mode]=episode(pool,features,prior,reveal,seed,mode,cfg)
        assert len(reveal.calls)==cfg['initial']+cfg['rounds']
    assert out['Always']['observed_indices'][:3]==out['Never']['observed_indices'][:3]
    return out

def gate(pairs):
    rows=[]
    for pair in pairs:
        for mode in ('Always','Never'):
            rows.append({**pair[mode],'top10_hit':0}) # not a gate endpoint; never query full oracle
    comparison=classic.paired_comparison(rows,'Always','Never')['best_so_far_auc']
    return {'accept':comparison['normal_95ci_low']>0,'paired_seeds':len(pairs),'auc_executor_units':comparison}

def mean_ci(deltas):
    x=np.asarray(deltas,dtype=float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Invalid analysis values')
    m=float(x.mean());h=1.96*float(x.std(ddof=0))/math.sqrt(len(x))
    return {'mean':m,'low':m-h,'high':m+h,'n':len(x)}

def truth(ci,delta):
    if ci['low']>delta:return 'beneficial'
    if ci['high']< -delta:return 'harmful'
    if ci['low']>=-delta and ci['high']<=delta:return 'near_zero'
    return 'uncertain'

def verify_mapping(mapping,root):
    for rel,expected in mapping.items():
        if digest(root/rel)!=expected: raise RuntimeError('Frozen input changed: '+rel)

def verify_freeze():
    path=HERE/'FREEZE.json'
    expected=(HERE/'FREEZE.sha256').read_text().strip()
    if digest(path)!=expected: raise RuntimeError('Freeze hash mismatch')
    lock=json.loads(path.read_text());verify_mapping(lock['artifacts'],BASE)
    if lock['status']!='execution_frozen_before_effects': raise RuntimeError('Not frozen')
    return expected

def verify_gate_barrier(freeze_sha):
    path=OUT/'GATES_LOCK.json'
    if not path.exists(): raise RuntimeError('Heldout blocked: all gates must be locked')
    if digest(path)!=(OUT/'GATES_LOCK.sha256').read_text().strip():raise RuntimeError('Gate lock changed')
    lock=json.loads(path.read_text())
    if lock['freeze_sha256']!=freeze_sha: raise RuntimeError('Wrong gate freeze')
    if set(lock['routes'])!={r['route_id'] for r in routes()}: raise RuntimeError('Incomplete gate barrier')
    verify_mapping(lock['artifacts'],OUT)
    return lock
