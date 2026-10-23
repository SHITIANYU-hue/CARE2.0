"""Offline pool provenance, full outcome export and trace consistency. No normalization."""
import offline
import os,sys,csv,io,gzip,json,math,random,hashlib,tarfile,zipfile,types,subprocess
from pathlib import Path
from collections import defaultdict
from statistics import mean
P=Path(__file__).resolve().parent;B=P.parents[1];ROOT=B.parents[1];OUT=B/'outputs/auc_reanalysis_v1';POOL=OUT/'pools'
INPUTS={};CERTS=[];GAPS=[]
def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):
 p=Path(p);x=p.read_bytes();INPUTS[str(p)]=sha(x);return x
def jd(p):return json.loads(read(p))
def checked(p,h):
 raw=read(p)
 if sha(raw)!=h:raise ValueError('Input hash mismatch: '+str(p))
 return raw
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=canonical(x)
 if p.exists() and p.read_bytes()!=data:raise ValueError('Refusing changed certification output: '+str(p))
 if not p.exists():p.write_bytes(data)
def git_read(path):
 ref='4c6f19f8cf62b25558f966604ce0b7644578f889:'+path
 q=subprocess.run(['git','show',ref],cwd=ROOT,capture_output=True,check=True)
 INPUTS['git:'+ref]=sha(q.stdout);return q.stdout

def module(data,label):
 name='auc_adapter_'+label;obj=types.ModuleType(name);obj.__file__=str(B/'scripts/run_synthetic_suzuki.py');sys.modules[name]=obj
 exec(compile(data,obj.__file__,'exec'),obj.__dict__)
 obj.ROOT=B;obj.RAW_DATA=B/'data/raw';obj.REACTION_DESCRIPTOR_PATH=B/'data/descriptors/reaction_component_descriptors.csv'
 def no_download(filename):
  path=obj.RAW_DATA/filename
  if not path.exists():raise FileNotFoundError(path)
  return path
 # Original builders use ensure_public_data; network guard remains a second barrier.
 for key in ['ensure_public_data','ensure_raw_file']:
  if hasattr(obj,key):setattr(obj,key,no_download)
 return obj

def pool_export(label,ids,values,metadata):
 if len(ids)!=len(set(ids)) or len(ids)!=len(values) or not all(math.isfinite(v) for v in values):raise ValueError('Pool schema')
 record={'id':label,'ordered_ids':ids,'values':values,'metadata':metadata}
 data_hash=sha(canonical(list(zip(ids,values))));ids_hash=sha(canonical(ids));path=POOL/(label+'.json');save(path,record)
 return {'pool_file':str(path.relative_to(B)),'pool_sha256':sha(path.read_bytes()),'ordered_ID_value_sha256':data_hash,'ordered_ids_sha256':ids_hash,'n':len(ids),'oracle':max(values),'minimum':min(values)}

def check_trace_groups(traces,ids,values,initial=5,rounds=10,metrics=None,initial_map=None):
 by=dict(zip(ids,values));groups=defaultdict(list)
 for x in traces:groups[(int(x['seed']),x['mode'])].append(x)
 checked_steps=0;checked_auc=0;initids={};fullaucs={}
 for (seed,mode),rows in groups.items():
  rows.sort(key=lambda x:x['round_index'])
  if [r['round_index'] for r in rows]!=list(range(rounds)):raise ValueError('Trace horizon '+str((seed,mode)))
  order=list(ids);random.Random(seed).shuffle(order);start=order[:initial];initids[str(seed)]=start
  if initial_map is not None and initial_map[str(seed)]!=start:raise ValueError('Recorded initial differs')
  observed=set(start);best=max(by[i] for i in start);trace=[]
  for t,row in enumerate(rows):
   cid=row['selected_candidate']
   if cid in observed or cid not in by:raise ValueError('Duplicate/unknown reveal')
   if 'public_observed_count' in row and int(row['public_observed_count'])!=initial+t:raise ValueError('Initial/horizon inconsistency')
   if abs(float(row['revealed_value'])-by[cid])>0.00005001:raise ValueError('Revealed value differs from reconstructed pool')
   observed.add(cid);best=max(best,by[cid]);trace.append(best)
   if abs(float(row['best_so_far'])-best)>0.00005001:raise ValueError('Best trace differs; pool or initialization invalid')
   checked_steps+=1
  auc=mean(trace);fullaucs[str(seed)+'|'+mode]=auc
  if metrics and (seed,mode) in metrics:
   if abs(float(metrics[(seed,mode)]['best_so_far_auc'])-auc)>0.00005001:raise ValueError('Archived AUC mismatch')
   checked_auc+=1
 return {'trajectories':len(groups),'steps':checked_steps,'metric_auc_checks':checked_auc,'seed_count':len(initids),'initial_ids_sha256':sha(canonical(initids))},initids,fullaucs

def certify_protein():
 old=B/'protocols/experiment3_execution_v1';freeze_raw=checked(old/'FREEZE.json',(old/'FREEZE.sha256').read_text().strip());lock=json.loads(freeze_raw)
 for rel,h in lock['artifacts'].items():checked(B/rel,h)
 cfg=jd(old/'config.json');panel=B/'outputs/experiment3_panel_amendment_v1';routes=jd(panel/'public/routes.json');public=B/'outputs/experiment3_recruitment_v1/public';meta={x['DMS_id']:x for x in jd(public/'assay_metadata.json')}
 resultdir=B/'outputs/experiment3_execution_v1';barrier_raw=checked(resultdir/'GATES_LOCK.json',(resultdir/'GATES_LOCK.sha256').read_text().strip());barrier=json.loads(barrier_raw)
 if barrier['freeze_sha256']!=sha(freeze_raw):raise ValueError('Gate/freeze mismatch')
 for rel,h in barrier['artifacts'].items():checked(resultdir/rel,h)
 bundles={};wanted=defaultdict(set)
 for route in routes:
  rid=route['route_id'];m=json.loads(gzip.decompress(checked(panel/route['memberships_file'],route['memberships_sha256'])))
  if len(m['target'])!=route['target_n']:raise ValueError('Target count')
  gate=jd(resultdir/'gates'/f'{rid}.json');bundles[rid]=(route,m,gate);wanted[route['assay']].update(m['target'])
 rawpath=B/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip';previous=rawpath.stat().st_mode&0o777;allvalues={}
 os.chmod(rawpath,0o400)
 try:
  # Stream hash rather than retaining the raw archive in memory.
  h=hashlib.sha256()
  with rawpath.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  if h.hexdigest()!=cfg['raw_zip_sha256']:raise ValueError('Protein raw archive changed')
  INPUTS[str(rawpath)]=h.hexdigest()
  with zipfile.ZipFile(rawpath) as z:
   for assay,ids in wanted.items():
    names=[n for n in z.namelist() if Path(n).name==meta[assay]['DMS_filename']]
    if len(names)!=1:raise ValueError('ZIP member ambiguity')
    found={}
    with z.open(names[0]) as f:
     for row in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')):
      sid=hashlib.sha256(row['mutated_sequence'].strip().upper().encode()).hexdigest()
      if sid not in ids:continue
      if sid in found:raise ValueError('Duplicate target sequence')
      value=float(row['DMS_score'])
      if not math.isfinite(value):raise ValueError('Nonfinite target')
      found[sid]=value
    if set(found)!=ids:raise ValueError('Missing target members')
    allvalues[assay]=found
 finally:os.chmod(rawpath,previous)
 for rid,(route,m,gate) in bundles.items():
  ids=m['target'];u=gate['units'];values=[100*((allvalues[route['assay']][i]-u['source_median'])/u['source_iqr']) for i in ids]
  export=pool_export(rid,ids,values,{'panel':'experiment3','raw_zip_sha256':cfg['raw_zip_sha256'],'source_units':u,'group':route['candidate_dependency_group'],'initial':3,'rounds':12})
  paths=list((resultdir/'episodes'/rid/'heldout').glob('*.json'));seeds=[];initials={};steps=0
  held=jd(resultdir/'heldout'/f'{rid}.json')
  for f in paths:
   checked(f,held['episode_artifacts'][str(f.relative_to(resultdir))]);d=jd(f);payload=d['payload']
   if sha(canonical(payload))!=d['sha256']:raise ValueError('Episode internal hash')
   ctx=payload['context'];seed=ctx['seed'];seeds.append(seed)
   if ctx['pool_ids_sha256']!=export['ordered_ids_sha256']:raise ValueError('Episode ordered pool differs')
   order=list(range(len(ids)));random.Random(seed).shuffle(order);expected=order[:3]
   for mode,x in payload['result'].items():
    obs=x['observed_indices']
    if len(obs)!=15 or len(set(obs))!=15 or obs[:3]!=expected:raise ValueError('Protein initial or observations mismatch')
    for index,v in zip(obs,x['revealed_executor_values']):
     if values[index]!=v:raise ValueError('Protein observed values differ from frozen pool')
    tr=[max(values[j] for j in obs[:t]) for t in range(4,16)]
    if tr!=x['best_trace'] or mean(tr)!=x['auc_full_precision']:raise ValueError('Protein AUC mismatch')
    steps+=15
   initials[str(seed)]=[ids[i] for i in expected]
  if sorted(seeds)!=sorted(cfg['seeds']['heldout']):raise ValueError('Protein seed set mismatch')
  save(OUT/'initials'/f'{rid}.json',initials)
  CERTS.append({'id':rid,'status':'certified_frozen_pool','panel':'experiment3',**export,'heldout_seed_count':len(seeds),'observed_values_verified':steps,'initial':3,'rounds':12})
  print('certified protein',rid,flush=True)
 return {'raw_mode_restored':(rawpath.stat().st_mode&0o777)==previous,'new_policy_runs':0}

def recent_chemistry():
 labels=[('BH_pilot','2026-10-18-bh-random-rule-controls-pilot-v1-rerun'),('BH_scale','2026-10-18-bh-random-rule-controls-scale-v1'),('BH_coverage','2026-10-19-bh-coverage-matched-rule-controls-v1'),('BH_anon_v2','2026-10-20-bh-semantic-anonymization-v2-evaluation-recovery-1'),('Suzuki_v1','2026-10-20-suzuki-continuation-amendment2-evaluation')]
 for label,folder in labels:
  d=B/'results'/folder;manifest=jd(d/'manifest.json')
  if label=='Suzuki_v1':lock=jd(B/'results/2026-10-20-suzuki-second-dataset-v1-generation/protocol_lock.json');cfg=jd(B/'results/2026-10-20-suzuki-second-dataset-v1-generation/config.json')
  else:lock=manifest;cfg=jd(d/'config.json')
  for name,h in lock['data_sha256'].items():checked(ROOT/name,h)
  key='experiments/care_replay/scripts/run_synthetic_suzuki.py';expected=lock['source_sha256'][key]
  candidates=[ROOT/key,d/'source_snapshot/run_synthetic_suzuki.py',d/'source_snapshot'/key,B/'results/2026-10-18-bh-random-rule-controls-scale-v1/source_snapshot/run_synthetic_suzuki.py',B/'results/2026-10-19-bh-coverage-matched-rule-controls-v1/source_snapshot/run_synthetic_suzuki.py']
  code=None
  for f in candidates:
   if f.exists() and sha(f.read_bytes())==expected:code=checked(f,expected);break
  if code is None:raise ValueError('Cannot find frozen adapter bytes for '+label)
  mod=module(code,label);adapter=mod.DATASET_BUILDERS[cfg['target_dataset']]();ids=[x.candidate_id for x in adapter.candidates];values=[x.objective_value for x in adapter.candidates]
  if len(ids)!=cfg['expected_candidate_count']:raise ValueError('Chemistry pool size')
  export=pool_export(label,ids,values,{'panel':'recent_chemistry','dataset':cfg['target_dataset'],'source_sha256':expected,'data_sha256':lock['data_sha256'],'initial':cfg['initial_observations'],'rounds':cfg['reveal_rounds'],'projection_note':'Suzuki approved semantic correction does not change IDs, order or objective values; corrected representation hash remains bound by generation lock.'})
  if label=='Suzuki_v1':
   rows=jd(d/'heldout_metrics.json');traces=[]
   for f in sorted((d/'heldout_seeds').glob('*.json')):
    rec=jd(f);tf=f.with_name(f.stem+'.traces.json.gz');data=checked(tf,rec['trace_sha256']);part=json.loads(gzip.decompress(data))
    traces.extend(part)
  else:
   rows=list(csv.DictReader(io.StringIO(read(d/'heldout_metrics.csv').decode())))
   tf=d/'heldout_traces.jsonl.gz';tf=tf if tf.exists() else d/'heldout_traces.jsonl';raw=read(tf);raw=gzip.decompress(raw) if tf.suffix=='.gz' else raw;traces=[json.loads(s) for s in raw.splitlines()]
  metrics={(int(x['seed']),x['mode']):x for x in rows}
  check,initials,aucs=check_trace_groups(traces,ids,values,metrics=metrics,initial_map=manifest.get('initial_candidates'))
  if check['seed_count']!=cfg['heldout_seed_count']:raise ValueError('Chemistry heldout seeds missing')
  save(OUT/'initials'/f'{label}.json',initials);save(OUT/'reconstructed_auc'/f'{label}.json',aucs)
  CERTS.append({'id':label,'status':'certified_frozen_pool',**export,'initial':5,'rounds':10,'trace_check':check})
  print('certified chemistry',label,check,flush=True)

def historical():
    matrix=jd(B/'outputs/auc_reanalysis_inventory_v1/comparability_matrix.json')
    specs={
      'Fixed_BH':('c1b976ec55c7d364b84e02b9b2a511da7783cc52','real_buchwald_hartwig','dreher_doyle_buchwald_hartwig.xlsx','bh_semantic_confirm500_audits.tar.gz'),
      'Fixed_bandgap':('c1b976ec55c7d364b84e02b9b2a511da7783cc52','real_matbench_expt_gap','matbench_expt_gap.json.gz','materials_semantic_confirm500_audits.tar.gz'),
      'Fixed_FreeSolv':('c2c86da99b4374636db296929c9076a6b5420eb7','real_moleculenet_freesolv','moleculenet_freesolv_sampl.csv','care2_kernel_transfer_500seed_audits.tar.gz'),
      'Fixed_Lipophilicity':('c2c86da99b4374636db296929c9076a6b5420eb7','real_moleculenet_lipophilicity','moleculenet_lipophilicity.csv','care2_universal_schedule_300seed_audits.tar.gz')}
    def git_at(commit,rel):
        ref=commit+':'+rel;data=subprocess.run(['git','show',ref],cwd=ROOT,capture_output=True,check=True).stdout;INPUTS['git:'+ref]=sha(data);return data
    for row in matrix:
        label=row['id']
        if label not in specs:continue
        commit,dataset,rawname,auditname=specs[label];archive,member=row['reference'].split('::')
        read(archive)
        with tarfile.open(archive) as t:
            metricbytes=t.extractfile(member).read()
            if metricbytes!=git_at(commit,member):raise ValueError('Historical metrics not identical to original result commit')
            am=next(m for m in t.getmembers() if Path(m.name).name==auditname);auditbytes=t.extractfile(am).read()
            if auditbytes!=git_at(commit,am.name):raise ValueError('Historical trace archive not identical to result commit')
        sourcepath='experiments/care_replay/scripts/run_synthetic_suzuki.py';code=git_at(commit,sourcepath)
        raw=git_at(commit,'experiments/care_replay/data/raw/'+rawname)
        desc=git_at(commit,'experiments/care_replay/data/descriptors/reaction_component_descriptors.csv')
        directory=OUT/'historical_inputs'/label;directory.mkdir(parents=True,exist_ok=True)
        for name,data in [('run_synthetic_suzuki.py',code),(rawname,raw),('reaction_component_descriptors.csv',desc)]:
            dest=directory/name
            if dest.exists() and dest.read_bytes()!=data:raise ValueError('Historical input changed')
            if not dest.exists():dest.write_bytes(data)
        mod=module(code,label);mod.RAW_DATA=directory;mod.REACTION_DESCRIPTOR_PATH=directory/'reaction_component_descriptors.csv'
        adapter=mod.DATASET_BUILDERS[dataset]();ids=[c.candidate_id for c in adapter.candidates];values=[c.objective_value for c in adapter.candidates]
        if len(ids)!=row['pool_n']:raise ValueError('Historical pool size mismatch')
        metricsrows=[x for x in csv.DictReader(io.StringIO(metricbytes.decode())) if x['split']=='heldout'];metrics={(int(x['seed']),x['mode']):x for x in metricsrows};seeds={int(x['seed']) for x in metricsrows}
        traces=[]
        with tarfile.open(fileobj=io.BytesIO(auditbytes)) as t:
            for m in t.getmembers():
                if not m.isfile() or not m.name.endswith('.jsonl'):continue
                for line in t.extractfile(m):
                    x=json.loads(line)
                    if int(x['seed']) in seeds and x.get('dataset_id',dataset)==dataset:traces.append(x)
        checks,initials,aucs=check_trace_groups(traces,ids,values,metrics=metrics)
        if checks['seed_count']!=row['heldout_seeds'] or checks['metric_auc_checks']==0:raise ValueError('Historical trace coverage')
        export=pool_export(label,ids,values,{'panel':'original_fixed','dataset':dataset,'original_result_commit':commit,'metrics_sha256':sha(metricbytes),'audit_archive_sha256':sha(auditbytes),'adapter_sha256':sha(code),'raw_sha256':sha(raw),'descriptor_sha256':sha(desc),'initial':5,'rounds':10})
        save(OUT/'initials'/f'{label}.json',initials);save(OUT/'reconstructed_auc'/f'{label}.json',aucs)
        CERTS.append({'id':label,'status':'certified_original_result_commit_reconstruction',**export,'initial':5,'rounds':10,'trace_check':checks,'evidence_limit':'Git result commit binds archived metrics, trace archive, adapter and raw data; entire archived heldout trace checked. Not a contemporaneously published preregistered pool hash.'})
        print('certified historical',label,checks,flush=True)

def main():
    OUT.mkdir(exist_ok=True);POOL.mkdir(exist_ok=True)
    if '--historical' in sys.argv:
        progress=jd(OUT/'certification_progress.json');CERTS.extend(progress['certified']);INPUTS.update(progress['inputs']);historical()
        if len(CERTS)!=32:raise ValueError('Expected all 32 pools')
        save(OUT/'certification_report.json',{'certified':CERTS,'gaps':GAPS,'inputs':INPUTS,'real_normalized_scores_computed':False,'model_api_calls':0})
    else:
        try:certify_protein();recent_chemistry()
        finally:save(OUT/'certification_progress.json',{'certified':CERTS,'gaps':GAPS,'inputs':INPUTS,'real_normalized_scores_computed':False,'model_api_calls':0})

if __name__=='__main__':main()
