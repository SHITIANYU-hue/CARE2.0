"""Freeze-verified two-stage offline runner. No outcomes are opened on import."""
from core import *
import csv, io, zipfile, time, datetime, argparse
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW=BASE/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip'

def load_values(archive, filename, wanted):
    """Parse numeric effects only after public sequence ID membership check."""
    wanted=set(wanted);found={}
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if Path(n).name==filename]
        if len(names)!=1:raise ValueError('Assay ZIP member ambiguity')
        with z.open(names[0]) as f:
            rows=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig'))
            if not {'mutated_sequence','DMS_score'}<=set(rows.fieldnames):raise ValueError('DMS schema mismatch')
            for row in rows:
                sid=hashlib.sha256(row['mutated_sequence'].strip().upper().encode()).hexdigest()
                if sid not in wanted:continue
                if sid in found:raise ValueError('Duplicate requested effect ID')
                value=float(row['DMS_score'])
                if not math.isfinite(value):raise ValueError('Nonfinite requested effect')
                found[sid]=value
    if set(found)!=wanted:raise ValueError('Missing requested effects')
    return found

def stage_ids(route, m, stage):
    if stage=='calibration':return list(dict.fromkeys(m['fit']+route['G0_selected_ids']+route['G1']['selected_ids']))
    if stage=='heldout':return list(dict.fromkeys(m['fit']+m['target']))
    raise ValueError(stage)

def run_pool(route_id, label, ids, public, source, raw, units, cfg, freeze_sha):
    phase='heldout' if label=='heldout' else 'calibration'
    pool=[public[i] for i in ids]
    prepared=(classic.feature_arrays(adapter(label,pool),pool),multi.additive_mutation_prior([source],pool)[0])
    rows=[]; files={}
    for seed in cfg['seeds'][phase]:
        path=OUT/'episodes'/route_id/label/f'{seed}.json'
        context={'freeze_sha256':freeze_sha,'route':route_id,'pool':label,'seed':seed,'pool_ids_sha256':object_hash(ids)}
        row=read_checkpoint(path,context)
        if row is None:
            row=paired_episodes(pool,source,raw,units,seed,cfg,prepared)
            save_checkpoint(path,context,row)
        for mode in ('Always','Never'):
            result=row[mode]
            if result['seed']!=seed or result['mode']!=mode or result['observations']!=15:
                raise RuntimeError('Checkpoint episode invariant')
        rows.append(row);files[str(path.relative_to(OUT))]=digest(path)
    return rows,files

def worker(route,stage,freeze_sha):
    start=time.monotonic();cfg=config();m=membership(route);rid=route['route_id']
    previous=None
    if stage=='heldout':
        verify_gate_barrier(freeze_sha)
        previous=json.loads((OUT/'gates'/f'{rid}.json').read_text())
        if previous['status']=='excluded_zero_source_iqr':
            write_once(OUT/'heldout'/f'{rid}.json',{'route':rid,'status':'excluded_zero_source_iqr','freeze_sha256':freeze_sha})
            return rid,'excluded',time.monotonic()-start
    ids=stage_ids(route,m,stage)
    public=build_public(route['assay'],ids)
    metadata=next(x for x in json.loads((PUBLIC/'assay_metadata.json').read_text()) if x['DMS_id']==route['assay'])
    raw=load_values(RAW,metadata['DMS_filename'],ids)
    source_values=[raw[i] for i in m['fit']]
    # Only this prespecified exclusion is allowed; invalid/nonfinite data already raise.
    q25,q75=np.quantile(np.asarray(source_values),[.25,.75],method='linear')
    if q75-q25==0:
        if stage!='calibration':raise RuntimeError('Source scale changed after calibration')
        write_once(OUT/'gates'/f'{rid}.json',{'route':rid,'status':'excluded_zero_source_iqr','freeze_sha256':freeze_sha,
            'G0':{'accept':False},'G1':{'accept':False},'source_count':1024})
        return rid,'excluded',time.monotonic()-start
    units,source=fit_source(public,m['fit'],raw)
    if stage=='calibration':
        g0,files=run_pool(rid,'G0',route['G0_selected_ids'],public,source,raw,units,cfg,freeze_sha)
        g0decision=gate(g0)
        if route['G1']['status']=='matched':
            g1,files1=run_pool(rid,'G1',route['G1']['selected_ids'],public,source,raw,units,cfg,freeze_sha)
            files.update(files1);g1decision={**gate(g1),'support':'matched'}
        else:g1decision={'accept':False,'support':'unsupported','paired_seeds':0,'reason':route['G1']['reason']}
        result={'route':rid,'status':'calibrated','freeze_sha256':freeze_sha,'units':asdict(units),
            'G0':g0decision,'G1':g1decision,'episode_artifacts':files,'source_ids_sha256':object_hash(m['fit'])}
        write_once(OUT/'gates'/f'{rid}.json',result)
    else:
        if previous['units']!=asdict(units):raise RuntimeError('Source units changed after gate lock')
        pairs,files=run_pool(rid,'heldout',m['target'],public,source,raw,units,cfg,freeze_sha)
        result={'route':rid,'status':'complete','freeze_sha256':freeze_sha,'units':asdict(units),
            'paired_seeds':len(pairs),'episode_artifacts':files,'gate_record_sha256':digest(OUT/'gates'/f'{rid}.json')}
        write_once(OUT/'heldout'/f'{rid}.json',result)
    return rid,'complete',time.monotonic()-start

def freeze_gates(freeze_sha):
    files={};ids=[]
    for route in routes():
        rid=route['route_id'];p=OUT/'gates'/f'{rid}.json';result=json.loads(p.read_text())
        if result['freeze_sha256']!=freeze_sha:raise RuntimeError('Wrong gate provenance')
        ids.append(rid);files[str(p.relative_to(OUT))]=digest(p);files.update(result.get('episode_artifacts',{}))
    write_once(OUT/'GATES_LOCK.json',{'freeze_sha256':freeze_sha,'routes':ids,'artifacts':files,
        'all_calibration_completed_before_heldout':True,'api_calls':0})
    write_once(OUT/'GATES_LOCK.sha256.json',{'sha256':digest(OUT/'GATES_LOCK.json')})
    checksum=OUT/'GATES_LOCK.sha256';text=digest(OUT/'GATES_LOCK.json')+'\n'
    if checksum.exists() and checksum.read_text()!=text:raise RuntimeError('Gate digest conflict')
    if not checksum.exists():checksum.write_text(text)
    verify_gate_barrier(freeze_sha)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['calibration','heldout','all'])
    args=parser.parse_args();freeze_sha=verify_freeze();cfg=config();OUT.mkdir(exist_ok=True)
    os.umask(0o077)
    # Open archive only after the complete execution freeze has been verified.
    os.chmod(RAW,0o400)
    try:
        if digest(RAW)!=cfg['raw_zip_sha256']:raise RuntimeError('Raw archive digest mismatch')
        for stage in (['calibration','heldout'] if args.stage=='all' else [args.stage]):
            if stage=='heldout':verify_gate_barrier(freeze_sha)
            print(json.dumps({'stage':stage,'event':'start','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'routes':23,'api_calls':0}),flush=True)
            with ProcessPoolExecutor(max_workers=cfg['workers']) as executor:
                futures=[executor.submit(worker,r,stage,freeze_sha) for r in routes()]
                for i,future in enumerate(as_completed(futures),1):
                    rid,status,elapsed=future.result()
                    print(json.dumps({'stage':stage,'completed':i,'route':rid,'status':status,'seconds':round(elapsed,2)}),flush=True)
            if stage=='calibration':freeze_gates(freeze_sha)
        print(json.dumps({'event':'requested_stages_complete','api_calls':0}),flush=True)
    finally:os.chmod(RAW,0o000)

if __name__=='__main__':main()
