"""Resolve existing logical rows to physical traces; do not compute normalized results."""
import offline
from certify_pools import B,OUT,INPUTS,read,jd,checked,canonical,sha,save
import json,csv,gzip,io,tarfile,copy

def signature(skill):
    x=copy.deepcopy(skill)
    for k in ('skill_id','hypothesis'):x.pop(k)
    for r in x['rules']:
        for k in ('rule_id','rationale'):r.pop(k)
    return sha(canonical(x))

def prepare():
    cert=jd(OUT/'certification_report.json');matrix=jd(B/'outputs/auc_reanalysis_inventory_v1/comparability_matrix.json');by={r['id']:r for r in matrix};report=[]
    for c in cert['certified']:
        label=c['id'];m=by[label];pool=jd(B/c['pool_file']);initials=jd(OUT/'initials'/f'{label}.json');binding=c['ordered_ID_value_sha256'];out=[]
        if m['panel']=='experiment3':
            directory=B/'outputs/experiment3_execution_v1';g=jd(directory/'gates'/f'{label}.json')
            for f in sorted((directory/'episodes'/label/'heldout').glob('*.json')):
                d=jd(f)['payload'];seed=d['context']['seed'];physical=d['result']
                for mode,physical_mode in [('Always','Always'),('Never','Never'),('G0','Always' if g['G0']['accept'] else 'Never'),('G1','Always' if g['G1']['accept'] else 'Never')]:
                    out.append(dict(seed=seed,mode=mode,physical_mode=physical_mode,auc=physical[physical_mode]['auc_full_precision'],stored_auc=physical[physical_mode]['best_so_far_auc'],initial=3,rounds=12,pool_binding=binding))
        else:
            ref=m['reference'];aliases={}
            if '::' in ref:
                a,member=ref.split('::');read(a)
                with tarfile.open(a) as t:rows=[r for r in csv.DictReader(io.StringIO(t.extractfile(member).read().decode())) if r['split']=='heldout']
            else:
                directory=__import__('pathlib').Path(ref);manifest=jd(directory/'manifest.json');aliases=manifest.get('execution_aliases',{})
                f=directory/'heldout_metrics.csv';rows=list(csv.DictReader(io.StringIO(read(f).decode()))) if f.exists() else jd(directory/'heldout_metrics.json')
            aucs=jd(OUT/'reconstructed_auc'/f'{label}.json')
            if label=='Suzuki_v1':
                modes={key.split('|',1)[1] for key in aucs};groups={}
                for skills in manifest['libraries'].values():
                    for skill in skills:groups.setdefault(signature(skill),[]).append('llm_semantic_'+skill['skill_id'])
                needed={r.get('selected_source_mode') or r['mode'] for r in rows}
                for signature_value,names in groups.items():
                    if not (set(names)&needed):continue
                    representatives=set(names)&modes
                    if len(representatives)!=1:raise ValueError('Ambiguous physical representative')
                    for name in names:aliases[name]=next(iter(representatives))
                # Bind the approved representation to the original lock. Pool values do not depend on the semantic view.
                lock=jd(B/'results/2026-10-20-suzuki-second-dataset-v1-generation/protocol_lock.json')
                key='experiments/care_replay/protocols/suzuki_second_dataset_v1/representation.py';checked(B.parents[1]/key,lock['source_sha256'][key])
            for row in rows:
                seed=str(row['seed']);mode=row['mode'];source=row.get('selected_source_mode') or mode;representative=aliases.get(source,source);key=seed+'|'+representative
                if key not in aucs:raise ValueError('Unresolved physical row '+label+' '+key)
                auc=aucs[key]
                if abs(auc-float(row['best_so_far_auc']))>.00005001:raise ValueError('Logical alias AUC mismatch')
                out.append(dict(seed=int(seed),mode=mode,physical_mode=representative,auc=auc,stored_auc=float(row['best_so_far_auc']),initial=5,rounds=10,pool_binding=binding))
        keys={(r['seed'],r['mode']) for r in out}
        if len(keys)!=len(out):raise ValueError('Duplicate logical rows')
        if {str(r['seed']) for r in out}!=set(initials):raise ValueError('Initial/row seed mismatch')
        raw=canonical(out);file=OUT/'prepared_rows'/f'{label}.json.gz';file.parent.mkdir(exist_ok=True);data=gzip.compress(raw,mtime=0)
        if file.exists() and file.read_bytes()!=data:raise ValueError('Prepared table changed')
        if not file.exists():file.write_bytes(data)
        report.append({'id':label,'logical_rows':len(out),'physical_rows':len({(r['seed'],r['physical_mode']) for r in out}),'prepared_file':str(file.relative_to(B)),'prepared_sha256':sha(data),'full_precision_reconstructed':True})
    save(OUT/'prepared_tables_report.json',{'tables':report,'inputs':INPUTS,'normalized_results_computed':False,'model_api_calls':0})
    print('Prepared',len(report),'tables;',sum(r['logical_rows'] for r in report),'logical rows;',sum(r['physical_rows'] for r in report),'physical rows. No normalization performed.')
if __name__=='__main__':prepare()
