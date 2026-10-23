"""Offline Suzuki matched-controls replay with immutable selection and seed checkpoints."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,gzip,random,time,argparse,hashlib
import representation as rep
import common,receipts,analysis
import run_matched_random_rule_controls as matched
import run_paired_anonymization_evaluation as legacy
import coverage_matched_rule_controls as coverage
ROOT=common.ROOT

def make_libraries(payloads,config):
    adapter=rep.build_adapter();catalog=rep.catalog(adapter);types=rep.field_types(catalog)
    libraries,provenance,diagnostics={},{},{}
    for identifier,payload in sorted(payloads.items()):
        pair=int(identifier.split("_")[1]);arm=identifier.split("_")[2]
        libraries[identifier],provenance[identifier]=legacy.normalize_without_loss(payload,catalog,pair,arm)
    for i in range(10):
        parent=f"pair_{i:02d}_labeled"
        if parent not in libraries: continue
        templates=libraries[parent]
        # Only the nine whitelisted public fields enter the null generator.
        public=[rep.view(c) for c in adapter.candidates]
        pools=coverage.build_pools(templates,catalog,types,public,tolerance=.005,max_configurations=1000000)
        for family in ("r1","r2"):
            diagnostics[parent+"_"+family]={s.skill_id:pools[family][s.skill_id].diagnostics for s in templates}
            for j in range(10):
                name=f"{family}_pair_{i:02d}_rep_{j:02d}"
                seed=int(common.object_hash([config["random_controls"]["seed"],family,i,j]),16)
                rng=random.Random(seed)
                libraries[name]=tuple(coverage.draw_skill(pools[family][s.skill_id],rng,f"{name}_slot_{slot:02d}") for slot,s in enumerate(templates))
                provenance[name]={"parent":parent,"family":family,"replicate":j,"derived_seed":str(seed)}
    return libraries,provenance,diagnostics

def select(rows,libraries,outcomes,seeds,config):
    valid=matched.freeze_selection(rows,libraries,seeds,config) if libraries else {}
    modes=(*matched.BASELINES,*(matched.semantic.skill_mode(s) for b in libraries.values() for s in b))
    matched.validate_rows(rows,modes,seeds,"calibration")
    means=matched.statistics.mode_means(rows,matched.BASELINES,seeds)
    anchor=max(matched.BASELINES,key=lambda m:(means[m]["composite"],m))
    choices={}
    for name,choice in valid.items():
        if choice["selection"]["target_anchor_mode"]!=anchor: raise ValueError("Anchor differs")
        family=name.split("_")[0] if name.startswith(("r1_","r2_")) else name.split("_")[2]
        choices[name]={**choice,"primary_mode":choice["ungated_calibration_winner"],"family":family,"status":"VALID"}
    for name,outcome in outcomes.items():
        if outcome["status"]=="INVALID_ANCHOR_FALLBACK":
            if name in choices: raise ValueError("Partial invalid library")
            choices[name]={"primary_mode":anchor,"selected_mode":anchor,"family":outcome["arm"],"status":outcome["status"],"reasons":outcome["reason_codes"]}
        elif name not in choices: raise ValueError("Missing valid library")
    return choices,anchor,means

def seed_job(seed,split,config,skills):
    rep.install()
    return matched.evaluate_seed(seed,split,config,skills)

def store_seed(directory,seed,batch,context):
    rows,traces,seconds=batch
    tracefile=directory/f"{seed}.traces.json.gz"
    raw=gzip.compress(json.dumps(traces,allow_nan=False,separators=(",",":")).encode(),mtime=0)
    tmp=tracefile.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        handle.write(raw);handle.flush();os.fsync(handle.fileno())
    tmp.replace(tracefile)
    common.save(directory/f"{seed}.json",{"seed":seed,"context_sha256":context,"rows":rows,"seconds":seconds,"trace_sha256":common.file_hash(tracefile)})

def read_seed(directory,seed,context,modes,split):
    payload=common.load(directory/f"{seed}.json")
    if payload["seed"]!=seed or payload["context_sha256"]!=context: raise ValueError("Checkpoint belongs to different plan")
    if common.file_hash(directory/f"{seed}.traces.json.gz")!=payload["trace_sha256"]: raise ValueError("Trace checkpoint changed")
    matched.validate_rows(payload["rows"],modes,{seed},split)
    return payload

def run_phase(split,seeds,config,skills,output,manifest_sha,workers=4):
    reps,aliases=matched.execution_plan(skills,True)
    modes=(*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in reps))
    context=common.object_hash({"manifest":manifest_sha,"split":split,"seeds":sorted(seeds),"representatives":[asdict(s) for s in reps],"aliases":aliases})
    directory=output/(split+"_seeds");directory.mkdir(exist_ok=True)
    pending=[]
    for seed in sorted(seeds):
        if (directory/f"{seed}.json").exists(): read_seed(directory,seed,context,modes,split)
        else: pending.append(seed)
    if workers==1:
        for seed in pending:
            store_seed(directory,seed,seed_job(seed,split,config,reps),context)
            print(json.dumps({"phase":split,"seed":seed,"executed_modes":len(modes)}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(seed_job,seed,split,config,reps):seed for seed in pending}
            for done,future in enumerate(as_completed(futures),1):
                seed=futures[future];store_seed(directory,seed,future.result(),context)
                print(json.dumps({"phase":split,"completed_seeds":len(seeds)-len(pending)+done,"total_seeds":len(seeds),"seed":seed,"executed_modes":len(modes)}),flush=True)
    rows=[];seconds=0
    for seed in sorted(seeds):
        payload=read_seed(directory,seed,context,modes,split)
        rows.extend(matched.expand_metric_aliases(payload["rows"],aliases));seconds+=payload["seconds"]
    logical=(*matched.BASELINES,*(matched.semantic.skill_mode(s) for s in skills))
    matched.validate_rows(rows,logical,set(seeds),split)
    common.save(output/(split+"_metrics.json"),rows)
    return rows,{"logical_trajectories":len(rows),"physical_trajectories":len(modes)*len(seeds),"summed_worker_seconds":seconds,"context_sha256":context}

def run(config_path,lock_path,freeze_dir):
    started=time.perf_counter()
    with matched.offline_guard():
        config,lock=common.verify_freeze(config_path,lock_path,freeze_dir)
        calibration,heldout=common.validate_config(config)
        generation=ROOT/config["generation_output_directory"];output=ROOT/config["evaluation_output_directory"]
        outcomes,payloads,hashes=receipts.verify_generation(config_path,lock_path,generation,config,lock)
        libraries,provenance,coverage_audit=make_libraries(payloads,config)
        skills=tuple(s for b in libraries.values() for s in b)
        manifest={"protocol_lock_sha256":common.file_hash(lock_path),"config_sha256":common.file_hash(config_path),"generation_files_sha256":hashes,
            "outcomes":outcomes,"libraries":{n:[asdict(s) for s in b] for n,b in libraries.items()},"id_provenance":provenance,"coverage_audit":coverage_audit,"model_api_calls":0}
        output.mkdir(parents=True,exist_ok=True)
        mp=output/"manifest.json"
        if mp.exists():
            if common.load(mp)!=json.loads(json.dumps(manifest)): raise ValueError("Resume manifest differs")
        else: common.save(mp,manifest,exclusive=True)
        manifest_sha=common.file_hash(mp)
        def verify():
            common.verify_freeze(config_path,lock_path,freeze_dir)
            if common.file_hash(mp)!=manifest_sha: raise ValueError("Manifest changed")
            for rel,sha in hashes.items():
                if common.file_hash(generation/rel)!=sha: raise ValueError("Generation changed")
        if (output/"completion.json").exists():
            verify();return common.load(output/"completion.json")
        cal_rows,cal_info=run_phase("calibration",calibration,config,skills,output,manifest_sha)
        choices,anchor,anchor_means=select(cal_rows,libraries,outcomes,calibration,config)
        selection={"manifest_sha256":manifest_sha,"calibration_sha256":common.file_hash(output/"calibration_metrics.json"),"choices":choices,"anchor":anchor,"anchor_means":anchor_means,"heldout_seeds":sorted(heldout)}
        sp=output/"selection_lock.json"
        if sp.exists():
            if common.load(sp)!=selection: raise ValueError("Frozen selection differs on resume")
        else: common.save(sp,selection,exclusive=True)
        selection_sha=common.file_hash(sp);verify()
        needed={v[k] for v in choices.values() for k in ("primary_mode","selected_mode")}
        heldskills=tuple(s for s in skills if matched.semantic.skill_mode(s) in needed)
        # No extra selection: evaluate exactly every precommitted primary/deployed policy.
        rows,held_info=run_phase("heldout",heldout,config,heldskills,output,manifest_sha+selection_sha)
        if common.file_hash(sp)!=selection_sha: raise ValueError("Selection changed during heldout")
        result=analysis.analyze(rows,choices,outcomes,anchor,config)
        verify()
        common.save(output/"analysis.json",result,exclusive=True)
        complete={"status":"complete","model_api_calls":0,"calibration":cal_info,"heldout":held_info,"wall_seconds":time.perf_counter()-started,"selection_sha256":selection_sha,"analysis_sha256":common.file_hash(output/"analysis.json"),"valid_libraries":len(payloads),"random_libraries":len(libraries)-len(payloads)}
        common.save(output/"completion.json",complete,exclusive=True)
        return complete

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--protocol-lock",type=Path,required=True);parser.add_argument("--freeze-dir",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(run(args.config,args.protocol_lock,args.freeze_dir)))
