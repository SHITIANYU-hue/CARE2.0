"""Original frozen scientific evaluation, with amendment-2 generation evidence."""
from bootstrap import *
from evaluation import *
import continuation_receipts as receipts
import continuation_integrity as integrity
def run(config_path,lock_path,freeze_dir):
    started=time.perf_counter()
    with matched.offline_guard():
        config,lock,_=integrity.verify()
        calibration,heldout=common.validate_config(config)
        generation=integrity.NEW;output=integrity.EVALUATION
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
            integrity.verify()
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
    print(json.dumps(run(integrity.CONFIG,integrity.LOCK,integrity.FREEZE)))
