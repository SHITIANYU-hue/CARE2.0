"""Offline validation, public audit and source/input snapshot before any model POST."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import json,sys,shutil,re,platform,importlib.metadata
import representation as rep
import contract,common
from run_matched_random_rule_controls import offline_guard


def seed_history_audit(config):
    cal,held=common.validate_config(config)
    proposed=cal|held|set(config["paired_design"]["alias_seed_by_pair"].values())|{config["bootstrap_seed"],config["source"]["seed"],config["source"]["seed"]+10000,config["random_controls"]["seed"],config["paired_design"]["order_seed"]}
    root=common.ROOT/"experiments/care_replay";files=[];matches=[]
    def inspect(value,path,key=""):
        if isinstance(value,dict):
            for k,v in value.items(): inspect(v,path,k)
            for k,v in value.items():
                if k.endswith("seed_start") and type(v) is int:
                    count=value.get(k[:-5]+"count")
                    if type(count) is int and any(v<=s<v+count for s in proposed): matches.append([path,k,v,count])
        elif isinstance(value,list):
            for item in value: inspect(item,path,key)
        elif type(value) is int and value in proposed and "seed" in key: matches.append([path,key,value])
    candidates=list((root/"configs").glob("*.json"))+list((root/"preregistrations").glob("*.json"))
    candidates+=list((root/"results").glob("*/config.json"))+list((root/"results").glob("*/protocol_lock.json"))
    for path in sorted(set(candidates)):
        if "suzuki_second_dataset" in path.name or "suzuki-second-dataset" in str(path): continue
        try: data=common.load(path)
        except Exception as error: raise ValueError("Cannot audit prior config "+str(path)) from error
        rel=common.relative_path(path);inspect(data,rel);files.append({"path":rel,"sha256":common.file_hash(path)})
    if matches: raise ValueError("Proposed seeds already registered: "+str(matches))
    return {"scope":"current local config/preregistration and result config/lock inventories; historical archived evidence reviewed separately","files":files,"collisions":matches,"proposed_seed_count":len(proposed)}

def preflight(config):
    with offline_guard():
        audit=rep.audit();target=rep.build_adapter()
        modified=replace(target,candidates=tuple(replace(c,objective_value=-123456.,metadata={**c.metadata,"yield_value":987654.}) for c in target.candidates))
        if contract.build_payload(target)!=contract.build_payload(modified): raise ValueError("Target oracle leaks into prompt")
        if [rep.view(c) for c in target.candidates]!=[rep.view(c) for c in modified.candidates]: raise ValueError("Target oracle leaks into null metadata")
        audit["target_outcome_noninterference"]=True
        source=rep.replay.DATASET_BUILDERS[rep.SOURCE]()
        observed=contract.transfer.source_observations(source,config["source"]["seed"],512)
        audit["source_sample"]={"count":len(observed),"seed_parameter":config["source"]["seed"],"effective_shuffle_seed":config["source"]["seed"]+10000,"candidate_ids":[c.candidate_id for c in observed]}
        audit["source_role_map"]=rep.ROLE_MAP
        audit["seed_history"]=seed_history_audit(config)
        audit["structural_scope"]="All eight nonmissing ligand structure mappings checked against named primary references; unknown three and true-none explicitly retained. Executable reaction roles/functional families checked against raw component identities; no unused whole-molecule reactant descriptors enter rules or source-target mapping."
        # Ensure the old preparation and its raw/source files have not been rewritten.
        prior=common.ROOT/"experiments/care_replay/outputs/suzuki_second_dataset_preparation_v1"
        old=common.load(prior/"manifest.json")
        for group in ("source_sha256","artifact_sha256"):
            for rel,sha in old[group].items():
                if common.file_hash(common.ROOT/rel)!=sha: raise ValueError("Historical preparation changed: "+rel)
        for rel,sha in common.load(prior/"data_audit.json")["data_sha256"].items():
            if common.file_hash(common.ROOT/rel)!=sha: raise ValueError("Historical data changed")
        return audit

def freeze_protocol(config_path,lock_path,output,validation_log):
    config_path,lock_path,output,validation_log=map(Path,(config_path,lock_path,output,validation_log))
    if common.relative_path(config_path)!=common.CONFIG_REL: raise ValueError("Wrong config")
    if output.exists() or lock_path.exists(): raise FileExistsError("Never overwrite freeze")
    config=common.load(config_path);common.validate_config(config)
    sources=common.source_inventory();text=validation_log.read_text()
    if not re.search(r"[1-9][0-9]* passed",text) or re.search(r"[1-9][0-9]* (failed|error)",text): raise ValueError("Successful pytest evidence required")
    if any((common.ROOT/f).stat().st_mtime_ns>validation_log.stat().st_mtime_ns for f in sources): raise ValueError("Tests predate source changes")
    audit=preflight(config)
    materials={f"pair_{i:02d}":contract.build_pair(i,config["paired_design"]["alias_seed_by_pair"][f"pair_{i:02d}"]) for i in range(10)}
    # Conservative serialized-byte bound avoids a tokenizer dependency: one UTF-8
    # byte per token plus generous framing allowance overestimates text token use.
    bounds={f"{pair}_{arm}":len(json.dumps(m["arms"][arm]["messages"]).encode())+2048+5000 for pair,m in materials.items() for arm in common.ARMS}
    if max(bounds.values())>=128000: raise ValueError("Prompt context bound exceeded; do not truncate")
    audit["context_byte_upper_bounds_with_output"]=bounds
    maps={"source_sha256":{f:common.file_hash(common.ROOT/f) for f in sources},"data_sha256":{f:common.file_hash(common.ROOT/f) for f in config["data_files"]},"input_sha256":{f:common.file_hash(common.ROOT/f) for f in common.input_inventory(config)}}
    output.mkdir(parents=True)
    for group,folder in (("source_sha256","source_snapshot"),("data_sha256","input_snapshot"),("input_sha256","input_snapshot")):
        for f,sha in maps[group].items():
            dest=output/folder/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(common.ROOT/f,dest)
            if common.file_hash(dest)!=sha: raise ValueError("Snapshot mismatch")
    (output/"messages").mkdir();(output/"private_pairs").mkdir();mh={};ph={}
    for pair,m in materials.items():
        dest=output/"private_pairs"/(pair+".json");common.save(dest,m,exclusive=True);ph[pair]=common.file_hash(dest)
        for arm in common.ARMS:
            identifier=pair+"_"+arm;dest=output/"messages"/(identifier+".json");common.save(dest,m["arms"][arm]["messages"],exclusive=True);mh[identifier]=common.file_hash(dest)
    shutil.copyfile(config_path,output/"config.json");shutil.copyfile(validation_log,output/"offline_validation.log")
    common.save(output/"preflight_audit.json",audit,exclusive=True)
    lock={"protocol_id":config["protocol_id"],"status":"frozen_before_generation","model_api_calls_at_freeze":0,"created_at_utc":datetime.now(timezone.utc).isoformat(),"config_sha256":common.file_hash(config_path),"freeze_directory":common.relative_path(output),**maps,"messages_sha256":mh,"pair_material_sha256":ph,"offline_validation_log_sha256":common.file_hash(output/"offline_validation.log"),"preflight_audit_sha256":common.file_hash(output/"preflight_audit.json"),"authorization_reference":config["resource_authorization"]["authorization_reference"],"new_authorized_model_calls":20,"suzuki_max_cost_usd":"1","environment":{"python":platform.python_version(),"packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","rdkit","pytest")}}}
    lock_path.parent.mkdir(parents=True,exist_ok=True);common.save(lock_path,lock,exclusive=True);shutil.copyfile(lock_path,output/"protocol_lock.json")
    common.verify_freeze(config_path,lock_path,output);return lock

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--protocol-lock",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--validation-log",type=Path,required=True)
    args=parser.parse_args();lock=freeze_protocol(args.config,args.protocol_lock,args.output,args.validation_log);print(json.dumps({"status":lock["status"],"api_calls":0,"sources":len(lock["source_sha256"])}))
