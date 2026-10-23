"""Freeze amendment-2 code, approval, original evidence, and successful offline log."""
from bootstrap import *
import continuation_integrity as I
import argparse,shutil

def freeze(log):
    config,_=common.verify_freeze(I.CONFIG,I.LOCK,I.FREEZE)
    I.validate_old(config);I.validate_plan(common.load(I.PLAN),config)
    log=Path(log)
    record=common.load(log.with_suffix(".result.json"))
    if record.get("exit_code")!=0 or record.get("log_sha256")!=common.file_hash(log) or record.get("network_disabled") is not True:
        raise ValueError("Successful verified offline validation required")
    if "98 passed" not in log.read_text(): raise ValueError("Expected full 98-test validation")
    dest=I.BASE/"results/2026-10-20-suzuki-continuation-amendment2-freeze"
    dest.mkdir(parents=True,exist_ok=False)
    amendment={"status":"frozen_before_continuation","frozen_at":generation.utc(),"new_post_attempts_at_freeze":0,
        "maximum_total_post_attempts":21,"maximum_new_post_attempts":11,"budget_usd":"1","unknown_reservation_usd":"0.045",
        "original_lock_sha256":I.ORIGINAL_LOCK_SHA,"plan_sha256":common.file_hash(I.PLAN),
        "source_sha256":I.source_inventory(),"original_files_sha256":I.inventory(I.OLD),
        "freeze_directory":common.relative_path(dest),"offline_validation_sha256":common.file_hash(log)}
    for rel,sha in amendment["source_sha256"].items():
        path=dest/"source_snapshot"/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(I.ROOT/rel,path)
        if common.file_hash(path)!=sha: raise ValueError("Source changed during freeze")
    shutil.copyfile(I.PLAN,dest/"plan.json");shutil.copyfile(log,dest/"offline_validation.log")
    shutil.copyfile(log.with_suffix(".result.json"),dest/"offline_validation.result.json")
    common.save(I.AMENDMENT_LOCK,amendment,exclusive=True);shutil.copyfile(I.AMENDMENT_LOCK,dest/"amendment_lock.json")
    I.verify()
    return {"status":amendment["status"],"lock_sha256":common.file_hash(I.AMENDMENT_LOCK),"source_files":len(amendment["source_sha256"]),"original_files":len(amendment["original_files_sha256"])}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--validation-log",type=Path,required=True);args=parser.parse_args()
    print(freeze(args.validation_log))
