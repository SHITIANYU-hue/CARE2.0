"""Approved one-slot continuation, preserving the failed attempt as unknown spend."""
from bootstrap import *
from decimal import Decimal
import shutil
HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/care_replay"
CONFIG=ROOT/common.CONFIG_REL
LOCK=BASE/"preregistrations/suzuki_second_dataset_v1.lock.json"
FREEZE=BASE/"results/2026-10-20-suzuki-second-dataset-v1-freeze"
OLD=BASE/"results/2026-10-20-suzuki-second-dataset-v1-generation"
PLAN=BASE/"configs/suzuki_continuation_amendment2.json"
AMENDMENT_LOCK=BASE/"preregistrations/suzuki_continuation_amendment2.lock.json"
NEW=BASE/"results/2026-10-20-suzuki-continuation-amendment2-generation"
EVALUATION=BASE/"results/2026-10-20-suzuki-continuation-amendment2-evaluation"
KNOWN=Decimal("0.0080829")
UNKNOWN=Decimal("0.045")
FAILED="pair_04_anonymous"
OLD_LEDGER_SHA="3c80799e4c408008e4f3d716cd88992f47413fecd51739c61a53c3444feddf8f"
ORIGINAL_LOCK_SHA="5b54fd62b962677c0ce5a30b15909c46861efcdd7df9c63331248f3df9d20473"

def inventory(directory):
    result={}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink(): raise ValueError("Symlinks forbidden")
        if path.is_file(): result[path.relative_to(directory).as_posix()]=common.file_hash(path)
    return result

def source_inventory():
    return {common.relative_path(p):common.file_hash(p) for p in sorted(HERE.rglob("*.py"))}

def expected_request(config,freeze,identifier):
    return {"model":config["model"]["id"],"messages":common.load(freeze/"messages"/(identifier+".json")),
            "temperature":config["model"]["temperature"],"max_tokens":config["model"]["max_tokens"],
            "response_format":{"type":"json_object"},"stream":False,"provider":config["api"]["provider"]}

def validate_old(config):
    if common.file_hash(LOCK)!=ORIGINAL_LOCK_SHA or common.file_hash(OLD/"ledger.json")!=OLD_LEDGER_SHA:
        raise ValueError("Original identity changed")
    ledger=common.load(OLD/"ledger.json");attempts=ledger["attempts"]
    if len(attempts)!=10 or ledger["actual_post_attempts"]!=10: raise ValueError("Expected ten original attempts")
    total=Decimal(0);seen=set()
    for i,attempt in enumerate(attempts):
        slot=config["generation_order"][i];identifier=f"pair_{slot['pair_index']:02d}_{slot['arm']}"
        if any(attempt[k]!=v for k,v in {**slot,"slot":i,"identifier":identifier,"reservation_usd":"0.045"}.items()): raise ValueError("Original attempt identity changed")
        receipt=OLD/"receipts"/identifier
        receipts.checked(receipt/"request.json",attempt["request_sha256"])
        if common.load(receipt/"request.json")!=expected_request(config,FREEZE,identifier): raise ValueError("Original request changed")
        if i==9:
            if identifier!=FAILED or attempt["status"]!="HALT" or "cost_usd" in attempt or "response_id" in attempt: raise ValueError("Wrong unknown attempt")
            if (receipt/"response.json").exists() or (OLD/"outcomes"/(identifier+".json")).exists(): raise ValueError("Unknown request has response/outcome")
            if common.load(receipt/"transport_error.json")!={"http_status":None,"response":"URLError"}: raise ValueError("Wrong transport failure")
            continue
        outcome_path=OLD/"outcomes"/(identifier+".json");receipts.checked(outcome_path,attempt["outcome_sha256"])
        outcome=common.load(outcome_path)
        for key in ("request_sha256","response_sha256","response_id","cost_usd","library_sha256","status"):
            if outcome[key]!=attempt[key]: raise ValueError("Original receipt/ledger differ")
        receipts.checked(receipt/"response.json",outcome["response_sha256"])
        receipts.checked(receipt/"billing.json",outcome["billing_sha256"])
        response=common.load(receipt/"response.json");billing=common.load(receipt/"billing.json");cost=generation.money(outcome["cost_usd"])
        if response["id"] in seen or response["id"]!=outcome["response_id"] or billing["response_id"]!=response["id"] or generation.money(billing["cost_usd"])!=cost: raise ValueError("Original response identity/cost differs")
        seen.add(response["id"])
        receipts.verify_billing_evidence(response,billing,receipt,cost)
        material=common.load(FREEZE/"private_pairs"/f"pair_{slot['pair_index']:02d}.json")
        decision=receipts.replay_classification(response,slot["arm"],material,cost)
        if decision.status!=outcome["status"] or list(decision.reasons)!=outcome["reason_codes"]: raise ValueError("Original classification differs")
        lib=OLD/"libraries"/(identifier+".json")
        if decision.library is not None:
            receipts.checked(lib,outcome["library_sha256"])
            if common.load(lib)!=decision.library: raise ValueError("Original library changed")
        elif lib.exists(): raise ValueError("Partial invalid library")
        total+=cost
    if total!=KNOWN or generation.money(ledger["cumulative_known_cost_usd"])!=KNOWN: raise ValueError("Original known costs differ")
    failure=common.load(OLD/"failure.json")
    if failure["actual_post_attempts"]!=10 or failure["terminal_outcome_count"]!=9 or generation.money(failure["unresolved_reservation_usd"])!=UNKNOWN: raise ValueError("Original halt accounting differs")
    return ledger

def validate_plan(plan,config):
    if plan.get("approval")!="批准这次续跑修订，总调用上限改为21次，费用仍不超过1美元，只替补这一次传输失败，保留已有结果和未知费用预留。先验证并冻结续跑方案，再继续完成实验，有新的方案变更再停下来问我。": raise ValueError("Missing exact approval")
    required={"maximum_total_post_attempts":21,"maximum_new_post_attempts":11,"budget_usd":"1","unknown_reservation_usd":"0.045","known_original_cost_usd":"0.0080829","replacement_identifier":FAILED,"additional_retry_allowed":False,"remaining_order":config["generation_order"][9:],"generation_output":common.relative_path(NEW),"evaluation_output":common.relative_path(EVALUATION)}
    if any(plan.get(k)!=v for k,v in required.items()): raise ValueError("Amendment exceeds authorization")
    if KNOWN+UNKNOWN+11*generation.RESERVE>generation.BUDGET or 21*generation.RESERVE>generation.BUDGET: raise ValueError("Budget exceeds cap")

def verify():
    config,lock=common.verify_freeze(CONFIG,LOCK,FREEZE)
    plan=common.load(PLAN);validate_plan(plan,config)
    amendment=common.load(AMENDMENT_LOCK)
    if amendment["status"]!="frozen_before_continuation" or amendment["plan_sha256"]!=common.file_hash(PLAN): raise ValueError("Amendment not frozen")
    if amendment["original_lock_sha256"]!=ORIGINAL_LOCK_SHA or amendment["original_files_sha256"]!=inventory(OLD): raise ValueError("Original evidence changed")
    if amendment["source_sha256"]!=source_inventory(): raise ValueError("Continuation source changed")
    frozen=ROOT/amendment["freeze_directory"]
    for rel,sha in amendment["source_sha256"].items(): receipts.checked(frozen/"source_snapshot"/rel,sha)
    receipts.checked(frozen/"plan.json",amendment["plan_sha256"])
    receipts.checked(frozen/"offline_validation.log",amendment["offline_validation_sha256"])
    receipts.checked(frozen/"amendment_lock.json",common.file_hash(AMENDMENT_LOCK))
    validate_old(config)
    return config,lock,amendment

def check_budget(attempts,known):
    unknown=sum((generation.RESERVE for a in attempts if "cost_usd" not in a),Decimal(0))
    if len(attempts)>=21 or known+unknown+generation.RESERVE>generation.BUDGET: raise generation.Halt("Continuation count/reservation cap reached")
    return unknown
