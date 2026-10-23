import copy,json
from decimal import Decimal
from pathlib import Path
import pytest
import continuation_integrity as I
import continuation_generation as G
import continuation_receipts as R
import continuation_evaluation as E
import common,generation

class FakeAPI:
    key="fixture-no-secret"
    def __init__(self,fail_at=None,duplicate=False,unbilled=False): self.posts=[];self.fail_at=fail_at;self.duplicate=duplicate;self.unbilled=unbilled
    def request(self,path,payload=None):
        if path=="/models": return {"data":[{"id":"openai/gpt-4o-mini","context_length":128000,"pricing":{"prompt":"0.00000015","completion":"0.0000006","request":"0"}}]}
        if path=="/key": return {"data":{}}
        if path.startswith("/generation?"): raise G.TransportError(404,"fixture")
        assert path=="/chat/completions"
        self.posts.append(copy.deepcopy(payload))
        if len(self.posts)==self.fail_at: raise G.TransportError(None,"fixture transport failure")
        ident="gen-1792494584-7tYWxk8aZIYpD2YNlBaD" if self.duplicate else "fixture-new-"+str(len(self.posts))
        return {"id":ident,"object":"chat.completion","model":"openai/gpt-4o-mini","provider":"OpenAI","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"{}"}}],"usage":{"cost":None if self.unbilled else .001,"is_byok":False}}

@pytest.fixture
def env(tmp_path,monkeypatch):
    config,lock=common.verify_freeze(I.CONFIG,I.LOCK,I.FREEZE)
    I.validate_old(config)
    amendment=tmp_path/"amendment_lock.json";common.save(amendment,{"test_fixture":True})
    monkeypatch.setattr(I,"AMENDMENT_LOCK",amendment)
    monkeypatch.setattr(I,"NEW",tmp_path/"generation")
    monkeypatch.setattr(I,"verify",lambda:(config,lock,{"test_fixture":True}))
    return config,lock

def run(api): return G.execute(I.CONFIG,I.LOCK,I.FREEZE,I.NEW,api=api,sleep=lambda _:None)

def test_original_and_amended_limits():
    config,_=common.verify_freeze(I.CONFIG,I.LOCK,I.FREEZE)
    original=I.validate_old(config)
    assert len(original["attempts"])==10
    assert I.check_budget(original["attempts"],I.KNOWN)==I.UNKNOWN
    with pytest.raises(G.Halt): I.check_budget(original["attempts"]*2+[{}],I.KNOWN)
    with pytest.raises(G.Halt): I.check_budget(original["attempts"],Decimal(".95"))

def test_completed_continuation_preserves_nine_and_unknown(env):
    config,lock=env;before=I.inventory(I.OLD);api=FakeAPI();result=run(api)
    assert len(api.posts)==11 and result["actual_post_attempts"]==21
    assert result["cumulative_known_cost_usd"]=="0.0190829"
    assert result["unknown_reservation_usd"]=="0.045" and result["budget_occupancy_usd"]=="0.0640829"
    assert I.inventory(I.OLD)==before
    assert I.inventory(I.NEW/"original_evidence")==before
    original=common.load(I.OLD/"ledger.json")["attempts"]
    assert common.load(I.NEW/"ledger.json")["attempts"][:10]==original
    for a in original[:9]:
        assert common.file_hash(I.NEW/"outcomes"/(a["identifier"]+".json"))==a["outcome_sha256"]
    assert api.posts==[I.expected_request(config,I.FREEZE,f"pair_{s['pair_index']:02d}_{s['arm']}") for s in config["generation_order"][9:]]
    outcomes,payloads,_=R.verify_generation(I.CONFIG,I.LOCK,I.NEW,config,lock)
    assert len(outcomes)==20 and len(payloads)==5
    with pytest.raises(G.Halt): run(api)
    assert len(api.posts)==11
    # An unknown reservation cannot silently disappear from the completed audit.
    ledger=common.load(I.NEW/"ledger.json");ledger["unknown_reservation_usd"]="0";common.save(I.NEW/"ledger.json",ledger)
    result["ledger_sha256"]=common.file_hash(I.NEW/"ledger.json");common.save(I.NEW/"completion.json",result)
    with pytest.raises(ValueError,match="reservation"): R.verify_generation(I.CONFIG,I.LOCK,I.NEW,config,lock)

@pytest.mark.parametrize("fail_at",[1,5,11])
def test_second_transport_failure_never_retries(env,fail_at):
    config,lock=env;before=I.inventory(I.OLD);api=FakeAPI(fail_at=fail_at)
    with pytest.raises(G.Halt): run(api)
    assert len(api.posts)==fail_at
    failure=common.load(I.NEW/"failure.json")
    assert failure["actual_post_attempts"]==10+fail_at
    assert failure["unresolved_reservation_usd"]=="0.090"
    assert failure["incomplete_run_not_evaluated"] and not failure["retry_performed"]
    assert I.inventory(I.OLD)==before
    with pytest.raises(ValueError): R.verify_generation(I.CONFIG,I.LOCK,I.NEW,config,lock)
    with pytest.raises(G.Halt): run(api)
    assert len(api.posts)==fail_at

@pytest.mark.parametrize("kwargs",[{"duplicate":True},{"unbilled":True}])
def test_unsettled_or_duplicate_no_fallback(env,kwargs):
    api=FakeAPI(**kwargs)
    with pytest.raises(G.Halt): run(api)
    assert len(api.posts)==1
    assert len(list((I.NEW/"outcomes").glob("*.json")))==9
    assert common.load(I.NEW/"failure.json")["unresolved_reservation_usd"]=="0.090"

def test_original_scientific_functions_unchanged():
    import evaluation
    for name in ["make_libraries","select","run_phase","seed_job"]: assert getattr(E,name) is getattr(evaluation,name)
    config=common.load(I.CONFIG);cal,held=common.validate_config(config)
    assert len(cal)==50 and len(held)==100 and not cal&held

def test_plan_mutations_rejected():
    config=common.load(I.CONFIG);plan=common.load(I.PLAN);I.validate_plan(plan,config)
    for key,value in [("maximum_total_post_attempts",22),("maximum_new_post_attempts",12),("unknown_reservation_usd","0"),("replacement_identifier","pair_00_labeled"),("additional_retry_allowed",True),("remaining_order",config["generation_order"]),("budget_usd","2")]:
        bad=copy.deepcopy(plan);bad[key]=value
        with pytest.raises(ValueError): I.validate_plan(bad,config)
