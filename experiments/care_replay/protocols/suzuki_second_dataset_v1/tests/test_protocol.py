import copy,json,socket,random
from pathlib import Path
from dataclasses import replace
from decimal import Decimal
import numpy as np
import pytest
import representation as rep
import contract,common,generation,evaluation,analysis,freeze,receipts

@pytest.fixture(scope="module")
def cfg(): return common.load(common.ROOT/common.CONFIG_REL)
@pytest.fixture(scope="module")
def target(): return rep.build_adapter()
@pytest.fixture(scope="module")
def pair(): return contract.build_pair(0,202610214000)

def fixture_library(catalog):
    field=next(iter(catalog));values=list(catalog[field]);skills=[]
    for i,n in enumerate((2,1,1,1,1)):
        skills.append({"skill_id":f"skill_{i}","rules":[{"rule_id":f"rule_{j}","conditions":{field:values[j%len(values)]},"weight":.5,"rationale":"fixture"} for j in range(n)],
            "ridge":1.,"prior_scale":.25,"semantic_mass_start":.3,"semantic_mass_end":.1,"ucb_weight":.6,"gp_beta_start":2.,"gp_beta_end":1.,"gp_xi":.01,"confidence":.7,"hypothesis":"synthetic fixture"})
    return {"skills":skills}

def response_for(pair,arm="labeled",content=None):
    if content is None: content=json.dumps(fixture_library(pair["arms"][arm]["payload"]["target"]["public_semantic_fields"]))
    return {"id":"fixture_response","object":"chat.completion","model":"openai/gpt-4o-mini","provider":"OpenAI","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":content}}],"usage":{"cost":.001,"is_byok":False}}

def classify(response,pair,arm="labeled"):
    return generation.classify_generation(response,arm,pair,response_sha256=common.object_hash(response),material_sha256=common.object_hash(pair),settled_cost=Decimal('.001'))

def test_approved_config(cfg):
    assert [len(x) for x in common.validate_config(cfg)]==[50,100]
    generation.validate_authorization(cfg)

@pytest.mark.parametrize("key,value",[("pair_count",11),("min_risk_adjusted_gain",0.0),("heldout_seed_count",101)])
def test_config_tamper(cfg,key,value):
    changed=copy.deepcopy(cfg);changed[key]=value
    with pytest.raises(ValueError): common.validate_config(changed)

def test_representation_and_baselines(target):
    original=rep.ORIGINAL_BUILDER();audit=rep.audit()
    assert audit["additional_verified_ligands"]==["SPhos","P(Cy)3","P(o-Tol)3","P(Ph)3","P(tBu)3"]
    assert len(target.candidates)==5760
    for a,b in zip(original.candidates,target.candidates):
        assert a.candidate_id==b.candidate_id and a.objective_value==b.objective_value
        assert rep.surrogate.candidate_features(original,a)==rep.surrogate.candidate_features(target,b)
    cat=rep.catalog(target)
    assert cat["ligand_mw_bin"]=={"mw_high":2880,"mw_mid":480,"mw_very_high":480,"none":480,"unknown":1440}
    assert cat["ligand_logp_bin"]=={"logp_high":1920,"logp_very_high":1920,"none":480,"unknown":1440}
    assert len(cat)==9 and "reagent_functional_class" not in cat
    types=rep.field_types(cat)
    assert types["ligand_mw_bin"]!=types["ligand_logp_bin"]
    assert types["ligand_has_phosphine"]==types["solvent_solvent_is_protic"]

def test_missing_and_roles(target):
    for c in target.candidates:
        md=c.metadata;v=rep.view(c)
        if md["ligand_raw_name"] in {"Xantphos","dppf","dtbpf"}: assert v["ligand_mw_bin"]=="unknown"
        if md["ligand_raw_name"]=="None": assert v["ligand_mw_bin"]=="none"
        if md["reactant_1_has_boron"]=="yes":
            assert v["electrophile_halide_type"]==md["reactant_2_halide_type"]
            assert v["boron_partner_boron_species"]==md["reactant_1_boron_species"]
    bad=replace(target.candidates[0],metadata={**target.candidates[0].metadata,"reactant_1_has_boron":"yes","reactant_2_has_boron":"yes"})
    with pytest.raises(ValueError): rep.view(bad)

def test_oracle_noninterference(target):
    altered=replace(target,candidates=tuple(replace(c,objective_value=-float(i),metadata={**c.metadata,"yield_value":12345+i}) for i,c in enumerate(target.candidates)))
    assert contract.build_payload(target)==contract.build_payload(altered)
    assert [rep.view(c) for c in target.candidates]==[rep.view(c) for c in altered.candidates]

def test_source_restriction(pair):
    e=pair["arms"]["labeled"]["payload"]["source_transfer_evidence"]
    assert e["source_observation_count"]==512 and e["source_dataset"]==rep.SOURCE
    assert {r["target_field"] for r in e["roles"]}==set(rep.ROLE_MAP.values())
    assert not any("additive" in r["source_field"] or "solvent" in r["target_field"] for r in e["roles"])
    assert all(r["target_field"] in rep.FIELDS for r in e["shared_value_priors"])

@pytest.mark.parametrize("index",range(10))
def test_pair_invariants(index):
    p=contract.build_pair(index,202610214000+index)
    assert p["invariants"]["inverse_mapping_exact"]
    assert p["invariants"]["numeric_leaves_equal"]
    assert p["invariants"]["explicit_label_denylist_hits"]==[]
    assert contract.base.map_tree(p["arms"]["anonymous"]["payload"],contract.base._combined(p["private_mapping"]["inverse"]))==p["arms"]["labeled"]["payload"]

@pytest.mark.parametrize("arm",["labeled","anonymous"])
def test_valid_response(pair,arm):
    decision=classify(response_for(pair,arm),pair,arm)
    assert decision.status==generation.VALID
    assert not contract.strict_validate_library(decision.library,pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])

@pytest.mark.parametrize("text",['{}','not JSON','{"skills":[],"skills":[]}','{"skills":NaN}'])
def test_invalid_whole_bank(pair,text):
    decision=classify(response_for(pair,content=text),pair)
    assert decision.status==generation.INVALID and decision.library is None

@pytest.mark.parametrize("field,value",[("model","different"),("provider","different"),("object","unexpected")])
def test_provenance_halts(pair,field,value):
    response=response_for(pair);response[field]=value
    with pytest.raises(generation.Halt): classify(response,pair)

@pytest.mark.parametrize("reason",["length","content_filter"])
def test_accounted_invalid_finish(pair,reason):
    response=response_for(pair);response["choices"][0]["finish_reason"]=reason
    assert classify(response,pair).status==generation.INVALID

def test_unknown_finish_halts(pair):
    response=response_for(pair);response["choices"][0]["finish_reason"]="unknown"
    with pytest.raises(generation.Halt): classify(response,pair)

def test_decoder_mismatch_halts(pair):
    bad=copy.deepcopy(pair);bad["private_mapping"]["inverse"]["values"]["V01"]="illegal"
    with pytest.raises(generation.Halt): classify(response_for(pair),bad)

def test_illegal_field_no_partial(pair):
    lib=fixture_library(pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])
    lib["skills"][0]["rules"][0]["conditions"]={"base_reagent_base_family":"guanidine_base"}
    decision=classify(response_for(pair,content=json.dumps(lib)),pair)
    assert decision.status==generation.INVALID and decision.library is None

def test_network_guard():
    with evaluation.matched.offline_guard():
        with pytest.raises(RuntimeError): socket.create_connection(("example.com",443))

def test_multiple_parent_nulls(cfg,pair):
    lib=fixture_library(pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])
    banks,prov,diag=evaluation.make_libraries({"pair_00_labeled":lib,"pair_01_labeled":lib,"pair_00_anonymous":lib},cfg)
    assert len(banks)==43 and all(len(b)==5 for b in banks.values())
    assert not any("r1_pair_02" in k for k in banks)
    for k,v in prov.items():
        if k.startswith(("r1_","r2_")):
            assert v["parent"] in ("pair_00_labeled","pair_01_labeled")
            assert [len(s.rules) for s in banks[k]]==[2,1,1,1,1]
    for pools in diag.values():
        for record in pools.values():
            assert record["max_accepted_joint_activation_fraction_deviation"]<=.005

def test_bootstrap_cluster_and_degenerate(cfg):
    c=copy.deepcopy(cfg);c["bootstrap_replicates"]=200
    result=analysis.summarize_tensor(np.ones((10,10,100))*2,c,conditional=True)
    assert result["interval_adjusted"]==[2.,2.] and result["stable_advantage"]
    zero=analysis.summarize_tensor(np.zeros((10,1,100)),c)
    assert zero["interval_adjusted"]==[0.,0.] and not zero["stable_advantage"]
    assert analysis.summarize_tensor(np.zeros((0,10,100)),c,conditional=True)["status"].startswith("unavailable")
    assert analysis.summarize_tensor(np.zeros((1,10,100)),c,conditional=True)["interval_adjusted"] is None
    # Ten parents, not 1000 independent cells: parent-constant outcomes retain uncertainty.
    a=np.arange(10)[:,None,None]*np.ones((10,10,100))
    out=analysis.summarize_tensor(a,c,conditional=True)
    assert out["interval_adjusted"][1]-out["interval_adjusted"][0]>1

def test_seed_history(cfg):
    audit=freeze.seed_history_audit(cfg)
    assert audit["files"] and not audit["collisions"]

def test_checkpoint_roundtrip_and_tamper(tmp_path):
    rows=[{"seed":7,"mode":m,"split":"calibration","final_best":2.,"best_so_far_auc":1.,"top10_hit":0.} for m in evaluation.matched.BASELINES]
    evaluation.store_seed(tmp_path,7,(rows,[],1.),"context")
    assert evaluation.read_seed(tmp_path,7,"context",evaluation.matched.BASELINES,"calibration")["rows"]==rows
    with pytest.raises(ValueError): evaluation.read_seed(tmp_path,7,"other",evaluation.matched.BASELINES,"calibration")
    (tmp_path/"7.traces.json.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError): evaluation.read_seed(tmp_path,7,"context",evaluation.matched.BASELINES,"calibration")

class FakeAPI:
    key="mock_not_a_real_credential"
    def __init__(self,invalid=False,unknown_billing=False): self.posts=0;self.invalid=invalid;self.unknown=unknown_billing
    def request(self,path,payload=None):
        if path=="/models": return {"data":[{"id":"openai/gpt-4o-mini","context_length":128000,"pricing":{"prompt":"0.00000015","completion":"0.0000006","request":"0"}}]}
        if path=="/key": return {"data":{"limit_remaining":2}}
        if path.startswith("/generation"): raise generation.TransportError(404,"not_ready")
        assert path=="/chat/completions"
        self.posts+=1
        data=json.loads(payload["messages"][1]["content"])
        content=json.dumps(fixture_library(data["target"]["public_semantic_fields"]))
        if self.invalid and self.posts==1: content="{}"
        return {"id":f"mock_{self.posts}","object":"chat.completion","model":"openai/gpt-4o-mini","provider":"OpenAI","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":content}}],"usage":{"cost":None if self.unknown else .001,"is_byok":False}}

def fake_freeze(tmp_path,cfg,monkeypatch):
    c=copy.deepcopy(cfg);c["generation_output_directory"]="generation"
    cp=tmp_path/"config.json";common.save(cp,c)
    fd=tmp_path/"freeze";fd.mkdir();(fd/"messages").mkdir();(fd/"private_pairs").mkdir()
    lock={"config_sha256":common.file_hash(cp),"source_sha256":{},"data_sha256":{},"input_sha256":{},"messages_sha256":{},"pair_material_sha256":{}}
    for i in range(10):
        pair=contract.build_pair(i,202610214000+i);name=f"pair_{i:02d}"
        path=fd/"private_pairs"/(name+".json");common.save(path,pair);lock["pair_material_sha256"][name]=common.file_hash(path)
        for arm in common.ARMS:
            name2=name+"_"+arm;path=fd/"messages"/(name2+".json");common.save(path,pair["arms"][arm]["messages"]);lock["messages_sha256"][name2]=common.file_hash(path)
    lp=tmp_path/"lock.json";common.save(lp,lock)
    monkeypatch.setattr(generation,"ROOT",tmp_path)
    monkeypatch.setattr(common,"verify_freeze",lambda *args:(c,lock))
    return c,cp,lp,fd,lock

@pytest.mark.parametrize("invalid",[False,True])
def test_twenty_mock_calls_and_receipts(tmp_path,cfg,monkeypatch,invalid):
    c,cp,lp,fd,lock=fake_freeze(tmp_path,cfg,monkeypatch)
    api=FakeAPI(invalid=invalid)
    result=generation.execute(cp,lp,fd,tmp_path/"generation",api=api,sleep=lambda x:None)
    assert api.posts==20 and result["actual_post_attempts"]==20
    assert result["invalid_library_count"]==int(invalid)
    outcomes,payloads,hashes=receipts.verify_generation(cp,lp,tmp_path/"generation",c,lock)
    assert len(outcomes)==20 and len(payloads)==20-int(invalid)
    with pytest.raises(generation.Halt): generation.execute(cp,lp,fd,tmp_path/"generation",api=api)
    assert api.posts==20

def test_unknown_cost_halts_without_post_retry(tmp_path,cfg,monkeypatch):
    c,cp,lp,fd,_=fake_freeze(tmp_path,cfg,monkeypatch);api=FakeAPI(unknown_billing=True)
    with pytest.raises(generation.Halt): generation.execute(cp,lp,fd,tmp_path/"generation",api=api,sleep=lambda x:None)
    assert api.posts==1
    failure=common.load(tmp_path/"generation/failure.json")
    assert failure["unresolved_reservation_usd"]=="0.045"
    assert not (tmp_path/"generation/completion.json").exists()


def test_selection_budget_and_invalid_fallback(cfg,pair):
    lib=fixture_library(pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])
    banks,_,_=evaluation.make_libraries({"pair_00_labeled":lib},cfg)
    outcomes={f"pair_{i:02d}_{arm}":{"pair_index":i,"arm":arm,"status":"VALID" if i==0 and arm=="labeled" else "INVALID_ANCHOR_FALLBACK","reason_codes":[]} for i in range(10) for arm in common.ARMS}
    modes=list(evaluation.matched.BASELINES)+[evaluation.matched.semantic.skill_mode(s) for b in banks.values() for s in b]
    seeds=set(range(50));rows=[]
    for seed in seeds:
        for idx,mode in enumerate(modes):
            score=50.0 if mode in evaluation.matched.BASELINES else 51.0+idx%5
            rows.append({"mode":mode,"seed":seed,"split":"calibration","final_best":score,"best_so_far_auc":score,"top10_hit":0.0})
    choices,anchor,_=evaluation.select(rows,banks,outcomes,seeds,cfg)
    assert len(choices)==40 # 20 scheduled L/A plus 20 null banks
    assert choices["pair_00_anonymous"]["primary_mode"]==anchor
    for name,bank in banks.items():
        allowed={evaluation.matched.semantic.skill_mode(s) for s in bank}
        assert choices[name]["primary_mode"] in allowed
        assert choices[name]["selected_mode"] in allowed|{anchor}
    corrupted=copy.deepcopy(rows);corrupted[0]["split"]="heldout"
    with pytest.raises(ValueError): evaluation.select(corrupted,banks,outcomes,seeds,cfg)


def test_real_executor_synthetic_fixture_and_resume(tmp_path,cfg,pair,monkeypatch):
    # Uses public component metadata only; all fixture outcomes below are synthetic.
    original=rep.ORIGINAL_BUILDER();candidates=[]
    for i in range(24):
        c=original.candidates[i*240]
        candidates.append(replace(c,objective_value=float((i*17)%100),metadata={**c.metadata,"yield_value":float((i*17)%100)}))
    fixture=replace(original,candidates=tuple(candidates))
    monkeypatch.setattr(rep,"ORIGINAL_BUILDER",lambda:fixture)
    monkeypatch.setitem(rep.replay.DATASET_BUILDERS,rep.TARGET,rep.build_adapter)
    lib=fixture_library(pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])
    bank,_=evaluation.legacy.normalize_without_loss(lib,pair["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"],0,"labeled")
    c=copy.deepcopy(cfg);c["reveal_rounds"]=2
    rows,info=evaluation.run_phase("calibration",{91,92},c,bank,tmp_path,"fixture",workers=1)
    assert len(rows)==16 and info["physical_trajectories"]<=16
    # Resume must read the same exact checkpoints without computing a new trajectory.
    monkeypatch.setattr(evaluation,"seed_job",lambda *args:(_ for _ in ()).throw(AssertionError("unexpected recompute")))
    rows2,_=evaluation.run_phase("calibration",{91,92},c,bank,tmp_path,"fixture",workers=1)
    assert rows2==rows
