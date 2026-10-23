"""Suzuki pair material with BH-only source evidence and strict whole-library validation."""
import copy, json, random, hashlib
from pathlib import Path
from dataclasses import asdict, replace
from functools import lru_cache
import representation as rep
import paired_semantic_anonymization as base
import run_transfer_ablation as transfer
from run_matched_random_rule_controls import offline_guard
parse_strict_json = base.parse_strict_json
strict_validate_library = base.strict_validate_library
ValidationError = base.ValidationError
canonical_json, json_sha256 = base.canonical_json, base.json_sha256
SCHEMA_VERSION = "suzuki_second_dataset_v1_amendment1"


def build_payload(target=None, source=None):
    with offline_guard():
        target = rep.build_adapter() if target is None else target
        source = rep.replay.DATASET_BUILDERS[rep.SOURCE]() if source is None else source
        cat = rep.catalog(target)
        # Public target projection is independently unable to carry an oracle value.
        public = replace(target, decision_columns=rep.FIELDS,
            candidates=tuple(replace(c,objective_value=0.0,metadata=rep.view(c)) for c in target.candidates))
        observed = transfer.source_observations(source,202610213000,512)
        key=(rep.SOURCE,rep.TARGET)
        prior_fields=transfer.VALUE_PRIOR_FIELDS[key]
        try:
            transfer.VALUE_PRIOR_FIELDS[key]=set(rep.ROLE_MAP.items())
            card=transfer.compile_transfer_card(source,public,observed,rep.ROLE_MAP,0.65,3)
        finally:
            transfer.VALUE_PRIOR_FIELDS[key]=prior_fields
        historical=json.loads(base.RECORD.read_text())["prompt_payload"]
        payload={k:copy.deepcopy(historical[k]) for k in ("task","executor","design_requirements","allowed_ranges","output_contract")}
        payload["target"]={"dataset":rep.TARGET,"title":target.title,"objective":target.objective,
            "candidate_count":len(target.candidates),"public_semantic_fields":cat,"hidden_target":target.hidden_target,
            "boundary":"Field values and counts are public; target outcomes are not included."}
        payload["source_transfer_evidence"]={"source_dataset":rep.SOURCE,"source_objective":source.objective,
            "source_observation_count":len(observed),"roles":[asdict(x) for x in card.roles],
            "shared_value_priors":[asdict(x) for x in card.value_priors]}
        return base.common_payload(payload)


def input_hashes():
    paths=[Path(__file__),Path(rep.__file__),rep.REFERENCES,base.RECORD]
    return {str(p.relative_to(rep.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def aliases(payload,pair_index,seed):
    # Same label-only intervention as BH, with a new deterministic RNG namespace.
    cat=payload["target"]["public_semantic_fields"]
    fields=set(cat);values={v for counts in cat.values() for v in counts}
    e=payload["source_transfer_evidence"]
    for row in e["roles"]+e["shared_value_priors"]:
        fields.update((row["source_field"],row["target_field"]))
        if "value" in row: values.add(row["value"])
    fields.add(payload["target"]["hidden_target"])
    tasks={payload["target"]["dataset"],payload["target"]["title"],e["source_dataset"]}
    result={}
    for family,labels,prefix in (("fields",fields,"F"),("values",values,"V"),("tasks",tasks,"T")):
        codes=[f"{prefix}{i:02d}" for i in range(1,len(labels)+1)]
        rng=random.Random(int(json_sha256([SCHEMA_VERSION,seed,pair_index,family]),16));rng.shuffle(codes)
        result[family]=dict(zip(sorted(labels),codes))
    result["objectives"]={"maximize_yield":"maximize_O01"}
    base._combined(result)
    return result


def pair_from_payload(payload,pair_index,alias_seed):
    if type(pair_index) is not int or not 0<=pair_index<10 or type(alias_seed) is not int: raise ValueError("Invalid pair")
    sections=aliases(payload,pair_index,alias_seed)
    coded=base.map_tree(payload,base._combined(sections))
    inverse={family:{v:k for k,v in mapping.items()} for family,mapping in sections.items()}
    if base.map_tree(coded,base._combined(inverse))!=payload: raise ValueError("Not reversible")
    if base.numeric_leaves(coded)!=base.numeric_leaves(payload): raise ValueError("Numbers changed")
    arms={}
    for arm,value in (("labeled",payload),("anonymous",coded)):
        messages=base._messages(value)
        arms[arm]={"payload":value,"messages":messages,"payload_sha256":json_sha256(value),"messages_sha256":json_sha256(messages)}
    wire=canonical_json(arms["anonymous"]["messages"]).lower()
    deny=base.EXPLICIT_LABEL_DENYLIST+("boronic","boron_partner","electrophile","protic_alcohol","ferrocenyl")
    hits=[x for x in deny if x in wire]
    if hits: raise ValueError("Anonymous semantic leakage: "+str(hits))
    return {"schema_version":SCHEMA_VERSION,"pair_index":pair_index,"alias_seed":alias_seed,"input_hashes":input_hashes(),
        "arms":arms,"private_mapping":{"never_send_to_model":True,"forward":sections,"inverse":inverse},
        "invariants":{"numeric_leaf_count":len(base.numeric_leaves(payload)),"numeric_leaves_equal":True,"inverse_mapping_exact":True,"explicit_label_denylist_hits":hits}}

@lru_cache(maxsize=1)
def _payload(): return build_payload()

def build_pair(pair_index,alias_seed):
    return pair_from_payload(copy.deepcopy(_payload()),pair_index,alias_seed)

def validate_and_decode(content,arm,pair_material):
    if arm not in ("labeled","anonymous"): raise ValidationError(["Unknown arm"])
    if canonical_json(pair_material)!=canonical_json(build_pair(pair_material["pair_index"],pair_material["alias_seed"])):
        raise ValidationError(["Pair material changed"])
    payload=parse_strict_json(content)
    errors=strict_validate_library(payload,pair_material["arms"][arm]["payload"]["target"]["public_semantic_fields"])
    if errors: raise ValidationError(errors)
    decoded=copy.deepcopy(payload)
    if arm=="anonymous":
        inverse=pair_material["private_mapping"]["inverse"]
        for skill in decoded["skills"]:
            for rule in skill["rules"]:
                rule["conditions"]={inverse["fields"][f]:inverse["values"][v] for f,v in rule["conditions"].items()}
    errors=strict_validate_library(decoded,pair_material["arms"]["labeled"]["payload"]["target"]["public_semantic_fields"])
    if errors: raise ValidationError(errors)
    return decoded
