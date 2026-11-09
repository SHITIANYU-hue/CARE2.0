"""Public projections and source-only evidence. No API entry point."""
from .common import *
from dataclasses import asdict, replace
from copy import deepcopy
from functools import lru_cache
from collections import Counter
import math
import run_synthetic_suzuki as replay
import llm_semantic_skills as semantic
import run_transfer_ablation as transfer
from protocols.suzuki_second_dataset_v1 import representation as suzuki
@lru_cache(maxsize=None)
def adapter(dataset):
    with offline():
        return suzuki.build_adapter() if dataset==suzuki.TARGET else replay.DATASET_BUILDERS[dataset]()
def catalog(a):return suzuki.catalog(a) if a.dataset_id==suzuki.TARGET else semantic.semantic_field_catalog(a)
def public_projection(a,fields):
    fields=tuple(sorted(set(fields)))
    return replace(a,decision_columns=fields,candidates=tuple(replace(c,objective_value=0.,x1=0.,x2=0.,x3=0.,numeric_features=(),metadata={f:str(c.metadata.get(f,'')) for f in fields}) for c in a.candidates))
def role_map(route):
    return dict(suzuki.ROLE_MAP) if route['target']==suzuki.TARGET else transfer.descriptor_transfer_role_map_for(route['source'],route['target'])
def evidence(route,mode,source=None,target=None):
    if mode=='target_only':return {'evidence_mode':mode,'source_dataset':None,'source_objective':None,'mapped_roles':[],'source_outcome_statistics':None,'shared_value_priors':[]}
    source=adapter(route['source']) if source is None else source
    roles=role_map(route)
    e={'evidence_mode':mode,'source_dataset':source.dataset_id,'source_objective':source.objective,'mapped_roles':[{'source_field':s,'target_field':t} for s,t in sorted(roles.items())],'source_outcome_statistics':None,'shared_value_priors':[]}
    if mode=='source_schema_only':return e
    if mode!='full':raise Halt('Unknown evidence mode')
    target=adapter(route['target']) if target is None else target
    public=public_projection(target,roles.values())
    observed=transfer.source_observations(source,route['source_seed_proposed'],route['source_observations_proposed'])
    if len(observed)!=route['source_observations_proposed']:raise Halt('Source observations truncated')
    key=(source.dataset_id,target.dataset_id)
    prior=transfer.VALUE_PRIOR_FIELDS.get(key)
    try:
        if target.dataset_id==suzuki.TARGET:transfer.VALUE_PRIOR_FIELDS[key]=set(roles.items())
        card=transfer.compile_transfer_card(source,public,observed,roles,.65,3)
    finally:
        if prior is None:transfer.VALUE_PRIOR_FIELDS.pop(key,None)
        else:transfer.VALUE_PRIOR_FIELDS[key]=prior
    e['source_outcome_statistics']={'source_observation_count':len(observed),'roles':[asdict(r) for r in card.roles]}
    e['shared_value_priors']=[asdict(r) for r in card.value_priors]
    return e

def payload(config,cell,source=None,target=None):
    target=adapter(cell['target']) if target is None else target
    cat=catalog(target)
    route=next((r for r in config['routes'] if r['source']==cell['source'] and r['target']==cell['target']),None)
    mode=cell['evidence_mode']
    e=evidence(route,mode,source,target)
    # Reuse the previously approved grammar/ranges, never historical outcomes or KB.
    historical=load(contract.RECORD)['prompt_payload']
    template={k:deepcopy(historical[k]) for k in ['task','executor','design_requirements','allowed_ranges','output_contract']}
    template['source_transfer_evidence']=e
    result=contract.common_payload(template)
    result['target']={'dataset':target.dataset_id,'title':target.title,'objective':target.objective,'candidate_count':len(target.candidates),'public_semantic_fields':cat,'hidden_target':target.hidden_target,'boundary':'Only public schema and counts are supplied; no target outcomes or candidate identifiers.'}
    # The same two instructions apply to all arms; the evidence section alone varies.
    req=result['design_requirements']
    req[7]='Ground proposals only in the supplied information. Do not invent source outcome evidence; source evidence may be absent.'
    req[8]='Include a counter-hypothesis that reverses or downweights an uncertain relationship.'
    result['source_transfer_evidence']=e
    result['executor']['selection']='Fifty calibration seeds select exactly one of five strategies per library. Freeze all choices before one hundred held-out seeds. No library-level winner selection.'
    return result

def messages(p): return contract._messages(p)
def certify(a):
    ids=[c.candidate_id for c in a.candidates]
    if len(ids)!=len(set(ids)):raise Halt('Duplicate candidate IDs')
    if not all(math.isfinite(c.objective_value) for c in a.candidates):raise Halt('Nonfinite oracle')
    cat=catalog(a)
    if not cat:raise Halt('Empty rule catalog')
    return {'dataset':a.dataset_id,'candidate_count':len(ids),'objective':a.objective,'hidden_target':a.hidden_target,'candidate_ids_sha256':digest(ids),'complete_candidate_records_sha256':digest([asdict(c) for c in a.candidates]),'catalog':cat,'public_rule_metadata_sha256':digest([{f:str(c.metadata.get(f,'')) for f in cat} for c in a.candidates])}

def prepare(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with offline():
        certs={d:certify(adapter(d)) for d in sorted({c[k] for c in config['generation_cells'] for k in ['source','target'] if c[k]})}
        save(output/'pool_certificates.json',certs)
        unique={c['cell']:c for c in config['generation_cells']}; manifest={}
        for key,c in unique.items():
            p=payload(config,c);m=messages(p);save(output/'prompts'/f'{key}.json',{'payload':p,'messages':m})
            manifest[key]={'payload_sha256':digest(p),'messages_sha256':digest(m),'serialized_utf8_bytes':len(json.dumps(m,ensure_ascii=False).encode()),'body_utf8_bytes':sum(len(x['content'].encode()) for x in m),'input_token_bound_certified':False}
        audits=[]
        for r in config['routes']:
            s,t=adapter(r['source']),adapter(r['target']);rs=role_map(r);e=evidence(r,'full');obs=transfer.source_observations(s,r['source_seed_proposed'],r['source_observations_proposed'])
            overlap=None
            if all('smiles' in c.metadata for c in (*s.candidates,*t.candidates)):
                from rdkit import Chem
                def canonical(c):
                    mol=Chem.MolFromSmiles(c.metadata['smiles'])
                    if mol is None:raise Halt('Entity identity parse failure')
                    return Chem.MolToSmiles(mol)
                overlap={'method':'RDKit canonical SMILES exact match, no salt/stereo standardization beyond canonicalization','source_fit_unique_entities':len({canonical(c) for c in obs}),'target_unique_entities':len({canonical(c) for c in t.candidates}),'shared_source_fit_target_entities':len({canonical(c) for c in obs}&{canonical(c) for c in t.candidates})}
            audits.append({'route':r['route_id'],'source_observations':len(obs),'source_fit_ids_sha256':digest([c.candidate_id for c in obs]),'roles':rs,'outside_executable_target_catalog':{s:tfield for s,tfield in rs.items() if tfield not in catalog(t)},'source_supported_roles':sum(x['source_support_count']>0 for x in e['source_outcome_statistics']['roles']),'shared_value_prior_count':len(e['shared_value_priors']),'entity_overlap':overlap})
        save(output/'prompt_manifest.json',manifest);save(output/'route_audit.json',audits)
    return manifest,audits
