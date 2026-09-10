#!/usr/bin/env python3
"""Live, auditable LLM skill revision with fixed and inference-budget controls."""
from __future__ import annotations
import argparse
import concurrent.futures
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import urllib.request
import urllib.error

import numpy as np
import generate_llm_semantic_skills as generator
import llm_semantic_skills as semantic
import run_rsi_memory_component as offline
import run_synthetic_suzuki as replay

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def clean_text(text):
    secret = os.environ.get('COMMONSTACK_API_KEY', '')
    return text.replace(secret, '[REDACTED_CREDENTIAL]') if secret else text


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config):
    n = config['development_seeds_per_generation'] * config['updates']
    dev = set(range(config['development_seed_start'], config['development_seed_start'] + n))
    heldout = set(range(config['evaluation_seed_start'], config['evaluation_seed_start'] + config['evaluation_seed_count']))
    if dev & heldout:
        raise ValueError('Development/evaluation seeds overlap')
    if config['updates'] != 2 or config['skills_per_generation'] != 2:
        raise ValueError('This frozen runner requires two recursive updates and two skills')
    if config['evaluation_seed_count'] < 2 or config['development_seeds_per_generation'] < 2:
        raise ValueError('Insufficient seeds')
    if len(set(config['model_replicates'])) != len(config['model_replicates']):
        raise ValueError('Duplicate replicate IDs')
    if config['arms'] != ['true_feedback', 'no_feedback', 'shuffled_feedback']:
        raise ValueError('Unexpected arms')
    if len({s['id'] for s in config['tasks']}) != len(config['tasks']):
        raise ValueError('Duplicate task')
    if any(s['id'] == s['source'] for s in config['tasks']):
        raise ValueError('Source and target must differ')
    return dev, heldout


def finite_tree(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite LLM numeric parameter')
    if isinstance(value, dict):
        for x in value.values(): finite_tree(x)
    if isinstance(value, list):
        for x in value: finite_tree(x)


def normalize_response(content, catalog, count):
    parsed = replay.extract_json_object(content)
    finite_tree(parsed)
    raw = parsed.get('skills', [])
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError('Response must have exactly two skills')
    ids = [x.get('skill_id') for x in raw if isinstance(x, dict)]
    if len(set(ids)) != count:
        raise ValueError('Skill IDs must be unique')
    for skill in raw:
        rules = skill.get('rules', [])
        if not rules:
            raise ValueError('Skill has no rules')
        for rule in rules:
            conditions = rule.get('conditions')
            if not isinstance(conditions, dict) or not conditions:
                raise ValueError('Conditions must be a nonempty field/value object')
            if any(field not in catalog or str(value) not in catalog[field]
                   for field, value in conditions.items()):
                raise ValueError('Rule contains unsupported public field/value')
    skills = semantic.normalize_skills(parsed, catalog, count)
    if len(skills) != count:
        raise ValueError('Compiler dropped a skill')
    choice = str(parsed.get('selected_skill_id', ''))
    if choice not in {s.skill_id for s in skills}:
        raise ValueError('Selected skill ID must match a normalized executable skill')
    return {'skills': [asdict(s) for s in skills], 'selected_skill_id': choice,
            'lesson': str(parsed.get('lesson', '')),
            'revision_summary': str(parsed.get('revision_summary', ''))}


def generate(config, prompt, catalog, directory):
    """First valid response wins; never select model attempts using target performance."""
    key = os.environ['COMMONSTACK_API_KEY']
    system = ('You are a scientific optimization researcher revising executable semantic skills. '
              'Use only the evidence explicitly supplied. Never claim model-weight training or unseen '
              'target measurements. Return one valid JSON object with exactly two distinct skills. '
              'Every rule conditions value must exactly match the supplied public catalog. '
              'You may retain a successful skill and propose a counter-hypothesis. '
              'Select exactly one skill for deployment using selected_skill_id. '
              'Use short lowercase snake_case skill IDs. Include lesson and revision_summary strings. '
              'Use at most four rules per skill. Keep rationale and lesson concise.')
    messages = [{'role':'system','content':system},
                {'role':'user','content':json.dumps(prompt,ensure_ascii=False)}]
    for attempt in range(config['max_attempts_per_generation']):
        body = {'model':config['model'], 'messages':messages,
                'temperature':config['temperature'], 'max_tokens':config['max_tokens'],
                'response_format':{'type':'json_object'}}
        record = {'attempt':attempt, 'request':body, 'request_sha256':fingerprint(body),
                  'started_at':datetime.now(timezone.utc).isoformat(),
                  'credential_included_in_archive':False}
        save(directory / f'attempt_{attempt}_request.json', record)
        request = urllib.request.Request(config['base_url'].rstrip('/') + '/chat/completions',
            data=json.dumps(body).encode(), headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request,timeout=120) as response:
                payload = json.load(response)
            record['response'] = payload
            record['elapsed_seconds'] = time.monotonic() - started
            content = payload['choices'][0]['message']['content']
            record['raw_model_content'] = content
            decision = normalize_response(content, catalog, config['skills_per_generation'])
            record['normalized'] = decision
            record['status'] = 'valid'
            save(directory / f'attempt_{attempt}_response.json', record)
            save(directory / 'decision.json', decision)
            return decision
        except Exception as exc:
            record['status'] = 'failed'
            record['error_type'] = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                record['http_status'] = exc.code
            elif isinstance(exc, ValueError):
                record['validation_error'] = str(exc)
            save(directory / f'attempt_{attempt}_response.json', record)
            if attempt + 1 < config['max_attempts_per_generation']:
                messages = [*messages, {'role':'user','content':
                    'The prior attempt failed transport or schema validation. Return exactly the requested JSON with valid public fields, values, and selected_skill_id. Do not change the experimental evidence.'}]
    raise RuntimeError('No valid response after retained generation attempts')


def evaluate(spec, decision, seeds, config, phase, path, include_gp=False):
    rows=[]
    compact = json.loads(json.dumps(decision['skills']))
    for seed in seeds:
        # The existing executor updates GP online while skill parameters stay fixed.
        result=offline.run_job((spec['id'], compact, seed, config,
                                'evaluation' if include_gp else 'development'))
        for item in result: item['phase']=phase
        save(path / f'seed_{seed}.json',result)
        rows.extend(result)
    return rows


def feedback(rows, decision, arm, shuffle_seed):
    if arm == 'no_feedback':
        return None
    if any(row.get('phase') != 'development' for row in rows):
        raise ValueError('Evaluation outcomes cannot enter feedback')
    ids=[s['skill_id'] for s in decision['skills']]
    evidence={skill_id:[] for skill_id in ids}
    for row in rows:
        if row['skill_id'] in evidence: evidence[row['skill_id']].append(row['metrics'])
    source_ids=list(ids)
    if arm == 'shuffled_feedback':
        # With two skills, the only nonidentity permutation swaps the bindings.
        offset=random.Random(shuffle_seed).randrange(1,len(ids))
        source_ids=ids[offset:]+ids[:offset]
    out=[]
    for skill_id, source_id in zip(ids,source_ids):
        measurements=evidence[source_id]
        if not measurements: raise ValueError('Missing development feedback')
        out.append({'skill_id':skill_id, 'seeds':[r['seed'] for r in measurements],
                    'mean_auc':float(np.mean([r['best_so_far_auc'] for r in measurements])),
                    'mean_final_best':float(np.mean([r['final_best'] for r in measurements])),
                    'per_seed':[{'seed':r['seed'],'auc':r['best_so_far_auc'],
                                 'final_best':r['final_best']} for r in measurements]})
    return {'development_only':True,'skill_results':out}


def build_revision_prompt(base, previous, memories, arm):
    payload=json.loads(json.dumps(base))
    payload['task']='Revise the preceding executable skills for the same scientific search task. Return two skills and select one for deployment.'
    payload['previous_skills']=previous
    payload['persistent_experimental_memory']=memories if arm != 'no_feedback' else []
    payload['feedback_boundary']='Only completed development-seed summaries may appear here. Held-out evaluation outcomes are unavailable. No global KB retrieval or promotion is performed.'
    payload['revision_instruction']=('Use the supplied feedback to identify failures and improve the next skill generation. '
                                     'Do not infer causal mechanisms from a small set of seeds alone.'
                                     if arm != 'no_feedback' else
                                     'No measured development feedback is available. Reconsider the previous proposals using only the original scientific evidence and schema.')
    return payload


def run_case(spec, replicate, config, output):
    directory=output / spec['id'] / f'replicate_{replicate}'
    try:
        adapter=replay.DATASET_BUILDERS[spec['id']]()
        catalog=semantic.semantic_field_catalog(adapter)
        base=generator.build_prompt_payload(spec['source'],spec['id'],
            config['source_observations'],config['discount'],config['skills_per_generation'],
            'full',knowledge_context=None,proposal_mode='parametric')
        base['deployment_requirement']='Also output selected_skill_id matching one of the two skill IDs, lesson, and revision_summary. Deployment is frozen before evaluation.'
        initial=generate(config,base,catalog,directory/'initial_call')
        save(directory/'initial_state.json',initial)
        dev_start=config['development_seed_start']
        width=config['development_seeds_per_generation']
        initial_rows=evaluate(spec,initial,range(dev_start,dev_start+width),config,
                              'development',directory/'initial_development',include_gp=True)
        final={'fixed_initial':initial}
        for arm in config['arms']:
            current=json.loads(json.dumps(initial)); current_rows=initial_rows; memories=[]
            for generation in range(1,config['updates']+1):
                supplied=feedback(current_rows,current,arm,config['shuffle_seed']+replicate+generation)
                if supplied is not None:
                    memories.append({'generation_observed':generation-1,'previous_skills':current,'feedback':supplied})
                prompt=build_revision_prompt(base,current,memories,arm)
                parent_hash=fingerprint(current)
                new=generate(config,prompt,catalog,directory/arm/f'generation_{generation}'/'llm_call')
                save(directory/arm/f'generation_{generation}'/'state.json',
                     {'parent_state_sha256':parent_hash,'new_state_sha256':fingerprint(new),
                      'generation':generation,'arm':arm,'state':new,
                      'memory_sha256':fingerprint(memories),'experimental_only':True})
                current=new
                print(f'{spec["id"]} replicate {replicate} {arm} generation {generation} completed',flush=True)
                if generation < config['updates']:
                    start=dev_start+generation*width
                    current_rows=evaluate(spec,current,range(start,start+width),config,
                                          'development',directory/arm/f'generation_{generation}'/'development')
            final[arm]=current
        frozen={'selected':{k:v['selected_skill_id'] for k,v in final.items()},
                'states':final,'frozen_at':datetime.now(timezone.utc).isoformat(),
                'evaluation_feedback_permitted':False,
                'heldout_seeds':list(range(config['evaluation_seed_start'],config['evaluation_seed_start']+config['evaluation_seed_count']))}
        save(directory/'deployment_lock.json',frozen)
        heldout=[]
        for arm,decision in final.items():
            selected={'skills':[s for s in decision['skills'] if s['skill_id']==decision['selected_skill_id']]}
            rows=evaluate(spec,selected,frozen['heldout_seeds'],config,'evaluation',directory/arm/'evaluation',include_gp=arm=='fixed_initial')
            for row in rows:
                row['arm']='gp_ucb' if row['skill_id']=='gp_ucb' else arm
                row['replicate']=replicate
            heldout.extend(rows)
        save(directory/'evaluation_metrics.json',
             [{k:r[k] for k in ('task','seed','arm','replicate','skill_id','metrics')} for r in heldout])
        save(directory/'status.json',{'status':'complete','replicate':replicate,'task':spec['id']})
        return {'status':'complete','task':spec['id'],'replicate':replicate}
    except Exception as exc:
        status={'status':'failed','task':spec['id'],'replicate':replicate,'error_type':type(exc).__name__}
        save(directory/'status.json',status)
        return status


def run(config_path,output,workers):
    config=json.loads(config_path.read_text());validate_config(config)
    if not os.environ.get('COMMONSTACK_API_KEY'):
        raise RuntimeError('Configure COMMONSTACK_API_KEY in the environment')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite experiment evidence')
    output.mkdir(parents=True,exist_ok=True)
    save(output/'config.json',config)
    save(output/'protocol_lock.json',{
        'created_at':datetime.now(timezone.utc).isoformat(), 'config_sha256':fingerprint(config),
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'implementation_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/'scripts').glob('*.py')},
        'python':platform.python_version(),'numpy':np.__version__,
        'planned_valid_generation_calls':len(config['tasks'])*len(config['model_replicates'])*(1+len(config['arms'])*config['updates']),
        'model_rng_seed_supported':False,'claim_boundary':config['claim_boundary']})
    jobs=[(spec,rep) for spec in config['tasks'] for rep in config['model_replicates']]
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_case,spec,rep,config,output) for spec,rep in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            save(output/'run_status.json',{'cases':results,'planned_cases':len(jobs)})
            print(json.dumps(result),flush=True)
    save(output/'run_status.json',{'cases':results,'planned_cases':len(jobs),'complete':all(r['status']=='complete' for r in results)})
    if any(r['status'] != 'complete' for r in results):
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();run(args.config,args.output_dir,args.workers)
