#!/usr/bin/env python3
"""Secondary diagnostic: final true-feedback update versus its first update."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import run_rsi_live_feedback as live


def evaluate(root, allow_posthoc=False):
    config=json.loads((root/'config.json').read_text())
    lock_path=root/'first_update_diagnostic_lock.json'
    if not lock_path.exists():
        if not allow_posthoc:
            raise ValueError('Missing diagnostic analysis lock')
        run_status=root/'run_status.json'
        live.save(lock_path,{
            'created_at':datetime.now(timezone.utc).isoformat(),
            'status':'posthoc_after_primary_execution_before_first_update_evaluation',
            'comparison':'second_true_feedback_update_minus_first_true_feedback_update',
            'primary_run_status_sha256':hashlib.sha256(run_status.read_bytes()).hexdigest(),
            'reason':'Descriptive replication of the earlier first-versus-second-update diagnostic',
            'extra_evaluation_trajectories':len(config['tasks'])*len(config['model_replicates'])*config['evaluation_seed_count'],
            'primary_protocol_unchanged':True,
            'claim_boundary':'Post-hoc secondary diagnostic; it cannot be treated as a preregistered endpoint.'})
    records=[]
    for spec in config['tasks']:
        for replicate in config['model_replicates']:
            folder=root/spec['id']/f'replicate_{replicate}'
            decision=json.loads((folder/'true_feedback/generation_1/llm_call/decision.json').read_text())
            selected={'skills':[s for s in decision['skills'] if s['skill_id']==decision['selected_skill_id']]}
            seeds=range(config['evaluation_seed_start'],config['evaluation_seed_start']+config['evaluation_seed_count'])
            rows=live.evaluate(spec,selected,seeds,config,'evaluation',folder/'first_update_only/evaluation')
            for row in rows:
                records.append({'task':spec['id'],'replicate':replicate,'seed':row['seed'],
                                'arm':'first_update_only','skill_id':row['skill_id'],'metrics':row['metrics']})
    live.save(root/'first_update_evaluation_metrics.json',records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--allow-posthoc',action='store_true')
    args=p.parse_args();evaluate(args.root,args.allow_posthoc)
