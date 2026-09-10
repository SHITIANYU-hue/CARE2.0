#!/usr/bin/env python3
"""Secondary diagnostic: final true-feedback update versus its first update."""
import argparse
import json
from pathlib import Path
import run_rsi_live_feedback as live


def evaluate(root):
    config=json.loads((root/'config.json').read_text())
    if not (root/'first_update_diagnostic_lock.json').exists():
        raise ValueError('Missing diagnostic analysis lock')
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
    evaluate(p.parse_args().root)
