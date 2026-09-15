#!/usr/bin/env python3
"""Matched frozen-base / trained-adapter RSI comparison, with pairing checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean

ARMS = ('fixed_initial', 'true_feedback', 'no_feedback', 'shuffled_feedback', 'gp_ucb')
MATCHED = ('base_model_path', 'temperature', 'max_tokens', 'model_replicates',
           'development_seed_start', 'development_seeds_per_generation',
           'evaluation_seed_start', 'evaluation_seed_count', 'initial_observations',
           'reveal_rounds', 'source_observations', 'discount', 'kernel', 'tasks', 'updates',
           'local_generation_seed', 'local_context_limit', 'generation_backend',
           'max_attempts_per_generation', 'skills_per_generation', 'arms',
           'shuffle_seed', 'constraints')


def read_complete(root):
    config = json.loads((root / 'config.json').read_text())
    if not json.loads((root / 'run_status.json').read_text()).get('complete'):
        raise ValueError('Incomplete RSI run cannot be compared')
    rows = []
    for spec in config['tasks']:
        for rep in config['model_replicates']:
            folder = root / spec['id'] / f'replicate_{rep}'
            lock = json.loads((folder / 'deployment_lock.json').read_text())
            if lock.get('evaluation_feedback_permitted') is not False:
                raise ValueError('Heldout feedback boundary was not locked')
            heldout = set(lock['heldout_seeds'])
            for request in folder.rglob('attempt_*_request.json'):
                prompt = json.loads(json.loads(request.read_text())['request']['messages'][1]['content'])
                for memory in prompt.get('persistent_experimental_memory', []):
                    for result in memory['feedback']['skill_results']:
                        if heldout.intersection(result['seeds']):
                            raise ValueError('Heldout seed entered revision prompt')
            rows.extend(json.loads((folder / 'evaluation_metrics.json').read_text()))
    index = {(r['task'], r['replicate'], r['seed'], r['arm']): r['metrics'] for r in rows}
    expected = len(config['tasks']) * len(config['model_replicates']) * config['evaluation_seed_count'] * len(ARMS)
    if len(rows) != len(index) or len(index) != expected:
        raise ValueError('Missing or duplicate paired evaluation metrics')
    for row in rows:
        arm = 'fixed_initial' if row['arm'] == 'gp_ucb' else row['arm']
        path = root / row['task'] / f"replicate_{row['replicate']}" / arm / 'evaluation' / f"seed_{row['seed']}.json"
        trajectory = next(x for x in json.loads(path.read_text()) if x['skill_id'] == row['skill_id'])
        if len(trajectory['events']) != config['reveal_rounds'] or len(trajectory['initial']['candidate_ids']) != config['initial_observations']:
            raise ValueError('Actual initial/reveal budget differs from protocol')
        key = row['task'], row['replicate'], row['seed'], row['arm']
        index[key] = {**index[key], '_initial': trajectory['initial']}
    return config, index


def matched_difference(values):
    return {'mean': mean(values), 'paired_wins': sum(x > 1e-8 for x in values),
            'paired_losses': sum(x < -1e-8 for x in values),
            'paired_ties': sum(abs(x) <= 1e-8 for x in values), 'paired_values': values}


def compare(base_root, adapter_root):
    base_config, base = read_complete(base_root)
    adapter_config, adapter = read_complete(adapter_root)
    for key in MATCHED:
        if base_config.get(key) != adapter_config.get(key):
            raise ValueError(f'Unmatched experimental setting: {key}')
    if base_config.get('adapter_path') or not adapter_config.get('adapter_path'):
        raise ValueError('Expected a frozen base and a trained adapter condition')
    if set(base) != set(adapter):
        raise ValueError('Evaluation seed/arm pairing differs')
    for key in base:
        fixed_key = key[:-1] + ('fixed_initial',)
        if base[key]['_initial'] != adapter[key]['_initial'] or base[key]['_initial'] != base[fixed_key]['_initial']:
            raise ValueError('Paired initial candidates or measured values differ')
    tasks = []
    for spec in base_config['tasks']:
        keys = [(spec['id'], rep, seed) for rep in base_config['model_replicates']
                for seed in range(base_config['evaluation_seed_start'],
                                  base_config['evaluation_seed_start'] + base_config['evaluation_seed_count'])]
        if any(abs(base[k + ('gp_ucb',)]['best_so_far_auc'] - adapter[k + ('gp_ucb',)]['best_so_far_auc']) > 1e-8 for k in keys):
            raise ValueError('Matched GP control changed across model conditions')
        result = {'task': spec['id'], 'mean_auc': {}, 'adapter_minus_base': {}, 'feedback_effects': {}}
        for arm in ARMS:
            result['mean_auc'][arm] = {
                'base': mean(base[k + (arm,)]['best_so_far_auc'] for k in keys),
                'adapter': mean(adapter[k + (arm,)]['best_so_far_auc'] for k in keys),
            }
            result['adapter_minus_base'][arm] = matched_difference([
                adapter[k + (arm,)]['best_so_far_auc'] - base[k + (arm,)]['best_so_far_auc'] for k in keys])
        for name, index in [('base', base), ('adapter', adapter)]:
            result['feedback_effects'][name] = {
                control: matched_difference([
                    index[k + ('true_feedback',)]['best_so_far_auc'] - index[k + (control,)]['best_so_far_auc'] for k in keys])
                for control in ('fixed_initial', 'no_feedback', 'shuffled_feedback')}
        result['change_in_recursive_update_gain'] = matched_difference([
            (adapter[k + ('true_feedback',)]['best_so_far_auc'] - adapter[k + ('fixed_initial',)]['best_so_far_auc'])
            - (base[k + ('true_feedback',)]['best_so_far_auc'] - base[k + ('fixed_initial',)]['best_so_far_auc']) for k in keys])
        tasks.append(result)
    return {
        'status': 'matched_diagnostic_complete', 'tasks': tasks,
        'primary_comparison': 'adapter true-feedback minus frozen-base true-feedback AUC',
        'checks': {'matched_settings': True, 'same_gp_control': True, 'same_initial_observations': True, 'heldout_feedback_excluded': True},
        'claim_boundary': 'Same-task diagnostic. Few model chains do not establish call-level reliability or task-family generalization. Training improves RSI only if matched base, no-feedback and shuffled-feedback comparisons replicate on families unused for training and checkpoint selection.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.base_dir, args.adapter_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
