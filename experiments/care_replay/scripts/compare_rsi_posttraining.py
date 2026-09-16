#!/usr/bin/env python3
"""Matched frozen-base / trained-adapter RSI comparison, with pairing checks."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from statistics import mean

ARMS = ('fixed_initial', 'true_feedback', 'no_feedback', 'shuffled_feedback', 'gp_ucb')
MATCHED = ('base_model_path', 'temperature', 'max_tokens', 'model_replicates',
           'development_seed_start', 'development_seeds_per_generation',
           'evaluation_seed_start', 'evaluation_seed_count', 'initial_observations',
           'reveal_rounds', 'source_observations', 'discount', 'kernel', 'tasks', 'updates',
           'local_generation_seed', 'local_context_limit', 'generation_backend',
           'max_attempts_per_generation', 'skills_per_generation', 'arms',
           'shuffle_seed', 'bootstrap_seed', 'bootstrap_draws', 'constraints')


def read_complete(root, task_ids=None):
    config = json.loads((root / 'config.json').read_text())
    overall_complete = json.loads((root / 'run_status.json').read_text()).get('complete')
    if not overall_complete and task_ids is None:
        raise ValueError('Incomplete RSI run cannot be compared without an explicit completed task subset')
    selected = [spec for spec in config['tasks'] if task_ids is None or spec['id'] in task_ids]
    if task_ids is not None and {spec['id'] for spec in selected} != set(task_ids):
        raise ValueError('Requested task subset is absent from config')
    rows = []
    for spec in selected:
        for rep in config['model_replicates']:
            folder = root / spec['id'] / f'replicate_{rep}'
            if json.loads((folder / 'status.json').read_text()).get('status') != 'complete':
                raise ValueError(f"Requested case is incomplete: {spec['id']} replicate {rep}")
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
    expected = len(selected) * len(config['model_replicates']) * config['evaluation_seed_count'] * len(ARMS)
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


def matched_difference(values, bootstrap_seed=20260910, bootstrap_draws=5000):
    if not values or bootstrap_draws < 1:
        raise ValueError('Bootstrap requires values and a positive draw count')
    rng = random.Random(bootstrap_seed)
    boot = sorted(mean(rng.choice(values) for _ in values) for _ in range(bootstrap_draws))
    lower = boot[max(0, int(0.025 * bootstrap_draws) - 1)]
    upper = boot[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))]
    return {'mean': mean(values), 'paired_wins': sum(x > 1e-8 for x in values),
            'paired_losses': sum(x < -1e-8 for x in values),
            'paired_ties': sum(abs(x) <= 1e-8 for x in values), 'paired_values': values,
            'paired_seed_bootstrap_mean_ci_95': [lower, upper],
            'bootstrap_seed': bootstrap_seed, 'bootstrap_draws': bootstrap_draws}


def compare(base_root, adapter_root, task_ids=None):
    base_config, base = read_complete(base_root, task_ids)
    adapter_config, adapter = read_complete(adapter_root, task_ids)
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
    selected_specs = [spec for spec in base_config['tasks'] if task_ids is None or spec['id'] in task_ids]
    if task_ids is not None and not selected_specs:
        raise ValueError('Requested task subset is absent from config')
    tasks = []
    for spec in selected_specs:
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
                adapter[k + (arm,)]['best_so_far_auc'] - base[k + (arm,)]['best_so_far_auc'] for k in keys], base_config.get('bootstrap_seed', 20260910), base_config.get('bootstrap_draws', 5000))
        for name, index in [('base', base), ('adapter', adapter)]:
            result['feedback_effects'][name] = {
                control: matched_difference([
                    index[k + ('true_feedback',)]['best_so_far_auc'] - index[k + (control,)]['best_so_far_auc'] for k in keys], base_config.get('bootstrap_seed', 20260910), base_config.get('bootstrap_draws', 5000))
                for control in ('fixed_initial', 'no_feedback', 'shuffled_feedback')}
        result['change_in_recursive_update_gain'] = matched_difference([
            (adapter[k + ('true_feedback',)]['best_so_far_auc'] - adapter[k + ('fixed_initial',)]['best_so_far_auc'])
            - (base[k + ('true_feedback',)]['best_so_far_auc'] - base[k + ('fixed_initial',)]['best_so_far_auc']) for k in keys], base_config.get('bootstrap_seed', 20260910), base_config.get('bootstrap_draws', 5000))
        tasks.append(result)
    return {
        'status': 'matched_task_subset_diagnostic_complete' if task_ids is not None else 'matched_diagnostic_complete', 'tasks': tasks,
        'primary_comparison': 'adapter true-feedback minus frozen-base true-feedback AUC',
        'checks': {'matched_settings': True, 'same_gp_control': True, 'same_initial_observations': True, 'heldout_feedback_excluded': True},
        'claim_boundary': base_config.get('claim_boundary', 'Diagnostic only; few model chains do not establish reliability or generalization.'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', action='append', help='Explicit completed task subset; permits recovery from an unrelated incomplete case')
    args = parser.parse_args()
    summary = compare(args.base_dir, args.adapter_dir, args.task)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
