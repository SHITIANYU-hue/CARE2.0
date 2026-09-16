#!/usr/bin/env python3
"""Distinguish rewritten prose from changed executable policies and experiments."""
import argparse
import json
from pathlib import Path

PARAMETERS = ('ridge', 'prior_scale', 'semantic_mass_start', 'semantic_mass_end',
              'ucb_weight', 'gp_beta_start', 'gp_beta_end', 'gp_xi')


def selected(lock, arm):
    return next(s for s in lock['states'][arm]['skills'] if s['skill_id'] == lock['selected'][arm])


def rules(skill):
    result = {}
    for rule in skill['rules']:
        key = json.dumps(sorted(dict(rule['conditions']).items()))
        result[key] = result.get(key, 0) + rule['weight']
    return result


def audit(root):
    config = json.loads((root / 'config.json').read_text())
    if not json.loads((root / 'run_status.json').read_text()).get('complete'):
        raise ValueError('Run is incomplete')
    chains = []
    for task in config['tasks']:
        for rep in config['model_replicates']:
            folder = root / task['id'] / f'replicate_{rep}'
            lock = json.loads((folder / 'deployment_lock.json').read_text())
            original = selected(lock, 'fixed_initial')
            before = rules(original)
            row = {'task': task['id'], 'replicate': rep, 'arms': {}}
            for arm in ('true_feedback', 'no_feedback', 'shuffled_feedback'):
                final = selected(lock, arm)
                after = rules(final)
                changes, rounds = 0, 0
                for seed in range(config['evaluation_seed_start'], config['evaluation_seed_start'] + config['evaluation_seed_count']):
                    get = lambda name: next(t for t in json.loads((folder / name / 'evaluation' / f'seed_{seed}.json').read_text()) if t['skill_id'] == lock['selected'][name])
                    old, new = get('fixed_initial'), get(arm)
                    if old['initial'] != new['initial'] or len(old['events']) != len(new['events']):
                        raise ValueError('Unmatched experiment trajectories')
                    for x, y in zip(old['events'], new['events']):
                        changes += x['selected_candidate'] != y['selected_candidate']
                        rounds += 1
                row['arms'][arm] = {
                    'added_rule_condition_groups': sorted(after.keys() - before.keys()),
                    'removed_rule_condition_groups': sorted(before.keys() - after.keys()),
                    'retained_condition_weight_changes': {k: {'before': before[k], 'after': after[k]} for k in before.keys() & after.keys() if before[k] != after[k]},
                    'controller_parameter_changes': {k: {'before': original[k], 'after': final[k]} for k in PARAMETERS if original[k] != final[k]},
                    'changed_experiment_selections': changes, 'compared_rounds': rounds,
                }
            chains.append(row)
    return {'status': 'execution_changes_audited', 'chains': chains,
            'claim_boundary': 'Candidate changes can result from earlier diverging observations. Changes alone do not establish improvement; compare the measured paired performance separately.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.result_dir)
    (args.result_dir / 'execution_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
