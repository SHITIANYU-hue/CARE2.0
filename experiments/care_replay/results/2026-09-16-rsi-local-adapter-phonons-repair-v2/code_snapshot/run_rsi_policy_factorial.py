#!/usr/bin/env python3
"""Post-hoc 2x2 replay of frozen semantic rules and search-controller parameters."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean

import audit_rsi_execution as execution
import compare_rsi_posttraining as paired
import run_rsi_live_feedback as live


def hybrid(rules_source, controller_source, name):
    skill = copy.deepcopy(rules_source)
    skill['skill_id'] = name
    for key in execution.PARAMETERS:
        skill[key] = controller_source[key]
    return skill


def run(base_dir, adapter_dir, output):
    # Validate the complete paired experiment before constructing any hybrids.
    paired.compare(base_dir, adapter_dir)
    config = json.loads((base_dir / 'config.json').read_text())
    if output.exists():
        raise FileExistsError('Refusing to overwrite attribution evidence')
    output.mkdir(parents=True)
    live.save(output / 'protocol_lock.json', {
        'experiment_class': 'posthoc_frozen_policy_factorial',
        'config': config, 'base_dir': str(base_dir), 'adapter_dir': str(adapter_dir),
        'controller_fields': list(execution.PARAMETERS),
        'semantic_factor': 'Rule conditions AND their weights; not conditions-only or weights-only.',
        'new_llm_calls': 0, 'model_weight_changes': False,
        'claim_boundary': 'Descriptive attribution using prior evaluation seeds. No checkpoint selection, policy tuning, or independent confirmation. Hybrid performance measures this executor and pool, not biological mechanisms.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    results = []
    for spec in config['tasks']:
        for rep in config['model_replicates']:
            folders = [root / spec['id'] / f'replicate_{rep}' for root in (base_dir, adapter_dir)]
            locks = [json.loads((p / 'deployment_lock.json').read_text()) for p in folders]
            policies = [execution.selected(lock, 'fixed_initial') for lock in locks]
            seeds = locks[0]['heldout_seeds']
            matrix = {}
            for rule_index, rule_name in enumerate(('base', 'adapter')):
                for controller_index, controller_name in enumerate(('base', 'adapter')):
                    name = f'{rule_name}_rules_{controller_name}_controller'
                    skill = hybrid(policies[rule_index], policies[controller_index], name)
                    live.save(output / spec['id'] / f'replicate_{rep}' / name / 'policy.json', skill)
                    rows = live.evaluate(spec, {'skills': [skill]}, seeds, config,
                                         'posthoc_attribution', output / spec['id'] / f'replicate_{rep}' / name / 'evaluation')
                    matrix[name] = {r['seed']: r['metrics']['best_so_far_auc'] for r in rows}
                    # Pure controls must reproduce original trajectories exactly.
                    if rule_index == controller_index:
                        for row in rows:
                            original = next(t for t in json.loads((folders[rule_index] / 'fixed_initial' / 'evaluation' / f"seed_{row['seed']}.json").read_text()) if t['skill_id'] == locks[rule_index]['selected']['fixed_initial'])
                            original_metrics = {k: v for k, v in original['metrics'].items() if k != 'mode'}
                            replay_metrics = {k: v for k, v in row['metrics'].items() if k != 'mode'}
                            if original['initial'] != row['initial'] or original_metrics != replay_metrics:
                                raise ValueError('Pure control replay changed original measurements or metrics')
                            if [e['selected_candidate'] for e in original['events']] != [e['selected_candidate'] for e in row['events']]:
                                raise ValueError('Pure control replay changed selected candidates')
            b_b, a_b, b_a, a_a = [matrix[x] for x in ('base_rules_base_controller', 'adapter_rules_base_controller', 'base_rules_adapter_controller', 'adapter_rules_adapter_controller')]
            differences = {
                'semantic_gain_with_base_controller': [a_b[s] - b_b[s] for s in seeds],
                'semantic_gain_with_adapter_controller': [a_a[s] - b_a[s] for s in seeds],
                'controller_gain_with_base_rules': [b_a[s] - b_b[s] for s in seeds],
                'controller_gain_with_adapter_rules': [a_a[s] - a_b[s] for s in seeds],
                'interaction': [a_a[s] - a_b[s] - b_a[s] + b_b[s] for s in seeds],
                'total_initial_policy_gain': [a_a[s] - b_b[s] for s in seeds],
            }
            results.append({'task': spec['id'], 'replicate': rep,
                            'mean_auc': {k: mean(v.values()) for k, v in matrix.items()},
                            'effects': {k: paired.matched_difference(v) for k, v in differences.items()},
                            'pure_control_trajectories_reproduced': True})
    live.save(output / 'factorial_summary.json', {'status': 'complete', 'results': results,
                                                 'evaluation_trajectories': len(results) * config['evaluation_seed_count'] * 4})
    live.save(output / 'run_status.json', {'complete': True})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir', type=Path, required=True)
    p.add_argument('--adapter-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    run(a.base_dir, a.adapter_dir, a.output_dir)
