#!/usr/bin/env python3
"""Trajectory-grounded RSI with an incumbent/challenger promotion gate.

This runner isolates two changes to the existing live-feedback loop:

1. rich feedback is compiled only from candidates actually revealed on
   development seeds; and
2. a proposed revision is deployed only after it beats the incumbent on a
   disjoint promotion seed split under a predeclared rule.

The final evaluation split is locked before any generation call and is never
serialized into an LLM prompt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

import generate_llm_semantic_skills as generator
import llm_semantic_skills as semantic
import run_rsi_live_feedback as live
import run_synthetic_suzuki as replay

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    required_positive = (
        'updates', 'development_seeds_per_generation',
        'promotion_seeds_per_generation', 'evaluation_seed_count',
        'bootstrap_draws',
    )
    for key in required_positive:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['updates'] != 2 or config['skills_per_generation'] != 2:
        raise ValueError('This diagnostic freezes two updates and two skills')
    if config['branches'] != ['sparse_ungated', 'rich_gated']:
        raise ValueError('Unexpected branches')
    if len(config['tasks']) != 1:
        raise ValueError('This bounded diagnostic requires exactly one task')
    development = set(range(
        config['development_seed_start'],
        config['development_seed_start']
        + config['updates'] * config['development_seeds_per_generation'],
    ))
    promotion = set(range(
        config['promotion_seed_start'],
        config['promotion_seed_start']
        + config['updates'] * config['promotion_seeds_per_generation'],
    ))
    evaluation = set(range(
        config['evaluation_seed_start'],
        config['evaluation_seed_start'] + config['evaluation_seed_count'],
    ))
    splits = {'development': development, 'promotion': promotion, 'evaluation': evaluation}
    for left, right in (('development', 'promotion'), ('development', 'evaluation'),
                        ('promotion', 'evaluation')):
        if splits[left] & splits[right]:
            raise ValueError(f'{left}/{right} seeds overlap')
    if config['promotion_min_wins'] > config['promotion_seeds_per_generation']:
        raise ValueError('promotion_min_wins exceeds the promotion seed count')
    return splits


def selected_only(decision):
    selected = decision['selected_skill_id']
    matches = [skill for skill in decision['skills'] if skill['skill_id'] == selected]
    if len(matches) != 1:
        raise ValueError('Selected skill must resolve uniquely')
    return {'skills': matches, 'selected_skill_id': selected}


def validate_initial_decision(decision, catalog, count):
    normalized = semantic.normalize_skills(decision, catalog, count)
    if len(normalized) != count:
        raise ValueError('Pinned initial decision does not compile to two skills')
    ids = {skill.skill_id for skill in normalized}
    if decision.get('selected_skill_id') not in ids:
        raise ValueError('Pinned initial selected skill is invalid')
    return {
        'skills': [semantic.asdict(skill) for skill in normalized],
        'selected_skill_id': decision['selected_skill_id'],
        'lesson': str(decision.get('lesson', '')),
        'revision_summary': str(decision.get('revision_summary', '')),
    }


def _candidate_features(candidate, catalog):
    return {
        field: str(candidate.metadata[field])
        for field in sorted(catalog)
        if field in candidate.metadata and str(candidate.metadata[field]) in catalog[field]
    }


def rich_feedback(rows, decision, adapter, catalog, examples_per_tail=3):
    """Compile measured trajectory evidence without reading unobserved labels."""
    if any(row.get('phase') != 'development' for row in rows):
        raise ValueError('Evaluation or promotion outcomes cannot enter trajectory feedback')
    by_id = {candidate.candidate_id: candidate for candidate in adapter.candidates}
    ids = [skill['skill_id'] for skill in decision['skills']]
    output = []
    for skill_id in ids:
        skill_rows = [row for row in rows if row['skill_id'] == skill_id]
        if not skill_rows:
            raise ValueError(f'Missing development trajectories for {skill_id}')
        observations = []
        rule_values = defaultdict(list)
        feature_values = defaultdict(list)
        for row in skill_rows:
            previous_best = max(row['initial']['revealed_values'])
            for event in row['events']:
                candidate_id = event['selected_candidate']
                candidate = by_id[candidate_id]
                improvement = max(0.0, float(event['best_so_far']) - float(previous_best))
                item = {
                    'seed': int(row['seed']),
                    'round': int(event['round_index']),
                    'candidate_id': candidate_id,
                    'public_features': _candidate_features(candidate, catalog),
                    'revealed_value': float(event['revealed_value']),
                    'improvement_over_previous_best': improvement,
                    'matched_rules': list(event['hypothesis_snapshot']['matched_rules']),
                }
                observations.append(item)
                for rule_id in item['matched_rules']:
                    rule_values[rule_id].append(item)
                for field, value in item['public_features'].items():
                    feature_values[(field, value)].append(item)
                previous_best = float(event['best_so_far'])

        def aggregate(items):
            values = [item['revealed_value'] for item in items]
            gains = [item['improvement_over_previous_best'] for item in items]
            return {
                'n': len(items),
                'mean_revealed_value': float(np.mean(values)),
                'mean_improvement': float(np.mean(gains)),
                'positive_discoveries': int(sum(value > 0 for value in gains)),
            }

        overall_mean = float(np.mean([item['revealed_value'] for item in observations]))
        feature_summary = []
        for (field, value), items in feature_values.items():
            if len(items) < 2:
                continue
            summary = aggregate(items)
            summary.update({
                'field': field,
                'value': value,
                'mean_value_lift_vs_all_selected': summary['mean_revealed_value'] - overall_mean,
            })
            feature_summary.append(summary)
        feature_summary.sort(key=lambda item: (-abs(item['mean_value_lift_vs_all_selected']),
                                               item['field'], item['value']))
        ordered = sorted(observations, key=lambda item: (
            item['revealed_value'], item['improvement_over_previous_best'],
            -item['seed'], -item['round']))
        weakest = ordered[:examples_per_tail]
        strongest = list(reversed(ordered[-examples_per_tail:]))
        metrics = [row['metrics'] for row in skill_rows]
        output.append({
            'skill_id': skill_id,
            'seed_count': len(skill_rows),
            'selected_candidate_count': len(observations),
            'mean_auc': float(np.mean([item['best_so_far_auc'] for item in metrics])),
            'mean_final_best': float(np.mean([item['final_best'] for item in metrics])),
            'per_seed': [{
                'seed': int(item['seed']),
                'auc': float(item['best_so_far_auc']),
                'final_best': float(item['final_best']),
            } for item in metrics],
            'strongest_observed_candidates': strongest,
            'weakest_observed_candidates': weakest,
            'matched_rule_evidence': [
                {'rule_id': rule_id, **aggregate(items)}
                for rule_id, items in sorted(rule_values.items())
            ],
            'largest_observed_feature_lifts': feature_summary[:12],
        })
    return {
        'development_only': True,
        'evidence_scope': {
            'candidate_outcomes': 'only candidates actually revealed by this skill',
            'unobserved_candidate_labels_included': False,
            'all_revealed_events_used_in_aggregates': True,
            'examples_are_descriptive_not_causal': True,
        },
        'skill_results': output,
    }


def promotion_gate(incumbent_rows, challenger_rows, config, generation):
    incumbent = {row['seed']: row['metrics'] for row in incumbent_rows}
    challenger = {row['seed']: row['metrics'] for row in challenger_rows}
    if set(incumbent) != set(challenger) or len(incumbent) < 2:
        raise ValueError('Promotion gate requires paired seeds')
    seeds = sorted(incumbent)
    auc_delta = np.asarray([
        challenger[seed]['best_so_far_auc'] - incumbent[seed]['best_so_far_auc']
        for seed in seeds
    ], dtype=float)
    final_delta = np.asarray([
        challenger[seed]['final_best'] - incumbent[seed]['final_best']
        for seed in seeds
    ], dtype=float)
    rng = np.random.default_rng(config['bootstrap_seed'] + generation)
    boot = auc_delta[rng.integers(0, len(auc_delta),
                                  (config['bootstrap_draws'], len(auc_delta)))].mean(axis=1)
    wins = int((auc_delta > 1e-8).sum())
    accepted = (
        float(auc_delta.mean()) >= float(config['promotion_min_mean_auc_delta'])
        and wins >= int(config['promotion_min_wins'])
        and (not config['promotion_require_nonnegative_final_delta']
             or float(final_delta.mean()) >= 0.0)
    )
    return {
        'accepted': bool(accepted),
        'rule': {
            'min_mean_auc_delta': config['promotion_min_mean_auc_delta'],
            'min_wins': config['promotion_min_wins'],
            'require_nonnegative_mean_final_delta': config['promotion_require_nonnegative_final_delta'],
        },
        'seed_ids': seeds,
        'mean_auc_delta': float(auc_delta.mean()),
        'mean_final_best_delta': float(final_delta.mean()),
        'auc_delta_bootstrap_ci80': np.quantile(boot, [0.1, 0.9]).tolist(),
        'auc_delta_bootstrap_ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
        'wins': wins,
        'ties': int((abs(auc_delta) <= 1e-8).sum()),
        'losses': int((auc_delta < -1e-8).sum()),
        'per_seed': [{
            'seed': seed,
            'auc_delta': float(challenger[seed]['best_so_far_auc']
                               - incumbent[seed]['best_so_far_auc']),
            'final_best_delta': float(challenger[seed]['final_best']
                                      - incumbent[seed]['final_best']),
        } for seed in seeds],
    }


def paired_summary(rows, left, right, config):
    lookup = {(row['arm'], row['seed']): row['metrics'] for row in rows}
    seeds = sorted({row['seed'] for row in rows if row['arm'] == left})
    output = {}
    for metric in ('best_so_far_auc', 'final_best'):
        delta = np.asarray([
            lookup[left, seed][metric] - lookup[right, seed][metric] for seed in seeds
        ], dtype=float)
        rng = np.random.default_rng(config['bootstrap_seed'])
        boot = delta[rng.integers(0, len(delta), (config['bootstrap_draws'], len(delta)))].mean(axis=1)
        output[metric] = {
            'n': len(delta), 'mean_delta': float(delta.mean()),
            'bootstrap_ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
            'wins': int((delta > 1e-8).sum()),
            'ties': int((abs(delta) <= 1e-8).sum()),
            'losses': int((delta < -1e-8).sum()),
        }
    return output


def run_branch(name, initial, base, spec, adapter, catalog, config, directory):
    current = json.loads(json.dumps(initial))
    memories = []
    width = config['development_seeds_per_generation']
    gate_width = config['promotion_seeds_per_generation']
    for generation in range(1, config['updates'] + 1):
        dev_start = config['development_seed_start'] + (generation - 1) * width
        dev_seeds = list(range(dev_start, dev_start + width))
        rows = live.evaluate(spec, current, dev_seeds, config, 'development',
                             directory / f'generation_{generation}' / 'development')
        if name == 'rich_gated':
            supplied = rich_feedback(rows, current, adapter, catalog,
                                     config['rich_examples_per_tail'])
        else:
            supplied = live.feedback(rows, current, 'true_feedback',
                                     config['shuffle_seed'] + generation)
        memory = {
            'generation_observed': generation - 1,
            'previous_skills': current,
            'feedback': supplied,
        }
        memories.append(memory)
        prompt = live.build_revision_prompt(base, current, memories, 'true_feedback')
        if name == 'rich_gated':
            prompt['revision_instruction'] = (
                'Diagnose the measured candidate trajectories and propose two conservative revisions. '
                'Treat feature and rule summaries as descriptive evidence from a small development split, '
                'not causal truth. Keep a useful incumbent hypothesis when the evidence is weak. A disjoint '
                'promotion split will decide whether the selected challenger replaces the incumbent.'
            )
        challenger = live.generate(config, prompt, catalog,
                                   directory / f'generation_{generation}' / 'llm_call')
        state = {
            'generation': generation,
            'branch': name,
            'parent_state_sha256': live.fingerprint(current),
            'challenger_state_sha256': live.fingerprint(challenger),
            'incumbent_selected_skill': current['selected_skill_id'],
            'challenger_selected_skill': challenger['selected_skill_id'],
        }
        if name == 'rich_gated':
            gate_start = config['promotion_seed_start'] + (generation - 1) * gate_width
            gate_seeds = list(range(gate_start, gate_start + gate_width))
            incumbent_rows = live.evaluate(
                spec, selected_only(current), gate_seeds, config, 'development',
                directory / f'generation_{generation}' / 'promotion' / 'incumbent')
            challenger_rows = live.evaluate(
                spec, selected_only(challenger), gate_seeds, config, 'development',
                directory / f'generation_{generation}' / 'promotion' / 'challenger')
            gate = promotion_gate(incumbent_rows, challenger_rows, config, generation)
            current = challenger if gate['accepted'] else current
            state['promotion_gate'] = gate
            state['deployed_state'] = 'challenger' if gate['accepted'] else 'incumbent'
            memory['proposal_and_gate'] = {
                key: gate[key] for key in (
                    'accepted', 'rule', 'seed_ids', 'mean_auc_delta',
                    'mean_final_best_delta', 'wins', 'ties', 'losses')
            }
        else:
            current = challenger
            state['deployed_state'] = 'challenger_without_gate'
        state['deployed_state_sha256'] = live.fingerprint(current)
        live.save(directory / f'generation_{generation}' / 'state.json', state)
        print(f'{name} generation {generation}: {state["deployed_state"]}', flush=True)
    return current


def run(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    splits = validate_config(config)
    if config.get('generation_backend') == 'local_qlora' and not config.get('base_model_path'):
        raise ValueError('Local backend requires base_model_path')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite experiment evidence')
    output.mkdir(parents=True, exist_ok=True)
    live.save(output / 'config.json', config)
    spec = config['tasks'][0]
    adapter = replay.DATASET_BUILDERS[spec['id']]()
    catalog = semantic.semantic_field_catalog(adapter)
    initial_path = REPO / config['initial_decision_path']
    if sha(initial_path) != config['initial_decision_sha256']:
        raise ValueError('Pinned initial decision hash mismatch')
    initial = validate_initial_decision(json.loads(initial_path.read_text()), catalog,
                                        config['skills_per_generation'])
    base = generator.build_prompt_payload(
        spec['source'], spec['id'], config['source_observations'], config['discount'],
        config['skills_per_generation'], 'full', knowledge_context=None,
        proposal_mode='parametric')
    base['deployment_requirement'] = (
        'Also output selected_skill_id matching one of the two skill IDs, lesson, and '
        'revision_summary. Deployment is frozen before final evaluation.'
    )
    try:
        base_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        base_commit = config.get('source_commit')
    live.save(output / 'protocol_lock.json', {
        'schema_version': 'care.rsi_evidence_gate_lock/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'base_commit': base_commit,
        'config_sha256': live.fingerprint(config),
        'initial_decision_path': config['initial_decision_path'],
        'initial_decision_sha256': config['initial_decision_sha256'],
        'implementation_sha256': {
            str(path.relative_to(REPO)): sha(path)
            for path in (ROOT / 'scripts').glob('*.py')
        },
        'python': platform.python_version(),
        'numpy': np.__version__,
        'seed_splits': {key: sorted(value) for key, value in splits.items()},
        'planned_valid_generation_calls': len(config['branches']) * config['updates'],
        'final_evaluation_feedback_permitted': False,
        'model_weights_frozen': True,
        'claim_boundary': config['claim_boundary'],
    })
    finals = {'fixed_initial': initial}
    for branch in config['branches']:
        finals[branch] = run_branch(
            branch, initial, base, spec, adapter, catalog, config, output / branch)
    heldout = []
    eval_seeds = sorted(splits['evaluation'])
    for arm, decision in finals.items():
        rows = live.evaluate(
            spec, selected_only(decision), eval_seeds, config, 'evaluation',
            output / 'evaluation' / arm, include_gp=arm == 'fixed_initial')
        for row in rows:
            row['arm'] = 'gp_ucb' if row['skill_id'] == 'gp_ucb' else arm
        heldout.extend(rows)
    live.save(output / 'deployment_lock.json', {
        'selected': {arm: decision['selected_skill_id'] for arm, decision in finals.items()},
        'state_sha256': {arm: live.fingerprint(decision) for arm, decision in finals.items()},
        'evaluation_seed_ids': eval_seeds,
        'evaluation_feedback_permitted': False,
        'frozen_at': datetime.now(timezone.utc).isoformat(),
    })
    live.save(output / 'evaluation_metrics.json', [
        {key: row[key] for key in ('task', 'seed', 'arm', 'skill_id', 'metrics')}
        for row in heldout
    ])
    comparisons = {}
    for left, right in (
        ('rich_gated', 'sparse_ungated'),
        ('rich_gated', 'fixed_initial'),
        ('sparse_ungated', 'fixed_initial'),
        ('rich_gated', 'gp_ucb'),
    ):
        comparisons[f'{left}_minus_{right}'] = paired_summary(
            heldout, left, right, config)
    means = {
        arm: {
            metric: float(np.mean([
                row['metrics'][metric] for row in heldout if row['arm'] == arm
            ]))
            for metric in ('best_so_far_auc', 'final_best')
        }
        for arm in sorted({row['arm'] for row in heldout})
    }
    gates = []
    for path in sorted((output / 'rich_gated').glob('generation_*/state.json')):
        state = json.loads(path.read_text())
        gates.append({'generation': state['generation'], **state['promotion_gate']})
    summary = {
        'status': 'complete', 'task': spec['id'], 'means': means,
        'comparisons': comparisons, 'promotion_gates': gates,
        'valid_generation_calls': len(config['branches']) * config['updates'],
        'claim_boundary': config['claim_boundary'],
    }
    live.save(output / 'summary.json', summary)
    paths = sorted(path for path in output.rglob('*')
                   if path.is_file() and path.name != 'SHA256SUMS')
    (output / 'SHA256SUMS').write_text(''.join(
        f'{sha(path)}  {path.relative_to(output)}\n' for path in paths))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output_dir)
