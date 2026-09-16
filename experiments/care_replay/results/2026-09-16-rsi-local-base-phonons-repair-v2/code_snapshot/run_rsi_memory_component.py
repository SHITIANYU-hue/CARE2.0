#!/usr/bin/env python3
"""Seed-disjoint, offline RSI skill-selection component; never makes an API call."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import warnings

import numpy as np
import llm_semantic_skills as semantic
import run_synthetic_suzuki as replay
import run_transfer_weighted_kernel as weighted

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(config):
    for key in ('initial_observations', 'reveal_rounds', 'generations',
                'development_seeds_per_generation', 'evaluation_seed_count', 'bootstrap_draws'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    dev = set(range(config['development_seed_start'], config['development_seed_start'] +
                    config['generations'] * config['development_seeds_per_generation']))
    test = set(range(config['evaluation_seed_start'], config['evaluation_seed_start'] +
                     config['evaluation_seed_count']))
    if dev & test:
        raise ValueError('Development and evaluation seeds overlap')
    ids = [t['id'] for t in config['tasks']]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Tasks must be unique and nonempty')
    return dev, test


def select_skill(scores, incumbent):
    """Selection consumes only accumulated development feedback."""
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    if not means or any(not np.isfinite(v) for v in means.values()):
        raise ValueError('Empty or nonfinite development feedback')
    best = max(means.values())
    if incumbent in means and means[incumbent] == best:
        return incumbent
    return min(key for key, value in means.items() if value == best)


def paired_stats(deltas, rng, draws, family_size):
    values = np.asarray(deltas, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError('At least two finite paired observations required')
    samples = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    alpha = 0.05 / family_size
    return {'n': len(values), 'mean': float(values.mean()),
            'bootstrap_ci95': np.quantile(samples, [.025, .975]).tolist(),
            'bootstrap_familywise_ci': np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist(),
            'family_size': family_size,
            'wins': int((values > 1e-8).sum()), 'ties': int((abs(values) <= 1e-8).sum()),
            'losses': int((values < -1e-8).sum())}


def run_job(job):
    task_id, skill_dicts, seed, config, phase = job
    adapter = replay.DATASET_BUILDERS[task_id]()
    task = replay.make_task(adapter, config['initial_observations'], config['reveal_rounds'])
    skills = semantic.normalize_skills({'skills': skill_dicts}, semantic.semantic_field_catalog(adapter), 100)
    if not skills:
        raise ValueError('No executable skills after worker serialization')
    kernel = config['kernel']
    shuffled = list(adapter.candidates)
    random.Random(seed).shuffle(shuffled)
    initial = shuffled[:config['initial_observations']]
    initial_event = {'candidate_ids': [c.candidate_id for c in initial],
                     'revealed_values': [c.objective_value for c in initial]}
    output = []
    for skill in [*skills, None]:
        if phase == 'development' and skill is None:
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if skill is None:
                metrics, audit = weighted.run_policy(
                    adapter, task, seed, 'gp_ucb', (1.,) * (len(adapter.decision_columns) + 1),
                    {}, 1.5, kernel['numeric_length_scale'], kernel['categorical_length_scale'], kernel['gp_noise'])
            else:
                metrics, audit = semantic.run_direct_prior_skill(
                    adapter, task, seed, skill, kernel['numeric_length_scale'],
                    kernel['categorical_length_scale'], kernel['gp_noise'])
        # Fail on actual nonfinite recorded metrics/decisions; retain numerical warnings.
        numeric = [metrics['best_so_far_auc'], metrics['final_best']]
        numeric += [a['selected_score'] for a in audit]
        if not np.all(np.isfinite(numeric)):
            raise FloatingPointError('Nonfinite metrics or selected scores')
        observed = set(initial_event['candidate_ids'])
        for event in audit:
            if event['selected_candidate'] in observed:
                raise ValueError('Repeated reveal')
            observed.add(event['selected_candidate'])
        if len(audit) != config['reveal_rounds']:
            raise ValueError('Unexpected reveal budget')
        output.append({'task': task_id, 'seed': seed, 'phase': phase,
                       'skill_id': skill.skill_id if skill else 'gp_ucb', 'metrics': metrics,
                       'initial': initial_event, 'events': audit,
                       'warnings': sorted(set(str(w.message) for w in caught))})
    return output


def run_phase(pool, task_spec, skills, seeds, config, phase, directory):
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(task_spec['id'], json.loads(json.dumps([asdict(s) for s in skills])), seed, config, phase) for seed in seeds]
    rows = []
    for result in pool.map(run_job, jobs):
        seed = result[0]['seed']
        write_json(directory / f'seed_{seed}.json', result)
        rows.extend(result)
    return rows


def run(config_path, output, workers):
    config = json.loads(config_path.read_text())
    dev, test = validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; refusing to overwrite evidence')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    implementations = {str(p.relative_to(REPO)): sha(p) for p in (ROOT / 'scripts').glob('*.py')}
    lock = {'schema_version': 'care.rsi_memory_component_lock/v1',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'config_sha256': canonical_hash(config),
            'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'implementation_sha256': implementations,
            'python': platform.python_version(), 'numpy': np.__version__,
            'new_llm_calls': 0, 'credential_access': False,
            'claim_boundary': config['claim_boundary']}
    write_json(output / 'protocol_lock.json', lock)
    all_rows, all_summaries = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for spec in config['tasks']:
            task_dir = output / spec['id']
            record_path = ROOT / spec['record']
            record = json.loads(record_path.read_text())
            if record['target_dataset'] != spec['id']:
                raise ValueError('Archived LLM target does not match replay task')
            adapter = replay.DATASET_BUILDERS[spec['id']]()
            skills = semantic.normalize_skills({'skills': record['normalized_skills']},
                                              semantic.semantic_field_catalog(adapter), 100)
            if len(skills) < 2:
                raise ValueError('At least two archived executable skills required')
            # Preserve complete existing public prompt/response trace by reference and hash.
            # No credentials are read or copied by this runner.
            write_json(task_dir / 'llm_trace_provenance.json',
                       {'path': spec['record'], 'sha256': sha(record_path), 'model': record.get('model'),
                        'usage': record.get('usage'), 'new_call': False,
                        'trace_kind': 'archived_prompt_and_raw_response',
                        'normalized_skills_used': [asdict(s) for s in skills]})
            write_json(task_dir / 'dataset_manifest.json',
                       {'task': spec['id'], 'candidate_count': len(adapter.candidates),
                        'adapter_candidates_sha256': canonical_hash([asdict(c) for c in adapter.candidates]),
                        'objective_units': 'existing adapter 0-100 objective, maximize',
                        'raw_files': {str(p.relative_to(ROOT)): sha(p) for p in replay.RAW_DATA.iterdir()
                                      if p.is_file() and ('freesolv' in p.name or 'lipophilicity' in p.name or 'expt_gap' in p.name)}})
            fixed = sorted(skills, key=lambda s: (-s.confidence, s.skill_id))[0].skill_id
            incumbent = fixed
            scores = {s.skill_id: [] for s in skills}
            shuffled_scores = {s.skill_id: [] for s in skills}
            versions = [{'generation': 0, 'selected_skill': fixed, 'feedback_seed_ids': []}]
            development_rows = []
            shuffle_rng = random.Random(config['shuffle_seed'])
            for gen in range(1, config['generations'] + 1):
                start = config['development_seed_start'] + (gen - 1) * config['development_seeds_per_generation']
                seeds = list(range(start, start + config['development_seeds_per_generation']))
                rows = run_phase(pool, spec, skills, seeds, config, 'development', task_dir / 'development')
                development_rows.extend(rows)
                for seed in seeds:
                    current = {r['skill_id']: r['metrics']['best_so_far_auc'] for r in rows if r['seed'] == seed}
                    values = [current[k] for k in scores]
                    shuffle_rng.shuffle(values)
                    for k, fake in zip(scores, values):
                        scores[k].append(current[k])
                        shuffled_scores[k].append(fake)
                previous = incumbent
                incumbent = select_skill(scores, incumbent)
                state = {'generation': gen, 'parent_state_sha256': canonical_hash(versions[-1]),
                         'previous_skill': previous, 'selected_skill': incumbent,
                         'feedback_seed_ids': sorted(set(r['seed'] for r in development_rows)),
                         'development_auc_by_skill': scores.copy(),
                         'development_mean_auc': {k: float(np.mean(v)) for k, v in scores.items()},
                         'promotion_status': 'experimental_only_not_active_global_kb'}
                write_json(task_dir / f'skill_bank_v{gen}.json', state)
                # Deep-copy protects the hash chain from later list mutation.
                versions.append(json.loads(json.dumps(state)))
                print(f'{spec["id"]} generation {gen}: {previous} -> {incumbent}', flush=True)
            selected = {'fixed': fixed, 'updated': incumbent,
                        'one_shot_first_generation': versions[1]['selected_skill'],
                        'batch_same_feedback': select_skill(scores, incumbent),
                        'shuffled_feedback': select_skill(shuffled_scores, fixed),
                        'gp_ucb': 'gp_ucb'}
            freeze = {'selected': selected, 'versions': versions,
                      'evaluation_seed_ids': sorted(test), 'development_seed_ids': sorted(dev),
                      'frozen_at': datetime.now(timezone.utc).isoformat(),
                      'evaluation_feedback_permitted': False}
            write_json(task_dir / 'selection_lock.json', freeze)
            test_rows = run_phase(pool, spec, skills, sorted(test), config, 'evaluation', task_dir / 'evaluation')
            all_rows.extend(development_rows + test_rows)
            lookup = {(r['seed'], r['skill_id']): r['metrics'] for r in test_rows}
            task_summary = {'task': spec['id'], 'selected': selected, 'skill_count': len(skills),
                            'development_trajectories': len(development_rows),
                            'development_reveal_cost': len(development_rows) * (config['initial_observations'] + config['reveal_rounds']),
                            'comparisons': {}}
            for name in ['fixed', 'one_shot_first_generation', 'batch_same_feedback', 'shuffled_feedback', 'gp_ucb']:
                task_summary['comparisons']['updated_vs_' + name] = {}
                for metric in ['best_so_far_auc', 'final_best']:
                    deltas = [lookup[seed, selected['updated']][metric] - lookup[seed, selected[name]][metric]
                              for seed in sorted(test)]
                    task_summary['comparisons']['updated_vs_' + name][metric] = paired_stats(
                        deltas, np.random.default_rng(config['bootstrap_seed']), config['bootstrap_draws'], len(config['tasks']))
            task_summary['numerical_warning_trajectories'] = sum(bool(r['warnings']) for r in development_rows + test_rows)
            write_json(task_dir / 'summary.json', task_summary)
            all_summaries.append(task_summary)
            print(f'{spec["id"]}: completed {len(development_rows)} development + {len(test_rows)} evaluation trajectories', flush=True)
    fields = ['task', 'phase', 'seed', 'skill_id', 'best_so_far_auc', 'final_best', 'simple_regret', 'top10_hit']
    with (output / 'raw_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] if k in row else row['metrics'][k] for k in fields})
    write_json(output / 'summary.json', {'tasks': all_summaries, 'claim_boundary': config['claim_boundary'],
                                        'new_llm_calls': 0, 'failed_jobs': 0,
                                        'uncertainty_scope': 'Seed resampling conditional on fixed finite pools, archived skills, and development data; not task or model-call uncertainty'})
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (output / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(output)}\n' for p in paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.output_dir, args.workers)
