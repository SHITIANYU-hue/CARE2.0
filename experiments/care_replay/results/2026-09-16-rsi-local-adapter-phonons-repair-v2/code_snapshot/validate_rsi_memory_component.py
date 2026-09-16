#!/usr/bin/env python3
"""Validate archive integrity, pairing, and NumPy scores against Python reference."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import run_rsi_memory_component as rsi
import run_surrogate_baselines as surrogate


def validate(root):
    config = json.loads((root / 'config.json').read_text())
    checked, finite_scores, versions = 0, 0, 0
    for spec in config['tasks']:
        task_dir = root / spec['id']
        lock = json.loads((task_dir / 'selection_lock.json').read_text())
        assert not set(lock['development_seed_ids']) & set(lock['evaluation_seed_ids'])
        for a, b in zip(lock['versions'], lock['versions'][1:]):
            assert rsi.canonical_hash(a) == b['parent_state_sha256']
            assert not set(b['feedback_seed_ids']) & set(lock['evaluation_seed_ids'])
            versions += 1
        for path in task_dir.glob('*/seed_*.json'):
            arms = json.loads(path.read_text())
            for arm in arms:
                assert arm['initial'] == arms[0]['initial']
                assert len(arm['initial']['candidate_ids']) == config['initial_observations']
                events = arm['events']
                assert len(events) == config['reveal_rounds']
                seen = set(arm['initial']['candidate_ids'])
                values = list(arm['initial']['revealed_values'])
                for event in events:
                    assert event['selected_candidate'] not in seen
                    seen.add(event['selected_candidate'])
                    values.append(event['revealed_value'])
                    assert np.isfinite(event['selected_score'])
                    assert abs(event['best_so_far'] - max(values)) < 1e-8
                    finite_scores += 1
                assert abs(arm['metrics']['best_so_far_auc'] - np.mean([e['best_so_far'] for e in events])) < .000051
                assert abs(arm['metrics']['final_best'] - max(values)) < .000051
                checked += 1
    comparisons = []
    vector = surrogate._gp_anchor_scores_vectorized
    scalar = surrogate._gp_anchor_scores_python
    max_error, candidate_scores_checked = 0., 0
    def compare(*args, **kwargs):
        nonlocal max_error, candidate_scores_checked
        actual, diagnostic = vector(*args, **kwargs)
        expected, _ = scalar(*args, **kwargs)
        for key in actual:
            assert actual[key].keys() == expected[key].keys()
            av = np.asarray(list(actual[key].values()))
            ev = np.asarray(list(expected[key].values()))
            assert np.all(np.isfinite(av)) and np.all(np.isfinite(ev))
            error = float(np.max(abs(av - ev)))
            max_error = max(max_error, error)
            candidate_scores_checked += len(av)
            np.testing.assert_allclose(av, ev, rtol=1e-9, atol=1e-10)
        return actual, diagnostic
    surrogate._gp_anchor_scores_vectorized = compare
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for spec in config['tasks']:
            task_dir = root / spec['id']
            selected = json.loads((task_dir / 'selection_lock.json').read_text())['selected']
            keep = {selected['fixed'], selected['updated']}
            record = json.loads((rsi.ROOT / spec['record']).read_text())
            skills = [s for s in record['normalized_skills'] if s['skill_id'] in keep]
            seed = config['evaluation_seed_start']
            original = {r['skill_id']: r for r in json.loads((task_dir / 'evaluation' / f'seed_{seed}.json').read_text())}
            job = (spec['id'], skills, seed, config, 'development')
            replayed = rsi.run_job(job)
            for row in replayed:
                assert row['metrics'] == original[row['skill_id']]['metrics']
                assert [e['selected_candidate'] for e in row['events']] == [e['selected_candidate'] for e in original[row['skill_id']]['events']]
                comparisons.append({'task': spec['id'], 'skill': row['skill_id'], 'seed': seed, 'archive_replay_matches': True})
    result = {'archive_trajectories_checked': checked, 'finite_selected_scores_checked': finite_scores,
              'hash_linked_versions_checked': versions, 'paired_initial_observations_match': True,
              'all_saved_metrics_recomputed': True, 'evaluation_feedback_excluded': True,
              'backend_validation': {'sample': comparisons, 'candidate_anchor_scores_checked': candidate_scores_checked,
                                     'max_absolute_numpy_vs_python_difference': max_error,
                                     'rtol': 1e-9, 'atol': 1e-10, 'status': 'passed'},
              'limitation': 'Backend comparison samples fixed and final selected skills on the first evaluation seed per task; it is not an exhaustive backend audit of every state.'}
    rsi.write_json(root / 'numerical_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    validate(p.parse_args().root)
