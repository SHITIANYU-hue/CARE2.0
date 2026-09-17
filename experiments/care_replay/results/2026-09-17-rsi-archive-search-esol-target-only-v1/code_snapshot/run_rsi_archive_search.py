#!/usr/bin/env python3
"""Small GEPA/ACE-inspired policy archive, tested against sparse recursive RSI.

This is an adaptation, not an implementation of the published algorithms. Each
model call returns two executable skills. Both are measured on the same fresh
promotion seeds against the incumbent, and only a passing skill enters the
bounded archive. The held-out split is opened after all choices are frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

import generate_llm_semantic_skills as generator
import llm_semantic_skills as semantic
import run_rsi_evidence_gate as evidence
import run_rsi_live_feedback as live
import run_synthetic_suzuki as replay

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    if config['updates'] != 2 or config['skills_per_generation'] != 2:
        raise ValueError('Frozen study requires two updates and two skills')
    if config['evidence_mode'] != 'target_only':
        raise ValueError('Frozen ESOL protocol requires target-only evidence')
    if config['arms'] != ['sparse_ungated', 'archive_search']:
        raise ValueError('Unexpected arms')
    for key in ('development_seeds_per_generation', 'promotion_seeds_per_generation',
                'evaluation_seed_count', 'bootstrap_draws'):
        if not isinstance(config[key], int) or config[key] < 2:
            raise ValueError(f'{key} must be an integer >= 2')
    if config['promotion_min_wins'] > config['promotion_seeds_per_generation']:
        raise ValueError('Impossible promotion win rule')
    splits = {
        'development': set(range(config['development_seed_start'],
                                 config['development_seed_start'] + config['updates']
                                 * config['development_seeds_per_generation'])),
        'promotion': set(range(config['promotion_seed_start'],
                               config['promotion_seed_start'] + config['updates']
                               * config['promotion_seeds_per_generation'])),
        'evaluation': set(range(config['evaluation_seed_start'],
                                config['evaluation_seed_start'] + config['evaluation_seed_count'])),
    }
    for a, b in (('development', 'promotion'), ('development', 'evaluation'),
                 ('promotion', 'evaluation')):
        if splits[a] & splits[b]:
            raise ValueError(f'{a}/{b} seeds overlap')
    return splits


def skill_decision(skill):
    return {'skills': [skill], 'selected_skill_id': skill['skill_id']}


def compact_evidence(rows, incumbent, adapter, catalog, config):
    card = evidence.rich_feedback(
        rows, evidence.selected_only(incumbent), adapter, catalog,
        examples_per_tail=config['examples_per_tail'])
    item = card['skill_results'][0]
    item['largest_observed_feature_lifts'] = item['largest_observed_feature_lifts'][:6]
    return card


def accepted_archive_card(decision, selected_skill, result, generation):
    """A bounded playbook item whose performance evidence came from promotion."""
    return {
        'generation': generation,
        'skill': selected_skill,
        'model_lesson': decision.get('lesson', '')[:350],
        'promotion_evidence': {
            key: result[key] for key in (
                'mean_auc_delta', 'mean_final_best_delta', 'wins', 'ties', 'losses')
        },
        'status': 'accepted_on_promotion_only_not_proven_on_final_evaluation',
    }


def archive_prompt(base, incumbent, current_card, archive):
    prompt = json.loads(json.dumps(base))
    prompt['task'] = (
        'Using measured development trajectories and a bounded archive of validated '
        'policies, propose exactly two distinct executable skills. Keep an incumbent-like '
        'candidate as one option when evidence for a large rewrite is weak.'
    )
    prompt['previous_skills'] = incumbent
    prompt['current_development_evidence'] = current_card
    prompt['validated_policy_archive'] = archive[-2:]
    prompt['feedback_boundary'] = (
        'Development trajectories contain revealed outcomes only. Archive scores came from '
        'disjoint promotion seeds. Final evaluation outcomes are unavailable.'
    )
    prompt['revision_instruction'] = (
        'Use trajectory evidence to diagnose concrete failures. Preserve validated insights '
        'incrementally. Do not infer causation from small descriptive feature summaries. '
        'The executor will measure both returned skills; selected_skill_id records your '
        'pre-measurement preference but does not determine archive promotion.'
    )
    return prompt


def choose_archive_candidate(incumbent_rows, candidates, config, generation):
    """Predeclared gate: choose the best passing candidate, otherwise incumbent."""
    scored = []
    for skill, rows in candidates:
        result = evidence.promotion_gate(incumbent_rows, rows, config, generation)
        scored.append({'skill_id': skill['skill_id'], 'gate': result})
    passing = [item for item in scored if item['gate']['accepted']]
    passing.sort(key=lambda item: (
        -item['gate']['mean_auc_delta'],
        -item['gate']['wins'],
        item['skill_id'],
    ))
    return (passing[0]['skill_id'] if passing else None), scored


def run_sparse(initial, base, spec, config, output):
    current = json.loads(json.dumps(initial))
    memory = []
    width = config['development_seeds_per_generation']
    for generation in range(1, config['updates'] + 1):
        seeds = range(config['development_seed_start'] + (generation - 1) * width,
                      config['development_seed_start'] + generation * width)
        rows = live.evaluate(spec, current, seeds, config, 'development',
                             output / f'generation_{generation}' / 'development')
        supplied = live.feedback(rows, current, 'true_feedback',
                                 config['shuffle_seed'] + generation)
        memory.append({'generation_observed': generation - 1,
                       'previous_skills': current, 'feedback': supplied})
        prompt = live.build_revision_prompt(base, current, memory, 'true_feedback')
        challenger = live.generate(config, prompt,
                                   semantic.semantic_field_catalog(replay.DATASET_BUILDERS[spec['id']]()),
                                   output / f'generation_{generation}' / 'llm_call')
        live.save(output / f'generation_{generation}' / 'state.json', {
            'generation': generation, 'selected': challenger['selected_skill_id'],
            'parent_sha256': live.fingerprint(current),
            'new_sha256': live.fingerprint(challenger),
            'deployment_rule': 'unconditional_model_selection',
        })
        current = challenger
        print(f'sparse generation {generation}: {current["selected_skill_id"]}', flush=True)
    return current


def run_archive(initial, base, spec, adapter, catalog, config, output):
    incumbent = json.loads(json.dumps(initial))
    archive = []
    width = config['development_seeds_per_generation']
    gate_width = config['promotion_seeds_per_generation']
    for generation in range(1, config['updates'] + 1):
        dev_seeds = range(config['development_seed_start'] + (generation - 1) * width,
                          config['development_seed_start'] + generation * width)
        current_rows = live.evaluate(
            spec, evidence.selected_only(incumbent), dev_seeds, config, 'development',
            output / f'generation_{generation}' / 'development')
        card = compact_evidence(current_rows, incumbent, adapter, catalog, config)
        prompt = archive_prompt(base, incumbent, card, archive)
        proposals = live.generate(config, prompt, catalog,
                                  output / f'generation_{generation}' / 'llm_call')
        gate_seeds = range(config['promotion_seed_start'] + (generation - 1) * gate_width,
                           config['promotion_seed_start'] + generation * gate_width)
        incumbent_rows = live.evaluate(
            spec, evidence.selected_only(incumbent), gate_seeds, config, 'development',
            output / f'generation_{generation}' / 'promotion' / 'incumbent')
        candidates = []
        for index, skill in enumerate(proposals['skills']):
            rows = live.evaluate(
                spec, skill_decision(skill), gate_seeds, config, 'development',
                output / f'generation_{generation}' / 'promotion' / f'candidate_{index}')
            candidates.append((skill, rows))
        chosen, comparisons = choose_archive_candidate(
            incumbent_rows, candidates, config, generation)
        state = {
            'generation': generation,
            'parent_sha256': live.fingerprint(incumbent),
            'model_selected_skill': proposals['selected_skill_id'],
            'candidate_comparisons': comparisons,
            'promoted_skill': chosen,
            'multiplicity_note': (
                'Two candidates share the promotion seeds; gate scores are selection-biased. '
                'Only disjoint final evaluation may assess the selected policy.'
            ),
        }
        if chosen:
            skill = next(item for item in proposals['skills'] if item['skill_id'] == chosen)
            chosen_gate = next(item['gate'] for item in comparisons if item['skill_id'] == chosen)
            incumbent = skill_decision(skill)
            archive.append(accepted_archive_card(proposals, skill, chosen_gate, generation))
        state['deployed_sha256'] = live.fingerprint(incumbent)
        state['archive_sha256'] = live.fingerprint(archive)
        live.save(output / f'generation_{generation}' / 'state.json', state)
        live.save(output / f'generation_{generation}' / 'accepted_archive.json', archive)
        print(f'archive generation {generation}: {chosen or "retain_incumbent"}', flush=True)
    return incumbent, archive


def run(config_path, output):
    config = json.loads(config_path.read_text())
    splits = validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite evidence')
    output.mkdir(parents=True, exist_ok=True)
    live.save(output / 'config.json', config)
    spec = config['task']
    adapter = replay.DATASET_BUILDERS[spec['id']]()
    catalog = semantic.semantic_field_catalog(adapter)
    base = generator.build_prompt_payload(
        spec['source'], spec['id'], config['source_observations'], config['discount'],
        config['skills_per_generation'], config['evidence_mode'], knowledge_context=None,
        proposal_mode='parametric')
    base['deployment_requirement'] = (
        'Return exactly two executable skills, selected_skill_id, lesson, and revision_summary.'
    )
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = config.get('source_commit')
    live.save(output / 'protocol_lock.json', {
        'schema_version': 'care.rsi_archive_search_lock/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'config_sha256': live.fingerprint(config),
        'base_commit': commit,
        'implementation_sha256': {
            str(path.relative_to(REPO)): sha(path)
            for path in (ROOT / 'scripts').glob('*.py')
        },
        'seed_splits': {key: sorted(value) for key, value in splits.items()},
        'model_weights_frozen': True,
        'valid_generation_calls_planned': 1 + 2 * config['updates'],
        'python': platform.python_version(), 'numpy': np.__version__,
        'claim_boundary': config['claim_boundary'],
    })
    initial = live.generate(config, base, catalog, output / 'initial_call')
    sparse = run_sparse(initial, base, spec, config, output / 'sparse_ungated')
    archived, archive = run_archive(initial, base, spec, adapter, catalog, config,
                                    output / 'archive_search')
    finals = {'fixed_initial': initial, 'sparse_ungated': sparse,
              'archive_search': archived}
    live.save(output / 'deployment_lock.json', {
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'selected': {arm: policy['selected_skill_id'] for arm, policy in finals.items()},
        'state_sha256': {arm: live.fingerprint(policy) for arm, policy in finals.items()},
        'heldout_seed_ids': sorted(splits['evaluation']),
        'heldout_feedback_permitted': False,
    })
    rows = []
    for arm, decision in finals.items():
        result = live.evaluate(
            spec, evidence.selected_only(decision), sorted(splits['evaluation']),
            config, 'evaluation', output / 'evaluation' / arm,
            include_gp=arm == 'fixed_initial')
        for row in result:
            row['arm'] = 'gp_ucb' if row['skill_id'] == 'gp_ucb' else arm
        rows.extend(result)
    live.save(output / 'evaluation_metrics.json', [
        {key: row[key] for key in ('task', 'seed', 'arm', 'skill_id', 'metrics')}
        for row in rows
    ])
    means = {
        arm: {metric: float(np.mean([
            row['metrics'][metric] for row in rows if row['arm'] == arm
        ])) for metric in ('best_so_far_auc', 'final_best')}
        for arm in sorted({row['arm'] for row in rows})
    }
    comparisons = {
        f'{left}_minus_{right}': evidence.paired_summary(rows, left, right, config)
        for left, right in (
            ('archive_search', 'sparse_ungated'),
            ('archive_search', 'fixed_initial'),
            ('archive_search', 'gp_ucb'),
            ('sparse_ungated', 'fixed_initial'),
        )
    }
    summary = {
        'status': 'complete', 'task': spec['id'], 'means': means,
        'comparisons': comparisons, 'accepted_archive_size': len(archive),
        'valid_generation_calls': 1 + 2 * config['updates'],
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
