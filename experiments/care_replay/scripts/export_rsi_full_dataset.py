#!/usr/bin/env python3
"""Convert the legacy RSI pilot split into canonical, prompt-disjoint SFT data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import run_rsi_live_feedback as live


def prompt_hash(row):
    return hashlib.sha256(json.dumps(row['messages'][:-1], sort_keys=True).encode()).hexdigest()


def build(source, output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite a dataset')
    rows = {s: [json.loads(x) for x in (source / f'{s}.jsonl').read_text().splitlines()]
            for s in ('train', 'validation')}
    training_hashes = {prompt_hash(r) for r in rows['train']}
    moved = [r for r in rows['validation'] if prompt_hash(r) in training_hashes]
    rows['validation'] = [r for r in rows['validation'] if prompt_hash(r) not in training_hashes]
    rows['train'].extend(moved)
    if not rows['validation']:
        raise ValueError('Prompt grouping leaves no validation examples')
    for split in rows:
        for row in rows[split]:
            payload = json.loads(next(m['content'] for m in row['messages'] if m['role'] == 'user'))
            catalog = payload['target']['public_semantic_fields']
            decision = live.normalize_response(row['messages'][-1]['content'], catalog, 2)
            # Compiler dataclasses use pair tuples internally; generation requires
            # field/value objects. Never teach the model the private storage form.
            for skill in decision['skills']:
                for rule in skill['rules']:
                    rule['conditions'] = dict(rule['conditions'])
            row['messages'][-1]['content'] = json.dumps(decision, ensure_ascii=False, separators=(',', ':'))
            live.normalize_response(row['messages'][-1]['content'], catalog, 2)
            row['metadata']['purpose'] = 'full_available_data_sft_same_task_prompt_disjoint_validation'
            row['metadata']['prompt_sha256'] = prompt_hash(row)
    hashes = {s: {prompt_hash(r) for r in rows[s]} for s in rows}
    if hashes['train'] & hashes['validation']:
        raise ValueError('Train/validation prompt leakage')
    output.mkdir(parents=True)
    files = {}
    for split, values in rows.items():
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
        files[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'examples': len(values), 'unique_prompts': len(hashes[split]),
                            'tasks': dict(Counter(r['metadata']['task'] for r in values))}
    manifest = {
        'schema_version': 'care.rsi_full_available_dataset/v1',
        'source_directory': str(source),
        'split_rule': 'Legacy validation groups that also occur in training move entirely to training',
        'overlapping_validation_rows_moved': len(moved), 'prompt_overlap': 0,
        'target_format': 'canonical generation-wire JSON; conditions objects; compiler round-trip checked',
        'claim_boundary': 'Complete available-data SFT only. Three task families and same-task validation do not satisfy the 1000-example, 8-training-family, 2-heldout-family efficacy gate.',
        'files': files,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), indent=2))
