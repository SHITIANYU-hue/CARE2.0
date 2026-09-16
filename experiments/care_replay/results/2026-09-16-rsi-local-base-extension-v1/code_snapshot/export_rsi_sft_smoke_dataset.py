#!/usr/bin/env python3
"""Export a provenance-preserving SFT dataset for an RSI pipeline smoke test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ALLOWED_ARMS = {'initial', 'true_feedback'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def location(path, root):
    parts = path.relative_to(root).parts
    task = parts[0]
    replicate = int(parts[1].split('_', 1)[1])
    if parts[2] == 'initial_call':
        return task, replicate, 'initial', 0
    return task, replicate, parts[2], int(parts[3].split('_', 1)[1])


def export(roots, output):
    rows = []
    for root in roots:
        for response_path in sorted(root.rglob('attempt_*_response.json')):
            response = json.loads(response_path.read_text())
            if response.get('status') != 'valid':
                continue
            task, replicate, arm, generation = location(response_path, root)
            if arm not in ALLOWED_ARMS:
                continue
            request_path = response_path.with_name(
                response_path.name.replace('_response.json', '_request.json'))
            request = json.loads(request_path.read_text())
            messages = request['request']['messages'] + [
                {'role': 'assistant', 'content': response['raw_model_content']}
            ]
            rows.append({
                'messages': messages,
                'metadata': {
                    'archive': str(root),
                    'task': task,
                    'replicate': replicate,
                    'arm': arm,
                    'generation': generation,
                    'request_sha256': sha256(request_path),
                    'response_sha256': sha256(response_path),
                    'purpose': 'format_and_training_pipeline_smoke_test_only',
                },
            })
    # A replicate-level split keeps each trajectory chain on only one side.
    train = [row for row in rows if row['metadata']['replicate'] % 5 != 0]
    validation = [row for row in rows if row['metadata']['replicate'] % 5 == 0]
    if not train or not validation:
        raise ValueError('Replicate-level smoke split produced an empty partition')
    output.mkdir(parents=True, exist_ok=True)
    for name, partition in [('train', train), ('validation', validation)]:
        with (output / f'{name}.jsonl').open('w') as handle:
            for row in partition:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    manifest = {
        'schema_version': 'care.rsi_sft_smoke_dataset/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_roots': [str(root) for root in roots],
        'allowed_arms': sorted(ALLOWED_ARMS),
        'split_rule': 'validation when model_replicate modulo 5 equals 0; otherwise training',
        'train_examples': len(train),
        'validation_examples': len(validation),
        'distinct_tasks': sorted({row['metadata']['task'] for row in rows}),
        'claim_boundary': (
            'This dataset is only for checking the A800 QLoRA pipeline and executable JSON '
            'conformance. It is too small and task-overlapping for an efficacy or generalization claim.'
        ),
    }
    manifest['files'] = {
        name: {'sha256': sha256(output / name),
               'bytes': (output / name).stat().st_size}
        for name in ('train.jsonl', 'validation.jsonl')
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output_dir), ensure_ascii=False, indent=2))
