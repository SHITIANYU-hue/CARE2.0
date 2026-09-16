#!/usr/bin/env python3
"""Audit completed RSI prompts against frozen SFT train/validation inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def fingerprint(messages, normalize_json=False):
    copied = json.loads(json.dumps(messages))
    if normalize_json:
        for message in copied:
            if message['role'] == 'user':
                try:
                    message['content'] = json.dumps(json.loads(message['content']), sort_keys=True)
                except ValueError:
                    pass
    return hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest()


def audit(run_dir, sft_dir):
    if not json.loads((run_dir / 'run_status.json').read_text()).get('complete'):
        raise ValueError('Audit requires a complete RSI run')
    exact, normalized = defaultdict(list), defaultdict(list)
    teacher_rows = []
    for name in ('train', 'validation'):
        for line in (sft_dir / f'{name}.jsonl').read_text().splitlines():
            row = json.loads(line)
            messages = row['messages'][:-1]
            payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
            info = {'split': name, 'target': payload['target']['dataset'],
                    'source': payload['source_transfer_evidence']['source_dataset']}
            teacher_rows.append(info)
            exact[fingerprint(messages)].append(info)
            normalized[fingerprint(messages, True)].append(info)
    requests = []
    for path in sorted(run_dir.rglob('attempt_*_request.json')):
        # Code snapshots contain only Python; every matched file is raw evidence.
        messages = json.loads(path.read_text())['request']['messages']
        payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        requests.append({'request': str(path.relative_to(run_dir)),
                         'target': payload['target']['dataset'],
                         'source': payload['source_transfer_evidence']['source_dataset'],
                         'exact_sft_rows': exact[fingerprint(messages)],
                         'normalized_sft_rows': normalized[fingerprint(messages, True)]})
    config = json.loads((run_dir / 'config.json').read_text())
    targets = {spec['id']: {
        'sft_target_rows': sum(r['target'] == spec['id'] for r in teacher_rows),
        'sft_source_rows': sum(r['source'] == spec['id'] for r in teacher_rows),
        'source_as_sft_source_rows': sum(r['source'] == spec['source'] for r in teacher_rows),
    } for spec in config['tasks']}
    return {'status': 'completed_prompt_exposure_audit', 'targets': targets,
            'requests': requests, 'request_count': len(requests),
            'exact_matching_requests': sum(bool(r['exact_sft_rows']) for r in requests),
            'normalized_matching_requests': sum(bool(r['normalized_sft_rows']) for r in requests),
            'sft_file_hashes': {name: hashlib.sha256((sft_dir / name).read_bytes()).hexdigest()
                               for name in ('train.jsonl', 'validation.jsonl')},
            'claim_boundary': 'No exact prompt match does not establish candidate or domain independence. Related schemas/source evidence and shared candidate pools may remain. Matching training prompts permit memorization. No evaluation-driven checkpoint selection or policy changes are performed.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--sft-dir', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run_dir, args.sft_dir)
    (args.run_dir / 'prompt_exposure_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}, indent=2))
