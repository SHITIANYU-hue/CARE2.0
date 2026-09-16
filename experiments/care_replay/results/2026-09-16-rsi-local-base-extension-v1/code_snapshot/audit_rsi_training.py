#!/usr/bin/env python3
"""Check completed epoch accounting, training targets, and saved adapter tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_rsi_live_feedback as live


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(root, verify_weights=False):
    m = json.loads((root / 'metrics.json').read_text())
    d, r, a = m['dataset'], m['result'], m['arguments']
    if m['status'] != 'full_training_complete' or not a['epochs']:
        raise ValueError('Expected a completed full-epoch run')
    epochs, n = a['epochs'], d['train_examples']
    steps = epochs * math.ceil(n / a['gradient_accumulation_steps'])
    if r['completed_epochs'] != epochs or r['examples_seen'] != epochs * n or r['optimizer_steps'] != steps:
        raise ValueError('Incomplete epoch/example/optimizer accounting')
    if d['train_validation_prompt_overlap']:
        raise ValueError('Exact prompts overlap')
    for split in ('train', 'validation'):
        stats = d[f'{split}_encoding']
        if stats['truncated_prompts'] or stats['truncated_answers']:
            raise ValueError('Full-context training contains truncation')
    curve = r['validation_curve']
    if len(curve) != epochs + 1 or not all(math.isfinite(x['loss']) for x in curve):
        raise ValueError('Missing or nonfinite validation curve')
    if any(not x['complete_epoch'] or x['optimizer_step'] != x['epoch'] * (steps // epochs) for x in curve[1:]):
        raise ValueError('Per-epoch checkpoint accounting differs')
    best = min(curve[1:], key=lambda x: x['loss'])
    if best['checkpoint'] != r['best_checkpoint'] or best['loss'] != r['best_validation_loss']:
        raise ValueError('Checkpoint was not selected by the declared validation loss')
    logs = [json.loads(x) for x in (root / 'training_log.jsonl').read_text().splitlines()]
    if [x['optimizer_step'] for x in logs] != list(range(1, steps + 1)) or any(not math.isfinite(x['loss']) for x in logs):
        raise ValueError('Missing or nonfinite optimizer log')
    val_rows, targets = [], 0
    for split in ('train', 'validation'):
        path = Path(a[f'{split}_file'])
        if sha256(path) != d[f'{split}_sha256']:
            raise ValueError('Dataset hash differs from completed run')
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            payload = json.loads(row['messages'][1]['content'])
            live.normalize_response(row['messages'][-1]['content'], payload['target']['public_semantic_fields'], 2)
            targets += 1
        if split == 'validation':
            val_rows = rows
    samples = json.loads((root / 'generated_samples.json').read_text())
    if len(samples) != r['generated_samples'] or len(samples) != min(a['generation_samples'], len(val_rows)):
        raise ValueError('Generated sample accounting differs')
    compiled = []
    for sample, row in zip(samples, val_rows):
        catalog = json.loads(row['messages'][1]['content'])['target']['public_semantic_fields']
        try:
            live.normalize_response(sample['text'], catalog, 2)
            compiled.append({'metadata': sample['metadata'], 'compiler_valid': True})
        except (ValueError, TypeError, KeyError) as exc:
            compiled.append({'metadata': sample['metadata'], 'compiler_valid': False, 'error': str(exc)})
    result = {'status': 'full_training_accounting_verified', 'completed_epochs': epochs,
              'examples_seen': epochs * n, 'optimizer_steps': steps,
              'exact_prompt_overlap': 0, 'truncated_examples': 0,
              'compiler_valid_teacher_targets': targets,
              'selected_epoch': best['epoch'], 'final_adapter_sample_compilation': compiled,
              'weights_verified': False,
              'claim_boundary': 'Accounting and executable format checks do not establish experimental efficacy or generalization. Samples come from the final epoch, not necessarily the selected best epoch.'}
    if verify_weights:
        import torch
        from safetensors import safe_open
        adapter = root / 'best_adapter' / 'adapter_model.safetensors'
        if adapter.parent.resolve() != (Path(best['checkpoint']) / 'adapter').resolve():
            raise ValueError('Best-adapter link differs from validation-selected checkpoint')
        state = torch.load(root / f'checkpoint-epoch-{epochs}' / 'training_state.pt', map_location='cpu', weights_only=False)
        if (state['epoch'], state['optimizer_step'], state['micro_step']) != (epochs, steps, epochs * n):
            raise ValueError('Saved training state differs from completed epoch accounting')
        del state
        finite, tensors, nonzero_b = True, 0, 0
        with safe_open(adapter, framework='pt', device='cpu') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                finite = finite and bool(torch.isfinite(tensor).all())
                tensors += 1
                if 'lora_B' in name and bool(tensor.abs().max() > 0):
                    nonzero_b += 1
        if not finite or not nonzero_b:
            raise ValueError('Selected adapter tensors are nonfinite or unchanged from zero-B initialization')
        result.update(weights_verified=True, adapter_sha256=sha256(adapter),
                      adapter_bytes=adapter.stat().st_size, adapter_tensors=tensors,
                      nonzero_lora_b_tensors=nonzero_b,
                      remote_best_adapter=str((root / 'best_adapter').resolve()))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--verify-weights', action='store_true')
    args = p.parse_args()
    result = audit(args.result_dir, args.verify_weights)
    (args.result_dir / 'training_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
