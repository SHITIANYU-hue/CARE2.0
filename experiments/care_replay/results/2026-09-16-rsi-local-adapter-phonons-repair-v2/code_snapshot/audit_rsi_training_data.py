#!/usr/bin/env python3
"""Audit live-RSI archives before any model-weight training is attempted."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_location(path, root):
    parts = path.relative_to(root).parts
    task = parts[0]
    replicate = int(parts[1].split('_', 1)[1])
    if parts[2] == 'initial_call':
        return task, replicate, 'initial', 0
    return task, replicate, parts[2], int(parts[3].split('_', 1)[1])


def audit(roots):
    examples = []
    archive_hashes = {}
    for root in roots:
        archive_hashes[str(root)] = {
            name: sha256(root / name)
            for name in ('config.json', 'protocol_lock.json', 'run_status.json')
            if (root / name).exists()
        }
        for response_path in sorted(root.rglob('attempt_*_response.json')):
            response = json.loads(response_path.read_text())
            if response.get('status') != 'valid':
                continue
            task, replicate, arm, generation = parse_location(response_path, root)
            request_path = response_path.with_name(
                response_path.name.replace('_response.json', '_request.json'))
            if not request_path.exists():
                raise FileNotFoundError(request_path)
            # Only generation 1 has a subsequent development-only evaluation.
            development_label_available = generation == 1 and (
                response_path.parents[1] / 'development').exists()
            examples.append({
                'archive': str(root),
                'task': task,
                'replicate': replicate,
                'arm': arm,
                'generation': generation,
                'request': str(request_path.relative_to(root)),
                'response': str(response_path.relative_to(root)),
                'request_sha256': sha256(request_path),
                'response_sha256': sha256(response_path),
                'development_label_available_without_heldout_outcomes': development_label_available,
                'total_tokens': int(response.get('response', {}).get('usage', {}).get('total_tokens', 0) or 0),
            })
    tasks = sorted({x['task'] for x in examples})
    labeled = [x for x in examples if x['development_label_available_without_heldout_outcomes']]
    true_labeled = [x for x in labeled if x['arm'] == 'true_feedback']
    return {
        'schema_version': 'care.rsi_training_data_audit/v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'archive_hashes': archive_hashes,
        'valid_prompt_response_examples': len(examples),
        'distinct_tasks': len(tasks),
        'task_ids': tasks,
        'examples_by_arm': dict(sorted(Counter(x['arm'] for x in examples).items())),
        'examples_by_generation': dict(sorted(Counter(str(x['generation']) for x in examples).items())),
        'development_labeled_revision_examples': len(labeled),
        'true_feedback_development_labeled_revision_examples': len(true_labeled),
        'total_archived_tokens': sum(x['total_tokens'] for x in examples),
        'efficacy_training_gate': {
            'minimum_examples': 1000,
            'minimum_distinct_training_task_families': 8,
            'heldout_task_families_required': 2,
            'gate_passed': len(examples) >= 1000 and len(tasks) >= 10,
        },
        'allowed_current_use': (
            'Pipeline and JSON-format smoke testing only. The archive is too small and too task-concentrated '
            'for a defensible model-weight efficacy claim.'
        ),
        'leakage_rule': (
            'Held-out evaluation outcomes must never select SFT examples, targets, weights, checkpoints, '
            'hyperparameters, or stopping points. A task family used for weight training cannot be used as '
            'the confirmatory evaluation family.'
        ),
        'examples': examples,
    }


def write_report(data, output):
    gate = data['efficacy_training_gate']
    lines = [
        '# RSI 模型训练数据审计',
        '',
        f"- 可用的有效提示—回答样本：{data['valid_prompt_response_examples']} 条。",
        f"- 独立任务：{data['distinct_tasks']} 个。",
        f"- 不读取测试集结果也能附带开发标签的修订样本：{data['development_labeled_revision_examples']} 条，其中真反馈 {data['true_feedback_development_labeled_revision_examples']} 条。",
        f"- 当前只能做训练管线和 JSON 格式稳定性的冒烟测试：{'是' if not gate['gate_passed'] else '否'}。",
        '',
        '## 为什么现在不能直接宣称“训练让 RSI 更好”',
        '',
        '现有样本少，而且集中在同三个任务。如果拿这些任务训练，再在同样任务上测试，模型可能只是记住任务和规则写法。即使分数上涨，也不能证明能迁移到新实验。',
        '',
        '## 正式训练前的门槛',
        '',
        f"- 至少 {gate['minimum_examples']} 条训练样本。",
        f"- 至少 {gate['minimum_distinct_training_task_families']} 个训练任务族。",
        f"- 至少 {gate['heldout_task_families_required']} 个完全不参与训练的测试任务族。",
        '- 模型选择、早停和超参数只能使用训练/验证任务；最终测试结果只能打开一次。',
        '',
        '详细样本路径和哈希见 `training_data_audit.json`。',
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'TRAINING_DATA_AUDIT_ZH.md').write_text('\n'.join(lines) + '\n')
    (output / 'training_data_audit.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    write_report(result, args.output_dir)
    print(json.dumps({key: result[key] for key in (
        'valid_prompt_response_examples', 'distinct_tasks',
        'development_labeled_revision_examples')}, indent=2))
