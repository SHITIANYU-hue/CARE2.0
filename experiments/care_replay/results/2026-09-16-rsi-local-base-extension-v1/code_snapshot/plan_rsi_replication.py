#!/usr/bin/env python3
"""Turn an archived live-RSI result into a reproducible replication decision aid."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


TASK_NAMES = {
    'real_moleculenet_freesolv': 'FreeSolv（水合自由能）',
    'real_moleculenet_lipophilicity': 'Lipophilicity（脂溶性）',
    'real_matbench_expt_gap': 'Experimental band gap（实验带隙）',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_effect(stats):
    values = [float(x) for x in stats['per_replicate_means']]
    mean = float(stats['mean'])
    between_replicate_sd = statistics.stdev(values)
    if abs(mean) < 1e-12:
        approximate_total_replicates_for_80pct_power = None
    else:
        approximate_total_replicates_for_80pct_power = math.ceil(
            ((1.96 + 0.84) * between_replicate_sd / abs(mean)) ** 2)
    leave_one_out_means = [
        statistics.mean(values[:index] + values[index + 1:])
        for index in range(len(values))
    ]
    return {
        'mean_auc_difference': mean,
        'exploratory_ci95': stats['ci95'],
        'per_model_replicate_means': values,
        'positive_replicates': sum(x > 1e-8 for x in values),
        'zero_replicates': sum(abs(x) <= 1e-8 for x in values),
        'negative_replicates': sum(x < -1e-8 for x in values),
        'between_replicate_sd': between_replicate_sd,
        'leave_one_replicate_out_means': leave_one_out_means,
        'approximate_total_replicates_for_80pct_power': approximate_total_replicates_for_80pct_power,
    }


def analyze(root):
    summary_path = root / 'summary.json'
    config_path = root / 'config.json'
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    action_index = {}
    for row in summary['true_vs_fixed_executed_action_changes']:
        action_index.setdefault(row['task'], []).append(float(row['fraction']))
    tasks = []
    for task in summary['tasks']:
        task_id = task['task']
        comparisons = {
            name: summarize_effect(task['comparisons'][name]['best_so_far_auc'])
            for name in ('fixed_initial', 'no_feedback', 'shuffled_feedback', 'first_update_only')
        }
        tasks.append({
            'task': task_id,
            'display_name': TASK_NAMES.get(task_id, task_id),
            'true_feedback_minus': comparisons,
            'executed_action_change_fraction_per_replicate': action_index[task_id],
            'mean_executed_action_change_fraction': statistics.mean(action_index[task_id]),
        })
    current_replicates = len(config['model_replicates'])
    return {
        'schema_version': 'care.rsi_replication_decision_aid/v1',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'source_result': str(root),
        'source_summary_sha256': sha256(summary_path),
        'source_config_sha256': sha256(config_path),
        'current_model_replicates': current_replicates,
        'planned_combined_model_replicates': 6,
        'expected_standard_error_multiplier_at_six_replicates': math.sqrt(current_replicates / 6),
        'planned_extension_valid_generation_calls': 63,
        'tasks': tasks,
        'interpretation_boundary': (
            '近似重复数使用正态近似，并且方差只来自 3 个已归档的模型重复均值。'
            '它只能帮助安排预算，不能当作正式的验证性功效计算；新时间点的模型调用方差也可能不同。'
        ),
    }


def write_report(data, output):
    by_id = {task['task']: task for task in data['tasks']}
    free = by_id['real_moleculenet_freesolv']
    lipo = by_id['real_moleculenet_lipophilicity']
    gap = by_id['real_matbench_expt_gap']

    def effect(task, comparison):
        value = task['true_feedback_minus'][comparison]
        lo, hi = value['exploratory_ci95']
        signs = (value['positive_replicates'], value['zero_replicates'], value['negative_replicates'])
        return f"{value['mean_auc_difference']:+.4f}（95% 探索区间 {lo:+.4f} 到 {hi:+.4f}；模型重复正/零/负={signs[0]}/{signs[1]}/{signs[2]}）"

    lines = [
        '# CARE 2.0 / RSI 续跑决策记录',
        '',
        '## 已有证据怎么读',
        '',
        f"- **FreeSolv 最值得继续复现。** 真反馈相对固定初始策略为 {effect(free, 'fixed_initial')}。方向有希望，但区间仍跨 0；真反馈相对无反馈为 {effect(free, 'no_feedback')}，还不能证明收益一定来自真实反馈。",
        f"- **Lipophilicity 已经证明“决策会变”，没有证明“结果会更好”。** 真反馈让执行动作平均改变 {lipo['mean_executed_action_change_fraction']:.1%}，但相对固定初始策略只有 {effect(lipo, 'fixed_initial')}。继续堆同一机制的重复价值较低。",
        f"- **带隙任务提示反馈格式可能在防止错误，而不是产生净增益。** 真反馈相对固定初始策略为 {effect(gap, 'fixed_initial')}；相对打乱反馈为 {effect(gap, 'shuffled_feedback')}，但相对无反馈为 {effect(gap, 'no_feedback')}。应先改进反馈摘要，再扩大重复。",
        '',
        '## 已冻结的下一步',
        '',
        '- 按相同任务、开发种子、测试种子和预算，再增加 3 个独立模型重复，把总重复数从 3 增加到 6。',
        '- 每个重复保留真反馈、无反馈、打乱反馈三组；预计需要 63 个有效模型生成。',
        f"- 如果未来调用的方差与历史相近，6 个重复的标准误约为当前的 {data['expected_standard_error_multiplier_at_six_replicates']:.3f}，即下降约 29%。这仍属于探索性证据。",
        '- 扩展结果必须单独报告，再与旧结果合并；不能只报合并后的较好数字。',
        '',
        '## 续跑后的判断门槛',
        '',
        '1. 先看真反馈相对固定初始策略是否稳定为正。',
        '2. 再看真反馈是否同时优于无反馈和打乱反馈；否则不能把收益归因于实验反馈。',
        '3. 如果只改变动作、不提高 AUC，就停止增加同类重复，转去修改反馈内容或增加新任务族。',
        '4. 湿实验阶段只采用在独立重复中方向稳定、且相对两个机制对照都占优的策略。',
        '',
        '## 统计边界',
        '',
        data['interpretation_boundary'],
        '',
        '完整数值和来源哈希见 `continuation_analysis.json`。',
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'CONTINUATION_PLAN_ZH.md').write_text('\n'.join(lines) + '\n')
    (output / 'continuation_analysis.json').write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    write_report(result, args.output_dir)
    print(json.dumps({
        'tasks': len(result['tasks']),
        'current_model_replicates': result['current_model_replicates'],
        'planned_extension_valid_generation_calls': result['planned_extension_valid_generation_calls'],
    }, indent=2))
