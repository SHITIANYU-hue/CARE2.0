#!/usr/bin/env python3
"""Combine complete live-RSI archives while retaining execution-period strata."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from report_rsi_live_feedback import paired_summary


TASK_NAMES = {
    'real_moleculenet_freesolv': 'FreeSolv',
    'real_moleculenet_lipophilicity': 'Lipophilicity',
    'real_matbench_expt_gap': 'Experimental band gap',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive(root):
    config=json.loads((root/'config.json').read_text())
    status=json.loads((root/'run_status.json').read_text())
    if not status.get('complete'):
        raise ValueError(f'Incomplete archive: {root}')
    rows=[]
    for spec in config['tasks']:
        for replicate in config['model_replicates']:
            path=root/spec['id']/f'replicate_{replicate}'/'evaluation_metrics.json'
            rows.extend(json.loads(path.read_text()))
    diagnostic_path=root/'first_update_evaluation_metrics.json'
    diagnostic=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else []
    summary=json.loads((root/'summary.json').read_text())
    return {'root':root,'config':config,'rows':rows,'diagnostic':diagnostic,'summary':summary}


def analyze(roots, output):
    archives=[load_archive(root) for root in roots]
    reference=archives[0]['config']
    task_ids=[x['id'] for x in reference['tasks']]
    scientific_fields=('updates','skills_per_generation','development_seed_start',
                       'development_seeds_per_generation','evaluation_seed_start',
                       'evaluation_seed_count','initial_observations','reveal_rounds','arms')
    for archive in archives[1:]:
        if [x['id'] for x in archive['config']['tasks']] != task_ids:
            raise ValueError('Task sets differ')
        for field in scientific_fields:
            if archive['config'][field] != reference[field]:
                raise ValueError(f'Scientific config differs: {field}')
    replicate_ids=[rep for archive in archives for rep in archive['config']['model_replicates']]
    if len(replicate_ids) != len(set(replicate_ids)):
        raise ValueError('Replicate IDs overlap across archives')
    rows=[row for archive in archives for row in archive['rows']]
    diagnostic=[row for archive in archives for row in archive['diagnostic']]
    if any(not archive['diagnostic'] for archive in archives):
        diagnostic=[]
    seeds=list(range(reference['evaluation_seed_start'],
                     reference['evaluation_seed_start']+reference['evaluation_seed_count']))
    methods=['fixed_initial','true_feedback','no_feedback','shuffled_feedback','gp_ucb']
    task_results=[]
    for task_id in task_ids:
        task_rows=[row for row in rows if row['task']==task_id]
        index={(row['replicate'],row['seed'],row['arm']):row['metrics'] for row in task_rows}
        expected=len(replicate_ids)*len(seeds)*len(methods)
        if len(index) != expected:
            raise ValueError(f'Incomplete or duplicated primary rows for {task_id}')
        for row in diagnostic:
            if row['task']==task_id:
                index[row['replicate'],row['seed'],'first_update_only']=row['metrics']
        comparisons={}
        comparator_names=['fixed_initial','no_feedback','shuffled_feedback','gp_ucb']
        if diagnostic:
            comparator_names.append('first_update_only')
        for comparator in comparator_names:
            comparisons[comparator]={}
            for metric in ('best_so_far_auc','final_best'):
                matrix=[[index[rep,seed,'true_feedback'][metric]-index[rep,seed,comparator][metric]
                         for seed in seeds] for rep in replicate_ids]
                comparisons[comparator][metric]=paired_summary(
                    matrix,reference['bootstrap_seed'],reference['bootstrap_draws'],len(task_ids))
        period_effects=[]
        for archive in archives:
            reps=archive['config']['model_replicates']
            period_effects.append({
                'root':str(archive['root']),
                'model_replicates':reps,
                'true_feedback_minus_fixed_auc':float(np.mean([
                    index[rep,seed,'true_feedback']['best_so_far_auc']-
                    index[rep,seed,'fixed_initial']['best_so_far_auc']
                    for rep in reps for seed in seeds])),
                'true_feedback_minus_no_feedback_auc':float(np.mean([
                    index[rep,seed,'true_feedback']['best_so_far_auc']-
                    index[rep,seed,'no_feedback']['best_so_far_auc']
                    for rep in reps for seed in seeds])),
                'true_feedback_minus_shuffled_feedback_auc':float(np.mean([
                    index[rep,seed,'true_feedback']['best_so_far_auc']-
                    index[rep,seed,'shuffled_feedback']['best_so_far_auc']
                    for rep in reps for seed in seeds])),
            })
        task_results.append({
            'task':task_id,
            'method_mean_auc':{method:float(np.mean([
                index[rep,seed,method]['best_so_far_auc']
                for rep in replicate_ids for seed in seeds])) for method in methods},
            'comparisons':comparisons,
            'execution_period_effects':period_effects,
        })
    usage={key:sum(archive['summary']['usage'][key] for archive in archives)
           for key in ('prompt_tokens','completion_tokens','total_tokens')}
    result={
        'schema_version':'care.rsi_live_feedback_combined/v1',
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'source_archives':[{
            'root':str(archive['root']),
            'summary_sha256':sha256(archive['root']/'summary.json'),
            'config_sha256':sha256(archive['root']/'config.json'),
            'model_replicates':archive['config']['model_replicates'],
            'valid_generations':archive['summary']['valid_generations'],
            'generation_attempts':archive['summary']['generation_attempts'],
        } for archive in archives],
        'model_replicates':replicate_ids,
        'model_replicate_count':len(replicate_ids),
        'seeds_per_replicate':reference['evaluation_seed_count'],
        'tasks':task_results,
        'usage':usage,
        'valid_generations':sum(x['summary']['valid_generations'] for x in archives),
        'generation_attempts':sum(x['summary']['generation_attempts'] for x in archives),
        'failed_generation_attempts':sum(len(x['summary']['failed_generation_attempts']) for x in archives),
        'primary_evaluation_trajectories':sum(x['summary']['evaluation_trajectories'] for x in archives),
        'development_trajectories':sum(x['summary']['development_trajectories'] for x in archives),
        'first_update_diagnostic_trajectories':sum(x['summary']['additional_first_update_diagnostic_trajectories'] for x in archives),
        'first_update_diagnostic_boundary':(
            '旧实验在生成过程中锁定了第一次与第二次更新对照；新扩展在主要运行完成后才补算。'
            '因此合并后的更新次数结果属于事后描述，不能当成预注册终点。'
            if diagnostic else '至少一个来源没有更新次数诊断，因此未合并。'),
        'claim_boundary':(
            '每个任务合并了 6 条独立模型重复链，每条链使用 40 个配对回放种子。由于模型服务状态和执行时间不同，'
            '旧实验与新扩展仍需分开报告。三个有限候选池任务都曾参与开发，因此这不是任务隔离验证、前瞻实验或湿实验结果。'),
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/'combined_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'combined_raw_evaluation_metrics.csv').open('w',newline='') as handle:
        fields=['task','replicate','seed','arm','skill_id','best_so_far_auc','final_best','simple_regret','top10_hit']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            writer.writerow({key:row[key] if key in row else row['metrics'][key] for key in fields})
    plot(result,output)
    write_report(result,output)
    return result


def plot(result, output):
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
    comparisons=['fixed_initial','no_feedback','shuffled_feedback']
    titles=['vs fixed initial skill','vs no-feedback revision','vs shuffled feedback']
    for axis,comparator,title in zip(axes,comparisons,titles):
        for index,task in enumerate(result['tasks']):
            stats=task['comparisons'][comparator]['best_so_far_auc']
            lo,hi=stats['ci95'];mean=stats['mean']
            axis.errorbar(mean,index,xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,
                          color='#176b7a',lw=2)
            axis.scatter(stats['per_replicate_means'],[index+.10]*len(result['model_replicates']),
                         marker='|',s=70,color='#7d858b')
            axis.annotate(f'{mean:+.2f}',(mean,index),xytext=(0,12),
                          textcoords='offset points',ha='center',fontsize=10)
        axis.axvline(0,color='gray',ls='--',lw=1);axis.set_title(title,fontsize=11)
        axis.set_xlabel('Paired AUC difference (points)')
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(len(result['tasks'])),
                       [TASK_NAMES[x['task']] for x in result['tasks']])
    axes[0].set_ylim(len(result['tasks'])-.45,-.55)
    fig.suptitle('Live LLM skill revision: combined independent runs',fontsize=14)
    fig.text(.5,.015,'6 model replicates × 40 paired replay seeds per task; crossed-bootstrap 95% intervals.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.93))
    fig.savefig(output/'combined_paired_auc.png',dpi=180)
    fig.savefig(output/'combined_paired_auc.pdf');plt.close(fig)


def write_report(result, output):
    def fmt(stats):
        return f"{stats['mean']:+.4f} [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"
    lines=['# CARE 2.0 / RSI 六模型重复合并结果','',
           '旧实验和本次扩展各有 3 个独立模型重复。合并分析使用 6 个模型重复，并在每个模型重复内保留相同的 40 个配对回放种子。','',
           '| 任务 | 真反馈 − 固定初始策略 | 真反馈 − 无反馈修订 | 真反馈 − 打乱反馈 | 真反馈 − GP-UCB |',
           '| --- | ---: | ---: | ---: | ---: |']
    for task in result['tasks']:
        lines.append('| '+TASK_NAMES[task['task']]+' | '+' | '.join(
            fmt(task['comparisons'][name]['best_so_far_auc'])
            for name in ('fixed_initial','no_feedback','shuffled_feedback','gp_ucb'))+' |')
    indexed={task['task']:task for task in result['tasks']}
    free=indexed['real_moleculenet_freesolv']['comparisons']
    lipo=indexed['real_moleculenet_lipophilicity']['comparisons']
    gap=indexed['real_matbench_expt_gap']['comparisons']
    lines+=['','## 主要结论','',
            f"- **FreeSolv 是目前最一致的 RSI 信号。** 真反馈相对固定初始策略为 {fmt(free['fixed_initial']['best_so_far_auc'])}，区间只略微跨过 0；相对打乱反馈为 {fmt(free['shuffled_feedback']['best_so_far_auc'])}。但相对无反馈仍为 {fmt(free['no_feedback']['best_so_far_auc'])}，所以反馈归因还不完整。",
            f"- **Lipophilicity 没有稳定的 RSI 净增益。** 真反馈相对固定初始策略为 {fmt(lipo['fixed_initial']['best_so_far_auc'])}，相对无反馈和打乱反馈也都跨 0。",
            f"- **带隙结果有明显批次差异。** 合并后相对固定初始策略为 {fmt(gap['fixed_initial']['best_so_far_auc'])}，但旧 3 次接近 0、新 3 次很大，且相对无反馈仍跨 0，暂时不能当作稳定收益。",
            '- 三个任务中真反馈策略都优于 GP-UCB，但这个对照同时包含初始技能本身的优势，不能单独证明两次反馈更新有效。','',
            '## 分时间批次检查','',
            '下表保留旧实验与新扩展的真反馈相对固定初始策略均值，防止合并结果掩盖模型或服务商随时间变化。','',
            '| 任务 | 旧 3 次 | 新 3 次 |','| --- | ---: | ---: |']
    for task in result['tasks']:
        periods=task['execution_period_effects']
        lines.append(f"| {TASK_NAMES[task['task']]} | {periods[0]['true_feedback_minus_fixed_auc']:+.4f} | {periods[1]['true_feedback_minus_fixed_auc']:+.4f} |")
    lines+=['','![六模型重复合并结果](combined_paired_auc.png)','',
            '## 怎么解释','',
            '- 真反馈相对固定初始策略回答“经过两次更新后是否更好”。',
            '- 真反馈相对无反馈和打乱反馈回答“收益是否来自正确的实验反馈”。只有三项一起成立，才适合把增益归因于反馈驱动 RSI。',
            '- 置信区间按模型重复和回放种子交叉重采样。40 个回放种子不能当成 40 次独立模型实验。',
            '',
            '## 执行记录','',
            f"- {result['valid_generations']} 次有效生成，{result['generation_attempts']} 次总尝试，{result['failed_generation_attempts']} 次失败尝试已保留。",
            f"- API 共报告 {result['usage']['total_tokens']} tokens。",
            f"- {result['development_trajectories']} 条开发轨迹，{result['primary_evaluation_trajectories']} 条主要测试轨迹，{result['first_update_diagnostic_trajectories']} 条第一次更新诊断轨迹。",
            '- 本次扩展没有训练模型权重；权重训练属于另一个待验证实验臂。','',
            '## 证据边界','',result['claim_boundary'],'',result['first_update_diagnostic_boundary']]
    (output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',action='append',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    data=analyze(args.root,args.output_dir)
    print(json.dumps({'model_replicates':data['model_replicate_count'],
                      'valid_generations':data['valid_generations'],
                      'primary_trajectories':data['primary_evaluation_trajectories']},indent=2))
