#!/usr/bin/env python3
"""Report only measured executions retained by run_rsi_memory_component.py."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    summary = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    rows = list(csv.DictReader((root / 'raw_metrics.csv').open()))
    names = {'real_moleculenet_freesolv': 'FreeSolv',
             'real_moleculenet_lipophilicity': 'Lipophilicity',
             'real_matbench_expt_gap': 'Experimental band gap'}
    tasks = summary['tasks']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, comparison, title in zip(axes, ['fixed', 'one_shot_first_generation', 'gp_ucb'],
                                    ['vs fixed archived skill', 'vs first update frozen', 'vs target-only GP']):
        for i, task in enumerate(tasks):
            stats = task['comparisons']['updated_vs_' + comparison]['best_so_far_auc']
            low, high = stats['bootstrap_ci95']
            ax.errorbar(stats['mean'], i, xerr=[[stats['mean']-low], [high-stats['mean']]],
                        fmt='o', color='#186a79', capsize=5, lw=2)
            ax.annotate(f"{stats['mean']:+.2f}", (stats['mean'], i),
                        xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
        ax.axvline(0, color='#8b9196', lw=1, ls='--')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Paired AUC difference (points)')
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(tasks)), [names[t['task']] for t in tasks])
    axes[0].set_ylim(-.55, len(tasks)-.45)
    axes[0].invert_yaxis()
    fig.suptitle('Offline skill-selection update: 40 held-out seeds per task', fontsize=14)
    fig.text(.5, .025, '95% paired seed-bootstrap intervals; same finite pools; no new LLM calls or task-disjoint confirmation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.92))
    fig.savefig(root / 'paired_auc.png', dpi=180)
    fig.savefig(root / 'paired_auc.pdf')
    plt.close(fig)
    lines = ['# RSI memory component: actual replay results', '',
             'The skill selector was updated in three development rounds using archived LLM proposals. '
             'This is an offline selection component, not a completed KB-to-LLM recursive self-improvement experiment.', '',
             '## Held-out result', '',
             '| Task | Fixed AUC | Updated AUC | Paired delta [95% CI] | Familywise interval* | Seed W/T/L | Final-best delta [95% CI] |',
             '| --- | ---: | ---: | --- | --- | --- | --- |']
    def fmt(s, key='bootstrap_ci95'):
        lo, hi = s[key]
        return f"{s['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    for task in tasks:
        eval_rows = [r for r in rows if r['task'] == task['task'] and r['phase'] == 'evaluation']
        averages = {m: np.mean([float(r['best_so_far_auc']) for r in eval_rows if r['skill_id'] == task['selected'][m]]) for m in ('fixed','updated')}
        a = task['comparisons']['updated_vs_fixed']['best_so_far_auc']
        b = task['comparisons']['updated_vs_fixed']['final_best']
        lo, hi = a['bootstrap_familywise_ci']
        lines.append(f"| {names[task['task']]} | {averages['fixed']:.4f} | {averages['updated']:.4f} | {fmt(a)} | [{lo:+.4f}, {hi:+.4f}] | {a['wins']}/{a['ties']}/{a['losses']} | {fmt(b)} |")
    lines += ['', '*Bonferroni-adjusted percentile bootstrap intervals across three primary task AUC comparisons '
              '(98.33% marginal coverage; bootstrap approximation, not exact finite-sample coverage). Secondary '
              'comparisons below are descriptive. Intervals are conditional on the fixed development sample and '
              'archived skills. They do not account for new-task or new-LLM-draw variability.', '',
              '## Attribution controls', '',
              '| Task | Updated − first update frozen | Updated − same-feedback batch | Updated − shuffled feedback | Updated − GP |',
              '| --- | --- | --- | --- | --- |']
    for task in tasks:
        stats = [fmt(task['comparisons']['updated_vs_' + c]['best_so_far_auc'])
                 for c in ['one_shot_first_generation', 'batch_same_feedback', 'shuffled_feedback', 'gp_ucb']]
        lines.append('| ' + names[task['task']] + ' | ' + ' | '.join(stats) + ' |')
    lines += ['', 'The cumulative updater and same-feedback batch selector select the same final skill. '
              'Their paired delta is exactly zero. Thus this experiment cannot attribute any improvement over '
              'the initial fixed choice to recursion itself, novel skill generation, or weight learning.', '',
              '![Measured paired AUC outcomes](paired_auc.png)', '', '## Protocol and cost', '',
              '- Three development generations × 10 seeds; 40 separate evaluation seeds per task.',
              '- Every trajectory uses identical paired initial observations and 10 additional reveals (15 observations total).',
              '- All archived normalized skills were included: no skill was removed after observing results.',
              '- Fixed choice: highest archived confidence, then lexicographic ID. Update: cumulative development AUC, incumbent retained on exact ties.',
              '- Selection locks and hash-linked version states precede evaluation on each task; held-out seeds never update the bank.',
              '- Bootstrap: 20,000 paired resamples, seed 20260909.', '']
    total_dev = sum(t['development_trajectories'] for t in tasks)
    lines += [f"Executed {len(rows)} trajectories: {total_dev} development and {len(rows)-total_dev} evaluation. "
              f"Development alone costs {sum(t['development_reveal_cost'] for t in tasks)} replay observations across skill arms. "
              'These are offline evaluations of measured datasets, not new wet-lab measurements. The fixed arm does '
              'not need this development expense; only the same-feedback batch control is development-budget matched.', '',
              f"New LLM calls: **0**. Three existing prompt/raw-response records are retained under `archived_llm_traces/`, "
              'with original paths and SHA-256 in each task provenance file. No model-call uncertainty can be estimated from one archived proposal set per task.', '',
              '## Failures and limitations', '',
              '- The first launch failed before completing any trajectory because tuple-valued normalized conditions were not round-tripped through JSON before worker normalization. The failure is retained under `failed_attempts/`; the corrected serialization has a regression test.',
              f"- {sum(t['numerical_warning_trajectories'] for t in tasks)} completed trajectories emitted NumPy numerical warnings. "
              'Warnings are retained per trajectory. All recorded metrics and selected scores are finite; see `numerical_validation.json` for independent backend checks.',
              '- Each task and its finite candidate pool have appeared in prior development. Seed disjointness does not mean task disjointness; positive intervals do not establish generalization.',
              '- No global candidate skill is promoted to active knowledge. Full fixed-KB versus updating-KB LLM experiments remain pending safe environment credentials and a frozen task-disjoint protocol.', '',
              '## Reproduction', '', 'From the repository root, with Python and the versions in `requirements.txt`:', '', '```bash',
              'python experiments/care_replay/scripts/run_rsi_memory_component.py \\',
              '  --config experiments/care_replay/configs/rsi_memory_component_v1.json \\',
              '  --output-dir /tmp/care-rsi-memory-reproduction --workers 3',
              'python experiments/care_replay/scripts/report_rsi_memory_component.py \\',
              '  --root /tmp/care-rsi-memory-reproduction', '```', '',
              '`raw_metrics.csv` is the complete per-skill/per-seed table. Task folders contain initial reveals, '
              'every subsequent selected candidate and measured value, skill snapshots, warnings, versioned '
              'selection banks, selection locks, dataset hashes and statistical summaries. `protocol_lock.json` '
              'records the base commit, implementation hashes, config and runtime. `AUDIT.md` records the prior implementation audit.']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    report(p.parse_args().root)
