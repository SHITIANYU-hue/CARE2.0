#!/usr/bin/env python3
"""Paired live-RSI analysis with crossed model-replicate / replay-seed bootstrap."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def paired_summary(matrix, seed, draws, family_size=3):
    values=np.asarray(matrix,dtype=float)
    if values.ndim!=2 or min(values.shape)<2 or not np.isfinite(values).all():
        raise ValueError('Need a complete finite replicate-by-seed matrix')
    rng=np.random.default_rng(seed);nrep,nseed=values.shape
    ri=rng.integers(0,nrep,size=(draws,nrep))
    si=rng.integers(0,nseed,size=(draws,nseed))
    # Shared seed indices preserve the crossed seed pairing across model draws.
    samples=values[ri[:,:,None],si[:,None,:]].mean(axis=(1,2))
    alpha=.05/family_size
    return {'mean':float(values.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'familywise_ci':np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),
            'model_replicates':nrep,'seeds_per_replicate':nseed,
            'per_replicate_means':values.mean(axis=1).tolist(),
            'paired_seed_wins':int((values>1e-8).sum()),'paired_seed_ties':int((abs(values)<=1e-8).sum()),
            'paired_seed_losses':int((values < -1e-8).sum())}


def analyze(root):
    config=json.loads((root/'config.json').read_text())
    status=json.loads((root/'run_status.json').read_text())
    if not status.get('complete'):
        raise ValueError('Full planned cases must finish before efficacy aggregation')
    names={'real_moleculenet_freesolv':'FreeSolv','real_moleculenet_lipophilicity':'Lipophilicity','real_matbench_expt_gap':'Experimental band gap'}
    diagnostic_path=root/'first_update_evaluation_metrics.json'
    diagnostic=json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else []
    action_changes=[]
    metrics=[];result=[];checks={'budget_and_pairing':True,'finite_metrics':True,'heldout_feedback_excluded':True}
    methods=['fixed_initial','true_feedback','no_feedback','shuffled_feedback','gp_ucb']
    for spec in config['tasks']:
        rows=[]
        for rep in config['model_replicates']:
            folder=root/spec['id']/f'replicate_{rep}'
            rows+=json.loads((folder/'evaluation_metrics.json').read_text())
            lock=json.loads((folder/'deployment_lock.json').read_text())
            assert not lock['evaluation_feedback_permitted']
            held=set(lock['heldout_seeds'])
            for p in folder.rglob('attempt_*_request.json'):
                request=json.loads(p.read_text())['request']
                prompt=json.loads(request['messages'][1]['content'])
                for memory in prompt.get('persistent_experimental_memory',[]):
                    for sk in memory['feedback']['skill_results']:
                        assert not held.intersection(sk['seeds'])
            baseline={}
            for p in (folder/'fixed_initial'/'evaluation').glob('seed_*.json'):
                arms=json.loads(p.read_text());baseline[arms[0]['seed']]=arms[0]['initial']
            changed=0;compared=0
            for seed in lock['heldout_seeds']:
                a=json.loads((folder/'fixed_initial'/'evaluation'/f'seed_{seed}.json').read_text())[0]
                b=json.loads((folder/'true_feedback'/'evaluation'/f'seed_{seed}.json').read_text())[0]
                changed+=sum(x['selected_candidate']!=y['selected_candidate'] for x,y in zip(a['events'],b['events']))
                compared+=len(a['events'])
            action_changes.append({'task':spec['id'],'replicate':rep,'changed_rounds':changed,'compared_rounds':compared,'fraction':changed/compared})
            for arm in ['fixed_initial','true_feedback','no_feedback','shuffled_feedback']:
                for p in (folder/arm/'evaluation').glob('seed_*.json'):
                    for r in json.loads(p.read_text()):
                        assert r['initial']==baseline[r['seed']]
                        assert len(r['events'])==config['reveal_rounds']
                        observed=set(r['initial']['candidate_ids']);values=list(r['initial']['revealed_values'])
                        for event in r['events']:
                            assert event['selected_candidate'] not in observed
                            observed.add(event['selected_candidate']);values.append(event['revealed_value'])
                            assert abs(event['best_so_far']-max(values))<1e-8
                            assert np.isfinite(event['selected_score'])
                        assert abs(r['metrics']['best_so_far_auc']-np.mean([e['best_so_far'] for e in r['events']]))<.000051
        index={(r['replicate'],r['seed'],r['arm']):r['metrics'] for r in rows}
        expected=len(config['model_replicates'])*config['evaluation_seed_count']*len(methods)
        assert len(index)==len(rows)==expected
        for row in diagnostic:
            if row['task']==spec['id']:
                index[row['replicate'],row['seed'],'first_update_only']=row['metrics']
        seeds=range(config['evaluation_seed_start'],config['evaluation_seed_start']+config['evaluation_seed_count'])
        summary={'task':spec['id'],'method_mean_auc':{},'comparisons':{}}
        for method in methods:
            summary['method_mean_auc'][method]=float(np.mean([index[rep,s,method]['best_so_far_auc'] for rep in config['model_replicates'] for s in seeds]))
        for comparator in ['fixed_initial','no_feedback','shuffled_feedback','gp_ucb'] + (['first_update_only'] if diagnostic else []):
            summary['comparisons'][comparator]={}
            for field in ['best_so_far_auc','final_best']:
                matrix=[[index[rep,s,'true_feedback'][field]-index[rep,s,comparator][field] for s in seeds] for rep in config['model_replicates']]
                summary['comparisons'][comparator][field]=paired_summary(matrix,config['bootstrap_seed'],config['bootstrap_draws'],len(config['tasks']))
        result.append(summary);metrics.extend(rows)
    calls=[json.loads(p.read_text()) for p in root.rglob('attempt_*_response.json')]
    usage={k:sum(int(r.get('response',{}).get('usage',{}).get(k,0) or 0) for r in calls)
           for k in ['prompt_tokens','completion_tokens','total_tokens']}
    failures=[{'path':str(p.relative_to(root)),'status':json.loads(p.read_text()).get('status'),
               'error_type':json.loads(p.read_text()).get('error_type')}
              for p in root.rglob('attempt_*_response.json') if json.loads(p.read_text()).get('status')!='valid']
    raw_trajectories=0;dev_trajectories=0;warning_trajectories=0
    for p in root.rglob('seed_*.json'):
        for row in json.loads(p.read_text()):
            raw_trajectories+=1
            dev_trajectories+=int(row['phase']=='development')
            warning_trajectories+=bool(row['warnings'])
    archive={'tasks':result,'generation_attempts':len(calls),'valid_generations':sum(r['status']=='valid' for r in calls),
             'failed_generation_attempts':failures,'usage':usage,'trajectory_count':raw_trajectories,
             'development_trajectories':dev_trajectories,'evaluation_trajectories':len(metrics),
             'additional_first_update_diagnostic_trajectories':len(diagnostic),
             'numerical_warning_trajectories':warning_trajectories,
             'checks':checks,'true_vs_fixed_executed_action_changes':action_changes,'claim_boundary':config['claim_boundary'],
             'inference_boundary':'Crossed bootstrap resamples three paired model replicates and forty paired replay seeds. Three model replicates are insufficient for strong call-level generalization; intervals are exploratory.'}
    (root/'summary.json').write_text(json.dumps(archive,indent=2)+'\n')
    with (root/'raw_evaluation_metrics.csv').open('w',newline='') as f:
        fields=['task','replicate','seed','arm','skill_id','best_so_far_auc','final_best','simple_regret','top10_hit']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in metrics:w.writerow({k:row[k] if k in row else row['metrics'][k] for k in fields})
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),sharey=True)
    for ax,comp,title in zip(axes,['fixed_initial','no_feedback','shuffled_feedback'],['vs fixed initial skill','vs no-feedback revision','vs shuffled feedback']):
        for i,task in enumerate(result):
            stats=task['comparisons'][comp]['best_so_far_auc'];lo,hi=stats['ci95'];m=stats['mean']
            ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color='#176b7a',lw=2)
            ax.scatter(stats['per_replicate_means'],[i+.10]*3,marker='|',s=70,color='#7d858b')
            ax.annotate(f'{m:+.2f}',(m,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
        ax.axvline(0,color='gray',ls='--',lw=1);ax.set_title(title,fontsize=11)
        ax.set_xlabel('Paired AUC difference (points)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    axes[0].set_yticks(range(len(result)),[names[t['task']] for t in result]);axes[0].set_ylim(len(result)-.45,-.55)
    fig.suptitle('Live LLM skill revision: two feedback updates',fontsize=14)
    fig.text(.5,.015,'3 model replicates × 40 replay seeds per task; exploratory crossed-bootstrap 95% intervals; gray ticks = model replicates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.93));fig.savefig(root/'paired_auc.png',dpi=180);fig.savefig(root/'paired_auc.pdf');plt.close(fig)
    def fmt(stats):return f"{stats['mean']:+.4f} [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"
    lines=['# Live LLM RSI feedback experiment', '',
           'This experiment executes new CommonStack Claude Opus 5 calls to revise executable skills twice. '
           'It compares true-feedback revisions against a shared frozen initial skill, equally frequent revisions without measured feedback, and revisions with swapped skill-feedback bindings.', '',
           '## Measured held-out AUC effects', '',
           '| Task | True feedback − fixed | True − no feedback | True − shuffled | True − GP |',
           '| --- | --- | --- | --- | --- |']
    for task in result:
        lines.append('| '+names[task['task']]+' | '+' | '.join(fmt(task['comparisons'][comp]['best_so_far_auc']) for comp in ['fixed_initial','no_feedback','shuffled_feedback','gp_ucb'])+' |')
    if diagnostic:
        lines += ['', '### First versus second update (secondary diagnostic)', '',
                  '| Task | Second − first true-feedback update | Model-replicate means |',
                  '| --- | --- | --- |']
        for task in result:
            st=task['comparisons']['first_update_only']['best_so_far_auc']
            lines.append('| '+names[task['task']]+' | '+fmt(st)+' | '+', '.join(f'{x:+.4f}' for x in st['per_replicate_means'])+' |')
        lines += ['', 'This additional diagnostic was locked during generation, before any final held-out case metrics existed. It requires 360 extra replay trajectories and no extra model calls; it is not a primary endpoint.']
    lines+=['','Intervals are exploratory 95% crossed-bootstrap intervals. Each task has only three independent '
            'initial model runs with paired revision arms and 40 common replay seeds. Model-call uncertainty remains weakly estimated. '
            'Familywise intervals for the three primary task comparisons are in `summary.json`; other contrasts are descriptive.', '',
            '![Actual held-out effects](paired_auc.png)', '', '## Execution and trace boundary', '',
            f"- {len(calls)} generation attempts, {archive['valid_generations']} valid generations, {len(failures)} failed attempts retained.",
            f"- API usage reported for these generation attempts: {usage['prompt_tokens']} input, {usage['completion_tokens']} output, {usage['total_tokens']} total tokens. This is not an invoiced-dollar estimate.",
            f"- {raw_trajectories} actual finite-pool trajectories: {dev_trajectories} development, {len(metrics)} primary held-out evaluation and {len(diagnostic)} secondary diagnostic evaluation. Each uses 5 initial observations + 10 reveals.",
            '- The initial model draw is shared among arms within each replicate. Each revised arm has two new calls; frozen initial has none. Development/revision overhead is extra relative to the frozen arm.',
            '- All initial candidate IDs, selected candidates, revealed measured values, model prompts and raw responses, normalized skills, memories, revisions and deployment locks are retained.',
            '- Model-selected final skill IDs are frozen before held-out execution. No held-out seed metrics appear in feedback prompts.',
            '- Two skills are evaluated per development generation. True feedback attaches scores to the correct skill; the shuffled arm swaps the two bindings. No-feedback receives its previous proposals but no measured development summaries.',
            '- The end-of-generation memory and executable skill states are experimental artifacts; no global active KB promotion or weight training occurs.', '',
            '## What this can and cannot show', '',
            'A gain over frozen initial alone can reflect extra inference or stochastic proposal variation. '
            'The no-feedback and shuffled-feedback contrasts are required to assess whether real measured feedback adds value. '
            'All tasks and finite data pools have appeared in earlier development. This is a same-task, seed-disjoint '
            'component test, not new-task confirmation or wet-lab evidence. It does not evaluate production KB retrieval. '
            'The frozen arm is the initial live-generated skill, not the separate historical CARE fixed-v2 warmstart controller.', '',
            f"Numerical warnings were recorded for {warning_trajectories} trajectories. Saved deployment metrics, finite selected scores, paired initial observations and reveal budgets were checked by this reporter.",
            'For backend sample checks see `numerical_validation.json` when present. Failed API/format attempts are not silently counted as valid model generations.', '',
            '## Reproduction', '', 'Use the versions in `requirements.txt` and configure `COMMONSTACK_API_KEY` in the environment. Then:', '', '```bash',
            'python experiments/care_replay/scripts/run_rsi_live_feedback.py \\',
            '  --config experiments/care_replay/configs/rsi_live_feedback_v1.json \\',
            '  --output-dir /tmp/care-rsi-live-reproduction --workers 3',
            'python experiments/care_replay/scripts/evaluate_rsi_first_update.py --root /tmp/care-rsi-live-reproduction',
            'python experiments/care_replay/scripts/report_rsi_live_feedback.py \\',
            '  --root /tmp/care-rsi-live-reproduction', '```', '',
            'Live regeneration is stochastic and can vary with provider/model changes. Archived prompt/response and replay records are the exact evidence for this run. Replay seeds do not imply a deterministic LLM seed.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    data=analyze(p.parse_args().root)
    print(json.dumps({'valid_generations':data['valid_generations'],'trajectories':data['trajectory_count'],'usage':data['usage']},indent=2))
