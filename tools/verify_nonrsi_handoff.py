#!/usr/bin/env python3
"""Verify restored strict/fence non-RSI results without network or old-directory writes.

Run from a restored checkout with the frozen WSL environment:
    .venv/bin/python -B tools/verify_nonrsi_handoff.py --output outputs/my-audit

This validates existing generation, locks, checkpoints, traces and supplemental
derivation. It does not regenerate rules, rerun policies or recompute statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/care_replay'
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(output):
    sys.path.insert(0, str(BASE))
    for name in tuple(os.environ):
        if name.endswith(('_API_KEY', '_AUTH_TOKEN')):
            os.environ.pop(name)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      MPLCONFIGDIR=str(output / 'matplotlib-config'))
    from protocols.nonrsi_cross_model_v1 import freeze, generation, inputs, pipeline
    from protocols.nonrsi_cross_model_v1.common import digest, load, offline, slots
    from protocols.nonrsi_cross_model_offline_resume_v1 import runner as original
    from protocols.nonrsi_fence_supplement_v1 import runner as supplement

    report = {'new_model_requests': 0, 'network_disabled': True,
              'statistics_recomputed': False, 'policies_rerun': False,
              'old_frozen_directories_written': False}

    def log(message):
        with (output / 'completion_audit.log').open('a', encoding='utf-8') as handle:
            handle.write(datetime.now(timezone.utc).isoformat() + ' ' + message + '\n')
        print(message, flush=True)

    def check_files(directory, manifest):
        for name, expected in manifest.items():
            require(freeze.file_hash(directory / name) == expected, 'File hash changed: ' + name)
        return len(manifest)

    def audit_evaluation(directory, config, attempts, prior=None):
        definition = load(directory / 'definition.json')
        lock = load(directory / 'CALIBRATION_LOCK.json')
        complete = load(directory / 'HELDOUT_COMPLETE.json')
        analysis = load(directory / 'analysis.json')
        require(len(definition['slots']) == 660, 'Expected 660 definition slots')
        require([a['slot'] for a in attempts] == slots(config), 'Generation slot order changed')
        require(definition['generation_sha256'] == digest(attempts), 'Generation binding changed')
        require(lock['config_sha256'] == complete['config_sha256'] == digest(config), 'Config binding changed')
        require(lock['definition_sha256'] == complete['definition_sha256'] == digest(definition), 'Definition binding changed')
        require(complete['calibration_lock_sha256'] == digest(lock), 'Completion/calibration binding changed')
        require(set(lock['selections']) == set(complete['metrics']) == set(definition['targets']), 'Target inventory changed')
        require(len(definition['targets']) == 6 and len(lock['checkpoints']) == 300, 'Expected six targets/300 calibration checkpoints')
        require(len(analysis['primary']) == 36, 'Expected 36 primary comparisons')
        expected_primary = {
            (c['cell'] + ' minus common BO', c['model'], c['target'])
            for c in config['generation_cells'] if c['core_fixed_task']
        } | {
            (r['route_id'] + ' F minus T', m['id'], r['target'])
            for m in config['models'] for r in config['routes']
        }
        require({(r['contrast'], r['model'], r['target']) for r in analysis['primary']} == expected_primary,
                'Primary comparison identities changed')
        require(analysis['statistical_scope'] == 'descriptive_cross_model_replication', 'Statistical scope changed')
        for name, expected in lock['checkpoints'].items():
            require(digest(load(directory / name)) == expected, 'Calibration checkpoint changed: ' + name)
        summary = {'primary_comparisons': 36, 'targets': {}, 'calibration_checkpoints': 0,
                   'heldout_checkpoints': 0, 'trace_hashes_verified': 0,
                   'metric_rows_verified': 0, 'original_rows_reused_exactly': 0,
                   'supplement_baseline_consistency_records': 0}
        for target, item in definition['targets'].items():
            require(inputs.certify(inputs.adapter(target))['complete_candidate_records_sha256'] == item['pool_sha256'], 'Candidate pool changed: ' + target)
            require(load(directory / target / 'selection.json') == lock['selections'][target], 'Selection changed: ' + target)
            per_target = {}
            for split, count in [('calibration', 50), ('heldout', 100)]:
                seeds = pipeline.seed_set(config, target, split)
                paths = sorted((directory / target / split).glob('*.json'), key=lambda p: int(p.stem))
                require(len(paths) == count and {int(p.stem) for p in paths} == set(seeds), 'Seed/checkpoint inventory changed: ' + target + '/' + split)
                skills = supplement.wanted_skills(definition, target, split, directory)
                modes = (*supplement.matched.BASELINES, *(supplement.matched.semantic.skill_mode(s) for s in skills))
                binding = digest([digest(config), digest(definition), target, split, [asdict(s) for s in skills]])
                combined = []
                for path in paths:
                    checkpoint = load(path)
                    require(checkpoint['binding'] == binding, 'Checkpoint binding changed: ' + str(path))
                    require(checkpoint['rows_sha256'] == digest(checkpoint['rows']), 'Metric hash changed: ' + str(path))
                    traces = json.loads(gzip.decompress(path.with_suffix('.traces.json.gz').read_bytes()))
                    require(checkpoint['traces_sha256'] == digest(traces), 'Trace hash changed: ' + str(path))
                    seed = int(path.stem)
                    supplement.matched.validate_rows(checkpoint['rows'], modes, {seed}, split)
                    require(len(traces) == 10 * len(checkpoint['rows']), 'Trace action count changed: ' + str(path))
                    trace_counts = Counter(e['mode'] for e in traces)
                    require(set(trace_counts) == set(modes) and all(n == 10 for n in trace_counts.values()), 'Trace mode inventory changed: ' + str(path))
                    if prior is not None:
                        oldpath = prior / target / split / path.name
                        old = load(oldpath)
                        require(checkpoint['source_checkpoint_sha256'] == freeze.file_hash(oldpath), 'Supplement source file changed')
                        require(checkpoint['source_binding'] == old['binding'], 'Supplement source binding changed')
                        old_modes = {r['mode']: r for r in old['rows']}
                        new_modes = {r['mode']: r for r in checkpoint['rows']}
                        require(old_modes.keys() <= new_modes.keys() and all(new_modes[k] == v for k, v in old_modes.items()), 'Previously available metrics changed')
                        require(checkpoint['reused_modes'] == len(old_modes) and checkpoint['new_modes'] == len(new_modes) - len(old_modes), 'Supplement mode accounting changed')
                        require(checkpoint['baseline_consistency_checked'] == bool(checkpoint['new_modes']), 'Supplement baseline consistency record changed')
                        summary['original_rows_reused_exactly'] += len(old_modes)
                        summary['supplement_baseline_consistency_records'] += int(checkpoint['baseline_consistency_checked'])
                    summary['trace_hashes_verified'] += 1
                    summary['metric_rows_verified'] += len(checkpoint['rows'])
                    combined.extend(checkpoint['rows'])
                per_target[split] = len(paths)
                summary[split + '_checkpoints'] += len(paths)
                if split == 'heldout':
                    metrics = load(directory / target / 'heldout_metrics.json')
                    require(metrics == combined, 'Aggregate metrics differ from checkpoints: ' + target)
                    require(digest(metrics) == complete['metrics'][target], 'Aggregate metrics hash changed: ' + target)
            summary['targets'][target] = per_target
            log('Verified ' + directory.parent.name + ': ' + target + ' (50 calibration, 100 heldout; all trace hashes)')
        require(set(definition['slots']) == {a['slot']['slot_id'] for a in attempts}, 'Definition slot IDs changed')
        summary['definition_file_sha256'] = freeze.file_hash(directory / 'definition.json')
        summary['analysis_file_sha256'] = freeze.file_hash(directory / 'analysis.json')
        return summary, definition, analysis

    with offline():
        for path in [original.SOURCE / 'journal/ATTEMPTS.json',
                     original.SOURCE / 'journal/science_ledger/ledger.json',
                     original.OUTPUT / 'FINAL_REPORT.txt',
                     supplement.EVAL / 'SUPPLEMENT_REPORT.txt']:
            require(path.is_file(), 'Restore completed result archives before verifying: ' + str(path.relative_to(ROOT)))
        captured = []
        def capture_original_audit(path, value):
            require(Path(path) == original.CONTROL / 'GENERATION_AUDIT.json', 'Unexpected old-directory write attempted: ' + str(path))
            captured.append(value)
        with patch.object(original, 'save', capture_original_audit):
            config, attempts, ledger = original.audit()
        require(len(captured) == 1 and len(attempts) == 660, 'Generation audit incomplete')
        require(captured[0] == load(original.CONTROL / 'GENERATION_AUDIT.json'), 'Saved generation accounting changed')
        report['generation'] = captured[0]
        log('Verified frozen protocol chain and all 660 generation slots; no unsettled request')
        for control in [original.CONTROL, supplement.OUT]:
            status = load(control / 'PIPELINE_STATUS.json')
            require(status['state'] == 'complete' and status['exit_code'] == 0, 'Offline pipeline not complete: ' + str(control))
        report['strict_final_file_hashes_verified'] = check_files(original.OUTPUT, load(original.CONTROL / 'FINAL_ARTIFACTS.json'))
        old_review = load(original.CONTROL / 'FINAL_REVIEW.json')
        require(freeze.file_hash(original.OUTPUT / 'FINAL_REPORT.txt') == old_review['human_report_sha256'], 'Strict human report changed')
        strict, old_definition, before = audit_evaluation(original.OUTPUT, config, attempts)
        report['strict'] = strict

        capsule = load(supplement.OUT / 'SUPPLEMENT_PROTOCOL.json')
        original.verify_hashes(capsule['preserved_original_files'])
        original.verify_hashes(capsule['code_files'])
        require(capsule['post_hoc'] and capsule['original_results_remain_primary'] and capsule['new_model_calls_authorized'] == 0, 'Supplement scope changed')
        report['supplement_preserved_original_files_verified'] = len(capsule['preserved_original_files'])
        report['supplement_final_file_hashes_verified'] = check_files(supplement.EVAL, load(supplement.OUT / 'FINAL_ARTIFACTS.json'))
        derived = load(supplement.OUT / 'DERIVED_GENERATION.json')
        require(len(derived['attempts']) == len(derived['provenance']) == 660, 'Supplement generation incomplete')
        require(derived['source_attempts_sha256'] == digest(attempts), 'Supplement source attempts changed')
        require(derived['protocol_sha256'] == freeze.file_hash(supplement.OUT / 'SUPPLEMENT_PROTOCOL.json'), 'Supplement protocol changed')
        models = {m['id']: m for m in config['models']}
        counts = {m: Counter() for m in models}
        for i, (old, new, prov) in enumerate(zip(attempts, derived['attempts'], derived['provenance'])):
            path = ledger / f'{i:04d}.response.json'
            response = load(path); content = response['choices'][0]['message']['content']
            body, applied = supplement.unwrap(content)
            modified = deepcopy(response); modified['choices'][0]['message']['content'] = body
            decision = generation.classify(modified, models[old['slot']['model']], inputs.catalog(inputs.adapter(old['slot']['target'])), old['reservation_usd'])
            expected = deepcopy(old); expected.update(decision)
            require(new == expected, 'Supplement response classification changed: ' + str(i))
            fields = {'index': i, 'slot_id': old['slot']['slot_id'], 'model': old['slot']['model'],
                      'source_response_sha256': old['response_sha256'], 'source_response_file_sha256': freeze.file_hash(path),
                      'source_content_sha256': supplement.text_sha(content), 'derived_content_sha256': supplement.text_sha(body),
                      'outer_fence_removed': applied, 'original_status': old['status'],
                      'supplemental_status': decision['status'], 'supplemental_reasons': decision['reasons']}
            require(all(prov[k] == v for k, v in fields.items()), 'Supplement provenance changed: ' + str(i))
            if applied:
                require((supplement.OUT / prov['derived_content_path']).read_bytes().decode('utf-8') == body, 'Derived body changed')
            if old['status'] == generation.VALID:
                require(not applied and new == old, 'Previously valid generation changed')
            c = counts[old['slot']['model']]
            c['slots'] += 1; c[decision['status']] += 1
            c['outer_fences_removed'] += int(applied)
            c['newly_valid'] += int(old['status'] != generation.VALID and decision['status'] == generation.VALID)
        require({m: dict(c) for m, c in counts.items()} == derived['counts'], 'Supplement count summary changed')
        after, definition, analysis = audit_evaluation(supplement.EVAL, config, derived['attempts'], original.OUTPUT)
        after['derived_counts'] = derived['counts']
        unchanged = {m for m, c in derived['counts'].items() if c['newly_valid'] == 0}
        require(analysis['draws_sha256'] == before['draws_sha256'], 'Bootstrap draw identity changed')
        for section in ['primary', 'secondary', 'random_controls']:
            select = lambda rows: [r for r in rows if all(m in unchanged for m in r['model'].split(' minus '))]
            require(select(analysis[section]) == select(before[section]), 'Unchanged-model results changed: ' + section)
        for target, choice in load(original.OUTPUT / 'CALIBRATION_LOCK.json')['selections'].items():
            new_choice = load(supplement.EVAL / 'CALIBRATION_LOCK.json')['selections'][target]
            require(choice['anchor'] == new_choice['anchor'], 'Anchor changed')
            for name, value in choice['choices'].items():
                if old_definition['targets'][target]['outcomes'][name] == definition['targets'][target]['outcomes'][name]:
                    require(value == new_choice['choices'][name], 'Unchanged library selection changed: ' + name)
        accounting = load(supplement.OUT / 'SUPPLEMENT_ACCOUNTING.json')
        require(accounting['new_model_requests'] == 0 and accounting['new_model_cost_usd'] == '0', 'Unexpected new model accounting')
        require(accounting['historical_accounting'] == load(supplement.OUT / 'SOURCE_AUDIT.json') == captured[0], 'Supplement source accounting changed')
        require(accounting['report_sha256'] == freeze.file_hash(supplement.EVAL / 'SUPPLEMENT_REPORT.txt'), 'Supplement report changed')
        require(accounting['analysis_sha256'] == freeze.file_hash(supplement.EVAL / 'analysis.json'), 'Supplement analysis changed')
        after['unchanged_models_verified'] = sorted(unchanged)
        report['supplement'] = after
        report['offline_phases_remaining'] = []
        report['interpretation_limits'] = [
            'Strict Haiku results are 220 invalid-format fallbacks, not valid three-model content replication.',
            'Outer-fence supplement is post hoc and reuses previously seen heldout seeds.',
            'Bootstrap intervals are descriptive sensitivity results, not calibrated confirmation.',
            'Missing experts, unseen-data and RSI studies are separate tasks, not unfinished phases of these pipelines.'
        ]
        original.verify_hashes(capsule['preserved_original_files'])
        original.verify_hashes(capsule['code_files'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New audit output directory; old frozen outputs are never written.')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(BASE):
        parser.error('--output must be outside experiments/care_replay; use a fresh directory under outputs/.')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': False}
    try:
        result.update(run(output)); result['passed'] = True
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / 'current_completion_audit.json').write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['passed', 'elapsed_seconds']}, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
