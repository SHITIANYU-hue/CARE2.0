"""Independent supplemental outputs; original responses/results are read-only."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import traceback

from protocols.nonrsi_cross_model_v1.common import BASE, ROOT, Halt, digest, load, offline, save, matched
from protocols.nonrsi_cross_model_v1 import evaluation, freeze, generation, inputs, pipeline
from protocols.nonrsi_cross_model_resume_v10 import evaluate
from protocols.nonrsi_cross_model_offline_resume_v1 import runner as original
from .normalization import unwrap

OUT = BASE / 'outputs/nonrsi_fence_supplement_v1_20261108'
EVAL = OUT / 'evaluation'
PHASES = ('derive', 'prepare', 'calibrate', 'heldout', 'analyze')
WORKERS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def text_sha(content):
    return None if content is None else hashlib.sha256(content.encode('utf-8')).hexdigest()


def source_audit():
    capsule = load(OUT / 'SUPPLEMENT_PROTOCOL.json')
    original.verify_hashes(capsule['preserved_original_files'])
    original.verify_hashes(capsule['code_files'])
    captured = []

    def capture(path, value):
        if Path(path) != original.CONTROL / 'GENERATION_AUDIT.json':
            raise Halt('Original audit attempted an unexpected write')
        captured.append(value)

    with patch.object(original, 'save', capture):
        config, attempts, ledger = original.audit()
    if len(captured) != 1 or captured[0]['unsettled_requests'] != 0:
        raise Halt('Original generation audit incomplete')
    pipeline.once(OUT / 'SOURCE_AUDIT.json', captured[0])
    return config, attempts, ledger


def freeze_supplement():
    if load(original.CONTROL / 'PIPELINE_STATUS.json')['state'] != 'complete':
        raise Halt('Original pipeline must be complete')
    OUT.mkdir(exist_ok=False)
    preserved = original.inventory(original.SOURCE)
    preserved.update(original.inventory(original.CONTROL))
    code = original.inventory(BASE / 'protocols/nonrsi_fence_supplement_v1')
    save(OUT / 'SUPPLEMENT_PROTOCOL.json', {
        'kind': 'post_hoc_outer_fence_sensitivity_v1', 'created_at_utc': now(),
        'user_instruction': 'Keep original results; separately re-evaluate saved responses after removing outer code fences. No new model requests. Do not run Experiments 2/3/4.',
        'post_hoc': True, 'new_model_calls_authorized': 0, 'network_disabled': True,
        'parser_policy': 'Uniform for all 660 saved responses: remove only one complete outer ```json or unlabelled ``` fence with no surrounding prose or nested fence. Preserve all enclosed bytes. No JSON repair, field repair, trimming/filling of libraries, retries, or regeneration. All original envelope, schema, field/value, range, and five-skill checks remain.',
        'evaluation_policy': 'Same frozen configuration, seeds, pools, 50 calibration/100 heldout per target, selector/gate, baselines, random-control construction, and descriptive bootstrap. Reuse all available original modes and traces; simulate missing modes only, plus verify identical baseline replay. Lock all calibration choices before heldout. Previously valid libraries and unchanged choices must be identical.',
        'original_results_remain_primary': True, 'source_output': str(original.SOURCE.relative_to(ROOT)),
        'workers': WORKERS, 'phases': list(PHASES),
        'preserved_original_files': preserved, 'code_files': code,
    })
    print('Frozen independent offline supplement; original files:', len(preserved), flush=True)


def derive(config, attempts, ledger):
    models = {m['id']: m for m in config['models']}
    derived = []; provenance = []; counts = {m: Counter() for m in models}
    for i, source in enumerate(attempts):
        response = load(ledger / f'{i:04d}.response.json')
        content = response['choices'][0]['message']['content']
        body, applied = unwrap(content)
        transformed = deepcopy(response)
        transformed['choices'][0]['message']['content'] = body
        decision = generation.classify(transformed, models[source['slot']['model']],
                                       inputs.catalog(inputs.adapter(source['slot']['target'])), source['reservation_usd'])
        if source['status'] == generation.VALID:
            if applied or any(source[k] != v for k, v in decision.items()):
                raise Halt('An originally valid response changed')
        row = deepcopy(source); row.update(decision)
        # response_sha256 still refers to the untouched ORIGINAL envelope, not a fabricated response.
        derived.append(row)
        item = {'index': i, 'slot_id': source['slot']['slot_id'], 'model': source['slot']['model'],
                'source_response_sha256': source['response_sha256'],
                'source_response_file_sha256': freeze.file_hash(ledger / f'{i:04d}.response.json'),
                'source_content_sha256': text_sha(content), 'derived_content_sha256': text_sha(body),
                'outer_fence_removed': applied, 'original_status': source['status'],
                'supplemental_status': decision['status'], 'supplemental_reasons': decision['reasons']}
        if applied:
            path = OUT / 'normalized_contents' / f'{i:04d}.txt'; path.parent.mkdir(exist_ok=True)
            if path.exists() and path.read_text() != body:
                raise Halt('Derived content changed')
            if not path.exists():
                path.write_text(body, encoding='utf-8')
            item['derived_content_path'] = str(path.relative_to(OUT))
        provenance.append(item)
        c = counts[source['slot']['model']]; c['slots'] += 1; c[decision['status']] += 1
        c['outer_fences_removed'] += int(applied)
        c['newly_valid'] += int(source['status'] != generation.VALID and decision['status'] == generation.VALID)
    pipeline.once(OUT / 'DERIVED_GENERATION.json', {'kind': 'derived_decisions_not_new_responses',
        'protocol_sha256': freeze.file_hash(OUT / 'SUPPLEMENT_PROTOCOL.json'),
        'source_attempts_sha256': digest(attempts), 'attempts': derived, 'provenance': provenance,
        'counts': {m: dict(c) for m, c in counts.items()}, 'new_model_requests': 0})
    print('Derived validation counts:', {m: dict(c) for m, c in counts.items()}, flush=True)


def prepare(config, attempts):
    old = load(original.OUTPUT / 'definition.json'); definition = deepcopy(old)
    definition['generation_sha256'] = digest(attempts)
    definition['supplement'] = {'post_hoc': True, 'protocol_sha256': freeze.file_hash(OUT / 'SUPPLEMENT_PROTOCOL.json'),
                                'source_definition_sha256': digest(old)}
    for a in attempts:
        slot = a['slot']; target = slot['target']; name = slot['slot_id']
        item = definition['targets'][target]
        if old['targets'][target]['outcomes'][name] == generation.VALID:
            if a['status'] != generation.VALID:
                raise Halt('Previously valid library lost')
            continue
        item['outcomes'][name] = a['status']
        if a['status'] != generation.VALID:
            continue
        adapter = inputs.adapter(target); catalog = inputs.catalog(adapter)
        bank = evaluation.compile_library(a['library'], catalog, name)
        item['banks'][name] = [asdict(s) for s in bank]
        if slot['evidence_mode'] == 'target_only' and target in ('real_buchwald_hartwig', 'real_suzuki_miyaura'):
            types = inputs.suzuki.field_types(catalog) if target == inputs.suzuki.TARGET else config['random_controls_proposed']['field_types_bh']
            children, diagnostics = evaluation.random_children(bank, catalog, types,
                [{f: str(c.metadata.get(f, '')) for f in catalog} for c in adapter.candidates], name, config['random_control_seed'])
            item['coverage_diagnostics'][name] = diagnostics
            for key, child in children.items():
                item['banks'][key] = [asdict(s) for s in child]; item['outcomes'][key] = generation.VALID
                definition['random_children'][key] = {'parent': name, 'family': key.split('_')[0], 'target': target}
    for target, old_item in old['targets'].items():
        if any(definition['targets'][target]['banks'][k] != v for k, v in old_item['banks'].items()):
            raise Halt('An existing bank changed')
    pipeline.once(EVAL / 'definition.json', definition)


def wanted_skills(definition, target, split, output):
    banks = pipeline.banks_for(definition, target)
    if split == 'calibration':
        return tuple(s for bank in banks.values() for s in bank)
    choices = load(Path(output) / 'CALIBRATION_LOCK.json')['selections'][target]['choices']
    wanted = {v[k] for v in choices.values() for k in ('primary_mode', 'selected_mode')}
    return tuple(s for bank in banks.values() for s in bank if matched.semantic.skill_mode(s) in wanted)


def merged_worker(seed, *, target, split, skills, output, binding, source_binding):
    path = Path(output) / target / split / f'{seed}.json'
    modes = (*matched.BASELINES, *(matched.semantic.skill_mode(s) for s in skills))
    if path.exists():
        cached = load(path); traces = json.loads(gzip.decompress(path.with_suffix('.traces.json.gz').read_bytes()))
        if cached['binding'] != binding or digest(cached['rows']) != cached['rows_sha256'] or digest(traces) != cached['traces_sha256']:
            raise Halt('Supplement checkpoint changed')
        pipeline.verify_trace(inputs.adapter(target), seed, cached['rows'], traces, modes, split)
        return cached['rows']
    source_path = original.OUTPUT / target / split / f'{seed}.json'
    cached = load(source_path); source_traces = json.loads(gzip.decompress(source_path.with_suffix('.traces.json.gz').read_bytes()))
    if (cached['binding'] != source_binding or digest(cached['rows']) != cached['rows_sha256']
            or digest(source_traces) != cached['traces_sha256']):
        raise Halt('Original checkpoint changed')
    by_mode = {row['mode']: row for row in cached['rows']}; trace_by_mode = {}
    for event in source_traces:
        trace_by_mode.setdefault(event['mode'], []).append(event)
    missing = tuple(s for s in skills if matched.semantic.skill_mode(s) not in by_mode)
    reps, aliases = matched.execution_plan(missing, enabled=True)
    duration = 0.
    if missing:
        rows, traces, duration = evaluation.replay_seed(target, seed, split, reps)
        rows = matched.expand_metric_aliases(rows, aliases)
        expanded = [dict(e, mode=alias) for alias, rep in aliases.items() for e in traces if e['mode'] == rep]
        expanded.extend(e for e in traces if e['mode'] in matched.BASELINES)
        fresh = {row['mode']: row for row in rows}; fresh_traces = {}
        for event in expanded:
            fresh_traces.setdefault(event['mode'], []).append(event)
        for mode in matched.BASELINES:
            if fresh[mode] != by_mode[mode] or fresh_traces[mode] != trace_by_mode[mode]:
                raise Halt('Replayed baseline differs from original')
        for skill in missing:
            mode = matched.semantic.skill_mode(skill)
            by_mode[mode] = fresh[mode]; trace_by_mode[mode] = fresh_traces[mode]
    combined = [by_mode[m] for m in modes]
    combined_traces = [e for mode in modes for e in trace_by_mode[mode]]
    pipeline.verify_trace(inputs.adapter(target), seed, combined, combined_traces, modes, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.traces.json.gz').write_bytes(gzip.compress(json.dumps(combined_traces, sort_keys=True, allow_nan=False).encode(), mtime=0))
    save(path, {'binding': binding, 'rows': combined, 'rows_sha256': digest(combined),
        'traces_sha256': digest(combined_traces), 'elapsed_seconds': duration,
        'executed_unique_skills': len(reps), 'logical_skills': len(skills),
        'reused_modes': len(modes) - len(missing), 'new_modes': len(missing),
        'source_checkpoint_sha256': freeze.file_hash(source_path), 'source_binding': source_binding,
        'baseline_consistency_checked': bool(missing)})
    return combined


def merged_phase(config, definition, target, split, skills, output, workers):
    if inputs.certify(inputs.adapter(target))['complete_candidate_records_sha256'] != definition['targets'][target]['pool_sha256']:
        raise Halt('Pool changed')
    old = load(original.OUTPUT / 'definition.json')
    old_skills = wanted_skills(old, target, split, original.OUTPUT)
    source_binding = digest([digest(config), digest(old), target, split, [asdict(s) for s in old_skills]])
    binding = digest([digest(config), digest(definition), target, split, [asdict(s) for s in skills]])
    worker = partial(merged_worker, target=target, split=split, skills=skills, output=str(output),
                     binding=binding, source_binding=source_binding)
    seeds = pipeline.seed_set(config, target, split)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for batch in pool.map(worker, seeds) for r in batch]
    matched.validate_rows(rows, (*matched.BASELINES, *(matched.semantic.skill_mode(s) for s in skills)), set(seeds), split)
    return rows


def check_unchanged_selections(definition):
    old_def = load(original.OUTPUT / 'definition.json')
    old = load(original.OUTPUT / 'CALIBRATION_LOCK.json'); new = load(EVAL / 'CALIBRATION_LOCK.json')
    checked = 0
    for target, choice in old['selections'].items():
        if choice['anchor'] != new['selections'][target]['anchor']:
            raise Halt('Common baseline anchor changed')
        for name, value in choice['choices'].items():
            if old_def['targets'][target]['outcomes'][name] != definition['targets'][target]['outcomes'][name]:
                continue
            if value != new['selections'][target]['choices'][name]:
                raise Halt('Selection for an unchanged library changed: ' + name)
            checked += 1
    pipeline.once(OUT / 'SELECTION_REUSE_AUDIT.json', {'unchanged_library_choices_verified': checked,
                  'anchors_identical': True, 'calibration_lock_sha256': digest(new)})


def worker(phase):
    with offline():
        config, source, ledger = source_audit()
        print(now(), phase, 'original complete run audited; network disabled', flush=True)
        if phase == 'derive':
            derive(config, source, ledger)
        else:
            derived = load(OUT / 'DERIVED_GENERATION.json')
            if derived['source_attempts_sha256'] != digest(source) or derived['protocol_sha256'] != freeze.file_hash(OUT / 'SUPPLEMENT_PROTOCOL.json'):
                raise Halt('Derived generation binding changed')
            attempts = derived['attempts']
            if phase == 'prepare':
                prepare(config, attempts)
            elif phase in ('calibrate', 'heldout'):
                with patch.object(pipeline, 'phase', merged_phase):
                    evaluate.run(phase, config, attempts, ledger, EVAL, WORKERS)
                if phase == 'calibrate':
                    check_unchanged_selections(load(EVAL / 'definition.json'))
            elif phase == 'analyze':
                check_unchanged_selections(load(EVAL / 'definition.json'))
                evaluate.run(phase, config, attempts, ledger, EVAL, WORKERS)
                from .report import export
                export(config)
            else:
                raise Halt('Unexpected phase')
        original.verify_hashes(load(OUT / 'SUPPLEMENT_PROTOCOL.json')['preserved_original_files'])
        print(now(), phase, 'complete; original files unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    if args.freeze:
        freeze_supplement(); return 0
    if args.phase:
        worker(args.phase); return 0
    with (OUT / 'pipeline.process.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = OUT / 'PIPELINE_STATUS.json'
        if status.exists():
            raise Halt('Existing supplement; inspect before restarting')
        phase = 'verify'
        try:
            for phase in PHASES:
                save(status, {'state': 'running', 'phase': phase, 'started_at_utc': now(), 'post_hoc': True, 'new_model_requests': 0})
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                for name in tuple(env):
                    if name.endswith(('_API_KEY', '_AUTH_TOKEN')):
                        env.pop(name)
                with (OUT / (phase + '.log')).open('x') as log:
                    code = subprocess.run([sys.executable, '-B', '-u', '-m', __package__ + '.runner', '--phase', phase],
                        cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if code:
                    save(status, {'state': 'stopped', 'phase': phase, 'exit_code': code, 'stopped_at_utc': now(), 'new_model_requests': 0})
                    return code
            required = ('definition.json', 'CALIBRATION_LOCK.json', 'HELDOUT_COMPLETE.json', 'analysis.json', 'generation_costs.json', 'SUPPLEMENT_REPORT.txt')
            if any(not (EVAL / name).is_file() for name in required):
                raise Halt('Missing final supplement artifact')
            save(OUT / 'FINAL_ARTIFACTS.json', {name: freeze.file_hash(EVAL / name) for name in required})
            save(status, {'state': 'complete', 'phase': 'analyze', 'exit_code': 0, 'finished_at_utc': now(), 'post_hoc': True, 'new_model_requests': 0})
            return 0
        except BaseException as exc:
            save(status, {'state': 'stopped', 'phase': phase, 'error': str(exc), 'error_type': type(exc).__name__, 'stopped_at_utc': now(), 'new_model_requests': 0})
            traceback.print_exc(); return 1


if __name__ == '__main__':
    sys.exit(main())
