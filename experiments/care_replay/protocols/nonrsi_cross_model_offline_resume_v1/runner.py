"""Offline-only continuation; existing generation and checkpoints stay immutable."""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import fcntl
import os
import subprocess
import sys
import traceback

from protocols.nonrsi_cross_model_v1 import freeze, generation
from protocols.nonrsi_cross_model_v1.common import BASE, ROOT, Halt, digest, load, offline, save
from protocols.nonrsi_cross_model_resume_v10 import evaluate, history, runner as recovery

SOURCE = BASE / 'outputs/nonrsi_cross_model_resume_v10'
OUTPUT = SOURCE / 'evaluation'
CONTROL = BASE / 'outputs/nonrsi_cross_model_offline_resume_v1_20261108'
PHASES = ('calibrate', 'heldout', 'analyze')
WORKERS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise Halt('Unexpected symlink in preserved evidence')
            result[str(path.relative_to(ROOT))] = freeze.file_hash(path)
    return result


def verify_hashes(files):
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or freeze.file_hash(path) != expected:
            raise Halt('Preserved evidence changed: ' + name)


def verify_static_chain(path, visited=None):
    visited = set() if visited is None else visited
    path = path.resolve()
    if path in visited:
        return
    visited.add(path)
    lock = load(path)
    verify_hashes(lock['files'])
    for name in lock['files']:
        child = ROOT / name
        if child.name.startswith('FROZEN_') and child.suffix == '.json':
            nested = load(child)
            if isinstance(nested.get('files'), dict):
                verify_static_chain(child, visited)


class ReadOnlyJournal(recovery.Journal):
    def sync(self):
        """Validate the existing projection instead of rewriting its metadata."""
        expected = list(self.prefix)
        for attempt in self.state()['attempts']:
            if attempt['status'] not in recovery.TERMINAL:
                continue
            if attempt['logical_index'] != len(expected):
                raise Halt('Scientific projection gap or duplicate')
            expected.append({k: attempt[k] for k in (
                'slot', 'request_sha256', 'reservation_usd', 'cost_usd', 'status',
                'test_only', 'response_sha256', 'response_id', 'library', 'reasons', 'fingerprint')})
        science = load(self.science / 'ledger.json')
        if science != {'config_sha256': digest(self.config), 'attempts': expected}:
            raise Halt('Scientific projection changed')


def audit():
    capsule = load(CONTROL / 'OFFLINE_RESUME.json')
    verify_hashes(capsule['preserved_files'])
    verify_hashes(capsule['launcher_files'])
    lock = load(SOURCE / 'FROZEN_RECOVERY.json')
    verify_static_chain(SOURCE / 'FROZEN_RECOVERY.json')
    config = freeze.verify(ROOT / lock['base_lock'])
    history.verify_stopped_history(lock['source_journal_hashes'])
    journal = ReadOnlyJournal(SOURCE / 'journal', ROOT / lock['original_ledger'], config)
    reports = []

    def capture_accounting(path, value):
        if Path(path) != SOURCE / 'journal/TOTAL_ACCOUNTING.json':
            raise Halt('Offline audit attempted an unexpected write')
        reports.append(value)

    with patch.object(recovery, 'save', capture_accounting):
        attempts = journal.audit(require_complete=True)
    if len(attempts) != 660 or len(reports) != 1:
        raise Halt('Expected complete generation with one accounting report')
    if load(OUTPUT / 'definition.json')['generation_sha256'] != digest(attempts):
        raise Halt('Prepared definition no longer binds to generation')
    save(CONTROL / 'GENERATION_AUDIT.json', {
        **reports[0], 'scientific_slots': len(attempts), 'unsettled_requests': 0,
        'network_disabled': True, 'new_model_requests': 0,
        'slot621_status': attempts[620]['status'], 'slot621_reasons': attempts[620]['reasons'],
    })
    return config, attempts, journal.science


def worker(phase):
    if phase not in PHASES:
        raise Halt('Only offline evaluation phases are allowed')
    with offline():
        config, attempts, ledger = audit()
        print(now(), phase, 'audited 660 saved generation slots; network disabled', flush=True)
        evaluate.run(phase, config, attempts, ledger, OUTPUT, WORKERS)
        verify_hashes(load(CONTROL / 'OFFLINE_RESUME.json')['preserved_files'])
        print(now(), phase, 'complete; original saved files unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=PHASES)
    args = parser.parse_args()
    if args.phase:
        worker(args.phase)
        return 0
    with (SOURCE / 'pipeline.process.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = CONTROL / 'PIPELINE_STATUS.json'
        if status.exists():
            raise Halt('Existing continuation; inspect before any restart')
        phase = 'verify'
        try:
            for phase in PHASES:
                save(status, {'state': 'running', 'phase': phase, 'started_at_utc': now(),
                              'evaluation_output': str(OUTPUT.relative_to(ROOT)),
                              'new_model_requests': 0, 'workers': WORKERS})
                command = [sys.executable, '-B', '-u', '-m', __package__ + '.runner', '--phase', phase]
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                # No credential values are read and no .env file is loaded.
                for name in tuple(env):
                    if name.endswith(('_API_KEY', '_AUTH_TOKEN')):
                        env.pop(name)
                with (CONTROL / (phase + '.log')).open('x') as log:
                    code = subprocess.run(command, cwd=BASE, env=env, stdout=log,
                                          stderr=subprocess.STDOUT).returncode
                if code:
                    save(status, {'state': 'stopped', 'phase': phase, 'exit_code': code,
                                  'stopped_at_utc': now(), 'new_model_requests': 0})
                    return code
            required = ('CALIBRATION_LOCK.json', 'HELDOUT_COMPLETE.json', 'analysis.json',
                        'generation_costs.json', 'SLOT_621_DISPOSITION.json')
            if any(not (OUTPUT / name).is_file() for name in required):
                raise Halt('Final evaluation artifacts missing')
            capsule = load(CONTROL / 'OFFLINE_RESUME.json')
            verify_hashes(capsule['preserved_files'])
            save(CONTROL / 'FINAL_ARTIFACTS.json', {name: freeze.file_hash(OUTPUT / name) for name in required})
            save(status, {'state': 'complete', 'phase': 'analyze', 'exit_code': 0,
                          'finished_at_utc': now(), 'new_model_requests': 0})
            return 0
        except BaseException as exc:
            save(status, {'state': 'stopped', 'phase': phase, 'error_type': type(exc).__name__,
                          'error': str(exc), 'stopped_at_utc': now(), 'new_model_requests': 0})
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
