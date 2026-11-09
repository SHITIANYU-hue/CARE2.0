"""Preflight the replacement key and execute only Experiment 1."""
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import os
import subprocess
import sys

from protocols.nonrsi_cross_model_v1.common import BASE, ROOT, Halt, load, save
from protocols.nonrsi_cross_model_resume_v3 import preflight
from .history import OUT
from .runner import verify

PHASES = ('generate', 'prepare', 'calibrate', 'heldout', 'analyze')


def commands(attestation):
    common = ['--lock', str(OUT / 'FROZEN_RECOVERY.json'),
              '--authorization', str(OUT / 'RECOVERY_AUTHORIZATION.json'),
              '--attestation', str(attestation), '--journal', str(OUT / 'journal')]
    return [(phase, [sys.executable, '-B', '-m', __package__ + '.runner', phase, *common]
             + ([] if phase == 'generate' else ['--output', str(OUT / 'evaluation'), '--workers', '4']))
            for phase in PHASES]


def main():
    lock = OUT / 'FROZEN_RECOVERY.json'
    auth = OUT / 'RECOVERY_AUTHORIZATION.json'
    status = OUT / 'PIPELINE_STATUS.json'
    with (OUT / 'pipeline.process.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists() or (OUT / 'journal').exists():
            raise Halt('Existing v9 run; inspect its journal before restarting')
        key = preflight.read_key()
        os.environ['OPENROUTER_API_KEY'] = key
        preflight_dir = OUT / 'preflight' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        try:
            preflight.check(ROOT / load(lock)['base_lock'], preflight_dir)
        finally:
            os.environ.pop('OPENROUTER_API_KEY', None)
        attestation = preflight_dir / 'ATTESTATION.json'
        verify(lock, auth, attestation)
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        for phase, command in commands(attestation):
            save(status, {'state': 'running', 'phase': phase, 'started_at_utc': datetime.now(timezone.utc).isoformat()})
            with (OUT / (phase + '.log')).open('x') as log:
                child_env = {**env, 'OPENROUTER_API_KEY': key} if phase == 'generate' else env
                code = subprocess.run(command, cwd=BASE, env=child_env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                save(status, {'state': 'stopped', 'phase': phase, 'exit_code': code})
                return code
        save(status, {'state': 'complete', 'phase': 'analyze', 'exit_code': 0})
        return 0


if __name__ == '__main__':
    sys.exit(main())
