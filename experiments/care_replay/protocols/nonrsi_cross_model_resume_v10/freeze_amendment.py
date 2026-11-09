"""Run offline recovery checks before publishing the v10 transport lock."""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys
import unittest

from protocols.nonrsi_cross_model_v1.common import ROOT, Halt, load, offline, save
from protocols.nonrsi_cross_model_v1 import freeze
from .history import OLD, OUT, PINNED, SOURCE_JOURNAL, journal_hashes, verify_stopped_history
from .policy import REASON


def main():
    lock_path = OUT / 'FROZEN_RECOVERY.json'
    auth_path = OUT / 'RECOVERY_AUTHORIZATION.json'
    if lock_path.exists() or auth_path.exists():
        raise Halt('Recovery bundle already exists')
    log = StringIO()
    with offline():
        source_hashes = journal_hashes()
        verify_stopped_history(source_hashes)
        previous = load(OLD / 'FROZEN_RECOVERY.json')
        freeze.verify(ROOT / previous['base_lock'])
        suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_recovery')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        print(log.getvalue(), end='', flush=True)
        if result.testsRun != 3 or not result.wasSuccessful():
            return 1
        verify_stopped_history(source_hashes)
    at = datetime.now(timezone.utc).isoformat()
    validation = OUT / 'OFFLINE_VALIDATION.json'
    save(validation, {
        'checked_at_utc': at, 'tests_passed': result.testsRun, 'network_guard_enabled': True,
        'inference_calls': 0, 'historical_attempts': 646, 'completed_slots': 620,
        'remaining_slots': 40, 'test_log': log.getvalue(),
    })
    code = Path(__file__).resolve().parent
    files = {str(p.relative_to(ROOT)): freeze.file_hash(p) for p in sorted(code.glob('*.py'))}
    files.update({str((OLD / name).relative_to(ROOT)): sha for name, sha in PINNED.items()})
    files[str(validation.relative_to(ROOT))] = freeze.file_hash(validation)
    save(lock_path, {
        'kind': 'nonrsi_cross_model_slot621_retry_v10', 'created_at_utc': at,
        'base_lock': previous['base_lock'], 'original_ledger': previous['original_ledger'],
        'source_journal': str(SOURCE_JOURNAL.relative_to(ROOT)), 'source_journal_hashes': source_hashes,
        'journal_directory': str((OUT / 'journal').relative_to(ROOT)),
        'evaluation_output': str((OUT / 'evaluation').relative_to(ROOT)),
        'inference_calls_at_freeze': 0, 'files': files,
    })
    save(auth_path, {
        'kind': 'slot621_retry_authorization', 'allow_execution': True,
        'resume_freeze_sha256': freeze.file_hash(lock_path),
        'previous_authorization_sha256': PINNED['RECOVERY_AUTHORIZATION.json'],
        'exact_retry_once': True, 'fallback_reason': REASON,
        'model_calls': 700, 'model_spend_cap_usd': '45', 'historical_unknown_charge_hold_usd': '26',
        'replacement_cap': 38, 'replacements_already_used': 23, 'per_slot_replacements_cap': 2,
        'backoff_seconds': [600, 1800], 'qwen_minimum_interval_seconds': 600,
        'user_message_reference': '2026-11-06: 那重试一下，如果不行就回退再继续。沿用原生成顺序、冷却、预算；仅实验一。',
    })
    print('v10 recovery frozen; fresh authenticated preflight required', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
