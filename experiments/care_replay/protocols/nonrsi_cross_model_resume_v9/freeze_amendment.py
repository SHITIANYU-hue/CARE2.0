"""Validate and pin the v9 recovery bundle without reading the API key."""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import sys
import unittest

from protocols.nonrsi_cross_model_v1.common import ROOT, Halt, load, offline, save
from protocols.nonrsi_cross_model_v1 import freeze
from .history import OLD, OUT, PINNED, SOURCE_JOURNAL, verify_stopped_history


def main():
    lock_path = OUT / 'FROZEN_RECOVERY.json'
    auth_path = OUT / 'RECOVERY_AUTHORIZATION.json'
    if lock_path.exists() or auth_path.exists():
        raise Halt('Recovery bundle already exists')
    log = StringIO()
    with offline():
        old_audit = verify_stopped_history()
        previous = load(OLD / 'FROZEN_RECOVERY.json')
        freeze.verify(ROOT / previous['base_lock'])
        suite = unittest.defaultTestLoader.loadTestsFromName(__package__ + '.test_recovery')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        print(log.getvalue(), end='', flush=True)
        if result.testsRun != 3 or not result.wasSuccessful():
            return 1
    at = datetime.now(timezone.utc).isoformat()
    save(OUT / 'OFFLINE_VALIDATION.json', {
        'checked_at_utc': at, 'tests_passed': result.testsRun, 'network_guard_enabled': True,
        'inference_calls': 0, 'historical_attempts': 539, 'completed_slots': 514,
        'remaining_slots': old_audit['remaining_logical_slots'], 'test_log': log.getvalue(),
    })
    code = Path(__file__).resolve().parent
    files = {str(p.relative_to(ROOT)): freeze.file_hash(p) for p in sorted(code.glob('*.py'))}
    files.update({str((OLD / name).relative_to(ROOT)): sha for name, sha in PINNED.items()})
    validation = OUT / 'OFFLINE_VALIDATION.json'
    files[str(validation.relative_to(ROOT))] = freeze.file_hash(validation)
    save(lock_path, {
        'kind': 'nonrsi_cross_model_openrouter_key_recovery_v9', 'created_at_utc': at,
        'base_lock': previous['base_lock'], 'original_ledger': previous['original_ledger'],
        'source_journal': str(SOURCE_JOURNAL.relative_to(ROOT)),
        'journal_directory': str((OUT / 'journal').relative_to(ROOT)),
        'evaluation_output': str((OUT / 'evaluation').relative_to(ROOT)),
        'resource_recovery_only': True, 'inference_calls_at_freeze': 0,
        'files': files,
    })
    save(auth_path, {
        'kind': 'openrouter_key_recovery_authorization', 'allow_execution': True,
        'resume_freeze_sha256': freeze.file_hash(lock_path),
        'previous_authorization_sha256': PINNED['RECOVERY_AUTHORIZATION.json'],
        'model_calls': 700, 'model_spend_cap_usd': '45', 'historical_unknown_charge_hold_usd': '25',
        'replacement_cap': 38, 'replacements_already_used': 22, 'per_slot_replacements_cap': 2,
        'backoff_seconds': [600, 1800], 'qwen_minimum_interval_seconds': 600,
        'user_message_reference': '2026-11-06: 接着跑实验一，其他三个先不跑；换 OpenRouter，key 填在 env；换新的了。沿用 v8 已授权的限额和原科学方案。',
    })
    print('v9 recovery frozen; fresh authenticated preflight required for execution', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
