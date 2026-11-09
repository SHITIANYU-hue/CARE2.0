from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protocols.nonrsi_cross_model_v1 import freeze, generation
from protocols.nonrsi_cross_model_v1.common import ROOT, Halt, digest, load, offline, save
from . import history, policy, runner


class RecoveryTests(unittest.TestCase):
    def test_historical_block_is_bound_and_other_errors_are_rejected(self):
        a = history.verify_stopped_history()
        self.assertTrue(history.check_stopped_attempt(a))
        raw = load(history.SOURCE_JOURNAL / '0595.response.json')
        self.assertTrue(policy.matches(raw))
        config = freeze.verify(ROOT / load(history.OLD / 'FROZEN_RECOVERY.json')['base_lock'])
        model = next(m for m in config['models'] if m['id'] == 'gpt6luna_public_id')
        with self.assertRaisesRegex(Halt, 'Invalid message'):
            generation.classify(raw, model, {}, '1')
        self.assertEqual(policy.fallback(raw, model, {}, '1')['reasons'], [policy.REASON])
        for code in (401, 429, 500, 502, '400'):
            changed = deepcopy(raw)
            changed['choices'][0]['error']['code'] = code
            self.assertFalse(policy.matches(changed), code)
        for bad_usage in ({}, {'is_byok': True, 'cost': 0}, {**raw['usage'], 'cost': None}, {**raw['usage'], 'cost': 2}):
            with self.assertRaises(Halt):
                policy.fallback({**raw, 'usage': bad_usage}, model, {}, '1')
        changed = deepcopy(raw)
        changed['choices'][0]['error']['message'] = 'Some other invalid request'
        self.assertFalse(policy.matches(changed))

    def test_exact_retry_falls_back_and_advances_without_mutating_history(self):
        prior = load(history.OLD / 'FROZEN_RECOVERY.json')
        with offline(), TemporaryDirectory() as temp:
            source_hashes = history.journal_hashes()
            cfg = freeze.verify(ROOT / prior['base_lock'])
            journal = runner.Journal(Path(temp) / 'journal', ROOT / prior['original_ledger'], cfg)
            audit = journal.audit()
            self.assertEqual((audit['logical_slots'], audit['total_http_attempts'], audit['unknown_charge_holds_usd']), (620, 646, '26'))
            plan = journal.plan()
            self.assertTrue(policy.is_retry(policy.SOURCE_ATTEMPTS, plan))
            self.assertEqual(plan['required_backoff_seconds'], 600)
            body = load(history.SOURCE_JOURNAL / '0595.request.json')
            self.assertEqual(digest(body), policy.REQUEST_DIGEST)
            idx = journal.reserve(plan, body, {'waited_seconds': 600}, test_only=True)
            raw = load(history.SOURCE_JOURNAL / '0595.response.json')
            with self.assertRaisesRegex(Halt, 'Duplicate response ID'):
                journal.response_identity(idx, raw)
            raw['id'] = 'synthetic-exp1-621-second-policy-block'
            save(journal.directory / f'{idx:04d}.http.json', {'status': 200})
            save(journal.directory / f'{idx:04d}.response.txt', raw)
            self.assertEqual(journal.settle(idx, raw, {}), None)
            copied = load(journal.science / '0620.response.json')
            self.assertEqual(copied, raw)
            a = load(journal.science / 'ledger.json')['attempts'][620]
            self.assertEqual((a['status'], a['reasons'], a['cost_usd']), (generation.INVALID, [policy.REASON], '0'))
            self.assertEqual(journal.audit(allow_test=True)['logical_slots'], 621)
            self.assertEqual(journal.plan()['logical_index'], 621)
            self.assertFalse(journal.plan()['is_replacement'])
            with self.assertRaisesRegex(Halt, 'Invalid message'):
                journal.classify_attempt(idx + 1, a, raw, {})
            self.assertEqual(history.journal_hashes(), source_hashes)

    def test_transport_anomaly_cannot_trigger_second_retry_of_slot_621(self):
        prior = load(history.OLD / 'FROZEN_RECOVERY.json')
        with offline(), TemporaryDirectory() as temp:
            cfg = freeze.verify(ROOT / prior['base_lock'])
            journal = runner.Journal(Path(temp) / 'journal', ROOT / prior['original_ledger'], cfg)
            plan = journal.plan()
            idx = journal.reserve(plan, load(history.SOURCE_JOURNAL / '0595.request.json'),
                                  {'waited_seconds': 600}, test_only=True)
            evidence = {'exception_type': 'URLError', 'transport_phase': 'open_or_read',
                        'http_status': None, 'response_received': False}
            self.assertTrue(runner.qualifying(plan['slot'], evidence))
            self.assertFalse(journal.fail(idx, evidence))
            self.assertEqual(journal.state()['attempts'][-1]['status'], 'HALTED')
            with self.assertRaisesRegex(Halt, 'non-retryable'):
                journal.plan()
