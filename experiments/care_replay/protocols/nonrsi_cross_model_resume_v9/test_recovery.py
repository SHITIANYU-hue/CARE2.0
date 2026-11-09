from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from protocols.nonrsi_cross_model_v1.common import ROOT, Halt, load, offline
from protocols.nonrsi_cross_model_v1 import freeze
from . import history, pipeline, runner


class RecoveryTests(unittest.TestCase):
    def test_exact_historical_stop_only(self):
        prior = load(history.SOURCE_JOURNAL / 'ATTEMPTS.json')['attempts']
        stop = prior[-1]
        self.assertTrue(runner.permitted_imported_stop(488, stop, prior))
        self.assertFalse(runner.permitted_imported_stop(489, stop, prior))
        self.assertFalse(runner.qualifying(stop['slot'], stop['error_evidence']))
        for field, value in [('total_attempt_index', 539), ('status', 'RESERVED'),
                             ('cost_usd', '0'), ('is_replacement', True), ('test_only', True)]:
            changed = deepcopy(stop)
            changed[field] = value
            self.assertFalse(history.check_stopped_attempt(changed), field)
        changed = deepcopy(stop)
        changed['error_evidence']['provider_metadata']['limit_source'] = 'other_limit'
        self.assertFalse(history.check_stopped_attempt(changed))

    def test_recover_without_mutating_stopped_journal(self):
        previous = load(history.OLD / 'FROZEN_RECOVERY.json')
        with offline(), TemporaryDirectory() as temp:
            original = history.verify_stopped_history()
            config = freeze.verify(ROOT / previous['base_lock'])
            journal = runner.Journal(Path(temp) / 'journal', ROOT / previous['original_ledger'], config)
            self.assertEqual(journal.audit(), {
                'logical_slots': 514, 'total_http_attempts': 539, 'new_replacement_attempts': 22,
                'known_model_cost_usd': '2.17578574', 'unknown_charge_holds_usd': '25',
                'conservative_accounted_usd': '27.17578574', 'cap_usd': '45',
            })
            plan = journal.plan()
            self.assertEqual(plan['logical_index'], 514)
            self.assertEqual(plan['total_attempt_index'], 539)
            self.assertEqual(plan['slot']['slot_id'], original['failed_slot']['slot_id'])
            self.assertTrue(plan['is_replacement'])
            self.assertEqual(plan['slot_replacement_ordinal'], 1)
            self.assertEqual(plan['required_backoff_seconds'], 600)
            with self.assertRaisesRegex(Halt, 'not authorized'):
                journal.reserve(plan, {}, {'waited_seconds': 600})
            self.assertEqual(history.verify_stopped_history(), original)

    def test_pipeline_is_experiment_one_only(self):
        commands = pipeline.commands(Path('fresh.json'))
        self.assertEqual([name for name, _ in commands], ['generate', 'prepare', 'calibrate', 'heldout', 'analyze'])
        for name, command in commands:
            self.assertEqual(command[3], 'protocols.nonrsi_cross_model_resume_v9.runner')
            self.assertIn(str(history.OUT / 'journal'), command)
            if name != 'generate':
                self.assertIn(str(history.OUT / 'evaluation'), command)
