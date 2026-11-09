"""Verify the v9 stop without writing to the source journal."""
from protocols.nonrsi_cross_model_v1.common import BASE, Halt, digest, load
from protocols.nonrsi_cross_model_v1.freeze import file_hash
from .policy import REQUEST_DIGEST, SLOT_ID, SOURCE_ATTEMPTS, matches

OLD = BASE / 'outputs/nonrsi_cross_model_resume_v9'
OUT = BASE / 'outputs/nonrsi_cross_model_resume_v10'
SOURCE_JOURNAL = OLD / 'journal'
PINNED = {
    'FROZEN_RECOVERY.json': '7c9994a7f0f9afed2701ccdebb08a52738059afeb173b5ca5241c24d94c4ad02',
    'RECOVERY_AUTHORIZATION.json': '8e9b89a67b51f655b5569063c2cf40b7b2a32832826d1d4933181358a7808157',
    'PIPELINE_STATUS.json': '21f069de58b0702edde9bb110f672bbf5903a214eb75ef38964640eabae13486',
    'journal/ATTEMPTS.json': '4917c53a026758a65d94761bba78888ff6985cb31b323973bda86ae428095776',
    'journal/0595.request.json': 'b65b865ebc68cc4fcc3b455b44323a3d1f1b0b3cef078c6fa7148664c95885c4',
    'journal/0595.response.json': '090f82314e3f2006f8b9efbc82e17663f1f52db0f0e6725adb92dba50d64d730',
    'journal/0595.http.json': '7388721bbb25abb53e8c39c4dc472e6bbd63ef35ef26cab3eeff5b9514425134',
}


def check_stopped_attempt(a):
    return (a['logical_index'] == 620 and a['total_attempt_index'] == 645
            and a['slot']['slot_id'] == SLOT_ID and a['status'] == 'HALTED'
            and a.get('halt_reason') == 'Invalid message' and a['test_only'] is False
            and a['is_replacement'] is False and a.get('cost_usd') is None
            and a['reservation_usd'] == '1' and a['request_sha256'] == REQUEST_DIGEST
            and a['response_sha256'] == '94e3b57495e3873a6813f6ddd7adfa7923e623db2caf162e4ecfef9b1c039f7b')


def journal_hashes():
    paths = sorted(p for p in SOURCE_JOURNAL.rglob('*') if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(SOURCE_JOURNAL) for p in paths):
        raise Halt('Unexpected path in stopped journal')
    return {p.relative_to(SOURCE_JOURNAL).as_posix(): file_hash(p) for p in paths}


def verify_stopped_history(expected_hashes=None):
    for name, expected in PINNED.items():
        if file_hash(OLD / name) != expected:
            raise Halt('Historical v9 recovery evidence changed: ' + name)
    attempts = load(SOURCE_JOURNAL / 'ATTEMPTS.json')['attempts']
    science = load(SOURCE_JOURNAL / 'science_ledger/ledger.json')['attempts']
    raw = load(SOURCE_JOURNAL / '0595.response.json')
    if (len(attempts) != SOURCE_ATTEMPTS or len(science) != 620
            or sum(a['is_replacement'] for a in attempts) != 23
            or any(a['status'] == 'RESERVED' or a['test_only'] for a in attempts)
            or not check_stopped_attempt(attempts[-1])
            or load(SOURCE_JOURNAL / '0595.response.txt') != raw
            or digest(raw) != attempts[-1]['response_sha256'] or not matches(raw)
            or load(SOURCE_JOURNAL / '0595.http.json')['status'] != 200
            or (OLD / 'evaluation').exists()):
        raise Halt('Expected exact v9 policy-block stop with 620 completed slots and no evaluation')
    if expected_hashes is not None and journal_hashes() != expected_hashes:
        raise Halt('Original v9 journal inventory or bytes changed')
    return attempts[-1]
