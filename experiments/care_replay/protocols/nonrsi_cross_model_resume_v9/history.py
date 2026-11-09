"""Pin the one eligible historical OpenRouter key-limit stop."""
from protocols.nonrsi_cross_model_v1.common import BASE, Halt, load
from protocols.nonrsi_cross_model_v1.freeze import file_hash

OLD = BASE / 'outputs/nonrsi_cross_model_resume_v8'
OUT = BASE / 'outputs/nonrsi_cross_model_resume_v9'
SOURCE_JOURNAL = OLD / 'journal'
SOURCE_ATTEMPTS = 489
PINNED = {
    'FROZEN_RECOVERY.json': 'e2ea6f2a37b947902f15a28bc66fc231b99998abd38eb6b337ab73d4555ceeb1',
    'RECOVERY_AUTHORIZATION.json': '7b7196d7a86c67161a94e400a51f0cd00a1a6fc04ddad8987dfc75ded41477c6',
    'STOP_AUDIT.json': '90e06faac05100b88f6187b5a0fc394a7d535a9c5d18edcfa6ea08e9e01f4d47',
    'journal/ATTEMPTS.json': 'ed1d26470b8e81477076d0005da0ab784d83a169ed8d842e9fc3da5bcc0c0235',
    'journal/0488.interruption.json': 'd4767307f59bcccf930bf814ffae99a8adb7b93e9312f99ce5bed8f7371f0a6d',
}


def check_stopped_attempt(attempt):
    evidence = attempt.get('error_evidence', {})
    return (
        attempt.get('logical_index') == 514 and attempt.get('total_attempt_index') == 538
        and attempt.get('status') == 'HALTED' and attempt.get('test_only') is False
        and attempt.get('is_replacement') is False and attempt.get('cost_usd') is None
        and attempt.get('reservation_usd') == '1'
        and attempt.get('request_sha256') == '781efdc95ed883b4587ef389d9df465f12aa10250a2a30697e35f1a5aa9c601b'
        and attempt.get('slot', {}).get('slot_id') == 'qwen3_instruct_2507__S_ESOL_to_Lipophilicity__07'
        and evidence.get('exception_type') == 'HTTPError' and evidence.get('http_status') == 403
        and evidence.get('provider_metadata', {}).get('limit_source') == 'openrouter_key_limit'
    )


def verify_stopped_history():
    for name, expected in PINNED.items():
        if file_hash(OLD / name) != expected:
            raise Halt('Historical v8 recovery evidence changed: ' + name)
    audit = load(OLD / 'STOP_AUDIT.json')
    paths = audit['journal_file_hashes']
    if len(paths) != 4018:
        raise Halt('Stopped journal inventory changed')
    actual = {p.relative_to(SOURCE_JOURNAL).as_posix() for p in SOURCE_JOURNAL.rglob('*') if p.is_file()}
    if actual != set(paths):
        raise Halt('Stopped journal membership changed')
    for name, expected in paths.items():
        path = (SOURCE_JOURNAL / name).resolve()
        if not path.is_relative_to(SOURCE_JOURNAL) or file_hash(path) != expected:
            raise Halt('Stopped journal bytes changed: ' + name)
    attempts = load(SOURCE_JOURNAL / 'ATTEMPTS.json')['attempts']
    science = load(SOURCE_JOURNAL / 'science_ledger/ledger.json')['attempts']
    if (len(attempts) != SOURCE_ATTEMPTS or len(science) != 514
            or sum(a['is_replacement'] for a in attempts) != 22
            or any(a['status'] == 'RESERVED' or a['test_only'] for a in attempts)
            or not check_stopped_attempt(attempts[-1])
            or attempts[-1]['error_evidence'] != load(SOURCE_JOURNAL / '0488.interruption.json')):
        raise Halt('Expected exact, settled 403 key-limit stop at logical slot 515')
    return audit
