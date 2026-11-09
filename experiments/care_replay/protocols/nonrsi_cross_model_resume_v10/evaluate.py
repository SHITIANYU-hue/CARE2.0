"""Use the frozen replay and reporting functions on the independently audited ledger."""
from protocols.nonrsi_cross_model_v1 import generation, pipeline, reporting
from protocols.nonrsi_cross_model_v1.common import Halt, digest, load
from .policy import SLOT_INDEX, REASON


def run(phase, config, attempts, ledger, output, workers):
    if phase == 'prepare':
        pipeline.prepare(config, attempts, output)
        return
    definition = load(output / 'definition.json')
    if definition['generation_sha256'] != digest(attempts):
        raise Halt('Generation inventory changed')
    if phase == 'calibrate':
        pipeline.calibrate(config, definition, output, workers)
        return
    if phase == 'heldout':
        pipeline.heldout(config, definition, output, workers)
        return
    if phase != 'analyze':
        raise Halt('Unexpected evaluation phase')
    lock = load(output / 'CALIBRATION_LOCK.json')
    complete = load(output / 'HELDOUT_COMPLETE.json')
    if (complete['config_sha256'] != digest(config) or complete['definition_sha256'] != digest(definition)
            or complete['calibration_lock_sha256'] != digest(lock)):
        raise Halt('Heldout completion binding changed')
    rows = {t: load(output / t / 'heldout_metrics.json') for t in definition['targets']}
    if {t: digest(r) for t, r in rows.items()} != complete['metrics']:
        raise Halt('Heldout results changed')
    reporting.analyze(config, definition, lock, rows, output)
    cost = {}
    for i, attempt in enumerate(attempts):
        model = attempt['slot']['model']
        item = cost.setdefault(model, {'attempts': 0, 'valid': 0, 'invalid': 0,
                                       'model_cost_usd': '0', 'input_tokens': 0, 'output_tokens': 0})
        usage = load(ledger / f'{i:04d}.response.json')['usage']
        item['attempts'] += 1
        item['valid' if attempt['status'] == generation.VALID else 'invalid'] += 1
        item['model_cost_usd'] = str(generation.money(item['model_cost_usd']) + generation.money(attempt['cost_usd']))
        item['input_tokens'] += usage['prompt_tokens']
        item['output_tokens'] += usage['completion_tokens']
    pipeline.once(output / 'generation_costs.json', cost)
    attempt = attempts[SLOT_INDEX]
    pipeline.once(output / 'SLOT_621_DISPOSITION.json', {
        'slot': SLOT_INDEX + 1, 'slot_id': attempt['slot']['slot_id'], 'status': attempt['status'],
        'provider_policy_block_fallback': attempt['reasons'] == [REASON],
        'request_sha256': attempt['request_sha256'], 'terminal_response_sha256': attempt['response_sha256'],
        'original_response_preserved': True, 'reasons': attempt['reasons'],
    })
