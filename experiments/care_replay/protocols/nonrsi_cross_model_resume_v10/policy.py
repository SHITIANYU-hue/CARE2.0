"""Map only the observed, identity-checked provider policy envelope to fallback."""
from copy import deepcopy

from protocols.nonrsi_cross_model_v1 import generation
from protocols.nonrsi_cross_model_v1.common import Halt

SLOT_INDEX = 620
SLOT_ID = 'gpt6luna_public_id__F_bandgap_to_dielectric__09'
SOURCE_ATTEMPTS = 596
REQUEST_DIGEST = '0ba6dc7b987d615be458bad910ffa4df5877a4fac28c3f18b120195dddf46d0a'
REASON = 'provider_prompt_policy_block_after_authorized_retry'
PREFIX = 'Invalid prompt: your prompt was flagged as potentially violating our usage policy.'


def is_retry(index, attempt):
    return (index == SOURCE_ATTEMPTS and attempt['logical_index'] == SLOT_INDEX
            and attempt['total_attempt_index'] == 646 and attempt['is_replacement'] is True
            and attempt['slot_replacement_ordinal'] == 1
            and attempt['slot']['slot_id'] == SLOT_ID
            and attempt['slot']['model'] == 'gpt6luna_public_id')


def matches(response):
    if not isinstance(response, dict) or response.get('model') != 'openai/gpt-6-luna' or response.get('provider') != 'OpenAI':
        return False
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    error = choice.get('error'); message = choice.get('message')
    if not isinstance(error, dict) or not isinstance(message, dict) or message.get('role') != 'assistant':
        return False
    metadata = error.get('metadata'); description = error.get('message')
    return (choice.get('finish_reason') == 'error' and type(error.get('code')) is int
            and error['code'] == 400 and isinstance(metadata, dict)
            and metadata.get('error_type') == 'invalid_request'
            and isinstance(description, str) and description.startswith(PREFIX)
            and isinstance(message.get('content'), str)
            and not any(message.get(k) for k in ('tool_calls', 'function_call', 'reasoning', 'refusal')))


def fallback(response, model, catalog, reservation):
    if not matches(response) or model['id'] != 'gpt6luna_public_id':
        raise Halt('Response does not match the authorized policy-block envelope')
    # Classify a projection in memory only. Keep original request and raw response.
    projected = deepcopy(response)
    choice = projected['choices'][0]
    del choice['error']
    choice['finish_reason'] = 'content_filter'
    choice['message']['content'] = None
    decision = generation.classify(projected, model, catalog, reservation)
    if decision['status'] != generation.INVALID or decision['reasons'] != ['content_filter']:
        raise Halt('Projected policy block did not resolve to frozen fallback')
    return {**decision, 'reasons': [REASON]}
