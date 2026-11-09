"""Deterministic provider requests. This module performs no network I/O."""
from .common import *
from .generation import money

def build(config,slot,prompt):
    model=next(m for m in config['models'] if m['id']==slot['model'])
    if not model.get('exact_endpoint_slug') or not model.get('observed_endpoint_id'):raise Halt('Unpinned endpoint')
    messages=prompt['messages']
    if len(messages)!=2 or [m['role'] for m in messages]!=['system','user'] or any(not isinstance(m['content'],str) for m in messages):raise Halt('Unexpected messages')
    request={'model':model['openrouter_id'],'messages':messages,'max_tokens':5000,'stream':False,'plugins':[],'transforms':[],
             'provider':{'only':[model['exact_endpoint_slug']],'order':[model['exact_endpoint_slug']],'allow_fallbacks':False,'require_parameters':True,
                         'max_price':{'prompt':float(money(model['price_cap_input'])),'completion':float(money(model['price_cap_output'])),'request':0}}}
    if model['temperature'] is not None:request['temperature']=model['temperature']
    if model['id']=='gpt6luna_public_id':
        if 'temperature' in request:raise Halt('Luna sampling amendment missing')
        request['reasoning']={'effort':'none'}
        request['prompt_cache_options']={'mode':'explicit','ttl':'30m'}
    elif model['id']=='haiku45_20251001':request['reasoning']={'enabled':False}
    elif model['id']!='qwen3_instruct_2507':raise Halt('Unknown model')
    return request

def check_endpoint(model,record):
    for key,expected in [('id',model['observed_endpoint_id']),('provider_slug',model['exact_endpoint_slug']),('provider_model_id',model['observed_provider_model_id']),('variant','standard'),('context_length',model['observed_context_length'])]:
        if record.get(key)!=expected:raise Halt('Endpoint changed: '+key)
    if any(record.get(k) for k in ['is_disabled','is_byok_only','is_private','is_pricing_unresolved','service_tier']):raise Halt('Ineligible endpoint')
    required={'max_tokens'}
    if model['temperature'] is not None:required.add('temperature')
    if model['id']!='qwen3_instruct_2507':required.add('reasoning')
    if not required<=set(record['supported_parameters']) or required&set(record['excluded_parameters']):raise Halt('Unsupported requested parameter')
    for k,cap in [('prompt',model['price_cap_input']),('completion',model['price_cap_output'])]:
        if money(record['pricing'][k])*1000000>money(cap):raise Halt('Endpoint price exceeds cap')
    # The complete tier schedule is bound into the freeze; no undocumented new tier.
    if record.get('pricing_version_id')!=model.get('pricing_version_id'):raise Halt('Pricing version changed')
    if record.get('quantization')!=model['observed_quantization']:raise Halt('Quantization changed')
    return True
