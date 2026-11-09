"""Generation accounting. Production reservations require external authorization."""
from .common import *
from decimal import Decimal,InvalidOperation
from contextlib import contextmanager
import fcntl
VALID='VALID';INVALID='INVALID_ANCHOR_FALLBACK'
def money(v):
    if isinstance(v,bool) or not isinstance(v,(int,float,str,Decimal)):raise Halt('Unknown cost')
    try:d=Decimal(str(v))
    except InvalidOperation as e:raise Halt('Malformed cost') from e
    if not d.is_finite() or d<0:raise Halt('Invalid cost')
    return d

def classify(response,model,catalog,reservation):
    if not isinstance(response,dict) or response.get('object')!='chat.completion' or not isinstance(response.get('id'),str) or not response['id'] or response.get('error'):raise Halt('Malformed response')
    if response.get('model') not in model.get('allowed_response_models',[model['openrouter_id']]) or response.get('provider')!=model['provider_candidate']:raise Halt('Identity mismatch')
    usage=response.get('usage')
    if not isinstance(usage,dict) or usage.get('is_byok') is not False:raise Halt('Billing provenance missing')
    cost=money(usage.get('cost'))
    if cost>money(reservation):raise Halt('Reservation exceeded')
    for name,maximum in [('prompt_tokens',12000),('completion_tokens',5000)]:
        v=usage.get(name)
        if type(v) is not int or not 0<=v<=maximum:raise Halt('Usage bound mismatch: '+name)
    details=usage.get('completion_tokens_details',{})
    if not isinstance(details,dict):raise Halt('Malformed completion details')
    if details.get('reasoning_tokens',0)!=0:raise Halt('Unexpected reasoning use')
    if model['id']=='gpt6luna_public_id':
        detail=usage.get('prompt_tokens_details')
        if not isinstance(detail,dict) or type(detail.get('cache_write_tokens')) is not int:raise Halt('Cache-write evidence missing')
        if detail['cache_write_tokens']!=0:raise Halt('Cache write despite disabled cache')
    choices=response.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):raise Halt('Invalid choice envelope')
    choice=choices[0];msg=choice.get('message');reason=choice.get('finish_reason')
    if not isinstance(msg,dict) or msg.get('role')!='assistant' or choice.get('error'):raise Halt('Invalid message')
    if reason not in ['stop','length','content_filter','tool_calls','function_call']:raise Halt('Unknown finish reason')
    content=msg.get('content');refusal=msg.get('refusal');tools=msg.get('tool_calls');function=msg.get('function_call')
    if content is not None and not isinstance(content,str):raise Halt('Non-text content envelope')
    if refusal is not None and not isinstance(refusal,str):raise Halt('Malformed refusal')
    if tools is not None and (not isinstance(tools,list) or any(not isinstance(t,dict) for t in tools)):raise Halt('Malformed tool envelope')
    if function is not None and (not isinstance(function,dict) or not isinstance(function.get('name'),str) or not isinstance(function.get('arguments'),str)):raise Halt('Malformed function envelope')
    if reason=='tool_calls' and not tools or reason=='function_call' and not function:raise Halt('Missing tool/function call')
    result={'cost_usd':str(cost),'response_id':response['id'],'status':INVALID,'library':None,'reasons':[]}
    if reason in ['length','content_filter']:result['reasons']=[reason];return result
    if refusal or tools or function or not content or not content.strip():result['reasons']=['non_library_completion'];return result
    try:library=contract.parse_strict_json(content)
    except contract.ValidationError as e:result['reasons']=['invalid_json',*e.errors];return result
    errors=contract.strict_validate_library(library,catalog)
    if errors:result['reasons']=errors;return result
    result.update(status=VALID,library=library);return result

class Ledger:
    """Single-writer, write-ahead reservations. An interrupted attempt is never replayed."""
    def __init__(self,directory,config,production_authorized=False):
        self.production_authorized=production_authorized
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.config=config;self.path=self.directory/'ledger.json'
        self.order=slots(config);self.models={m['id']:m for m in config['models']}
        self.reserves={b['model']:money(b['reservation_per_attempt_usd']) for b in config['budget']}
    @contextmanager
    def transaction(self):
        with (self.directory/'writer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=load(self.path) if self.path.exists() else {'config_sha256':digest(self.config),'attempts':[]}
            if state['config_sha256']!=digest(self.config):raise Halt('Config changed')
            yield state
            save(self.path,state)
    def reserve(self,slot,request,*,test_only=False):
        if not test_only and not self.production_authorized:raise Halt('Production execution has not been authorized')
        with self.transaction() as state:
            attempts=state['attempts'];n=len(attempts)
            if any(a['status'] not in [VALID,INVALID] for a in attempts):raise Halt('Unresolved earlier attempt')
            if n>=660 or slot!=self.order[n]:raise Halt('Unexpected next slot or attempt cap')
            reserve=self.reserves[slot['model']]
            spent=sum((money(a['cost_usd']) for a in attempts),Decimal(0))
            if spent+reserve>money(self.config['proposed_model_inference_spend_cap_usd']):raise Halt('Budget cap')
            attempts.append({'slot':slot,'request_sha256':digest(request),'reservation_usd':str(reserve),'cost_usd':None,'status':'RESERVED','test_only':test_only})
            save(self.directory/f'{n:04d}.request.json',request)
            return n
    def settle(self,index,response,catalog):
        error=None
        with self.transaction() as state:
            a=state['attempts'][index]
            if a['status']!='RESERVED':raise Halt('Attempt already settled or halted')
            save(self.directory/f'{index:04d}.response.json',response)
            a['response_sha256']=digest(response)
            try:
                decision=classify(response,self.models[a['slot']['model']],catalog,a['reservation_usd'])
                fingerprints={x.get('fingerprint') for x in state['attempts'][:index] if x['slot']['model']==a['slot']['model'] and x.get('fingerprint')}
                fp=response.get('system_fingerprint')
                if fp is not None and not isinstance(fp,str):raise Halt('Malformed fingerprint')
                if fingerprints and fp and fp not in fingerprints:raise Halt('Fingerprint changed')
                if any(x.get('response_id')==response['id'] for x in state['attempts'][:index]):raise Halt('Duplicate response ID')
                a.update(decision,fingerprint=fp)
            except Exception as exc:
                a.update(status='HALTED',halt_reason=str(exc))
                error=exc
        if error is not None:raise error
        return decision

    def complete(self,allow_test=False,catalogs=None):
        state=load(self.path)
        if state['config_sha256']!=digest(self.config):raise Halt('Ledger config changed')
        attempts=state['attempts']
        if len(attempts)!=len(self.order):raise Halt('Incomplete generation inventory')
        from . import inputs,request_contract
        seen_ids=set();seen_fingerprints={}
        for i,a in enumerate(attempts):
            if a['slot']!=self.order[i] or a['status'] not in [VALID,INVALID]:raise Halt('Unsettled or reordered generation')
            if a['test_only'] and not allow_test:raise Halt('Synthetic ledger cannot enter scientific replay')
            if digest(load(self.directory/f'{i:04d}.request.json'))!=a['request_sha256']:raise Halt('Request changed')
            raw=load(self.directory/f'{i:04d}.response.json')
            if digest(raw)!=a['response_sha256']:raise Halt('Response changed')
            if a.get('library') is not None and a['status']!=VALID:raise Halt('Invalid library retained')
            model=self.models[a['slot']['model']]
            cat=catalogs[a['slot']['target']] if allow_test and catalogs is not None else inputs.catalog(inputs.adapter(a['slot']['target']))
            decision=classify(raw,model,cat,a['reservation_usd'])
            if any(a[k]!=v for k,v in decision.items()):raise Halt('Ledger decision differs from raw response')
            if raw['id'] in seen_ids:raise Halt('Duplicate response ID in complete ledger')
            seen_ids.add(raw['id']);fp=raw.get('system_fingerprint')
            if fp:
                if model['id'] in seen_fingerprints and seen_fingerprints[model['id']]!=fp:raise Halt('Fingerprint drift')
                seen_fingerprints[model['id']]=fp
            if not allow_test:
                prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(a['slot']['cell']+'.json'))
                if a['request_sha256']!=digest(request_contract.build(self.config,a['slot'],prompt)):raise Halt('Recorded request differs from frozen contract')
        if sum((money(a['cost_usd']) for a in attempts),Decimal(0))>money(self.config['proposed_model_inference_spend_cap_usd']):raise Halt('Cost exceeds cap')
        return attempts
