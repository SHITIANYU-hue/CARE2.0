from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze,generation,request_contract,inputs,transport
from decimal import Decimal
from datetime import datetime,timezone
import shutil,fcntl,urllib.request,urllib.error,re,argparse,subprocess,time
from protocols.nonrsi_cross_model_resume_v1 import runner as previous

IMPORTED=25
UNKNOWN_HOLD=Decimal('2')
MAX_ATTEMPTS=662
CAP=Decimal('15')

def verify(resume_lock,authorization,attestation):
    lock=load(resume_lock)
    if lock.get('kind')!='nonrsi_cross_model_second_replacement_freeze':raise Halt('Wrong resume freeze')
    for name,value in lock['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or freeze.file_hash(path)!=value:raise Halt('Resume evidence changed: '+name)
    base=ROOT/lock['base_lock']
    if freeze.file_hash(base)!=lock['base_freeze_sha256']:raise Halt('Base freeze changed')
    config=freeze.authorize(base,ROOT/lock['original_authorization'],attestation)
    previous.verify(ROOT/lock['previous_lock'],ROOT/lock['previous_authorization'],attestation)
    auth=load(authorization)
    expected={'kind':'explicit_second_replacement_authorization','resume_freeze_sha256':freeze.file_hash(resume_lock),'base_freeze_sha256':freeze.file_hash(base),'model_calls':662,'model_spend_cap_usd':'15','unknown_charge_reservation_usd':'2','replacement_slot_id':slots(config)[25]['slot_id'],'allow_execution':True,'qwen_minimum_interval_seconds':60}
    if any(auth.get(k)!=v for k,v in expected.items()) or not auth.get('user_message_reference'):raise Halt('Resume authorization mismatch')
    old=load(ROOT/lock['original_ledger']/'ledger.json')
    if old['config_sha256']!=digest(config) or len(old['attempts'])!=26:raise Halt('Original inventory mismatch')
    if any(x['status'] not in [generation.VALID,generation.INVALID] or x.get('test_only') for x in old['attempts'][:25]):raise Halt('Original settled prefix mismatch')
    failed=old['attempts'][25]
    if failed['status']!='RESERVED' or failed['cost_usd'] is not None or failed['reservation_usd']!='1' or failed['slot']!=slots(config)[25]:raise Halt('Wrong failed attempt')
    if lock.get('replacement_slot_id')!=failed['slot']['slot_id']:raise Halt('Replacement mapping mismatch')
    return config,lock

def initialize(directory,original,config):
    directory=Path(directory);original=Path(original)
    state=load(original/'ledger.json')
    if len(state['attempts'])!=26:raise Halt('Original prefix length changed')
    if directory.exists():
        check_prefix(directory,original,config);return
    directory.mkdir(parents=True)
    # Imported successes retain their exact request, raw response and classification.
    for i in range(IMPORTED):
        for path in original.glob(f'{i:04d}.*'):
            shutil.copyfile(path,directory/path.name)
    save(directory/'ledger.json',{'config_sha256':digest(config),'attempts':state['attempts'][:IMPORTED]})
    save(directory/'ATTEMPT_MAPPING.json',{'imported_original_indices':list(range(25)),'excluded_original_attempt_index':25,'excluded_slot_id':slots(config)[25]['slot_id'],'unknown_charge_reservation_usd':'2','new_logical_index_to_total_attempt_index':'i + 2 for i >= 25; i + 1 for 16 <= i < 25; i for i < 16','maximum_total_http_attempts':662})
    check_prefix(directory,original,config)

def check_prefix(directory,original,config):
    state=load(Path(directory)/'ledger.json');old=load(Path(original)/'ledger.json')
    if state['config_sha256']!=digest(config) or state['attempts'][:25]!=old['attempts'][:25]:raise Halt('Imported prefix changed')
    if not 25<=len(state['attempts'])<=660:raise Halt('Logical inventory outside bounds')
    for i in range(25):
        for suffix in ['request.json','response.json','response.txt']:
            name=f'{i:04d}.{suffix}'
            if freeze.file_hash(Path(directory)/name)!=freeze.file_hash(Path(original)/name):raise Halt('Imported evidence changed')

class ResumeLedger(generation.Ledger):
    def reserve(self,slot,request,*,test_only=False):
        state=load(self.path);a=state['attempts'];n=len(a)
        if not IMPORTED<=n<660 or n+3>MAX_ATTEMPTS:raise Halt('Total attempt cap')
        if any(x['status'] not in [generation.VALID,generation.INVALID] for x in a):raise Halt('Unresolved attempt; no retry')
        accounted=sum((generation.money(x['cost_usd']) for x in a),Decimal(0))+UNKNOWN_HOLD
        if accounted+generation.money('1')>CAP:raise Halt('Budget cap including unknown charge')
        return super().reserve(slot,request,test_only=test_only)
    def complete(self,allow_test=False,catalogs=None):
        a=super().complete(allow_test=allow_test,catalogs=catalogs)
        if len(a)+2>MAX_ATTEMPTS or sum((generation.money(x['cost_usd']) for x in a),Decimal(0))+UNKNOWN_HOLD>CAP:raise Halt('Final budget/attempt cap')
        return a

def redact(value,key):
    if not isinstance(value,str):return value
    if key:value=value.replace(key,'[REDACTED]')
    value=re.sub(r'(?i)Bearer\s+[^\s"\']+','Bearer [REDACTED]',value)
    value=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',value)
    return value[:2000]

def error_evidence(exc,key):
    result={'exception_type':type(exc).__name__,'action':'halt_without_retry','unknown_charge_reservation_retained':True}
    if isinstance(exc,urllib.error.HTTPError):
        result['http_status']=exc.code
        result['headers']={k:redact(exc.headers[k],key) for k in ['date','x-request-id','content-type','retry-after'] if exc.headers and k in exc.headers}
        try:
            raw=exc.read(65537);result['error_body_sha256']=hashlib.sha256(raw).hexdigest();result['body_truncated']=len(raw)>65536
            obj=json.loads(raw[:65536].decode('utf-8'));error=obj.get('error',{})
            if isinstance(error,dict):
                result['error']={k:redact(error[k],key) for k in ['code','message'] if isinstance(error.get(k),(str,int,float))}
                metadata=error.get('metadata',{})
                if isinstance(metadata,dict):
                    result['provider_metadata']={k:redact(metadata[k],key) for k in ['provider_name','provider_code','provider_error_code','limit_source','retry_after','remedy_hint'] if isinstance(metadata.get(k),(str,int,float))}
        except Exception as parse_error:result['error_body_parse_exception']=type(parse_error).__name__
    return result

class QwenThrottle:
    def __init__(self,clock=None,sleeper=None):
        self.clock=clock or time.monotonic;self.sleeper=sleeper or time.sleep
        # Restart always waits a fresh 60 seconds before its first Qwen request.
        self.ready_at=self.clock()+60
    def wait(self,slot):
        if slot['model']!='qwen3_instruct_2507':return
        while True:
            remaining=self.ready_at-self.clock()
            if remaining<=0:return
            self.sleeper(min(30,remaining))
    def started(self,directory,index,slot):
        save(Path(directory)/f'{index:04d}.request_timing.json',{'model':slot['model'],'logical_index':index,'total_attempt_index':index+2,'started_at_utc':datetime.now(timezone.utc).isoformat(),'started_monotonic_seconds':self.clock(),'minimum_qwen_interval_seconds':60})
    def finished(self,slot):
        # End-to-start cooldown is conservative: starts are at least 60s apart.
        if slot['model']=='qwen3_instruct_2507':self.ready_at=self.clock()+60

def run(resume_lock,authorization,attestation,directory):
    config,lock=verify(resume_lock,authorization,attestation)
    if Path(directory).resolve()!=(ROOT/lock['resume_ledger']).resolve():raise Halt('Unapproved ledger destination')
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise Halt('OPENROUTER_API_KEY missing')
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    # One lock covers initialization, reservation, request, settlement and completion.
    with (directory.parent/'resume.process.lock').open('a') as process_lock:
        fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=ROOT/lock['original_ledger'];initialize(directory,original,config)
        ledger=ResumeLedger(directory,config,production_authorized=True)
        if any(a.get('test_only') for a in load(ledger.path)['attempts']):raise Halt('Synthetic ledger prohibited')
        certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
        for name,cert in certs.items():
            if inputs.certify(inputs.adapter(name))!=cert:raise Halt('Pool changed: '+name)
        opener=urllib.request.build_opener(transport.NoRedirect());throttle=QwenThrottle()
        if any(a['status'] not in [generation.VALID,generation.INVALID] for a in load(ledger.path)['attempts']):raise Halt('Unresolved attempt; no retry')
        for slot in ledger.order[len(load(ledger.path)['attempts']):]:
            throttle.wait(slot)
            verify(resume_lock,authorization,attestation);check_prefix(directory,original,config)
            prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(slot['cell']+'.json'))
            body=request_contract.build(config,slot,prompt);index=ledger.reserve(slot,body)
            request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            try:
                throttle.started(directory,index,slot)
                with opener.open(request,timeout=180) as response:
                    raw=response.read();save(directory/f'{index:04d}.http.json',{'status':response.status,'headers':{k:response.headers[k] for k in ['date','x-request-id','content-type'] if k in response.headers},'total_attempt_index':index+2})
                (directory/f'{index:04d}.response.txt').write_bytes(raw)
                ledger.settle(index,contract.parse_strict_json(raw.decode('utf-8')),inputs.catalog(inputs.adapter(slot['target'])))
            except Exception as exc:
                save(directory/f'{index:04d}.interruption.json',error_evidence(exc,key))
                raise Halt('Resume interrupted; no additional replacement authorized') from None
            finally:throttle.finished(slot)
        a=ledger.complete();write_costs(directory,a)

def write_costs(directory,attempts):
    known=sum((generation.money(x['cost_usd']) for x in attempts),Decimal(0))
    save(Path(directory)/'TOTAL_ACCOUNTING.json',{'logical_slots':len(attempts),'total_http_attempts':len(attempts)+2,'known_cost_usd':str(known),'unknown_charge_reservation_usd':'2','conservative_accounted_usd':str(known+UNKNOWN_HOLD),'cap_usd':'15','excluded_attempts':['Initial execution index 16; HTTPError; cost unknown','First resume index 25; HTTP429; cost unknown']})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','generate','prepare','calibrate','heldout','analyze']);p.add_argument('--lock',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--attestation',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    config,lock=verify(a.lock,a.authorization,a.attestation)
    if a.ledger.resolve()!=(ROOT/lock['resume_ledger']).resolve():raise Halt('Unapproved ledger destination')
    if a.output is not None and a.output.resolve()!=(ROOT/lock['evaluation_output']).resolve():raise Halt('Unapproved evaluation destination')
    if a.phase=='verify':print('Resume freeze and authorization verified');return
    if a.phase=='generate':run(a.lock,a.authorization,a.attestation,a.ledger);return
    if a.output is None:p.error('--output required')
    with offline():
        check_prefix(a.ledger,ROOT/lock['original_ledger'],config)
        attempts=ResumeLedger(a.ledger,config).complete();write_costs(a.ledger,attempts)
    # Existing frozen scientific CLI receives the complete 660-slot inventory.
    command=[sys.executable,'-B','-m','protocols.nonrsi_cross_model_v1',a.phase,'--lock',str(ROOT/lock['base_lock']),'--ledger',str(a.ledger),'--output',str(a.output),'--workers',str(a.workers)]
    subprocess.run(command,cwd=BASE,check=True)

if __name__=='__main__':main()
