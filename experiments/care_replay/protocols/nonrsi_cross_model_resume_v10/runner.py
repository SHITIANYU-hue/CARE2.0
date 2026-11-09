from protocols.nonrsi_cross_model_v1.common import *
from protocols.nonrsi_cross_model_v1 import freeze,generation,request_contract,inputs,transport
from protocols.nonrsi_cross_model_resume_v2 import runner as error_helper
from protocols.nonrsi_cross_model_resume_v8 import response_transport
from protocols.nonrsi_cross_model_resume_v9 import runner as previous
from . import policy, evaluate
from .history import OLD, OUT, SOURCE_JOURNAL, SOURCE_ATTEMPTS, check_stopped_attempt, verify_stopped_history
from decimal import Decimal
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import math,shutil,fcntl,urllib.request,urllib.error,argparse,subprocess,time
IMPORTED=47
HISTORICAL_ATTEMPTS=50
HISTORICAL_HOLD=Decimal('3')
MAX_ATTEMPTS=700
MAX_REPLACEMENTS=38
PER_SLOT_REPLACEMENTS=2
CAP=Decimal('45')
TRANSPORT_EXCEPTIONS={'URLError','TimeoutError','ConnectionResetError','ConnectionAbortedError','BrokenPipeError','RemoteDisconnected','IncompleteRead'}
TERMINAL=(generation.VALID,generation.INVALID)


def shared_pool_429(slot,evidence):
    meta=evidence.get('provider_metadata',{})
    return (slot['model']=='qwen3_instruct_2507' and evidence.get('exception_type')=='HTTPError' and evidence.get('http_status')==429 and evidence.get('error',{}).get('code')==429 and isinstance(meta,dict) and meta.get('provider_name')=='Parasail' and meta.get('limit_source')=='upstream_provider_shared_pool')

def qualifying(slot,evidence):
    return response_transport.qualifies(slot,evidence) or shared_pool_429(slot,evidence) or (evidence.get('exception_type') in TRANSPORT_EXCEPTIONS and evidence.get('transport_phase')=='open_or_read' and evidence.get('http_status') is None and not evidence.get('response_received'))

def permitted_imported_stop(index,attempt,source):
    if index==595:
        return index<len(source) and attempt==source[index] and check_stopped_attempt(attempt)
    return previous.permitted_imported_stop(index,attempt,source)


def retry_hint(evidence,now=None):
    now=now or datetime.now(timezone.utc);values=[]
    for value in [evidence.get('headers',{}).get('retry-after'),evidence.get('provider_metadata',{}).get('retry_after')]:
        if isinstance(value,bool) or value is None:continue
        try:
            seconds=float(value)
            if math.isfinite(seconds) and seconds>=0:values.append(math.ceil(seconds));continue
        except (ValueError,TypeError):pass
        try:
            date=parsedate_to_datetime(str(value))
            if date.tzinfo is not None:values.append(max(0,math.ceil((date-now).total_seconds())))
        except (ValueError,TypeError,OverflowError):pass
    return max(values,default=0)

def verify(lock_path,auth_path,attestation):
    if Path(lock_path).resolve()!=OUT/'FROZEN_RECOVERY.json' or Path(auth_path).resolve()!=OUT/'RECOVERY_AUTHORIZATION.json':
        raise Halt('Unapproved recovery bundle path')
    lock=load(lock_path)
    if lock.get('kind')!='nonrsi_cross_model_slot621_retry_v10':raise Halt('Wrong resume freeze')
    if lock.get('source_journal')!=str(SOURCE_JOURNAL.relative_to(ROOT)) or lock.get('journal_directory')!=str((OUT/'journal').relative_to(ROOT)) or lock.get('evaluation_output')!=str((OUT/'evaluation').relative_to(ROOT)):
        raise Halt('Recovery path binding changed')
    for name,value in lock['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or freeze.file_hash(path)!=value:raise Halt('Frozen resume evidence changed: '+name)
    auth=load(auth_path)
    expected={'kind':'slot621_retry_authorization','exact_retry_once':True,'fallback_reason':policy.REASON,'resume_freeze_sha256':freeze.file_hash(lock_path),'previous_authorization_sha256':freeze.file_hash(OLD/'RECOVERY_AUTHORIZATION.json'),'allow_execution':True,'model_calls':700,'model_spend_cap_usd':'45','historical_unknown_charge_hold_usd':'26','replacement_cap':38,'replacements_already_used':23,'per_slot_replacements_cap':2,'backoff_seconds':[600,1800],'qwen_minimum_interval_seconds':600}
    if any(auth.get(k)!=v for k,v in expected.items()) or not auth.get('user_message_reference'):raise Halt('Authorization mismatch')
    config,prior=previous.verify(OLD/'FROZEN_RECOVERY.json',OLD/'RECOVERY_AUTHORIZATION.json',attestation)
    if lock['base_lock']!=prior['base_lock'] or lock['original_ledger']!=prior['original_ledger']:raise Halt('Original protocol binding changed')
    verify_stopped_history(lock['source_journal_hashes'])
    return config,lock


def copy_exact(source,target):
    source=Path(source);target=Path(target)
    if target.exists():
        if freeze.file_hash(source)!=freeze.file_hash(target):raise Halt('Derived evidence changed: '+target.name)
    else:shutil.copyfile(source,target)

class Journal:
    def __init__(self,directory,original,config,production_authorized=False):
        self.directory=Path(directory);self.original=Path(original);self.config=config;self.authorized=production_authorized
        self.path=self.directory/'ATTEMPTS.json';self.science=self.directory/'science_ledger';self.order=slots(config);self.models={m['id']:m for m in config['models']}
        self.imported_source=load(SOURCE_JOURNAL/'ATTEMPTS.json')['attempts']
        if not self.directory.exists():
            self.directory.parent.mkdir(parents=True,exist_ok=True)
            temp=Path(tempfile.mkdtemp(prefix='.transport-journal-',dir=self.directory.parent))
            shutil.copytree(SOURCE_JOURNAL,temp,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.lock'))
            os.rename(temp,self.directory)

        self.original_state=load(self.original/'ledger.json');self.prefix=self.original_state['attempts'][:IMPORTED]
        if len(self.original_state['attempts'])!=48 or any(a['status'] not in TERMINAL for a in self.prefix):raise Halt('Invalid original ledger')
        self.original_sha=freeze.file_hash(self.original/'ledger.json')
        self.base_cost=sum((generation.money(a['cost_usd']) for a in self.prefix),Decimal(0))
        if not self.directory.exists():
            self.directory.mkdir(parents=True);self.science.mkdir()
            for i in range(IMPORTED):
                for f in self.original.glob(f'{i:04d}.*'):copy_exact(f,self.science/f.name)
            save(self.science/'ledger.json',{'config_sha256':digest(config),'attempts':self.prefix})
            save(self.path,{'config_sha256':digest(config),'original_ledger_sha256':self.original_sha,'attempts':[]})
        self.state();self.sync()
    def state(self):
        state=load(self.path)
        if state['config_sha256']!=digest(self.config) or state['original_ledger_sha256']!=self.original_sha:raise Halt('Journal binding changed')
        if state['attempts'][:SOURCE_ATTEMPTS]!=self.imported_source:raise Halt('Imported 596-attempt prefix changed')
        return state
    def totals(self,rows):
        known=self.base_cost+sum((generation.money(a['cost_usd']) for a in rows if a.get('cost_usd') is not None),Decimal(0))
        holds=HISTORICAL_HOLD+sum((Decimal('1') for a in rows if a.get('cost_usd') is None),Decimal(0))
        return known,holds
    def plan(self):
        rows=self.state()['attempts'];logical=IMPORTED;replacements=0;local={}
        retry=True;hint=retry_hint(load(self.original/'0047.interruption.json'))
        for j,a in enumerate(rows):
            expected_ordinal=local.get(logical,0)+1 if retry else 0
            if a['logical_index']!=logical or a['slot']!=self.order[logical] or a['total_attempt_index']!=HISTORICAL_ATTEMPTS+j or a['is_replacement']!=retry or a['slot_replacement_ordinal']!=expected_ordinal:raise Halt('Attempt ordering/mapping changed')
            schedule=[60,180] if j<273 else [180,600] if j<487 else [600,1800]
            required=max(schedule[expected_ordinal-1],hint) if retry and 1<=expected_ordinal<=2 else 0
            if a['required_backoff_seconds']!=required or a['wait_record']['waited_seconds']+1e-6<required:raise Halt('Recorded backoff invalid')
            if retry:
                local[logical]=expected_ordinal;replacements+=1
                if expected_ordinal>PER_SLOT_REPLACEMENTS or replacements>MAX_REPLACEMENTS:raise Halt('Recorded replacement cap exceeded')
            if a['status'] in TERMINAL:logical+=1;retry=False;hint=0
            elif a['status'] in ['RETRYABLE_429','RETRYABLE_TRANSPORT','RETRYABLE_RESPONSE_TRANSPORT']:
                if policy.is_retry(j,a) or not qualifying(a['slot'],a['error_evidence']):raise Halt('Nonqualifying error marked retryable')
                retry=True;hint=a['retry_after_seconds']
            elif permitted_imported_stop(j,a,self.imported_source):retry=True;hint=0
            else:raise Halt('Unresolved or non-retryable attempt; stopped')
        known,holds=self.totals(rows)
        if len(rows)+HISTORICAL_ATTEMPTS>MAX_ATTEMPTS or known+holds>CAP:raise Halt('Recorded budget/call cap exceeded')
        if logical==660:return None
        if len(rows)+HISTORICAL_ATTEMPTS>=MAX_ATTEMPTS:raise Halt('Total HTTP attempt cap reached')
        ordinal=local.get(logical,0)+1 if retry else 0
        if retry and replacements>=MAX_REPLACEMENTS:raise Halt('Global replacement allowance exhausted')
        if retry and ordinal>PER_SLOT_REPLACEMENTS:raise Halt('Per-slot replacement allowance exhausted')
        if known+holds+1>CAP:raise Halt('Budget cannot reserve next dollar')
        wait=max([600,1800][ordinal-1],hint) if retry else 0
        return {'logical_index':logical,'slot':self.order[logical],'total_attempt_index':HISTORICAL_ATTEMPTS+len(rows),'is_replacement':retry,'slot_replacement_ordinal':ordinal,'required_backoff_seconds':wait}
    def reserve(self,plan,body,wait_record,*,test_only=False):
        if not self.authorized and not test_only:raise Halt('Production execution not authorized')
        if plan!=self.plan():raise Halt('Plan changed before reservation')
        if wait_record['waited_seconds']+1e-6<plan['required_backoff_seconds']:raise Halt('Backoff not honored')
        prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(plan['slot']['cell']+'.json'))
        if body!=request_contract.build(self.config,plan['slot'],prompt):raise Halt('Request differs from frozen contract')
        state=self.state();n=len(state['attempts'])
        if n==SOURCE_ATTEMPTS and (not policy.is_retry(n,plan) or digest(body)!=policy.REQUEST_DIGEST):
            raise Halt('Authorized retry differs from slot 621 or frozen request')
        entry={**plan,'status':'RESERVED','reservation_usd':'1','cost_usd':None,'request_sha256':digest(body),'test_only':test_only,'wait_record':wait_record}
        state['attempts'].append(entry)
        save(self.directory/f'{n:04d}.request.json',body);save(self.path,state)
        return n
    def response_identity(self,index,response):
        state=self.state();model_id=state['attempts'][index]['slot']['model']
        prior=self.prefix+state['attempts'][:index]
        ids={a.get('response_id') for a in prior if a.get('response_id')}
        fps={a.get('fingerprint') for a in prior if a['slot']['model']==model_id and a.get('fingerprint')}
        for a in prior:
            e=a.get('error_evidence') or {}
            if e.get('response_id'):ids.add(e['response_id'])
            if a['slot']['model']==model_id and e.get('fingerprint'):fps.add(e['fingerprint'])
        if index>456:ids.add(load(SOURCE_JOURNAL/'0456.response.json')['id'])
        if index>595:ids.add(load(SOURCE_JOURNAL/'0595.response.json')['id'])
        if response.get('id') in ids:raise Halt('Duplicate response ID including failed network response')
        if response.get('system_fingerprint') and fps and response['system_fingerprint'] not in fps:raise Halt('Fingerprint changed')
    def classify_attempt(self,index,attempt,response,catalog):
        model=self.models[attempt['slot']['model']]
        if policy.is_retry(index,attempt) and policy.matches(response):
            if attempt['request_sha256']!=policy.REQUEST_DIGEST or load(self.directory/f'{index:04d}.http.json')['status']!=200:
                raise Halt('Policy-block retry lacks matching request/HTTP evidence')
            return policy.fallback(response,model,catalog,'1')
        return generation.classify(response,model,catalog,'1')
    def settle(self,index,response,catalog):
        state=self.state();a=state['attempts'][index]
        if index!=len(state['attempts'])-1 or a['status']!='RESERVED':raise Halt('Attempt already settled')
        save(self.directory/f'{index:04d}.response.json',response);a['response_sha256']=digest(response)
        try:
            self.response_identity(index,response)
            d=self.classify_attempt(index,a,response,catalog)
            prior=self.prefix+[x for x in state['attempts'][:index] if x['status'] in TERMINAL]
            if any(x.get('response_id')==response['id'] for x in prior):raise Halt('Duplicate response ID')
            fingerprints={x.get('fingerprint') for x in prior if x['slot']['model']==a['slot']['model'] and x.get('fingerprint')};fp=response.get('system_fingerprint')
            if fp is not None and not isinstance(fp,str):raise Halt('Malformed fingerprint')
            if fingerprints and fp and fp not in fingerprints:raise Halt('Fingerprint changed')
            a.update(d,fingerprint=fp)
        except Exception as exc:
            a.update(status='HALTED',halt_reason=str(exc));save(self.path,state);raise
        save(self.path,state);self.sync()
    def fail(self,index,evidence):
        state=self.state();a=state['attempts'][index]
        if index!=len(state['attempts'])-1:raise Halt('Failure out of order')
        if a['status']=='HALTED':return False
        if a['status']!='RESERVED':raise Halt('Cannot replace settled outcome')
        allowed=not policy.is_retry(index,a) and qualifying(a['slot'],evidence)
        a.update(status=('RETRYABLE_RESPONSE_TRANSPORT' if response_transport.qualifies(a['slot'],evidence) else 'RETRYABLE_429' if shared_pool_429(a['slot'],evidence) else 'RETRYABLE_TRANSPORT') if allowed else 'HALTED',error_evidence=evidence,retry_after_seconds=retry_hint(evidence),failure_observed_at_utc=datetime.now(timezone.utc).isoformat())
        save(self.directory/f'{index:04d}.interruption.json',evidence);save(self.path,state)
        return allowed
    def sync(self):
        rows=self.state()['attempts'];expected=list(self.prefix);mapping=[]
        current=load(self.science/'ledger.json')
        if current['config_sha256']!=digest(self.config):raise Halt('Science config changed')
        for i in range(IMPORTED):
            for suffix in ['request.json','response.json','response.txt']:copy_exact(self.original/f'{i:04d}.{suffix}',self.science/f'{i:04d}.{suffix}')
        for j,a in enumerate(rows):
            if a['status'] not in TERMINAL:continue
            i=a['logical_index']
            if i!=len(expected):raise Halt('Scientific slot gap or duplicate')
            record={k:a[k] for k in ['slot','request_sha256','reservation_usd','cost_usd','status','test_only','response_sha256','response_id','library','reasons','fingerprint']}
            expected.append(record);mapping.append({'logical_index':i,'journal_index':j,'total_attempt_index':a['total_attempt_index']})
            for suffix in ['request.json','response.json','response.txt']:
                copy_exact(self.directory/f'{j:04d}.{suffix}',self.science/f'{i:04d}.{suffix}')
        if current['attempts']!=expected[:len(current['attempts'])]:raise Halt('Derived science ledger changed')
        if len(current['attempts'])!=len(expected):save(self.science/'ledger.json',{'config_sha256':digest(self.config),'attempts':expected})
        save(self.directory/'PROJECTION_MAPPING.json',{'imported_prefix_count':47,'historical_http_attempts':50,'historical_unknown_holds_usd':'3','new_terminal_attempts':mapping})
    def audit(self,allow_test=False,catalogs=None,require_complete=False):
        rows=self.state()['attempts'];self.sync()
        # Verify every payload and successful/failed outcome before scientific use.
        seen={a['response_id'] for a in self.prefix};fingerprints={}
        for x in self.prefix:
            if x.get('fingerprint'):fingerprints[x['slot']['model']]=x['fingerprint']
        for j,a in enumerate(rows):
            if a['test_only'] and not allow_test:raise Halt('Synthetic attempt in production')
            body=load(self.directory/f'{j:04d}.request.json');prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(a['slot']['cell']+'.json'))
            if digest(body)!=a['request_sha256'] or body!=request_contract.build(self.config,a['slot'],prompt):raise Halt('Recorded request changed')
            if a['status'] in TERMINAL:
                raw=load(self.directory/f'{j:04d}.response.json')
                if digest(raw)!=a['response_sha256'] or load(self.directory/f'{j:04d}.response.txt')!=raw:raise Halt('Raw response changed')
                cat=catalogs[a['slot']['target']] if allow_test and catalogs else inputs.catalog(inputs.adapter(a['slot']['target']))
                if raw['id'] in seen:raise Halt('Duplicate response ID in audit')
                seen.add(raw['id']);fp=raw.get('system_fingerprint');model=a['slot']['model']
                if fp!=a['fingerprint'] or fp is not None and not isinstance(fp,str):raise Halt('Fingerprint evidence changed')
                if fp:
                    if model in fingerprints and fingerprints[model]!=fp:raise Halt('Fingerprint drift')
                    fingerprints[model]=fp
                d=self.classify_attempt(j,a,raw,cat)
                if any(a[k]!=v for k,v in d.items()):raise Halt('Decision differs from frozen classifier')
            elif a['status'] in ['RETRYABLE_429','RETRYABLE_TRANSPORT','RETRYABLE_RESPONSE_TRANSPORT']:
                if policy.is_retry(j,a) or not qualifying(a['slot'],a['error_evidence']) or load(self.directory/f'{j:04d}.interruption.json')!=a['error_evidence']:raise Halt('Retry evidence changed')
                if a['status']=='RETRYABLE_RESPONSE_TRANSPORT':
                    raw=load(self.directory/f'{j:04d}.response.json')
                    if load(self.directory/f'{j:04d}.response.txt')!=raw or response_transport.evidence(raw,self.models[a['slot']['model']])!=a['error_evidence']:raise Halt('Response-network evidence changed')
                    if raw['id'] in seen:raise Halt('Duplicate failed response ID')
                    seen.add(raw['id'])
                    fp=raw.get('system_fingerprint');model=a['slot']['model']
                    if fp and model in fingerprints and fingerprints[model]!=fp:raise Halt('Failed response fingerprint drift')
                    if fp:fingerprints[model]=fp
            elif permitted_imported_stop(j,a,self.imported_source):
                if j in (197,488):
                    if load(self.directory/f'{j:04d}.interruption.json')!=a['error_evidence']:raise Halt('Imported stop evidence changed')
                else:
                    raw=load(self.directory/f'{j:04d}.response.json')
                    if digest(raw)!=a['response_sha256'] or load(self.directory/f'{j:04d}.response.txt')!=raw:raise Halt('Imported response evidence changed')
                    if j==595:
                        if not policy.matches(raw) or load(self.directory/f'{j:04d}.http.json')['status']!=200:raise Halt('Imported policy-block evidence changed')
                    elif response_transport.evidence(raw,self.models[a['slot']['model']]) is None:raise Halt('Imported response-network evidence changed')
                    if raw['id'] in seen:raise Halt('Duplicate imported failed response ID')
                    seen.add(raw['id'])
                    fp=raw.get('system_fingerprint');model=a['slot']['model']
                    if fp and model in fingerprints and fingerprints[model]!=fp:raise Halt('Failed response fingerprint drift')
                    if fp:fingerprints[model]=fp
        next_plan=self.plan()
        known,holds=self.totals(rows)
        report={'logical_slots':len(load(self.science/'ledger.json')['attempts']),'total_http_attempts':50+len(rows),'new_replacement_attempts':sum(x['is_replacement'] for x in rows),'known_model_cost_usd':str(known),'unknown_charge_holds_usd':str(holds),'conservative_accounted_usd':str(known+holds),'cap_usd':'45'}
        save(self.directory/'TOTAL_ACCOUNTING.json',report)
        if require_complete:
            if next_plan is not None:raise Halt('Generation inventory incomplete')
            return self.complete_science(allow_test=allow_test,catalogs=catalogs)
        return report

    def complete_science(self,allow_test=False,catalogs=None):
        # The preceding audit binds this one outcome to the exact authorized retry.
        # Let the original complete-ledger validation also check the unmodified raw digest.
        attempts=load(self.science/'ledger.json')['attempts']
        a=attempts[policy.SLOT_INDEX]
        ledger=generation.Ledger(self.science,self.config)
        if a['reasons']!=[policy.REASON]:
            return ledger.complete(allow_test=allow_test,catalogs=catalogs)
        rows=self.state()['attempts']; candidate=rows[SOURCE_ATTEMPTS]
        if not policy.is_retry(SOURCE_ATTEMPTS,candidate) or a['response_sha256']!=candidate['response_sha256'] or a['slot']!=candidate['slot']:
            raise Halt('Fallback no longer binds to the authorized slot')
        original_classify=generation.classify
        def scoped(response,model,catalog,reservation):
            if digest(response)==a['response_sha256']:
                return policy.fallback(response,model,catalog,reservation)
            return original_classify(response,model,catalog,reservation)
        generation.classify=scoped
        try:
            return ledger.complete(allow_test=allow_test,catalogs=catalogs)
        finally:
            generation.classify=original_classify

class Scheduler:
    def __init__(self,clock=None,sleeper=None):
        self.clock=clock or time.monotonic;self.sleeper=sleeper or time.sleep;self.qwen_ready=self.clock()+600
    def wait(self,plan):
        start=self.clock();deadline=start+plan['required_backoff_seconds']
        if plan['slot']['model']=='qwen3_instruct_2507':deadline=max(deadline,self.qwen_ready)
        while self.clock()<deadline:self.sleeper(min(30,deadline-self.clock()))
        return {'waited_seconds':self.clock()-start,'required_backoff_seconds':plan['required_backoff_seconds'],'ready_monotonic_seconds':self.clock()}
    def finished(self,slot):
        if slot['model']=='qwen3_instruct_2507':self.qwen_ready=self.clock()+600

def run(lock_path,auth_path,attestation,directory):
    config,lock=verify(lock_path,auth_path,attestation)
    if Path(directory).resolve()!=(ROOT/lock['journal_directory']).resolve():raise Halt('Unapproved journal destination')
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise Halt('OPENROUTER_API_KEY missing')
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/'generation.process.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=Journal(directory,ROOT/lock['original_ledger'],config,production_authorized=True);journal.audit()
        certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
        for name,cert in certs.items():
            if inputs.certify(inputs.adapter(name))!=cert:raise Halt('Pool changed: '+name)
        opener=urllib.request.build_opener(transport.NoRedirect());scheduler=Scheduler()
        while True:
            plan=journal.plan()
            if plan is None:break
            wait_record=scheduler.wait(plan);verify(lock_path,auth_path,attestation)
            prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(plan['slot']['cell']+'.json'));body=request_contract.build(config,plan['slot'],prompt)
            n=journal.reserve(plan,body,wait_record)
            request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            transport_phase='preparing'
            try:
                save(directory/f'{n:04d}.request_timing.json',{'total_attempt_index':plan['total_attempt_index'],'logical_index':plan['logical_index'],'model':plan['slot']['model'],'started_at_utc':datetime.now(timezone.utc).isoformat(),'started_monotonic_seconds':time.monotonic(),'wait_record':wait_record})
                transport_phase='open_or_read'
                with opener.open(request,timeout=180) as response:
                    raw=response.read();save(directory/f'{n:04d}.http.json',{'status':response.status,'headers':{k:response.headers[k] for k in ['date','x-request-id','content-type'] if k in response.headers}})
                transport_phase='response_received'
                (directory/f'{n:04d}.response.txt').write_bytes(raw)
                parsed=contract.parse_strict_json(raw.decode('utf-8'))
                save(directory/f'{n:04d}.response.json',parsed)
                journal.response_identity(n,parsed)
                embedded=response_transport.evidence(parsed,journal.models[plan['slot']['model']])
                if embedded is not None:
                    journal.fail(n,embedded)
                else:journal.settle(n,parsed,inputs.catalog(inputs.adapter(plan['slot']['target'])))

            except Exception as exc:
                evidence=error_helper.error_evidence(exc,key)
                evidence.update(action='bounded_transport_retry_if_qualified_and_allowance_remains',transport_phase=transport_phase,response_received=transport_phase=='response_received')
                allowed=journal.fail(n,evidence)
                if not allowed:raise Halt('Nonqualifying anomaly; stopped without retry') from None
            finally:scheduler.finished(plan['slot'])
        journal.audit(require_complete=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify','generate','prepare','calibrate','heldout','analyze']);p.add_argument('--lock',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--attestation',type=Path,required=True);p.add_argument('--journal',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    config,lock=verify(a.lock,a.authorization,a.attestation)
    if a.journal.resolve()!=(ROOT/lock['journal_directory']).resolve():raise Halt('Unapproved journal destination')
    if a.output is not None and a.output.resolve()!=(ROOT/lock['evaluation_output']).resolve():raise Halt('Unapproved evaluation destination')
    if a.phase=='verify':print('Bounded transport freeze and authorization verified');return
    if a.phase=='generate':run(a.lock,a.authorization,a.attestation,a.journal);return
    if a.output is None:p.error('--output required')
    with offline():
        j=Journal(a.journal,ROOT/lock['original_ledger'],config);attempts=j.audit(require_complete=True)
        evaluate.run(a.phase,config,attempts,j.science,a.output,a.workers)

if __name__=='__main__':main()
