"""Explicitly authorized sequential POST, no SDK, retry, redirect or model fallback."""
from .common import *
from . import freeze,request_contract,inputs,generation
import urllib.request,urllib.error

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise Halt('HTTP redirect prohibited')

def run(lock_path,authorization_path,attestation_path,directory):
    # All permission/integrity checks occur BEFORE reading a key or creating transport.
    config=freeze.authorize(lock_path,authorization_path,attestation_path)
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise Halt('OPENROUTER_API_KEY is missing')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (directory/'generation.process.lock').open('a') as process_lock:
        fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=generation.Ledger(directory,config,production_authorized=True)
        if ledger.path.exists() and any(a.get('test_only') for a in load(ledger.path)['attempts']):raise Halt('Cannot resume synthetic ledger')
        # Verify all input pools once per process, without producing effects.
        certs=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/pool_certificates.json')
        for name,cert in certs.items():
            if inputs.certify(inputs.adapter(name))!=cert:raise Halt('Pool changed: '+name)
        opener=urllib.request.build_opener(NoRedirect())
        for slot in ledger.order[len(load(ledger.path)['attempts']) if ledger.path.exists() else 0:]:
            # Recheck authorization freshness, file integrity and the same route attestation.
            freeze.authorize(lock_path,authorization_path,attestation_path)
            prompt=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/prompts'/(slot['cell']+'.json'))
            body=request_contract.build(config,slot,prompt)
            index=ledger.reserve(slot,body)
            encoded=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=encoded,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            try:
                with opener.open(request,timeout=180) as response:
                    raw=response.read()
                    save(directory/f'{index:04d}.http.json',{'status':response.status,'headers':{k:response.headers[k] for k in ['date','x-request-id','content-type'] if k in response.headers}})
                (directory/f'{index:04d}.response.txt').write_bytes(raw)
                parsed=contract.parse_strict_json(raw.decode('utf-8'))
                ledger.settle(index,parsed,inputs.catalog(inputs.adapter(slot['target'])))
            except Exception as exc:
                # Never record exception text that could include headers/key. Reservation remains.
                save(directory/f'{index:04d}.interruption.json',{'exception_type':type(exc).__name__,'action':'halt_without_retry','unknown_charge_reservation_retained':True})
                raise Halt('Generation interrupted; preserve response/reservation and request review') from None
        ledger.complete()
