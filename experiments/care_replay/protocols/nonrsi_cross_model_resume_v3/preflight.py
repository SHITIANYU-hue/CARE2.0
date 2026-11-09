"""Read-only public endpoint and account checks; no inference or retry."""
from .runner import *
import re
import requests,concurrent.futures

def read_key():
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:
        for line in (ROOT/'.env').read_text().splitlines():
            s=line.strip()
            if s.startswith('export '):s=s[7:].lstrip()
            if s.startswith('OPENROUTER_API_KEY='):key=s.split('=',1)[1].strip().strip('"\'')
    if not key:raise Halt('Key missing')
    return key

def check(base_lock,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);config=freeze.verify(base_lock)
    old=load(BASE/'outputs/nonrsi_cross_model_preparation_v1/endpoint_page_evidence.json')
    def fetch(m):
        response=requests.get('https://openrouter.ai/'+m['openrouter_id'],timeout=25,allow_redirects=False)
        if response.status_code!=200:raise Halt('Public endpoint lookup failed: '+str(response.status_code))
        stem=m['openrouter_id'].replace('/','__');(out/(stem+'.html')).write_bytes(response.content)
        parts=[]
        for match in re.finditer(r'self\.__next_f\.push\((\[.*?\])\)</script>',response.text):
            parts.extend(v for v in json.loads(match[1]) if isinstance(v,str))
        text=''.join(parts);decoder=json.JSONDecoder();found={}
        for match in re.finditer(r'"provider_slug":"'+re.escape(m['exact_endpoint_slug'])+'"',text):
            start=max(0,match.start()-14000)
            for rel in reversed([v.start() for v in re.finditer(r'\{',text[start:match.start()])]):
                pos=start+rel
                try:obj,end=decoder.raw_decode(text[pos:])
                except ValueError:continue
                if isinstance(obj,dict) and 'provider_slug' in obj and pos+end>match.end():found[obj.get('id',str(pos))]=obj;break
        eligible=[v for v in found.values() if v.get('variant')=='standard' and not any(v.get(k) for k in ['is_disabled','is_byok_only','is_private','is_pricing_unresolved','service_tier'])]
        if len(eligible)!=1:raise Halt('Not one eligible endpoint: '+m['id'])
        rec=eligible[0];request_contract.check_endpoint(m,rec)
        prior=[v for v in old[stem+'.flight.txt'] if v.get('id')==m['observed_endpoint_id']]
        if not prior or any(v['pricing']!=rec['pricing'] for v in prior):raise Halt('Complete pricing schedule changed: '+m['id'])
        return m['id'],{'endpoint':rec,'url':response.url,'html_sha256':hashlib.sha256(response.content).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:routes=dict(pool.map(fetch,config['models']))
    save(out/'routes.json',routes)
    key=read_key();results={}
    for endpoint in ['key','credits']:
        response=requests.get('https://openrouter.ai/api/v1/'+endpoint,headers={'Authorization':'Bearer '+key},timeout=20,allow_redirects=False)
        if response.status_code!=200:raise Halt('Account preflight HTTP '+str(response.status_code))
        results[endpoint]=response.json()['data']
    k=results['key'];credits=results['credits'];remaining=Decimal(str(credits['total_credits']))-Decimal(str(credits['total_usage']))
    ready=k.get('is_free_tier') is False and remaining>=15 and (k.get('limit_remaining') is None or Decimal(str(k['limit_remaining']))>=15)
    if k.get('expires_at'):ready=ready and datetime.fromisoformat(k['expires_at'].replace('Z','+00:00'))>datetime.now(timezone.utc)
    save(out/'account_check.json',{'account_ready':ready,'key_http_status':200,'credits_http_status':200,'account_remaining_at_least_15_usd':remaining>=15,'key_limit_remaining_at_least_15_usd':k.get('limit_remaining') is None or Decimal(str(k['limit_remaining']))>=15,'inference_calls':0})
    if not ready:raise Halt('Account not ready')
    attest={'freeze_sha256':freeze.file_hash(base_lock),'account_ready':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'endpoints':{k:v['endpoint'] for k,v in routes.items()}}
    save(out/'ATTESTATION.json',attest);print('Fresh route, full pricing and account checks passed; no inference calls')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();check(a.base_lock,a.output)
