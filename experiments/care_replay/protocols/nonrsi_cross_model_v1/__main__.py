"""All production entry points verify the future formal freeze first."""
import argparse
from .common import *
from . import freeze,generation,pipeline,reporting,transport

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['verify','generate','prepare','calibrate','heldout','analyze'])
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--ledger',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--authorization',type=Path)
    parser.add_argument('--attestation',type=Path)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();config=freeze.verify(args.lock)
    if args.phase=='verify':print('Formal freeze verified; no API authorization implied');return
    if args.ledger is None:parser.error('--ledger is required')
    if args.phase=='generate':
        if args.authorization is None or args.attestation is None:parser.error('Explicit future authorization and fresh route attestation are required')
        transport.run(args.lock,args.authorization,args.attestation,args.ledger);return
    if args.output is None:parser.error('--output is required')
    ledger=generation.Ledger(args.ledger,config);attempts=ledger.complete()
    with offline():
        if args.phase=='prepare':pipeline.prepare(config,attempts,args.output);return
        definition=load(args.output/'definition.json')
        if definition['generation_sha256']!=digest(attempts):raise Halt('Generation inventory changed')
        if args.phase=='calibrate':pipeline.calibrate(config,definition,args.output,args.workers)
        elif args.phase=='heldout':pipeline.heldout(config,definition,args.output,args.workers)
        elif args.phase=='analyze':
            lock=load(args.output/'CALIBRATION_LOCK.json');complete=load(args.output/'HELDOUT_COMPLETE.json')
            if complete['config_sha256']!=digest(config) or complete['definition_sha256']!=digest(definition) or complete['calibration_lock_sha256']!=digest(lock):raise Halt('Heldout completion binding changed')
            rows={t:load(args.output/t/'heldout_metrics.json') for t in definition['targets']}
            if {t:digest(r) for t,r in rows.items()}!=complete['metrics']:raise Halt('Heldout results changed')
            reporting.analyze(config,definition,lock,rows,args.output)
            cost={}
            for i,a in enumerate(attempts):
                m=a['slot']['model'];item=cost.setdefault(m,{'attempts':0,'valid':0,'invalid':0,'model_cost_usd':'0','input_tokens':0,'output_tokens':0})
                usage=load(args.ledger/f'{i:04d}.response.json')['usage'];item['attempts']+=1;item['valid' if a['status']=='VALID' else 'invalid']+=1
                item['model_cost_usd']=str(generation.money(item['model_cost_usd'])+generation.money(a['cost_usd']))
                item['input_tokens']+=usage['prompt_tokens'];item['output_tokens']+=usage['completion_tokens']
            pipeline.once(args.output/'generation_costs.json',cost)

if __name__=='__main__':main()
