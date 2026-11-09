from pathlib import Path
import sys, os, json, hashlib, tempfile
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/care_replay'
sys.path.insert(0,str(BASE/'scripts'))
import run_matched_random_rule_controls as matched
import paired_semantic_anonymization as contract
class Halt(RuntimeError): pass
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path): return contract.parse_strict_json(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def slots(config):
    import random
    out=[]
    for replicate in range(10):
        block=[dict(c,replicate=replicate,slot_id=f"{c['model']}__{c['cell']}__{replicate:02d}") for c in config['generation_cells']]
        random.Random(config['generation_order_seed']+replicate).shuffle(block);out+=block
    if len(out)!=660 or len({r['slot_id'] for r in out})!=660:raise Halt('Slot inventory mismatch')
    return out
@contextmanager
def offline():
    with matched.offline_guard():yield
