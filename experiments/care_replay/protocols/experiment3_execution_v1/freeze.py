"""Formal execution freeze; must run after offline preflight, before effects access."""
from core import *
import stat,datetime,platform,subprocess,tarfile

def main():
    if (HERE/'FREEZE.json').exists():
        print('Already frozen: '+verify_freeze());return
    preflight=json.loads((OUT/'preflight.json').read_text())
    if preflight['status']!='passed' or preflight['real_effects_opened']:raise RuntimeError('Invalid preflight')
    raw=BASE/'outputs/experiment3_recruitment_v1/sealed/DMS_ProteinGym_substitutions.zip'
    if stat.S_IMODE(raw.stat().st_mode)!=0:raise RuntimeError('Effects not sealed')
    old=json.loads((BASE/'outputs/experiment3_gate_preparation_v1/input_manifest.json').read_text());verify_mapping(old,REPO)
    public=json.loads((BASE/'protocols/experiment3_public_preparation_v1/lock.json').read_text());verify_mapping(public['artifacts'],BASE)
    artifacts=dict(public['artifacts'])
    # All local Python dependency sources and current protocol files are locked.
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.md'))+list((BASE/'scripts').glob('*.py'))
    files += [OUT/'preflight.json',OUT/'offline_tests.txt',BASE/'protocols/experiment3_public_preparation_v1/lock.json',BASE/'protocols/experiment3_public_preparation_v1/lock.json.sha256',BASE/'outputs/experiment3_gate_preparation_v1/input_manifest.json']
    for p in files:artifacts[str(p.relative_to(BASE))]=digest(p)
    record={'status':'execution_frozen_before_effects','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'python':sys.version,'numpy':np.__version__,'artifacts':artifacts,'raw_archive_expected_sha256':config()['raw_zip_sha256'],
        'approved_score_transform':json.loads((HERE/'approved_score_units_amendment.json').read_text()),
        'routes':23,'candidate_dependency_groups':13,'tests_passed':preflight['tests'],'api_calls_allowed':0,
        'execution_authorized_after_this_freeze':True,'all_gates_before_heldout':True,'preexisting_214_inputs_unchanged':True}
    write_once(HERE/'FREEZE.json',record);(HERE/'FREEZE.sha256').write_text(digest(HERE/'FREEZE.json')+'\n')
    freeze_sha=verify_freeze()
    archive=OUT/'execution_freeze_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for rel in sorted(artifacts):tar.add(BASE/rel,arcname=rel)
        for name in ('FREEZE.json','FREEZE.sha256'):tar.add(HERE/name,arcname=str((HERE/name).relative_to(BASE)))
    (OUT/'execution_freeze_bundle.sha256').write_text(digest(archive)+'\n')
    print(json.dumps({'status':record['status'],'freeze_sha256':freeze_sha,'artifacts':len(artifacts),'sealed_raw_mode':oct(stat.S_IMODE(raw.stat().st_mode)),'api_calls_allowed':0},indent=2))

if __name__=='__main__':main()
