"""Immutable protocol verification; authorization is a separate future artifact."""
from .common import *
from datetime import datetime,timezone
import platform,importlib.metadata

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(lock_path):
    lock=load(lock_path)
    if lock.get('kind')!='nonrsi_cross_model_v1_formal_freeze' or not lock.get('offline_only'):raise Halt('Not an offline protocol freeze')
    for name,expected in lock['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_hash(path)!=expected:raise Halt('Frozen file changed: '+name)
    env=lock['environment']
    if platform.python_version()!=env['python']:raise Halt('Python version changed')
    for name,version in env['packages'].items():
        if importlib.metadata.version(name)!=version:raise Halt('Dependency version changed: '+name)
    config=load(ROOT/lock['config_path'])
    if digest(config)!=lock['config_sha256'] or not config['freeze_complete'] or not config.get('statistical_scope_resolved'):raise Halt('Config or statistical scope not frozen')
    if config['generation_enabled'] or config['api_calls_authorized']!=0:raise Halt('Freeze must retain zero API authorization')
    if config['statistics_proposed'].get('scope')!='descriptive_cross_model_replication' or config['statistics_proposed'].get('familywise_error_control_claim') is not False:raise Halt('Statistical scope changed')
    return config

def authorize(lock_path,authorization_path,attestation_path):
    config=verify(lock_path)
    auth=load(authorization_path)
    if auth.get('kind')!='explicit_user_execution_authorization' or auth.get('freeze_sha256')!=file_hash(lock_path) or auth.get('model_calls')!=660 or str(auth.get('model_spend_cap_usd'))!='15' or auth.get('allow_execution') is not True:raise Halt('Missing matching explicit execution authorization')
    if not auth.get('user_message_reference'):raise Halt('Human authorization reference missing')
    fresh=load(attestation_path)
    if fresh.get('freeze_sha256')!=file_hash(lock_path) or fresh.get('account_ready') is not True:raise Halt('Execution preflight missing')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(fresh['checked_at_utc'])).total_seconds()
    if not 0<=age<=86400:raise Halt('Route/price preflight stale')
    from .request_contract import check_endpoint
    for model in config['models']:check_endpoint(model,fresh['endpoints'][model['id']])
    return config
