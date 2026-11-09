"""Recognize one explicit provider network-error envelope, never partial rules."""
from protocols.nonrsi_cross_model_v1.common import Halt,digest
from protocols.nonrsi_cross_model_v1.generation import money
def evidence(raw,model):
 if not isinstance(raw,dict):return None
 choices=raw.get('choices')
 if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):return None
 c=choices[0];error=c.get('error')
 if not (isinstance(error,dict) and type(error.get('code')) is int and error['code']==502 and error.get('message')=='Network connection lost.' and error.get('metadata')=={'error_type':'provider_unavailable'} and c.get('finish_reason')=='error'):return None
 if raw.get('object')!='chat.completion' or not isinstance(raw.get('id'),str) or not raw['id'] or raw.get('error'):raise Halt('Network response envelope invalid')
 if model['id']!='qwen3_instruct_2507' or raw.get('model') not in model['allowed_response_models'] or raw.get('provider')!='Parasail':raise Halt('Network response identity mismatch')
 msg=c.get('message')
 if not isinstance(msg,dict) or msg.get('role')!='assistant' or not isinstance(msg.get('content'),str):raise Halt('Network response message invalid')
 if any(msg.get(k) for k in ['tool_calls','function_call','refusal','reasoning']):raise Halt('Unexpected network response content type')
 u=raw.get('usage')
 if not isinstance(u,dict) or u.get('is_byok') is not False:raise Halt('Network response billing provenance missing')
 cost=money(u.get('cost'))
 if cost>1:raise Halt('Network response reservation exceeded')
 for key,maximum in [('prompt_tokens',12000),('completion_tokens',5000)]:
  if type(u.get(key)) is not int or not 0<=u[key]<=maximum:raise Halt('Network response token bound')
 details=u.get('completion_tokens_details',{})
 if not isinstance(details,dict) or details.get('reasoning_tokens',0)!=0:raise Halt('Network response reasoning evidence')
 fp=raw.get('system_fingerprint')
 if fp is not None and not isinstance(fp,str):raise Halt('Network response fingerprint malformed')
 return {'exception_type':'ProviderResponseNetworkError','http_status':200,'transport_phase':'response_received','response_received':True,'provider_name':'Parasail','error':error,'finish_reason':'error','response_id':raw['id'],'response_sha256':digest(raw),'fingerprint':fp,'reported_cost_usd':str(cost),'unknown_charge_reservation_retained':True,'partial_content_not_used':True}
def qualifies(slot,e):
 return (slot['model']=='qwen3_instruct_2507' and e.get('exception_type')=='ProviderResponseNetworkError' and e.get('http_status')==200 and e.get('transport_phase')=='response_received' and e.get('response_received') is True and e.get('provider_name')=='Parasail' and e.get('finish_reason')=='error' and e.get('error')=={'code':502,'message':'Network connection lost.','metadata':{'error_type':'provider_unavailable'}} and isinstance(e.get('response_id'),str) and bool(e['response_id']) and isinstance(e.get('response_sha256'),str) and e.get('partial_content_not_used') is True)
