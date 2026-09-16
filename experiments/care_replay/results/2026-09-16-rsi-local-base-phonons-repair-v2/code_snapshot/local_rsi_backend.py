"""Local frozen-base / trained-adapter generation for the existing RSI runner.

Model weights remain frozen during each search. Heavy dependencies are imported
only when this backend is used, leaving API-backed RSI dependency requirements
unchanged. Invalid output is checked by the same compiler as API output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_client = None
_signature = None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class LocalRSIClient:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('Local QLoRA RSI requires CUDA')
        model_path = config['base_model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={'': 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        self.provenance = {
            'backend': 'local_qlora', 'base_model_path': model_path,
            'base_config_sha256': file_sha256(Path(model_path) / 'config.json'),
            'adapter_path': config.get('adapter_path'),
            'weights_frozen_during_search': True,
            'quantization': '4-bit NF4 double quantization',
            'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
            'gpu': torch.cuda.get_device_name(0),
        }
        if config.get('adapter_path'):
            from peft import PeftModel
            path = Path(config['adapter_path'])
            self.model = PeftModel.from_pretrained(self.model, path, is_trainable=False)
            self.provenance['adapter_sha256'] = file_sha256(path / 'adapter_model.safetensors')
        self.model.eval()
        self.model.config.use_cache = True
        self.context_limit = min(
            int(config.get('local_context_limit', 32768)),
            int(self.model.config.max_position_embeddings),
        )
        self.seed = int(config.get('local_generation_seed', 20260915))

    def complete(self, body):
        torch = self.torch
        text = self.tokenizer.apply_chat_template(
            body['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        prompt_tokens = encoded['input_ids'].shape[1]
        max_tokens = int(body['max_tokens'])
        if prompt_tokens + max_tokens > self.context_limit:
            raise RuntimeError('Local RSI context budget exceeded; refusing silent evidence truncation')
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        seed_hash = hashlib.sha256(json.dumps(body['messages'], sort_keys=True).encode()).hexdigest()
        seed = (self.seed + int(seed_hash[:8], 16)) % (2**31)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        temperature = float(body.get('temperature', 0))
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion = output[0, prompt_tokens:]
        content = self.tokenizer.decode(completion, skip_special_tokens=True)
        return {
            'model': body['model'],
            'choices': [{'message': {'role': 'assistant', 'content': content},
                         'finish_reason': 'length' if len(completion) >= max_tokens else 'stop'}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': len(completion),
                      'total_tokens': prompt_tokens + len(completion)},
            'local_provenance': {**self.provenance, 'generation_seed': seed,
                                 'schema_validation': 'Post-generation compiler, not constrained decoding'},
        }


def complete(config, body):
    global _client, _signature
    signature = (config['base_model_path'], config.get('adapter_path'),
                 config.get('local_context_limit'), config.get('local_generation_seed'))
    if _client is not None and _signature != signature:
        raise RuntimeError('Run each base/adapter condition in its own process')
    if _client is None:
        _client = LocalRSIClient(config)
        _signature = signature
    return _client.complete(body)
