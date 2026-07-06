# GPT LLM Transfer Descriptor Follow-up

Date: 2026-07-05

This run uses the same replay prompts and default 500-token LLM cap, with OpenAI-compatible structured output enabled at the API layer.

## Takeaways

- GPT fixed the structured-output issue seen with `moonshotai/kimi-k2.7-code`: all 778 GPT calls returned via tool-call JSON, with 0 parse errors.
- The run completed both formal directions: BH->Suzuki with 30 seeds and Suzuki->BH with 50 seeds.
- LLM modes were format-valid but not materially better than `incumbent`. BH->Suzuki produced 0 LLM interventions; Suzuki->BH produced only 0.02 mean interventions for `llm_transfer_gate_v1`, with no aggregate metric gain.
- Descriptor transfer still intervenes much more often, but remains risky in Suzuki->BH because bad interventions are high.

## Summary

| direction | mode | final_best | AUC | top10_hit | interventions | bad_interventions | llm_calls | parse_errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BH->Suzuki | incumbent | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| BH->Suzuki | transfer_gate_v1 | 91.3879 | 86.4187 | 0.1667 | 3.2000 | 1.6333 | 0.0000 | 0.0000 |
| BH->Suzuki | transfer_strict_gate_v1 | 92.8770 | 87.7002 | 0.1333 | 1.4333 | 0.8333 | 0.0000 | 0.0000 |
| BH->Suzuki | transfer_descriptor_value_prior_gate_v1 | 92.5510 | 88.1918 | 0.0667 | 7.0000 | 4.1667 | 0.0000 | 0.0000 |
| BH->Suzuki | transfer_descriptor_value_prior_strict_gate_v1 | 92.0322 | 88.3821 | 0.1000 | 7.4333 | 4.1000 | 0.0000 | 0.0000 |
| BH->Suzuki | llm_transfer_gate_v1 | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 | 7.0000 | 0.0000 |
| BH->Suzuki | llm_audit_transfer_gate_v1 | 92.6085 | 87.5533 | 0.1000 | 0.0000 | 0.0000 | 3.3667 | 0.0000 |
| Suzuki->BH | incumbent | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Suzuki->BH | transfer_gate_v1 | 89.0866 | 81.0779 | 0.2400 | 2.0800 | 0.8600 | 0.0000 | 0.0000 |
| Suzuki->BH | transfer_strict_gate_v1 | 87.6158 | 80.2528 | 0.1600 | 0.6200 | 0.1800 | 0.0000 | 0.0000 |
| Suzuki->BH | transfer_descriptor_value_prior_gate_v1 | 75.8008 | 72.3470 | 0.1000 | 6.5000 | 4.8400 | 0.0000 | 0.0000 |
| Suzuki->BH | transfer_descriptor_value_prior_strict_gate_v1 | 73.3883 | 70.5371 | 0.1000 | 6.1200 | 4.8000 | 0.0000 | 0.0000 |
| Suzuki->BH | llm_transfer_gate_v1 | 86.6477 | 79.9713 | 0.1600 | 0.0200 | 0.0000 | 7.0000 | 0.0000 |
| Suzuki->BH | llm_audit_transfer_gate_v1 | 86.6477 | 79.9713 | 0.1600 | 0.0000 | 0.0000 | 2.3400 | 0.0000 |

## Files

- `transfer_summary.csv`: aggregate metrics across both directions.
- `tables/`: seed-level metrics.
- `runs/`: summary JSON, audit JSONL, knowledge JSON, and transfer-card JSON artifacts.
- `logs/`: terminal logs and `gpt_llm_calls.jsonl` per-call LLM tracing.

## LLM Trace

`logs/gpt_llm_calls.jsonl` records call timing, model, token usage, retry events, and tool-call counts. It does not store API keys, prompts, or raw LLM responses.
