# LLM-audited BH to Suzuki transfer replay

Date: 2026-07-03

This run tests whether a real LLM can act as an auditor on top of the CARE 2.0 cross-domain transfer gate.

The source domain is real Buchwald-Hartwig HTE. The target domain is real Suzuki-Miyaura HTE. For each seed, the source side exposes 48 observations to build a role-level transfer card, then the target side runs 10 replay rounds after 5 initial target observations. All experiment execution was done on the GPU server, not on the laptop.

## Modes

- `no_care_random`: random no-CARE baseline.
- `incumbent`: target-only CARE incumbent policy.
- `transfer_gate_v1`: target policy with transferred role-level score adjustments.
- `transfer_strict_gate_v1`: transfer gate with stricter role support checks.
- `llm_audit_transfer_gate_v1`: `transfer_gate_v1` proposes a challenger, then a real LLM audits whether to approve it.
- `llm_audit_transfer_strict_gate_v1`: strict transfer gate proposes a challenger, then a real LLM audits whether to approve it.

The LLM auditor uses `openai/gpt-4o-mini` through a CommonStack-compatible endpoint. The prompt receives only revealed target evidence, public candidate features, the source-to-target transfer card, and the deterministic gate proposal. Hidden target outcomes for unobserved candidates are not included.

This is the v2 prompt. A first smoke version used an overly suggestive approval-shaped JSON example; that made the auditor approve every proposed challenger. The v2 prompt removes that example and tells the auditor to reject when evidence is close, sparse, or mostly inherited from broad transfer priors.

## Result summary

| mode | final_best mean | best_so_far_auc mean | bad intervention mean | LLM calls mean |
| --- | ---: | ---: | ---: | ---: |
| `no_care_random` | 88.0639 | 81.3556 | 0.0 | 0.0 |
| `incumbent` | 89.7358 | 86.6963 | 0.0 | 0.0 |
| `transfer_gate_v1` | 87.6407 | 85.1042 | 1.4 | 0.0 |
| `transfer_strict_gate_v1` | 89.7358 | 86.6963 | 0.4 | 0.0 |
| `llm_audit_transfer_gate_v1` | 89.7358 | 86.6963 | 0.0 | 3.2 |
| `llm_audit_transfer_strict_gate_v1` | 89.7358 | 86.6963 | 0.0 | 0.6 |

## What this says

The cross-domain transfer interface works end to end: source-side evidence is converted into a transfer card, the target policy applies it, the gate emits certificates, and the LLM auditor is called in the replay loop with parseable JSON responses. There were no LLM parse errors in this run.

The scientific result is conservative. Plain transfer is not yet better than the target-only incumbent on this 5-seed BH to Suzuki replay. It also introduces more bad interventions. The strict transfer gate avoids most of that damage and recovers incumbent-level performance, but does not create a clear gain.

The LLM auditor works as a safety filter in this version. It rejects all challengers that reach it: 16/16 for the ordinary transfer gate and 3/3 for the strict transfer gate. That removes the bad transfer interventions and brings the ordinary transfer mode back to the incumbent baseline. The tradeoff is that it is too conservative: the aggregate `rejected_good_challenger_count` is 2.0 for ordinary LLM audit and 0.2 for strict LLM audit. So the current LLM-audit version is useful for preventing negative transfer, but it does not yet unlock positive cross-domain transfer.

The next useful step is a middle path: keep the v2 rejection discipline, but give the auditor a more explicit checklist for when a transfer challenger is actually worth trying. A good follow-up is to require JSON evidence fields such as matched positive roles, matched negative roles, incumbent-vs-challenger support, and uncertainty reason, then approve only when those fields pass a deterministic rule.

## Files

- `server_llm_audit_transfer_5seed_v2_metrics.csv`: per-mode, per-seed metrics copied from the canonical output table.
- `transfer_summary.csv`: compact aggregate table for discussion.
- `../../outputs/runs/*server_llm_audit_transfer_5seed_v2*`: raw audit logs, transfer cards, and per-seed knowledge snapshots.
- `../../outputs/tables/transfer_real_buchwald_hartwig_to_real_suzuki_miyaura_server_llm_audit_transfer_5seed_v2_metrics.csv`: canonical metrics table.
