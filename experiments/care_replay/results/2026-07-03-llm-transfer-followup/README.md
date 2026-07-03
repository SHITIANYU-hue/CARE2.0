# LLM Transfer Follow-up

Date: 2026-07-03

This snapshot follows the 50-seed transfer advantage sweep with real
CommonStack LLM calls. The goal is to separate three questions:

1. Can an LLM proposer produce usable transfer adjustments from the transfer
   card and revealed target evidence?
2. Can an LLM auditor reduce unsafe transfer without destroying all useful
   signal?
3. Does the LLM path improve the same positive transfer directions found by
   the deterministic sweep?

## Runs

Both runs used 10 seeds, 5 initial observations, and 10 replay rounds.

| Pair | Source observations | Modes |
| --- | ---: | --- |
| FreeSolv -> Lipophilicity | 192 | `no_care_random`, `incumbent`, `transfer_value_prior_gate_v1`, `llm_transfer_gate_v1`, `llm_audit_transfer_gate_v1` |
| Suzuki-Miyaura -> Buchwald-Hartwig | 96 | `no_care_random`, `incumbent`, `transfer_gate_v1`, `llm_transfer_gate_v1`, `llm_audit_transfer_gate_v1` |

The LLM model was `openai/gpt-4o-mini` through the OpenAI-compatible
CommonStack endpoint. The API key is not stored in the repository.

## Results

### FreeSolv -> Lipophilicity

| Mode | Final best | Delta vs incumbent | AUC | AUC delta | Top10 hit | Total LLM calls | Parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 84.8625 | -1.7875 | 84.0900 | -0.8200 | 0.0000 | 0 | 0 |
| `incumbent` | 86.6500 | 0.0000 | 84.9100 | 0.0000 | 0.0000 | 0 | 0 |
| `transfer_value_prior_gate_v1` | 87.2000 | +0.5500 | 85.4838 | +0.5738 | 0.1000 | 0 | 0 |
| `llm_transfer_gate_v1` | 87.1750 | +0.5250 | 85.6050 | +0.6950 | 0.1000 | 70 | 0 |
| `llm_audit_transfer_gate_v1` | 86.6500 | 0.0000 | 84.9100 | 0.0000 | 0.0000 | 9 | 0 |

This is the positive LLM result. The LLM proposer is close to the
deterministic value-prior transfer mode and improves both final best and AUC
over the incumbent. It also raises top10 hit from 0 to 10% in this 10-seed
follow-up. The LLM auditor is much more conservative and mostly collapses back
to incumbent behavior.

### Suzuki-Miyaura -> Buchwald-Hartwig

| Mode | Final best | Delta vs incumbent | AUC | AUC delta | Top10 hit | Total LLM calls | Parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_care_random` | 81.8903 | -4.7357 | 78.5358 | -3.4488 | 0.0000 | 0 | 0 |
| `incumbent` | 86.6260 | 0.0000 | 81.9846 | 0.0000 | 0.0000 | 0 | 0 |
| `transfer_gate_v1` | 90.0980 | +3.4720 | 82.9136 | +0.9290 | 0.2000 | 0 | 0 |
| `llm_transfer_gate_v1` | 85.6227 | -1.0033 | 81.4474 | -0.5372 | 0.0000 | 70 | 0 |
| `llm_audit_transfer_gate_v1` | 86.6260 | 0.0000 | 81.9179 | -0.0667 | 0.0000 | 28 | 0 |

Here the deterministic transfer card remains the strongest policy. The LLM
proposer parses cleanly but does not match the hand-coded transfer rule on this
reaction HTE direction. The LLM auditor is conservative: it prevents the
strong transfer policy from taking effect and returns roughly to incumbent.

## Interpretation

The useful takeaway is not "LLM always wins." It is more specific:

- In shared descriptor spaces, an LLM proposer can read the transfer card and
  target evidence well enough to produce a small positive transfer gain.
- In reaction HTE transfer, the deterministic role-level transfer rule is still
  stronger than the current LLM proposer.
- The current LLM audit prompt is too conservative for advantage-seeking
  transfer. It may be useful as a safety check, but not as the main policy.
- All real LLM calls in this follow-up parsed successfully, so the next issue
  is policy quality, not API plumbing.

## Files

- `llm_transfer_followup_summary.csv`: compact aggregate comparison.
- `runs/*_summary.json`: canonical summary JSON copied from `outputs/runs`.
- `tables/*_metrics.csv`: per-seed metric tables copied from `outputs/tables`.

Full per-seed audit logs, knowledge snapshots, and transfer cards are tracked
under `../../outputs/runs/` using the same output tags:

- `llm_transfer_freesolv_to_lipo_10seed`
- `llm_transfer_suzuki_to_bh_10seed_v2`
