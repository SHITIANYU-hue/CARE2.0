# Zero-shot Suzuki -> Buchwald-Hartwig

This is a frozen-record, no-target-calibration replay. The source record is
`outputs/llm_semantic/suzuki_to_bh_semantic_record.json`; every LLM skill and
three matched random-rule replicates were evaluated on the same 30 target
seeds. No skill was selected after seeing target outcomes.

## Protocol

- Target calibration seeds: `0`
- Pre-decision target outcomes: `0`
- Online target budget: `5` initial observations + `10` rounds per seed
- Baselines: GP-UCB and mixed-kernel GP-EI
- Audits: `660` JSON records in `audits.jsonl`

## Main signal

The strongest frozen skill was `high_mw_ligand_effect`:

| Comparison | Final best | AUC | Composite |
| --- | ---: | ---: | ---: |
| vs GP-UCB | +2.3813 (95% CI [-0.1614, +4.9240]) | +5.6299 (CI [+2.3568, +8.9030]) | +8.0112 (CI [+2.8732, +13.1492]) |
| vs mixed-kernel GP-EI | +2.6003 (CI [-0.0822, +5.2829]) | +4.4146 (CI [+1.4128, +7.4165]) | +7.0149 (CI [+1.6881, +12.3418]) |

The composite and AUC signals are positive, but Final Best is not decisive
and its paired win rate is only `43.3%`. Other LLM skills are mostly negative.
The matched random nulls do not reproduce this specific skill's positive
composite, but this is one reaction pair, not cross-domain generalization.

The raw files are `summary.json`, `metrics.csv`, and `audits.jsonl`.
