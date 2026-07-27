# Materials semantic-skill replication and router audit

Date: 2026-07-27

This archive tests the frozen dielectric -> experimental band-gap record on
new seed ranges. It contains both a single-skill control and an automatic
frozen-skill router run.

## Automatic router run

- Calibration: 10 seeds (`86000-86009`).
- Held out: 30 seeds (`87000-87029`).
- Target: real Matbench experimental band gap.
- Source: real Matbench dielectric.
- Candidate skills: all 8 skills in the frozen `gpt-4o-mini` record.
- Baselines: GP-UCB, mixed-kernel GP-EI, target acquisition portfolio.
- Replay budget: 5 initial observations plus 10 reveal rounds.

Calibration selected the target acquisition portfolio as the anchor and
rejected every LLM semantic skill. The held-out CARE strategy router therefore
deployed target-only and exactly matched the target acquisition portfolio. No
LLM skill was promoted, and no positive LLM-transfer claim is made from this
run.

## Single-skill control

The `counter_transition_metal` skill was evaluated separately on calibration
seeds `84000-84009` and held-out seeds `85000-85029`. On held-out seeds it was
numerically above the target acquisition portfolio by Final best `+2.3375`,
AUC `+0.9738`, and top-10 hit `+0.0333`, but all confidence intervals crossed
zero. The calibration gate rejected it and the strategy router fell back to
GP-UCB. This is an uncertain/negative control, not a confirmed gain.

The result is useful because it shows the intended behavior of the router:
material skills that look promising on a small calibration sample are not
automatically deployed without fold stability and risk-adjusted evidence.

## Larger router check

To check whether the 10-seed calibration was simply too small, the full frozen
skill set was rerun with 30 calibration seeds (`88000-88029`) and 50 held-out
seeds (`89000-89049`). Calibration selected `counter_transition_metal`. On the
held-out seeds its strategy-router delta versus mixed-kernel GP-EI was Final
best `+5.11` (95% CI `[-3.78, +14.00]`), AUC `+3.42`
(`[-2.65, +9.49]`), and top-10 hit `+0.08` (`[-0.05, +0.21]`). The point
estimate is positive, but none of the intervals excludes zero. This is an
exploratory candidate-positive result, not a confirmed material transfer win.

The larger run is stored under `router_30x50/` and should be read together
with the existing 500-seed material confirmation rather than replacing it.

## 30-seed calibration and 100-seed held-out check

The same frozen router was then evaluated with 30 calibration seeds
(`90000-90029`) and 100 held-out seeds (`91000-91099`). Calibration again
selected `counter_transition_metal`. Against the strongest target-only
mixed-kernel GP-EI baseline, the held-out strategy-router deltas were:

| Metric | Delta | 95% CI |
| --- | ---: | ---: |
| Final best | `+6.52` | `[+0.13, +12.91]` |
| Best-so-far AUC | `+2.59` | `[-1.60, +6.77]` |
| Top-10 hit | `+0.16` | `[+0.05, +0.25]` |

This is stronger than the 30/50 point estimate and gives positive evidence
for final best and top-10 discovery in this material pair. AUC remains
inconclusive, and this is still one source-target pair; it should not be
reported as a universal cross-domain result. The raw metrics, frozen model
record, per-seed audit archive, and SHA256 manifest are under `router_30x100/`.

## Reproduction

The two subdirectories contain per-seed metrics, selection summaries, frozen
LLM records, compressed audit traces, and local SHA256 manifests. Replay uses
zero new LLM calls; the generation record is frozen before evaluation.
