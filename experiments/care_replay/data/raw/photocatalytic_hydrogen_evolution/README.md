# Photocatalytic hydrogen-evolution benchmark

`data.csv` is pinned from the public supplementary repository for:

> M. A. Cisse et al., "Can We Automate Scientific Reasoning in Closed-Loop
> Experiments using Large Language Models?", *Digital Discovery* (2026),
> DOI: 10.1039/D5DD00520E.

Upstream repository:
<https://github.com/Ablatif6c/llm-closed-loop-experiments>

Source file:
`hydrogen_evolution_problem/data.csv` at upstream commit
`a1bc3a821568dafe01c4f1b9ba03982229d9341f`.

Pinned SHA-256:
`c0ec0a6ddbe5b34c19dea8f6a9412d244f11e6583ead427af4bc785b7aeb022d`.

The table contains 1,109 rows, ten photocatalytic formulation variables, and
the measured hydrogen-evolution-rate target used to construct the public
closed-loop benchmark. CARE uses the CSV as a finite-pool replay and does not
load the upstream `model.pkl` file.

The upstream repository marks the material as CC BY-NC 4.0. See
`LICENSE.txt` in this directory and the upstream repository for attribution
and use restrictions.
