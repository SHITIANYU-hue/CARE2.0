# Awesome Repository Notes for CARE 2.0

This note records public resources that can feed CARE 2.0 as knowledge cards,
benchmark adapters, or agent design references.

## Directly Useful for CARE Replay

### rxn4chemistry/rxn_yields

Status: already connected.

Use:

- Dreher-Doyle Buchwald-Hartwig HTE workbook.
- Perera Suzuki-Miyaura HTE workbook.

Why it fits CARE:

- Each row is a real measured experiment.
- Reaction conditions are discrete candidates.
- Yield can be treated as the hidden target during replay.

Current adapters:

- `real_buchwald_hartwig`
- `real_suzuki_miyaura`

## Useful Dataset Leads from the Awesome Repositories

### sherrylixuecheng/awesome-ai4chem

Useful sections:

- Generalized Model/Datasets.
- Reaction design/Reactivity.
- Robotic chemist/Automation.

Candidate datasets to evaluate next:

- Open Reaction Database: likely useful for reaction knowledge extraction and
  possibly yield replay if suitable measured-yield subsets are selected.
- QM9/QM7/QM7b: useful for molecular property optimization, but not a direct
  experimental replay unless framed as finite-pool property search.
- HOPV15 and related photovoltaic/materials datasets: useful for materials-like
  finite-pool property optimization.

### REAL-Lab-NU/Awesome-LLM-Centric-Molecular-Discovery

Useful sections:

- Datasets table for molecule generation, optimization, docking, and property
  prediction.
- Agent and benchmark papers for molecular discovery workflows.

Candidate datasets to evaluate next:

- Dockstring: practical finite-pool virtual-screening objective.
- MoleculeNet: property-prediction datasets that can be reframed as finite-pool
  optimization. ESOL/Delaney is now connected as `real_moleculenet_esol`.
- QM9: molecular property search with clean tabular targets.
- ChEMBL/PubChem/ZINC: too large for immediate replay; better as retrieval or
  pretraining corpora unless downsampled.

### PEESEgroup/Awesome-Materials-Aware-Large-Language-Models

Useful sections:

- Data extraction.
- Property prediction.
- Synthesis planning.
- Agent-driven laboratory.

Candidate use:

- Not primarily a dataset repo.
- Best used to populate the CARE knowledge base with materials-aware agent
  patterns, extraction methods, and literature-to-structured-data workflows.
- For runnable datasets, follow the linked papers/projects such as LLaMP,
  MatKG, MaterialsBERT, PolyBERT, and related materials benchmarks.

### Thinklab-SJTU/Awesome-LLM4EDA

Useful sections:

- Verification loops.
- Benchmarking generated artifacts.
- Agentic design and review workflows.

Candidate use:

- Not a chemistry/materials dataset source.
- Useful as an analogy for CARE's gate: generated proposal, deterministic
  checks, simulator/verifier, audit trail, and rejection on failed evidence.

## Practical Priority

1. Keep the connected real HTE datasets as the first real replay baseline.
2. Use the connected MoleculeNet ESOL adapter as the first molecular property
   replay baseline.
3. Add Dockstring for discovery-style docking optimization.
4. Add QM9 or a Matbench/materials property dataset for cross-domain materials
   search.
5. Use Open Reaction Database as a knowledge source first; only use it as replay
   data after selecting a clean measured-yield subset.
6. Add papers/projects from the materials and molecular discovery lists as
   knowledge cards for retrieval and skill generation.
