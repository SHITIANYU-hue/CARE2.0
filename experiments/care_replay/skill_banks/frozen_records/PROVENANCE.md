# Frozen target-only LLM records

These five records are runtime inputs for the seven declared pairs in
`../../configs/source_outcome_benchmark.json`. They are retained outside the
historical results archives so the canonical suite can load its frozen
target-only controls without restoring old experiment outputs.

Source commit: `4c6f19f8cf62b25558f966604ce0b7644578f889`.

Each JSON file was extracted directly from the Git blob at the source commit.
File names and every byte are unchanged; no JSON fields were regenerated or edited.
Only the record paths in the benchmark configuration were changed.
Original paths below are relative to the repository root.

| Frozen file | Original path | SHA-256 | Bytes |
| --- | --- | --- | ---: |
| [materials_target_only.json](materials_target_only.json) | `experiments/care_replay/results/2026-07-21-llm-evidence-causality/model_calls/materials_target_only.json` | `a7354341538f4180993de974edbb5e7bfb80699d74d5525ac337837d4297f835` | 24816 |
| [materials_dielectric_shared_target_only.json](materials_dielectric_shared_target_only.json) | `experiments/care_replay/results/2026-07-23-source-evidence-extension/model_calls/materials_dielectric_shared_target_only.json` | `c0ed1f4bde84ded793ae056dcd643af757086437b9ef83d5a660fa3013bf9201` | 51365 |
| [molecular_esol_to_lipophilicity_target_only.json](molecular_esol_to_lipophilicity_target_only.json) | `experiments/care_replay/results/2026-07-23-source-evidence-extension/model_calls/molecular_esol_to_lipophilicity_target_only.json` | `969903d2dc7a3e13bd3789bf2074be4f91da319e2d6c15b540838f2090b5daf0` | 54708 |
| [molecular_lipophilicity_to_freesolv_target_only.json](molecular_lipophilicity_to_freesolv_target_only.json) | `experiments/care_replay/results/2026-07-23-source-evidence-extension/model_calls/molecular_lipophilicity_to_freesolv_target_only.json` | `25902f1b40d89942ec5e5f8b30fe6017408237f6d9312a707d5a462621ebb065` | 51002 |
| [reaction_chemlex_to_buchwald_hartwig_target_only.json](reaction_chemlex_to_buchwald_hartwig_target_only.json) | `experiments/care_replay/results/2026-07-23-source-evidence-extension/model_calls/reaction_chemlex_to_buchwald_hartwig_target_only.json` | `82f5e6867e0606d083abcdcab5837953d977a7e65a412a8582a7acf0a602349c` | 53183 |
