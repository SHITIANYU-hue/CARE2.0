# Hidden-target non-interference audit

All 170/170 archived online decision states passed exact hidden-label non-interference.

| Domain | Route | States | Changed hidden labels | Result |
| --- | --- | ---: | ---: | --- |
| molecular_property | molecular_freesolv_to_esol | 10/10 | 11190 | PASS |
| molecular_property | molecular_freesolv_to_lipophilicity | 10/10 | 41765 | PASS |
| materials_property | materials_expt_gap_to_dielectric | 10/10 | 47550 | PASS |
| materials_property | materials_phonons_to_bulk_modulus | 10/10 | 109285 | PASS |
| materials_property | materials_dielectric_to_jdft2d | 10/10 | 6276 | PASS |
| materials_property | materials_expt_gap_to_mp_gap | 10/10 | 860039 | PASS |
| reaction_optimization_cn | baumgartner_aniline_to_phenethylamine_alphos | 10/10 | 151 | PASS |
| reaction_optimization_cn | baumgartner_aniline_to_benzamide_tbuxphos | 10/10 | 236 | PASS |
| reaction_optimization_cn | baumgartner_aniline_to_phenethylamine_tbubrettphos | 10/10 | 259 | PASS |
| reaction_optimization_cn | baumgartner_benzamide_tbubrettphos_to_alphos | 10/10 | 73 | PASS |
| reaction_optimization_suzuki | reizman_cases_123_to_case4 | 10/10 | 887 | PASS |
| molecular_property | molecular_lipophilicity_to_freesolv | 10/10 | 6335 | PASS |
| materials_property | materials_bulk_to_shear_modulus | 10/10 | 108619 | PASS |
| materials_property | materials_phonons_to_perovskites | 10/10 | 187569 | PASS |
| materials_property | materials_expt_gap_to_steels | 10/10 | 3034 | PASS |
| molecular_property | molecular_freesolv_to_bace | 10/10 | 14987 | PASS |
| materials_property | materials_perovskites_to_mp_e_form | 10/10 | 1327434 | PASS |

For each state, all outcomes not yet revealed to the controller were permuted while the revealed history and public candidate attributes were held fixed. The production code then rebuilt the source prior, candidate menu, diagnostics, and full LLM prompt. Exact equality is required.

Archived prompts are separately checked against their recorded SHA-256. Exact byte-for-byte reconstruction is reported as a version-drift diagnostic, not used as the non-interference endpoint. All archived states predate the added public outcome-semantics field: 110 reproduce after removing only that field, while 60 states from six earlier routes also predate the current eligibility and safety fields.

This is an implementation-level audit over archived decision states. It supports the stated information boundary but does not establish scientific reasoning quality, transfer efficacy, or prospective generalization.
