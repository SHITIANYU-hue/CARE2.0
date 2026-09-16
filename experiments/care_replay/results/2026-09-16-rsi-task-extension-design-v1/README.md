# Paired post-training task extension

Frozen protocol: Qwen2.5-32B base versus the already selected epoch-2 QLoRA adapter. Tasks: Lipophilicity from ESOL and Matbench phonons from dielectric. Each has one greedy generation chain, two revisions, ten evaluation seeds, five initial observations, and ten reveals. Controls are fixed initial policy, no feedback, shuffled feedback, and GP-UCB.

The 54 SFT rows include 18 Lipophilicity target rows and no phonons target/source rows. Dielectric source evidence and related materials schemas appeared in band-gap SFT; phonons is a previously untrained target, not an independent domain. Evaluation seeds 202616200–202616209 are excluded from development prompts and checkpoint selection. Development uses 202616100–202616103.

Run both conditions sequentially on a free A800 to avoid other active jobs. Preserve every request, response, validation failure, development trace, evaluation trajectory, code hash and deployment lock. Complete all cases before comparisons. Do not select a checkpoint or revise policies using evaluation results. No new wet-lab data are produced.

## Audits after run start

The Lipophilicity initial prompt exactly matches five training rows; it is a training-task / repeated-input diagnostic, not a prompt-disjoint confirmation. The phonons initial prompt has no exact or JSON-normalized SFT input match. Every request will be audited after both runs complete.

An exact adapter-composition-string audit finds 112 of the 1,265 phonon candidates share a composition with the 512 source observations. This does not establish identical crystal structures, and the source property is refractive index, not the target phonon measurement. Source and target are not compound-disjoint. See `composition_overlap_audit.json`.
