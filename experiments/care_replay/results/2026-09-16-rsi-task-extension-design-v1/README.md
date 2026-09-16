# Paired post-training task extension

Frozen protocol: Qwen2.5-32B base versus the already selected epoch-2 QLoRA adapter. Tasks: Lipophilicity from ESOL and Matbench phonons from dielectric. Each has one greedy generation chain, two revisions, ten evaluation seeds, five initial observations, and ten reveals. Controls are fixed initial policy, no feedback, shuffled feedback, and GP-UCB.

The 54 SFT rows include 18 Lipophilicity target rows and no phonons target/source rows. Dielectric source evidence and related materials schemas appeared in band-gap SFT; phonons is a previously untrained target, not an independent domain. Evaluation seeds 202616200–202616209 are excluded from development prompts and checkpoint selection. Development uses 202616100–202616103.

Run both conditions sequentially on a free A800 to avoid other active jobs. Preserve every request, response, validation failure, development trace, evaluation trajectory, code hash and deployment lock. Complete all cases before comparisons. Do not select a checkpoint or revise policies using evaluation results. No new wet-lab data are produced.
