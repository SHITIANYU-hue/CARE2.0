# Reizman Suzuki reaction tasks

The four CSV files in this directory come from the public repository for
Taylor et al., *Accelerated Chemical Reaction Optimization Using Multi-Task
Learning* (ACS Central Science, 2023):

- repository: https://github.com/sustainable-processes/multitask
- pinned source commit: `99cd89b378cd8d7a624f22ae138fb6f0a85ef440`
- original path: `data/reizman_suzuki/`
- license: MIT, copyright 2023 University of Cambridge

CARE 2.0 uses cases 1-3 only as completed source experiments and freezes case
4 as the unseen target task. The public decision variables are catalyst,
residence time, temperature, and catalyst loading. `yld` is revealed only
after a candidate has been selected by the replay policy.
