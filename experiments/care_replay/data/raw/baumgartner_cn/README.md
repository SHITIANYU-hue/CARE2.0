# Baumgartner C-N reaction campaigns

`op9b00236_si_002.xlsx` is the public supplementary workbook used by the
multi-task reaction-optimization repository:

- repository: https://github.com/sustainable-processes/multitask
- pinned source commit: `99cd89b378cd8d7a624f22ae138fb6f0a85ef440`
- original path: `data/baumgartner_cn/op9b00236_si_002.xlsx`
- SHA-256: `a99cb5a29732f043055dacd115eef7af82c5fae8225cfde200677daf229bd969`
- repository license: MIT, copyright 2023 University of Cambridge

CARE 2.0 treats each named `Optimization` campaign as a task. Public candidate
features are base identity, base equivalents, temperature, residence time, and
precatalyst loading. Reaction yield remains hidden until a candidate is selected.

The frozen warm-start benchmark uses the first nine campaigns in workbook order
for development and the final four for confirmation. The preliminary Morpholine
campaign is kept, labeled explicitly, and reported separately rather than merged
with the main campaign.
