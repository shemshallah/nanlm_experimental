# v48 full-vocabulary signed scout
Component diagnostic, not a replacement checkpoint. Build signed 128-D
distributional coordinates from exactly the 384 reference books plus the
reference wordlist. Eight causal context rails project through independently
learned positive and negative 128x128 maps. Full vocabulary cross-entropy
trains the query on unindexed books; fixed prior, identical reference index,
calibration and held-out cohorts. 262153 trainable parameters at 128D.
Measure equal candidate budgets 32/128/160 versus exact suffix count
baseline, fixed lexical similarity and unrelated history. OOV targets count
as misses. No target insertion. No claim about integrated LM accuracy,
byte BPB or v37 checkpoint parity. The scout produces proposals only; it can
later supply the v47 conserved decoder without changing its law.
Local CPU test at 16D on constructed task: 512 train / 256 held-out,
192 updates, NLL 2.830252 -> .008405, accuracy .09765625 -> 1,
wrong-history .06640625, all four parameter groups nonzero. This is not
language accuracy. Actual corpus diagnostic is launched with no artifact
polling and existing public release SHA checks. Its standalone reference
codebook is not the trained v37 codebook; future integration must adapt
the scout to inherited coordinates, not replace them.
