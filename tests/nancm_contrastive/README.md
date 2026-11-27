# v55
v54 saved provisional results but failed strict historical v53 replay:
story NLL difference -1.07937e-5, rank counts equal. Both runs used numpy
2.5.5 and torch 2.15.0+cpu; source of numerical drift unconfirmed.
Sentence wrong-history NLL 6.36704 beat true 6.59464. Do not promote.
v55 keeps frozen v53 and compares pure recency-copy, word contrast against
reference frequency, and class contrast (smoothed, bounded log ratios
over all known vocabulary). Exact normalization, zero-strength parent
identity through the inherited gate. Local NumPy test executed: maximum
mass error 1.1102230246251565e-16, unseen same-class word 0.1 ->
0.21840507431488246 vs opposing history 0.04925496815076465.
This is a constructed probability test, not language accuracy.
Three unrelated-history controls, donor selection by distinct ID and
closest observed prefix length, final four words held fixed. These
splices remain artificial, not natural documents. Generic frequency
control substitutes reference prior for copy and parent for contrast
under same mixture weights. Target presence counted after selection.
16/32-word exact tuple overlap audit against all reference training
prefixes and among first 1200 words of fit/cal/DEV documents; does not
certify entire corpus or approximate overlap. Record historical replay
delta; paired results diagnostic even if prior gate fails. No promotion
in this runner. Existing optimizer states immutable; new gates trained
on secondary unindexed books. Categorical law only, not raw BPB.
