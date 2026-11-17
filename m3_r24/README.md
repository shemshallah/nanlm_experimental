# M3 R24 — frozen independent TEST for the R23 copy specialist

R23's five-scalar recurrent readout was trained on NaNROUTER book indices
188–203 and selected by indices 204–207. On 12 new DEV books (208–219),
the selected copy-only specialist improved R21 from 3.547176 to 3.470165 BPB
(top-1 34.82% to 37.89%). The torus wave did **not** beat shuffled geometry;
it is **not** included in the promoted candidate.

`r23_frozen_candidate.json` records the exact scalar, inherited optimizer
moments, training artifact digest and **predeclared** untouched TEST indices
220–243. It is fixed before independent TEST is scored. The TEST evaluates
R21 versus causal copy and a wrong-successor control using the original
training prior and full inherited predictor. Observations update memory only
after scoring; every book gets a fresh state. Whole-book bootstrap and
nonspace loss must pass; at least 18 of 24 books must improve. No parameter
may be retrained, scaled or selected using this TEST.

If the locked TEST passes, R24 is eligible to embed the exact same causal
prefix-copy equation into the canonical default, with R21 still accessible as
an ablation. If it fails, retain R21. This experiment makes **no claim**
that the toroidal wave contributes predictive value, and remains separate
from FA1, FA2, Terminal-RSI and NaNLMind.
