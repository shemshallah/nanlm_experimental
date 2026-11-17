# M3 R24 — accepted default after independent TEST

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

The locked [TEST run #44001403746](https://github.com/shemshallah/nanlm_experimental/actions/runs/44001403746)
PASSED: 24/24 new books improved, 73,726 positions; R21 3.43648662 →
R24 3.34083570 BPB (gain 0.09565093), nonspace 4.04008615 → 3.95073300,
top-1 35.496% → 39.332%, top-5 70.474% → 71.508%. The descriptive
book-bootstrap interval is [+0.082683,+0.106767] BPB. Wrong-successor
control worsened to 3.903432 BPB. No TEST adaptation occurred.

R24 now becomes `m3_default.M3DefaultPredictor`'s default mode; the
`M3R21Predictor` class continues the original frozen R21 exact path. The
R23 research model still has its full recurrent apex, but that apex remains
unpromoted and is not involved in the accepted default. This architecture is
separate from FA1, FA2, Terminal-RSI and NaNLMind.
