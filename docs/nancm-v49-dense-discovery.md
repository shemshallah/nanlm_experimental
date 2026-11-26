# v48 development promotion and v49 full-vocabulary event credit
v48 actual corpus completed successfully (run 45912532334). 49,960 vocabulary,
1,961,325 reference words, 384 indexed books. Unindexed 12/12 fitting cohorts,
12 calibration books, exact full-file hash check. Scout selection recall32
161/768 -> 218/768 after 256 updates, count 449/768. Held-out 1728 targets:
count32 822, scout32 528; count160 1074, count+scout160 1070. No demonstrated
rank-matched scout benefit; the underperforming weights are not promoted.
User authorized development promotion: v48 unindexed fit discipline and signed
scout design are the development baseline. Keep the actual v37 checkpoint
step 7998 immutable. v48 standalone codebook cannot be substituted for v37.
v49 uses the existing inherited word_code projected by the v41 Graph into
128-D coordinates, the v48 eight-rail Scout, v44 Retention, and signed
projections of the v37 root hidden state. All inherited known words eligible,
no top-k gradient censoring; parent retains positive escape weight. Exact
raw-event likelihood with full dense marginal and capped-prefix intervals.
Actual local test executed with current PyTorch: 7 event types x 9 weight
combinations, literal error 6.938893903907228e-18, folded error 0, token
marginal error 3.552713678800501e-15, 28 finite differences max
3.262673464732302e-11. Zero-strength returns parent object; empty lexical
events yield zero gradients. Constructed 16-dimensional task, 384 training,
192 held-out, 160 steps: raw BPB 1.7470540699176151 -> .8594858437591973,
candidate accuracy .0625 -> 1, wrong context BPB 2.1462705936531496,
wrong accuracy .0364583320915699; nine parameter groups nonzero.
Not actual LM accuracy. Corpus pilot49 uses v37 legacy solver to avoid
unverified portable kernel, two same-seed 192-update controls (with/without
adaptive graph evidence), shared frozen donor, calibration scales incl zero,
true vs wrong scout contexts, folded event metrics, prefix parity and codec.
Inherited params, moments, lexical optimizer, RNG and step digest checks.
No v46 triangle gradient or v47 finite allocation is enabled in this trial:
first establish genuine dense discovery on actual language, then integrate
those mechanisms only if beneficial. Inherited field state is frozen.
Existing recovery is still pinned to earlier source and recovering artifacts.
A later workflow dispatch of consolidated recovery will include v49 with
preflight and corrected split integrity. No heavy duplicate workflow launched.
