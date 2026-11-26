# v44 adaptive admission/retention diagnostic
Learn positive soft admission weights for first-seen versus repeated
associations and age-bin weights (16, 64, 256, 1024), independently for each
of eight coordinate rails. 48 additional parameters. All weights initially one.
The fixed arm uses identical candidates, stored events, signed EdgeCredit
readout, batch sampling and 384 Adam updates; only retention weights are fixed.
Hard storage bounds remain 64/1024 words; this is differentiable effective
admission/forgetting, NOT learned physical eviction. Weights may increase the
effective count above the stored count; report and calibrate that behavior.
The local Retention preflight executed with torch on CPU: synthetic held-out
NLL 2.250522491716958 -> 0.07559497978040719, four-way rank accuracy 0 -> 1,
wrong-history accuracy 0.2734375. This deliberately contrived task contains three
recent distractors and one older recurring correct association, not language
data. Empty state exactly zero; initial weights match fixed. Only the standalone
Retention module was executed locally; corpus and integration checks pending.
The integrated graph tracks repeat status at write time, checks age-basis
count parity and reset/restore, and shares v43 signed readout and v42 folded
event law. Real-corpus fitting uses 12 reference books and separate 12
calibration books; no target-dependent candidates. Shared raw mixture weight
is selected on the fixed arm's calibration only. Actual raw/byte-token
marginals and codec checked on DEV; no end-to-end byte gradient yet.
Uses immutable completed v36 as diagnostic checkpoint, not a replacement
lineage. Learned admission/retention, geometric readout, and their optimizer
states are saved in the diagnostic artifact. General Unicode and full
recurrent-triangle feedback remain outstanding.
