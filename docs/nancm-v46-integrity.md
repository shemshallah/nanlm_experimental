# v46 persistent triangle bridge and benchmark integrity
v42 reached report serialization after actual corpus evaluation and failed
on np.int64 byte_t1. Its results were not preserved; no corpus score claimed.
Shared metric_io writes NumPy scalars and arrays as JSON numbers and rejects
nonfinite values, atomically replacing the preceding snapshot. The v42,
v41/v43/v44/v45 and v39/v40 reports now checkpoint completed evaluation
cohorts. Local JSON scalar/array/boolean and nonfinite-preservation tests passed.
The portable recurrent solver now accepts an optional persistent edge callback
computed once at entry. It blends persistent memory anchors into the same
2816 edge states and adds bounded learned corrections to the existing 1024
triangles, rather than adding a separate predictor. The original no-hook
execution is preserved. The legacy implementation explicitly rejects active
hooks; CPU/GPU full-model parity still needs testing.
Local executed test with the actual triangle topology/operator, the v44
Retention and the v45 objective: 54,288 new anchor parameters + 48 retention
parameters; 9 groups nonzero gradients; maximum error 1.3080363182912902e-12;
minimum gradient norm 7.007859551739222e-08; exact empty and zero-scale
operator outputs and edge values; max gate .211682391, max edge magnitude
.542327409; wrong history changes readout (norm 1.8832e-5). The test used a
simplified recurrence around the real triangle operator, not the full corpus
model. No accuracy claim. Convex blending cannot amplify edge update
sensitivity at fixed anchor, and corrected triangle gains remain tanh-bounded;
full solver contraction still must be verified.
This repair uses [skip ci] to avoid duplicate heavy workflows while the prior
consolidated recovery and v41/v43 diagnostics run. Active jobs are pinned to
their original commits and will not receive these fixes until a new checkout.
