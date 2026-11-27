# v52 signed pair interaction and cross-fitted continuation
v50 actual held-out 1728 words: count top1 235, top5 477, top32 822;
coupled mixture 246/479/837, NLL improved each domain. Coupling itself
did not convincingly beat uncoupled. v51 actual audit 24576 positions:
indexed NLL 1.920468357 -> leave-document-out 6.068002636; 99.96745%
lower target probability after exclusion. Separate calibration NLL
5.802136212. Both workflows completed successfully.
v52 launches exact v51 continuation now that v50 artifact exists, plus
one matched arm adding multiplicative signed interactions over all 28
rail pairs. The existing 8-rail state, tied lexical identity and normalized
hierarchy remain the same. Pair residual bounded by .5 per component,
disabled path exact. Separate optimizer for newly added pair parameters,
all old AdamW state restored from v50 and continued 768 -> 1024 updates.
Three arms share samples, budget and parent; fixed gate, adaptive gate,
adaptive gate + pair interactions. Pair scale chosen on calibration with
zero included; inference ablation and wrong-context control. Full 128-D
pair module has 71,452 new trainable parameters (two 8x128x32 maps,
two 28x32 signed gains, 28 routing values, 32x128 readout).
Actual local executed 16-D test on real v50 Field implementation:
exact disabled prediction parity; 4.440892098500626e-16 target dense
logprob discrepancy, normalization 3.3306690738754696e-16, six new group
gradients nonzero, max FD error 3.749990829267206e-11. Constructed XOR
test 512 train/256 validation, 192 updates, accuracy .4765625 -> 1,
NLL .694504559 -> 0 (float32). NOT language accuracy.
New corpus workflow fetches only completed v50 artifact; no polling.
Prior v37 production lineage and v48 development baseline unchanged.
Categorical word/unknown law only; v49 open-vocabulary raw bridge not
yet integrated. Actual corpus v52 results pending.
