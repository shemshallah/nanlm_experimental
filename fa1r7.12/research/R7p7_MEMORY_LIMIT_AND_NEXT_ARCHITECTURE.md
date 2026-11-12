# PAL R7.7 — negative results, structural memory limit, and next controlled experiments

## Measured contributions

R7.7 14,416-step checkpoint, 4,096 **historically consulted** positions in each split. Development active NLL 5.13890015; semantic-off 5.13936456; all-retrieval-off 5.23104214; outer-off 5.13900310. Test active 5.77411379; semantic-off 5.77799413; all-retrieval-off 5.88877235; outer-off 5.77422115. The new path's benefit is measurable but small; exact retrieval dominates; outer compressed memory adds essentially nothing.

## A concrete information bottleneck (provable from the implementation)

For a history of `6144 = 96 × 64` token IDs, `CrossWindowMirror.initial` constructs a masked mean embedding per 64-position block,

\[ c_b = \frac{\sum_{i=1}^{64} m_{b,i} E(x_{b,i})}{\max(1,\sum_i m_{b,i})} \]

and then writes each of the 96 block means to the **same** physical rank-64 matrix. Any permutation of tokens (including padding-mask values moved together with their tokens) within block b leaves that block's mean c_b identical. All subsequent deterministic state updates and final memory state are therefore equal for such permuted histories. Thus the initial compressed memory, irrespective of learned parameters, cannot disambiguate ordered pairs, temporal direction within a 64-word block, or who-did-what-to-whom within that block solely from the initialized matrix. This does not imply all other PAL features are order-invariant, nor does it apply to later ordered `advance()` updates.

## Rejected auxiliary experiment

The opt-in rank-64 contrastive event objective (`source/r77_outer_aux.py`) attempted to make the initialized matrix distinguish present history keys from negatives. On a fixed 128-position historical development probe before experimental training, correct-state CE = 3.61177835; shuffled-state CE = 3.60727829, so `shuffled − correct = −0.00450006`. After 128 *unpromoted* auxiliary updates, correct = 3.25079092 and shuffled = 3.23760156, yielding **−0.01318936**. The lower absolute CE did not demonstrate retrieval of correct old history; even wrong memory improved more. Rejected state 14,544 is **not included** as a second model or checkpoint. We restored the accepted 14,416 state.

## Next architecture to implement with lossless migration

1. Preserve one PAL checkpoint and five optimizer histories. Add an **order-sensitive event writer into the same rank-64 S+/S− matrix**, not a second future branch or independent optimizer. A candidate: per-block causal, position-weighted or sequential rank-1 keyed-value write where ordering changes the resulting memory. `advance` still updates that one state. Consider an eight-bin subblock sequence first to control cost; later full 64-token unroll.
2. **Zero-gate lossless migration:** initialize any replacement write pathway with output gain zero (or blend coefficient zero) so the old logits match to the supported floating-point tolerance; then train the new gate gradually. Record optimizer group changes without resetting old moments.
3. **Gate on actual state utility**, not contrastive CE alone: evaluate correct historical state vs shuffled state, blanked state, and previous-window-only baseline; require correct-state CE < shuffled-state CE, and outer-on NLL < outer-off NLL on disjoint whole-book development and test splits. Include a randomized-order within-block discriminability unit test.
4. On matching held-out books, compare exact-only, exact+semantic, correct-state outer, shuffled-state outer, out-of-domain and OOV rates. Record bootstrap CIs by **book**, not correlated adjacent words.
5. Increase document/history length only after evidence; 6144 old + 2048 recent already gives 8192 words, but only 96 unordered 64-token means feed the old rank64 state. Wider context without preserving event order does not fix this bottleneck.
6. Optimize CPU throughput by caching historical projected keys, using segment-level vectorized updates, and benchmarking actual step/s, with no NEON-specific or fused-recurrence dependency.

**Status:** The order-sensitive writer is a proposed direction, not an implemented or validated R7.7 result. The present release ends at 14,416.
