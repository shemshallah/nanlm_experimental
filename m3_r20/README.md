# M3 R20 — Signed-pair and ordered-triangle successor memory

This is a major M3 architecture extension. R13–R19 used spatial waves primarily
as a phase/energy residual to an exact-window memory. R20 makes the geometric
addresses carry **causal predictive content** as well:

- 256 heads, including 256 diagonal self-transition stores.
- C(256,2) = 32,640 off-diagonal unordered mirror addresses.
- Separate directed S+ and S− integer rails, one count per orientation.
- Sparse ordered head triples (previous-previous, previous, successor) with an
  8,192-transition causal history, updated after scoring each target.
- Factorized feedback: previous head selects its directed mirror-pair row;
  previous-previous plus previous head selects an ordered-triangle successor.
- The result feeds the exact frozen R19 next-byte distribution with two
  independently calibrated trust weights. Zero/zero returns R19 exactly.

The model scores 256 candidate bytes before observing the target and updates
the pair and triangle states only afterward. Each book resets its own history.
The original R16 wave state and R19 gated readout remain intact; neither is
discarded, altered, nor retrained.

## Explicit probability coupling

Let P19 be R19's frozen byte probabilities, Pnear the inherited
ExactWindowMemory distribution, Ppair the 8192-transition directed signed-pair
successor distribution, and Ptriangle the ordered-triple distribution.

The two additive residuals are:
- pair_delta[b] = log Ppair[b] - log Pnear[b]
- triangle_delta[b] = log Ptriangle[b] - log Ppair[b]

Next-byte logits equal log P19 plus pair_scale times pair_delta plus
triangle_scale times triangle_delta, followed by exact normalization.

The scale grid is 0, .25, .5, 1 for both components. Screening uses sixteen
fresh TRAIN books, final choice uses four different TRAIN gate books, and
a zero-scale fallback is always eligible. No DEV/TEST gate re-selection.

## Book separation

NaNROUTER batch 0004, fixed seed 160016, 4,096-byte UTF-8-aligned prefixes,
1,024-byte warmup and release checksum.

- Indices 0–7 original TRAIN: frozen prior only.
- Indices 8–51 R16 DEV/R17 TEST/R18 TEST: original 52 SHA fingerprints verified.
- Indices 52–83 R19 TRAIN/DEV: not used in R20 training or evaluation.
- Indices 84–99 **reserved for future R19 independent TEST**: not used in R20.
- Indices 100–115 new R20 TRAIN screen.
- Indices 116–119 new R20 TRAIN inner gate.
- Indices 120–131 new R20 DEV (12 books).
- Indices 132+ remain eligible for a future independent R20 TEST.

## Matched controls and interpretation

Compare exact R19 baseline, pair-only, triangle-only, selected combination,
wrong-address (rotate both successor channels by 37 candidate bytes), and
wrong-context (rotate only triangle candidate indices).

Report BPB, BPC, non-space BPB, top-1/5, per-book bootstrap, throughput,
sign counts, mass conservation and address/triangle attribution.

**Scientific caveat:** directed pair and ordered-triple count stores are also
an exact conditional n-gram representation. A gain would show useful causal
content in M3's geometric address space, but not superiority to a
capacity-matched conventional n-gram or recurrent network. Independent TEST
and capacity-matched controls remain required before any model promotion.

Run .github/workflows/m3-r20-signed-successor-memory.yml on master.
