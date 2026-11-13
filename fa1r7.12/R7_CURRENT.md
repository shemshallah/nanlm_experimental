# FA1 R7 — one unified architecture on `master`

**Current source architecture: R7.31 (layered causal byte/word Diamond decoder; one unified optimizer).**
**Accepted model: step 14,416, with all five historical optimizer-engine states preserved.**
The accepted checkpoint is still the R7.13/R7.22 ancestor; no geometric
research candidate has passed the independent-book promotion gates.

- Accepted ancestor SHA-256:
  `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
- Latest recovered pre-meta R7.24 candidate: 384 geometry updates, five
  optimizer histories, SHA-256
  `5d07b1ef66aa9f870d05c6fda257b1dcdf36ecd4623999dff4a347cf82e4ed65`;
  **unpromoted**. R7.25 can migrate and resume this state without resetting
  inherited weights/moments or replacing the learned network.
- Unified hierarchy: original 256 points -> 32,640 signed S+/S- pairs ->
  2,763,520 exact-address triangles -> 174,792,640 rankable tetrahedra ->
  four of 256 meta specialists -> triangle/pair/point descent -> inherited
  R7 decoder. Activations are sparse: 8 points, 28 pairs, 56 triangles and
  70 tetrahedra per causal query, not a dense 174M-tetra memory bank.
- The R7.22 near/far/semantic million-token **addressable** retrieval and
  R7.23 byte/word/sentence Diamond towers remain attached to the same R7
  prediction model. Addressability is not demonstrated 1M reasoning.
- R7.25 fixes the four-gate geometry-off baseline, strictly foreign-book
  donor controls, and 9->11 face-projection/Adam checkpoint migration.
- R7.26 isolates the inherited R7.23 triangle address, orientation-even/odd
  face and pair channels, odd edge connection, tetrahedral closure versus
  signed volume, and each geometry/apex gate. It also checks triangle/geometry
  interactions and book-level confidence intervals without training weights.
- R7.27 preserves the exact legacy AdaGrad-style accumulator and merges the
  four AdamW engines into one optimizer interface with 13 parameter groups.
  Real-checkpoint bitwise state and update parity passed; the new unpromoted
  single-state checkpoint format has explicit restore-time tokenizer hashes.
- The R7.27 canonical source suite passes seven structural test groups. A
  512-target NaNROUTER book-disjoint DEV benchmark reported first-byte top-1
  15.63%, first-byte top-5 39.45%, trunk-word top-1 16.60%, trunk-word top-5
  37.70%, and full-system vocabulary NLL 5.098015 nats/word. The tiny
  foreign-history difference (+0.000056 nats/word) does not justify promotion.
  Full-system normalized word rankings and true raw-byte BPB remain unavailable.
- R7.28 forces the existing address-based S+/S- face memories and apex/pair
  downpath to learn from next-word, first-byte and per-example causal contrasts.
  A 96-update CPU pilot wrote ~34.5K signed face-code rows; its DEV-best
  candidate at update 32 preserved all 13 optimizer groups and original
  non-Diamond tensors. On six independent TEST books (512 positions), word
  NLL regressed 5.440382 -> 5.444346, first-byte top-5 improved 37.70% ->
  42.77%, first-byte top-1 fell 13.09% -> 11.33%. Geometry alone failed its
  DEV gate. **R7.28 rejected; accepted parent unchanged.**
- R7.29 extends the SAME Diamond with top-8 sparse context-addressed reads
  over the 56 active triangles. It adds only one zero-initialized trainable
  scalar as optimizer group 14; actual-weight migration preserved all 231
  existing parameter identities/tensors and exact word predictions. The
  corrected operator repaired a self-cancelling message and gave the gate
  ~24.6x stronger next-word gradient. Training used 300 previously unused
  NaNROUTER books, SHA-disjoint from the 240 used for R7.28. In fixed 512 DEV
  positions, total word NLL improved 5.151152 -> 5.150056, but the isolated
  new context readout harmed word NLL by 0.00002942 nats/word with a strictly
  negative four-book interval. First-byte top-5 fell 43.55% -> 42.58%.
  **R7.29 rejected and fresh TEST left sealed.**
- R7.30 source builds on R7.29 with 256-point factorized triangle keys/values,
  an apex/pair/point-dependent branch decoder correction and two zero-initialized
  gains. Its 15th optimizer group preserves all 14 inherited groups/moments.
  The earlier 64-update pilot reportedly had full word NLL 4.955495 and
  first-byte top-1/top-5 10.94%/32.81%, unchanged from its parent; correct-book
  advantage was negative. That pilot's recoverable checkpoint is not on GitHub
  master. The new source exposes factor-off branch and final-mixture gains with
  whole-book CIs; **no new full-checkpoint result or promotion is claimed**.
- R7.31 source integrates causal byte/subword-pattern/token/word/sentence
  attention loops into that same R7.30 Diamond, plus sparse point/pair/triangle
  attention and descending geometry into both word and autoregressive spelling
  logits. Six zero-initialized gates guarantee source migration parity; group
  16 appends without resetting optimizer moments. Conditional next-word UTF-8
  bytes and EOW can now be scored without gold-byte leakage, with top-1/top-5,
  conditional bits/byte and book-level geometry ablations. This does **not**
  establish raw text BPB, full-mixture word top-5 or a held-out gain yet.
- Local 1,024-position **historical** smoke still showed a geometry regression
  of 0.000874 nats/vocabulary-word. That corpus has been consulted before;
  it is not a sealed generalization test. No new frontier capability or
  raw-byte BPB gain has been demonstrated.

The only valid promotion route remains independently held-out NaNROUTER books,
book-level confidence bounds, correct-history superiority, informative
S-/tetra/apex ablations, matched nongeneric controls, and unchanged original
weights/moments. No GPU escalation yet. FA2 and NaNLMind remain distinct.

Read `FA1_R7p27_UNIFIED_SYSTEM.md` for checkpoint continuity, the new optimizer,
the unified suite and the honest byte/word measurement protocol.
Read `FA1_R7p28_GEOMETRIC_MEMORY.md` for the new causal training objective,
the saved rejected candidate, matched independent-book results, and the
next required context-dependent geometry readout.
Read `FA1_R7p29_CONTEXTUAL_GEOMETRY.md` for context-addressed read equations,
zero-gain and optimizer restoration checks, book-hash reservation protocol,
and the failed new-memory causal ablation.
Read `FA1_R7p30_FACTORIZED_GEOMETRY.md` for the factorized read, direct branch
diagnostic, source-only CI coverage, and strict DEV-before-TEST promotion gate.
Read `FA1_R7p31_LAYERED_DIAMOND.md` for the integrated multiscale attention
loops, autoregressive spelling head, migration parity and conditional-byte
metrics versus raw byte BPB.
Read `FA1_R7p25_CAUSAL_GEOMETRY_CONTINUATION.md` for checkpoint lineage and
`FA1_R7p26_FACTOR_ABLATION_PROTOCOL.md` for the new diagnostics. The implementation
is in `source/fa1_r723_diamond.py`, `source/fa1_r724_geometry.py`,
`source/fa1_r724_unified.py`, `source/fa1_r724_train.py`,
`source/fa1_r724_eval.py`, `source/fa1_r725_controls.py`, and
`source/fa1_r726_factor_audit.py`.
