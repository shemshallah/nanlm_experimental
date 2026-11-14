# FA1 R7 — one unified architecture on `master`

**R7.36 source integration:** optional zero-initialized selected-triangle
read normalization and query-only apex-feedback centering, strict far-eligible
wrong-book donors, corrected eligible-book counting and per-path gradient
health checks are now wired into the canonical FA1 stack. These preserve
R7.35 predictions and inherited optimizer moments at migration; the new
20th optimizer group does not imply a newly trained or promoted checkpoint.
See `FA1_R7p36_DEEP_WIRING.md`. The strict independent-book promotion gate
still fails pending a valid R7.36 continuation and held-out evaluation.

**R7.35 research continuation:** causal distant modal values now enter only
top-8 sparse triangle value addresses before the existing apex -> pair ->
point readout. Writer-only donor swaps and book-level word/byte ablations,
candidate-only gate warm starts, group-19 optimizer restoration and an honest
four-eligible-book CPU DEV benchmark are documented in
`FA1_R7p35_HISTORY_VALUE_REPAIR.md`. The final unpromoted checkpoint is
recoverable; **the strict independent-book gate still fails** and the accepted
step-14,416 ancestor is protected.

**Current source architecture: one FA1 calibrated, multi-horizon Diamond, with causal apex feedback, byte-accurate evaluation and checkpoint guards; one unified optimizer.**
**Accepted model: step 14,416, with all five historical optimizer-engine states preserved.**
The accepted checkpoint is still the R7.13/R7.22 ancestor; no geometric
research candidate has passed the independent-book promotion gates.

Run the current source through **`source/fa1.py`**, the canonical FA1 CLI.
It enables inherited contextual, factorized, layered, calibrated and matched
far-history paths, plus the new zero-gated apex-to-word/byte attention loop,
together on the *same model*. The numbered source modules
are checkpoint-migration components and reproducibility history, not separate
FA1 projects or independently accepted models. The original `fa1_r728_geometric_memory.py`
CLI remains available for historical ablations and explicit source parity tests.
The current CLI requires `--root`, `--checkpoint`, `--corpus-dir`, and `--output`
and never promotes a candidate automatically. Use a distinct
`--candidate-checkpoint` path for optional unpromoted saves.

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
- R7.32 adds a *training/evaluation objective*, not another replacement model:
  latest eight word IDs, 32 literal bytes, two sentence heads, all eight
  active geometric vertices, inherited hidden state, near retrieval, and
  targets stay unchanged, while another book supplies older history and far
  memory. A difference-of-differences tests whether geometric *gain* is larger
  for the true older history than for the foreign one, separately for word
  branch, final mixture and conditional spelling bytes. Book-level lower
  confidence bounds must all be positive for promotion. Full R7.31 resume now
  preserves learned gates rather than reapplying migration initialization.
  The matched-history-only R7.31 continuation retains 16 optimizer groups.
- The parallel R7.32 calibration change adds identity-initialized per-example
  geometric trust for word and byte predictions, preserving all inherited
  outputs and optimizer moments at migration while adding group 17 to the
  *same* unified optimizer. Geometry-off diagnostics now preserve the separate
  language-context loop, and candidate selection prioritizes strict DEV
  admissibility over raw word-loss gains. The two R7.32 changes are integrated:
  matched-history constraints also apply to calibrated candidates.
  **No new accepted checkpoint or measured integrated R7.32 training result.**
- The R7.33 multi-horizon controls and the follow-up historically labeled R7.34 are now integrated in **the single FA1 source path**. The three far-history horizons (8/32/2, 16/64/4, 24/96/6 words/bytes/sentences), worst-horizon DEV contrast, pooled-literal-byte whole-book bootstrap, and checkpoint alias guards are enforced together. The reusable guards and byte validation live in `source/fa1_validation.py`, not a separate R7.34 implementation. These changes add **no model weights or optimizer groups** and claim no promoted model or independent-book gain.
- The next *in-place* FA1 source upgrade adds query-conditioned cross-attention
  from descended geometric point state, apex specialists, and their interaction
  back to the existing word and autoregressive byte decoders. A zero-initialized
  gate makes attachment prediction-identical. An 18th group is appended to the
  **same** optimizer, without discarding any of the prior 17 groups or the five
  inherited optimizer engines. Resuming a fully trained FA1 checkpoint does
  not reopen inherited gates. Strict DEV eligibility additionally requires
  positive whole-book apex-feedback word/byte ablations. No real-checkpoint
  training result, independent-book improvement, or promotion is claimed.
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
Read `FA1_R7p32_MATCHED_HISTORY.md` for the matched-history counterfactual,
the new loss, resume protections and the next candidate-only CPU protocol.
Read `FA1_R7p32_CALIBRATED_GEOMETRY.md` for the two trust routers,
unconfounded geometry ablation and strict DEV checkpoint selection.
Read `FA1_R7p25_CAUSAL_GEOMETRY_CONTINUATION.md` for checkpoint lineage and
`FA1_R7p26_FACTOR_ABLATION_PROTOCOL.md` for the new diagnostics. The implementation
is in `source/fa1_r723_diamond.py`, `source/fa1_r724_geometry.py`,
`source/fa1_r724_unified.py`, `source/fa1_r724_train.py`,
`source/fa1_r724_eval.py`, `source/fa1_r725_controls.py`, and
`source/fa1_r726_factor_audit.py`.

Read `FA1_R7p33_MATCHED_FAR_HISTORY.md` for the multi-horizon causal curriculum and `FA1_R7p34_BYTE_WEIGHTED_BOOTSTRAP.md` for the historical audit of the now-integrated byte-level book bootstrap and checkpoint protection.
Read `FA1_INPLACE_APEX_FEEDBACK.md` for the current source upgrade and its
function-preserving migration, causal feedback, and diagnostic requirements.
