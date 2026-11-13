# FA1 R7 — one unified architecture on `master`

**Current source architecture: R7.26 (read-only geometric factor diagnostic).**
**Accepted model: step 14,416, with five continuous inherited optimizer histories.**
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
- Local 1,024-position **historical** smoke still showed a geometry regression
  of 0.000874 nats/vocabulary-word. That corpus has been consulted before;
  it is not a sealed generalization test. No new frontier capability or
  raw-byte BPB gain has been demonstrated.

The only valid promotion route remains independently held-out NaNROUTER books,
book-level confidence bounds, correct-history superiority, informative
S-/tetra/apex ablations, matched nongeneric controls, and unchanged original
weights/moments. No GPU escalation yet. FA2 and NaNLMind remain distinct.

Read `FA1_R7p25_CAUSAL_GEOMETRY_CONTINUATION.md` for checkpoint lineage and
`FA1_R7p26_FACTOR_ABLATION_PROTOCOL.md` for the new diagnostics. The implementation
is in `source/fa1_r723_diamond.py`, `source/fa1_r724_geometry.py`,
`source/fa1_r724_unified.py`, `source/fa1_r724_train.py`,
`source/fa1_r724_eval.py`, `source/fa1_r725_controls.py`, and
`source/fa1_r726_factor_audit.py`.
