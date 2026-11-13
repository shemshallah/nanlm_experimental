# FA1 R7.31 — unified layered Diamond decoder

**Source continuation on `master`; accepted step 14,416 unchanged.**

R7.31 is appended directly to R7.30's existing network. It does not clone the
96D learned trunk, its 32K-word tree, 256 points, original signed S+/S- 32,640
pairs, C(256,3)=2,763,520 exact triangle addresses, tetrahedra, sparse apex
specialists, or downward pair-to-point route. The existing R7.30 factor memory
and 15 optimizer groups remain; the new parameters form one **16th group** in
the same single optimizer. Their six geometric/decoder gains all initialize
at zero, preserving inherited word NLL and original first-byte logits.

## Layers and causal attention

| Level | What enters | Learned operation |
|---|---|---|
| Byte | Last 128 bytes of **past** token spellings | Byte embeddings + attention over eight ordered 16-byte chunks |
| Subword | The same 128 past bytes | Trainable width-3/5 byte-pattern convolutions and self-attention over 32 four-byte blocks; **not BPE** |
| Token | The existing R7 32 past word IDs | Reused inherited 96D embedding table, projection and self-attention |
| Word | Groups of the past token states | Attention over eight word blocks |
| Sentence | Eight inherited past sentence-start IDs | Reused embeddings and self-attention |
| Geometry up | Eight active points, 28 pairs, 56 triangles | Separate 4-head self-attention loops, individually gated; pre-existing S+/S-, tetrahedron and apex computation remain active |
| Geometry down | Apex -> triangle -> pair -> point | Existing bidirectional Diamond mail; additional word and byte residuals require **both** nonzero apex and descended point state |
| Output word | Same R7 word-tree branch | A zero-gated additive 96D hidden residual; final probability mixture remains the inherited normalized R7 bridge |
| Output bytes | Causal spelling prefix plus descended geometry | Two causal attention passes, 257-class byte/EOW softmax; teacher forcing is isolated from word routing |

The byte head scores each next vocabulary word's complete UTF-8 spelling,
predicting the literal bytes and an explicit end-of-word (EOW) symbol. Gold
bytes are shifted one step before input and future positions are attention-
masked. Long spellings over 64 UTF-8 bytes are excluded, not truncated.

The resulting `conditional_spelling_bits_per_byte`, all-position byte top-1,
and byte top-5 are genuine **conditional word-spelling** measurements.
They are **not raw contiguous-text bits/byte**, because word boundaries and
tokenization are supplied. The original first-byte head remains available
for direct comparisons, and `raw_contiguous_text_bits_per_byte` is `null`.
Similarly, the inherited word tree has normalized word top-1/top-5; the
full retrieval/geometry mixture still needs all-vocabulary normalization.

## Protection, testing and promotion

- `attach_layered` verifies every prior parameter remains in one optimizer,
  appending one group without resetting learned Adam or legacy moments.
- The full R7.31 loader reconstructs R7.29 -> R7.30 -> R7.31 before strict
  model and optimizer restoration; a pre-R7.31 parent gets zero-gain migration.
- The source suite checks exact zero-gate identity, sparse level shapes,
  future-byte nonleakage, byte probability normalization, geometric gradient
  paths, apex/pair necessity and optimizer-moment preservation.
- The TRAIN-only objective includes conditional spelling CE and a
  geometry-off contrast. DEV diagnostics measure word gains, correct-book
  advantage, factor/point/apex ablations, word bootstrap and now byte geometry
  gain, byte top-1/top-5, EOW and byte-advantage book bootstrap.
- The new positive-byte gates are **additional** constraints; existing positive
  full-word/factor/geometry/history gates stay mandatory. TEST is not read for
  model selection and no checkpoint is auto-promoted.

## Repeatable CPU candidate (requires private full checkpoint and disjoint books)

```bash
PYTHONPATH=fa1r7.12/source python fa1r7.12/source/fa1_r728_geometric_memory.py \
  --layered --root R713_ANCESTOR --checkpoint R727_OR_R730_PARENT.pt \
  --corpus-dir SHA_DISJOINT_BOOK_CORPUS --split dev \
  --train 1024 --dev 512 --steps 96 --minibatch 12 --batch 8 \
  --eval-every 16 --ablate-every 4 --layered-gate-init .12 \
  --byte-ar-weight .35 --byte-geometry-weight .2 \
  --output R731_DEV.json --candidate-checkpoint R731_UNPROMOTED.pt
```

CI is source-only without the private large parent. Until a complete R7.31
candidate has trained and passed independent DEV books, **this is an
architectural implementation, not a measured predictive advance.**
