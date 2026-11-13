# FA1 R7.24 — Oriented geometric Diamond (single R7 lineage)

**Status:** architectural integration and CPU evidence complete; **UNPROMOTED**. The accepted R7.22 ancestor at step 14,416, its five optimizer histories, and the unpromoted R7.23 Diamond baseline remain intact. No independent sealed-test improvement has been established for R7.24.

## User-directed architecture

The single R7 language model now supports simultaneous grade-2 and grade-3 mirror channels and tetrahedral geometric feedback:

```
256 original R7 points (H, with a learned latent 3-coordinate projection)
      <-> C(256,2) = 32,640 inherited R7 P/M pair addresses
      <-> C(256,3) = 2,763,520 exact-address triangle specialists
      <-> C(256,4) = 174,792,640 possible tetrahedra (sparse critic)
      <-> triangle -> pair -> point reconstruction -> SAME R7 decoder
```

- **Grade 2 S+ / S−:** inherited R7 `P` is the orientation-even pair source and `M` supplies orientation-odd phase and direction. These are separate learned channels: `S+(i,j)=S+(j,i)` and `S-(i,j)=-S-(j,i)`. Original R7 tensors are never reset.
- **Grade 3 S+ / S−:** every triangle has distinct exact-address even and odd scalar embeddings. Trigonometric edge circulation `A_ij + A_jk - A_ik` yields even `cos(curl)` and odd `sin(curl)` features. Triangle side lengths are sorted before geometric readout, so the intrinsic metric is orientation independent. The signed triangle component changes sign under odd vertex permutations.
- **Grade 4:** every possible tetrahedron has a distinct combinadic index, but only `C(8,4)=70` locally active tetrahedra are materialized per target. Its 4 oriented faces yield `F_bcd - F_acd + F_abd - F_abc`, plus a signed latent 3D volume. Shared sparse learned tetra operators send corrections back to the triangle states. No 174-million-entry dense tetra expert bank is allocated.
- **Causal connectivity:** 8 active points -> 28 local edges -> 56 local triangles -> 70 local tetrahedra. Apex and old-memory features can affect the original R7 decoder only via descending triangle -> edge -> point messages. The 256 meta specialists and all inherited R7 memory routes remain in one model.
- **Mathematics:** the alternating tetrahedral boundary obeys `d(dA)=0` for any edge 1-cochain, tested to floating-point precision; nonlinear face holonomy and learned S− residues can yield nonzero tetra closure. The latent 3D projections are learned representations, **not** an assertion of physical/manifold curvature.

## Identity-preserving migration and exact tests

The geometric module is an optional extension of the existing `CombinatorialDiamond`, so the original R7.23 checkpoint still loads unchanged. With geometric gates exactly closed, all three Diamond outputs (hidden, byte logits, word vector) match the original R7.23 checkpoint **bitwise in eval mode**. All original R7 tensors and five inherited Adam histories were recovered and round-tripped; only geometric parameters are new. The fifth inherited optimizer holds these new parameters in its existing final group.

Verified tests: full edge/rank index, maximum tetra address 174,792,639, signed pair and triangle reversal, 70 tetra-to-face incidences, `d²=0`, finite gradients for all geometric submodules, frozen original tensors, no pair-to-point bypass, full 5-optimizer checkpoint roundtrip, and inherited R7.22/R7.23 synthetic regression.

The geometric extension adds **5,537,379** trainable parameters, primarily two independent length-2,763,520 scalar grade-3 mirror tables. Sparse actual tetra feedback remains light on CPU.

## CPU experiments — original NaNROUTER book collection

All metrics below are next-*vocabulary-word* NLL in nats/word, **not** byte-accurate BPB. Inherited R7.23 (itself unpromoted) is the matched baseline with geometric gates closed.

| Probe | Train / dev targets | Updates | Best geometry gain (nats/word) | Interpretation |
|---|---:|---:|---:|---|
| Preliminary unsorted-shape prototype | 256 / 96 | 128 | +0.000489 | Positive development book CI; superseded when orientation-invariant triangle metric was corrected |
| Corrected geometry, strict book-CI gate | 384 / 160 | 128 | none selected | Mean gain +0.000216, book CI includes zero; stopped |
| Corrected geometry, continued exploration | 384 / 160 | 384 | **+0.002946** | Best at update 384; six-book development CI [+0.000096,+0.005770] |
| **Fresh development positions** from the same six books | 0 / 256, read-only | — | **−0.000653** | Book bootstrap CI [−0.004965,+0.005206]; not reproducible |

The freshly sampled development-position ablations from the third run are:

| Removed module | Mean change in NLL (no-module − full), nats/word | 95% book CI |
|---|---:|---|
| Edge S+/S− geometry | +0.000149 | [−0.001660,+0.002599] |
| Triangle S+/S− geometry | −0.000764 | [−0.002522,+0.001415] |
| Tetrahedral consistency module | −0.000267 | [−0.000845,+0.000371] |

The full candidate's correct-vs-unrelated-history advantage on the fresh positions is **+0.000547 nats/word**, but the book-level uncertainty is not resolved. No positive grade-3 or grade-4 advantage generalizes independently in this round. The six books have previously influenced R7.23 architecture development, so even fresh target positions do **not** establish genuinely unseen-book transfer. The previously sealed TEST books were not consulted for the R7.24 experiment.

**Decision:** preserve R7.24 as an unpromoted, recoverable research candidate; keep the accepted R7.22 checkpoint untouched. **Do not spend GPU time yet**: require repeated gains on entirely new books with a parameter-matched non-geometric control, clear S−/tetra ablation benefits, and actual BPB measurements.

## Reproduction

```bash
export PYTHONPATH=fa1r7.12/source
python fa1r7.12/source/test_fa1_r724_geometry.py
python fa1r7.12/source/test_fa1_r723_topology.py
# Needs the genuine R7.23 unpromoted checkpoint, original tokenizer and
# prepared, book-disjoint NaNROUTER corpus; no surrogate model is permitted.
python fa1r7.12/source/fa1_r724_train.py --root /path/to/original_r7_metadata --parent /path/to/R723_DIAMOND_STRICT_DOWNPATH_UNPROMOTED.pt --corpus-dir /path/to/processed --train 384 --dev 160 --steps 384 --seed 7241 --lr 0.0003 --contrast 0.5
python fa1r7.12/source/fa1_r724_eval.py --candidate /path/to/unpromoted_r724.pt --root /path/to/original_r7_metadata --corpus-dir /path/to/processed --targets 256 --split dev --seed 19017
```

Source modules: `fa1_r723_diamond.py` (optional geometry hooks), `fa1_r724_geometry.py` (grade2/3 S+/S−, grade4 tetra), `fa1_r724_unified.py` (five-optimizer identity/restore), `fa1_r724_train.py` (CPU gates), `fa1_r724_eval.py` (read-only fresh-position ablations), and `test_fa1_r724_geometry.py`.