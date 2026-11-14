# FA2 R35 — hyperbolic triangle rotation and curvature memory (experimental)

R35 extends the **same FA2/R34 model** without changing its byte tokenizer,
its exact point/pair/triangle/tetra addresses, its 256 meta experts, or its
optimizer structure. It must be enabled with `--triangle-curvature` when
growing or continuing a checkpoint that already has `--simplex-transfer`.
The canonical R30 checkpoint and the unpromoted R34 step-2,849 checkpoint
are not modified by this source update.

## Stored geometry and physical interpretation

For every token, R34 routes to eight learned point embeddings and its 56
oriented triangle faces. R35 learns a **2D Poincare disk coordinate** for
each of those points, with norm below 0.75; the disk metric has constant
Gaussian curvature K=-1. This is a trainable latent geometry, not a claim
about physical spacetime or externally grounded semantics.

1. For each face (a,b,c), compute its three interior angles alpha, beta,
   gamma from Mobius-translated geodesic tangent directions.
2. Gauss--Bonnet gives an unsigned angular defect
   `D = pi - alpha - beta - gamma`, equal to that face's hyperbolic area.
   It is **not** a vertex-curvature estimate over the non-manifold complete
   eight-point simplex complex.
3. Use a smooth oriented signed phase `phi = tanh(20 * cross(b-a,c-a)) * D`.
   Encode sine and cosine of all three angles and of phi, plus D/pi and
   orientation, in a trainable 10-to-rank feature map.
4. Rotate each triangle state in rank/2 complex coordinate pairs:
   `R(phi)(x,y)=(x cos(phi)-y sin(phi), x sin(phi)+y cos(phi))`.
   This preserves the feature norm; rotation/inverse tests verify that.
5. Pool the 56 face vectors using the same learned intake weights as R34.
   Rotate the pooled write by the weighted phase. A per-document causal
   prefix of these writes stores phase-aligned history. The current phase
   conjugates the read (inverse rotation). The current write is strictly
   excluded from the memory read, so no later byte can enter an earlier read.
6. Send the read into the pre-existing geometric upward/downward feedback
   with a zero-initialized learned residual gate. Use an auxiliary next-byte
   head on **retrieved history only**, not on the current write. This gives
   direct prediction gradients to the curvature and storage weights even
   at migration, before the main residual gate has opened.

The disk is just 2D per selected point, not a dense 2,763,520-row memory.
New weights per geometric MoE block: `2*r*r+(12+vocab_size)*r+1`, where
`r` is the existing geometric rank. All persistent old tensors, Adam
moments, signed mirror weights, and tetrahedral transport remain untouched.

## Verification

`python -m unittest discover -s poincare_mirror/tests -p 'test_*.py' -q`

**61 tests passed; 2 were skipped** in the local CPU environment. The six
new R35 checks cover:

- Positive hyperbolic triangle defect and invariance under disk rotation.
- Rotation norm preservation and exact inverse to floating-point tolerance.
- Strictly previous-position memory reads and independent batch documents.
- Migration from a trained toy R34 checkpoint with **exact +1/+2/+3 logits**
  and all inherited Adam optimizer states retained.
- Finite, nonzero supervision gradients through coordinate projection,
  angular encoding, stored values, and the new auxiliary vocabulary head.
- Full-model causal prefixes under changes to future input bytes, with a
  deliberately activated curvature gate.

## Paired CPU pilot (NOT NaNROUTER)

Reproduce via `python poincare_mirror/scripts/r35_toy_pilot.py 640`.
Both models begin from the **same trained tiny R34-style checkpoint** (32D,
rank 8, 2 layers, 4 mirror pairs, 48-symbol synthetic vocabulary), with
51 populated Adam parameter states. Migration preserves identical logits.
Both receive identical sampled windows and 640 optimizer updates; distinct
generated motif documents are held out for evaluation, 240 byte positions.

| Metric on toy DEV | Before (both) | R34 control (640) | R35 curvature (640) |
| --- | ---: | ---: | ---: |
| CE (nats/byte) | 3.657665 | **0.598628** | 0.601526 |
| bits/byte | 5.276895 | **0.863638** | 0.867819 |
| next-byte top-1 | 37.08% | 90.00% | **90.83%** |
| next-byte top-5 | 62.50% | 95.83% | **96.25%** |

At the R35 endpoint, **disabling only curvature memory** changes DEV CE
from 0.601526 to 0.600564, an improvement of 0.000962 nats/byte.
Thus curvature memory is active but does **not** demonstrate a positive
predictive ablation. Mixed metric movement is not proof of superiority.
This tiny synthetic workload is not comparable to the historical R34
512D/615-book metrics; it is an integration/optimizer/causality training test.

## Real-checkpoint status and next gate

The R34 step-2,849 `latest.pt` was not available in the current environment
or the accessible FA2 checkpoint inventory. The older recoverable R30 file
cannot substitute for it without losing the R33/R34 learned continuation.
Therefore **no real R35 NaNROUTER held-out benchmark, checkpoint advancement,
or model promotion** is claimed here. Source and tests are safe to commit to
`master`; training should resume only against the exact recoverable R34
candidate with its optimizer history.

Next: restore step 2,849, perform cold logit/Adam-state parity checks, run a
fixed-DEV short pilot and stratified independent-book plus 32K causal-memory
comparisons, report +1/+2/+3 CE/top-1/top-5/bits per byte, and accept only
genuine improvement versus matched geometric-memory-off ablations.
