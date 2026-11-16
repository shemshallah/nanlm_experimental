# FA2 R59 — Causal compound-key geometric memory (candidate)

Author: Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
Date: 2026-11-16 · Branch: `master` · Architecture: FA2, not FA1 or Terminal-RSI

## Major architecture change

R54/R55 retrieved observed successor bytes using an exact `C(256,2)` pair
address. R58 learned a distinct zero-initialized four-parameter content rail,
which conserved the total probability mass on the 255 non-space/non-EOS bytes.
Real-book R58 DEV did **not** improve over R55; R58 is *not promoted*.

R59 gives the **same** FA2 model a *third* independent trainable four-parameter
trust rail, keyed by the joint content-address:

`(geometric mirror-pair address, current observed byte)`.

Optional `pair_bigram` adds the prior observed byte and disallows u=0.
Earlier memory locations u<t qualify only when their stored pair and observed
byte(s) match the current query; each matched successor x[u+1] is already
observed at prediction time t. Nothing reads x[t+1] or any held-out label.

Both R54/R55 and R58 rails remain present and unchanged; the R59 rail is
zero-initialized and exactly preserves inherited predictions. Existing R58
checkpoint and optimizer tensor history are migrated in place by adding one
new tensor and one slot in the existing optimizer group. No replacement model.

R59 adjusts only the *conditional distribution* on non-space/non-EOS bytes:
the log-sum-exp over this group is preserved and literal-space/EOS logits are
unchanged. The ordinary FA2 next-byte/+2/+3 heads are still shared.

## Predeclared causal and counterfactual controls

- **Neutral R59 rail:** zero the R59 weights, leaving R55 and R58 frozen.
- **Exact compound address:** preserve pair and observed byte key identity.
- **Wrong pair:** permute query pair ID only; historical keys untouched.
- **Wrong byte:** permute query byte only; historical keys untouched.
- **Wrong successor:** rotate *observed* historical successor values while
  retaining identical compound-match locations, hit counts, and coverage.

With all prior rails frozen, train only R59 `joint_weight` (four parameters);
do not step inherited Adam tensors. The candidate's DEV comparison uses 12
whole-book-disjoint development books and bootstraps by whole book, with a
familywise correction across **four** comparisons (neutral, wrong pair,
wrong byte, wrong successor). TEST stays sealed until corrected DEV,
long-context, and retrieval-control gates all pass.

## Continuation command

Use the verified R58 **step 3185** cumulative checkpoint as `--checkpoint`.
Always give separate new candidate output paths, and verify the exact input
SHA-256: `63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7`.

```sh
python poincare_mirror/scripts/train_r55_pair_successor.py \
  --checkpoint /path/to/fa2_r58_step3185.pt \
  --expected-parent-sha256 63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7 \
  --data-dir /path/to/verified_gutenberg38 \
  --joint-content --updates 96 --context 64 \
  --train-windows-per-book 8 --dev-windows-per-book 8 \
  --output-checkpoint /path/to/fa2_r59_candidate.pt \
  --output-report /path/to/fa2_r59_candidate.json
```

The source CI is **not** an actual corpus training or checkpoint-backed
measurement. No R59 DEV/TEST scores or promotion should be claimed until
executed against the verified R58 parent. The previous real-book R58 result
remains the last empirical FA2 successor-memory measurement.

## Acceptance criteria

1. Verified exact SHA and inherited 196 optimizer moments preserved.
2. Parity before/after zero-init migration; no changed historical parameters.
3. R59 reaches a positive corrected whole-book DEV gain in next-byte CE/BPB
   and non-space CE, with top-1/top-5 and +2/+3 not regressing.
4. Exact address/content must beat all three isolated negative controls.
5. Long-context regression passes; only then open independent TEST.

No trainable parameters from FA1, M3, or Terminal-RSI are imported.
