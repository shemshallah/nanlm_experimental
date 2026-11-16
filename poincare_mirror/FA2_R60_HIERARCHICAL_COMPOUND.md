# FA2 R60 — Causal hierarchical endpoint backoff

Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
2026-11-16 · GitHub `master` · FA2 only

## Reason for the change

R59 joins a canonical mirror pair (one of C(256,2) = 32,640 edges)
to the *current observed byte* before retrieving a historical observed successor.
Its strict `pair_byte` match is sound but might be extremely sparse inside
the model's 32-token memory window. Adding more gates cannot help a memory
that almost never retrieves any matching evidence.

R60 implements a causal **exact-to-shared-head geometric backoff**, inside
the SAME R59 trust rail and SAME FA2 neural/Adam lineage:

1. First require matching `(exact mirror pair, observed byte)`.
2. If *an individual query edge* has zero exact matches, search historical
   keys that share at least one of its two 256-point endpoints AND have
   the same observed byte.
3. Head-only matches contribute at 1/4 the trust coverage of an exact
   compound match, suppressing less-specific retrieval.
4. No historical matched successor at u+1 can lie after current input x[t].
   The model never inspects future target x[t+1].
5. If no head-level match is available, that query edge makes no retrieval.

Canonical lookup uses `pair_address(a,b)=a*(511-a)//2 + b-a-1`,
with 0<=a<b<256. An exact 32,640-entry ID is always preferred over
a weaker shared endpoint, preserving the C(256,2) address identity.

The existing independent R55/R58 rails are untouched; R60 is configured
by `pair_successor_key_mode="hierarchical"`, saved in the same FA2
checkpoint config so it survives strict restart. R59's original
`pair_byte` mode remains the default. No new neural parameter tensors
are needed for this experiment; R59's new rail remains zero-init.

## Honest controls and targeted measurement

R60 source CI verifies:

- Hierarchical fallback retrieves in a toy prefix where exact pair+byte has
  zero hits, and exact matches override coarse hits when present.
- Query-only wrong-pair and wrong-byte maps, wrong-successor content controls.
- Prefix/suffix causality, inverse pair address enumeration, mass preservation.
- Configuration reload and a CPU synthetic known-copy learning diagnostic.
- Earlier R54/R55/R58/R59 regression tests.

The checkpoint driver now also reports independent DEV coverage diagnostics:
exact edge coverage and positions with hits vs hierarchical edge coverage and
positions with hits. These are *not* predictions and cannot claim lower BPB.

The existing corrected whole-book development gate still requires next-byte
and non-space CE improvement and beating each wrong-address, wrong-byte,
and wrong-successor control, with +2/+3 and top-1/top-5 non-regression.
No TEST data enters training or pilot selection.

## R60 real-book continuation command (when restored)

Use the complete previously verified R58 step-3185 **model and optimizer**
checkpoint. The required SHA-256 is:
`63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7`.

```sh
python poincare_mirror/scripts/train_r55_pair_successor.py \
  --checkpoint /path/to/fa2_r58_step3185.pt \
  --expected-parent-sha256 63cde9aa1875d9f5e3db78e92678b63057550584752cf20ffe8859c890895dc7 \
  --data-dir /path/to/verified_gutenberg38 \
  --joint-content --joint-key-mode hierarchical \
  --updates 96 --context 64 \
  --train-windows-per-book 8 --dev-windows-per-book 8 \
  --output-checkpoint /path/to/fa2_r60_candidate.pt \
  --output-report /path/to/fa2_r60_candidate.json
```

Do not promote a checkpoint unless the independent long-context regression
and sealed TEST gates pass AFTER book-disjoint DEV selection. At this writing
the checkpoint is in the cumulative ChatGPT Library R58 archive but is not
accessible to GitHub Actions and the local CPU executor is unavailable;
R60's real-book gain is UNKNOWN.

This track is isolated from FA1, M3, and Terminal-RSI.
