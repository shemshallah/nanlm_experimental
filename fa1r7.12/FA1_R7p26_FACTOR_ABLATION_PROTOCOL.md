# FA1 R7.26 — separating geometry from inherited Diamond learning

**Source-only read-only diagnostic; no model promotion or optimizer updates.**
One R7 architecture on GitHub `master`, retaining the accepted step-14,416
checkpoint and five original optimizer histories. FA2 stays separate.

The R7.25 four-gate ablations did not separate the R7.23 exact-address
triangle hierarchy from R7.24 geometric additions. R7.26 masks nine factors
individually, then tests the four geometry gates and a triangle/geometry
difference-in-differences interaction. All factor masks are temporary forward
hooks: no 2.76M-row address table or Adam state is modified.

| Factor | What is removed |
| --- | --- |
| `r723_triangle_address` | Original learned R7.23 triangle codes |
| `face_splus_address` | Orientation-even geometry triangle code |
| `face_sminus_address` | Orientation-odd geometry triangle code |
| `edge_splus` | New even pair channel |
| `edge_sminus` | New odd pair channel |
| `edge_connection` | Learned oriented edge holonomy |
| `edge_odd_joint` | Odd pair channel and edge holonomy together |
| `tetra_closure` | Tetra input `[closure,sin,cos]` set to `[0,0,1]` |
| `tetra_volume` | Signed volume and squared volume set to zero |

The tetrahedral boundary of pure edge curl is exactly zero (`d(dA)=0`).
Independent S- face residues can produce nonzero tetra closure. A geometric
critic may separately rely on volume, so both signals must be ablated.

Each case reports `ablated NLL - full NLL` (positive means the component
helps), with whole-book bootstrap intervals. Book-donor controls replace all
target-free Diamond context but keep the inherited next-word target fixed.
The audit requires multiple DEV books, rejects TEST selection, checks all
mask/gate restoration, and does **not** claim raw-byte bits/byte.

Run once the verified 384-update unpromoted R7.24 candidate (SHA-256
`5d07b1ef66aa9f870d05c6fda257b1dcdf36ecd4623999dff4a347cf82e4ed65`)
and a genuinely whole-book-disjoint NaNROUTER DEV corpus are co-located:

```bash
export PYTHONPATH=fa1r7.12/source:/path/to/accepted_r7_root/source
python fa1r7.12/source/fa1_r726_factor_audit.py \
  --root /path/to/accepted_r7_root \
  --candidate /path/to/FA1_R7p24_GEOMETRIC_UNPROMOTED.pt \
  --corpus-dir /path/to/book_disjoint_nanrouter/processed \
  --targets 256 --seed 726 --threads 2 \
  --output /path/to/r726_factor_dev.json
```

**No R7.26 real-weight DEV loss result yet.** Structural tests demonstrate
correct ablations and closure representational capacity, not better language
prediction. R7.25's consulted historical 1,024-target evaluation regressed by
`0.000874` nats/vocabulary-word. Promotion still requires new book-disjoint
DEV BPB improvements and confidence bounds, correct-history superiority,
matched-parameter nongeneric controls, then a single sealed TEST check.
