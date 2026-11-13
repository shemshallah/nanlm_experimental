# FA2 R30: sparse diamond Overmind (experimental source)

The R30 source extends the **same** FA2 PAL model; it does not release or
promote new weights. The last reproducible checkpoint remains the R27/R28
lineage supplied by the original R19 ancestor, R27 release and R29 overlay.

## Exact combinatorial topology

| Layer | Logical elements | Actual work per token (defaults) |
| --- | ---: | ---: |
| Base resonance heads | 256 | 12 selected |
| Sparse abstract bridge | 64 learned prototypes | 8 selected |
| Pair lattice `C(256,2)` | 32,640 canonical pair addresses | Up to 66 candidate pairs, 24 initially selected |
| Triangle lattice `C(256,3)` | 2,763,520 canonical triple addresses | Up to 220 candidates, 24 selected |
| Apex Overmind heads | 256 | 8 selected |

The complete pair lattice has a unique low-rank embedding per unordered pair.
Triangle IDs are exactly ranked, but their states are factorized from their
three pair edges and geometric coordinates, with a hashed learned residual;
there is no physically allocated 2.76-million-vector triple table.
Triangle hash collisions may occur and therefore do not imply independent
learned embeddings for every triangle. Different triples remain addressable and
receive different factorized signals from their constituent heads/edges.

## Upward attention and downward feedback

1. PAL recurrent/attention states route to twelve of 256 base heads and eight
   sparse abstract concepts. All selections use content-dependent scores.
2. Active heads instantiate canonical pair IDs. Their S+ symmetric and S−
   antisymmetric features, plus sine/cosine pair phase, form contextual edges.
3. Three edges induce a unique triangular concept, including signed oriented
   curvature/area from the three head phases. A sparse selection ascends.
4. Eight of 256 apex heads attend to those selected triangles. Two iterative
   attention/feedback passes mix top-down instructions with bottom-up evidence.
5. Triangle feedback is scattered through actual pair incidence; each pair
   returns **S+ + S−** to its first head and **S+ − S−** to its second head.
6. The resulting head readout returns to PAL's inherited MoE hidden features
   and to its existing output projection, without replacing either component.

This is an *instantaneous within-token feedback loop*, not a claim of
persistent, cross-token Overmind state. Each token remains causally isolated
from later inputs. The next design stage may explicitly attach a document-bound
recurrent apex carry; it must pass reset, stream and future-isolation tests.

## Checkpoint and optimization invariants

- `attach_r30(model, optimizer)` adds one new parameter group **after** the
  original 13 R28 groups; all previous weights and optimizer states are intact.
- `output_gain` starts at **exactly zero**, so R30 leaves inherited logits
  bitwise identical at initialization, even when downstream memory is active.
- Training can freeze all inherited parameters and train `r30.*` first; the
  inherited R27/R28 checkpoints must never be overwritten.
- A source-level architecture is not a trained model. No development or
  test performance improvement is claimed until trained and evaluated.

Run structural validation:

```bash
python fa2r19/source/test_r30_diamond_overmind.py
```

Run the **guarded exploratory CPU pilot**, only after restoring the complete
R19/R27/R28 lineage and book corpus into the same root:

```bash
python fa2r19/source/r30_cpu_pilot.py --root . --steps 8
```

An R30 candidate delta is saved only when development BPB, non-space loss, and
per-book limits all pass. This pilot has not yet been executed in this source
publication, so there are no R30 real-book performance claims.

## Mandatory promotion controls

1. Exact parity before the first update and inherited Adam audit afterward.
2. Matched CPU token/compute budget versus a conventional residual and a
   randomly routed Overmind, with a zero-gain control.
3. Overall BPB **and** non-space NLL improve on independent development books;
   reject candidates that regress either measure or any predefined book guard.
4. Validate correctly replayed 33K/65K/132K long prefixes, scrambled memory,
   unrelated-document history, hard semantic negatives, and future isolation.
5. Hold out genuinely unseen authors for a final one-time test; no test-driven
   hyperparameter selection. Only then consider a promoted R30 checkpoint.
