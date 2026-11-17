# FA1 R7.57: a trainable, leave-one-out geometric transition router

R7.57 extends the existing FA1 R7.53/R7.54 model. It is **not** FA2, M3,
Terminal-RSI, or a separate language model. It appends **one** optimizer group
(30 → 31), preserving all inherited weights and Adam moments without update.

## Memory construction

From **TRAIN books only**, count causal predecessor → successor pairs, never
crossing a document boundary. Retain two independently normalized sparse
conditional distributions:

1. `Q_word(next_word | previous_word)` over 32,000 lexical groups.
2. `Q_signed(next_word | signed C(256,2) address(previous_word))` over
   65,280 oriented pair addresses, corresponding to 32,640 mirror pairs.

The existing fixed head lookup maps each prior word to its 256-head pair.
Sparse training counts, totals, and lookup keys are immutable buffers saved
**inside the full FA1 checkpoint**. No external memory file is needed to reload
the predictor. The independent TEST split is never accessed by the driver.

## Integration with the inherited normalized predictor

The original R7.54 distribution and its causal word-copy specialist remain
unchanged. A newly trainable router receives the inherited 32-dimensional apex
descended state and three causal count features. It emits two gains, `w` and
`g`, whose sum cannot exceed 0.24:

```
P_R757 = (1-w-g) P_R754 + w Q_word + g Q_signed.
```

Both new gains are exactly zero when attached, preserving the full parent
prediction **bit-for-bit**. Only the router's three new tensors are trainable.
The inherited master optimizer retains all 30 groups and obtains one new 31st.

For an in-TRAIN supervised target, the current `(predecessor,target)` transition
is removed from its own count and its address total before probability scoring.
Thus the new memory cannot give itself credit merely for containing the
current supervised label. The gate sees only causal inherited state and
pre-target support totals. The DEV and TEST values use full TRAIN counts.

## Required evaluation and promotion controls

- Next-word DEV NLL must improve over reloaded R7.54 on matched book-disjoint
  positions, with whole-book bootstrap 95% lower bound positive.
- A `lexical_matched` control transfers *all* geometric mixture weight to the
  exact-word transition at the same total new-memory strength. A geometrical
  claim requires beating this control at independent-book confidence.
- `wrong_pair_addresses` scrambles geometric address identity while keeping
  the lexical memory intact. `wrong_pair_values` corrupts only the geometric
  successor values, keeping the gate's observed support features in place.
- `no_apex` tests whether the apex features matter to the router.
- A complete saved checkpoint must reload all 31 optimizer groups, all frozen
  historical parameters/moments, and three newly trained tensors/moments.
- Strict gating is development-only; promotion and independent TEST remain
  closed until a separately authorized validation protocol succeeds.

## Reproduction

`fa1_r757_cpu_pilot.py` needs the previously accepted R7.54 checkpoint,
verified NaNROUTER TRAIN/DEV corpus, and original tokenizer/split metadata.
It supports `--threads 1`, `2`, `4`. A typical pilot uses 4,096 TRAIN positions,
2,048 DEV positions, and 1,000 updates with 512-word causal histories.

`test_fa1_r757_geometric_transition_router.py` checks signed pair identity,
count normalization, leave-one-out removal and default zero-gain parity.
The GitHub CPU workflow runs these and inherited R7.49/R7.53/R7.54 tests.

**Metric scope:** this is word-level conditional NLL. Continuous byte BPB and
full normalized top-5 word rankings are not supplied by this inherited
forward API, so neither is fabricated.

## Measured 1,000-update CPU pilot

The pilot used 4 CPU threads, 4,096 TRAIN positions from 270 books, and 2,048
DEV positions from 15 disjoint books, with a full 512-word causal history.
It ran 1,000 *additional* updates, taking approximately 125.55 seconds.

| Matched DEV word metric | NLL (nats/word) |
| --- | ---: |
| Frozen R7.54 parent | 5.18260956 |
| R7.57 trained lexical + signed-pair router | **5.01347685** |
| R7.57 matched lexical-only control | **5.01346779** |

The new trained router improves the existing candidate by **0.16913286
nats/word** on this DEV sample. However, `geometry_over_matched_lexical` is
**−0.00000915** nats/word: geometry did not beat lexical memory at matched total
weight. The learned mean new lexical mixture gain is **0.23153837**, while the
mean signed geometric gain is only **0.00023156**. Its training algorithm has
effectively learned to suppress this geometrical pathway.

Checkpoint reload confirmed all 31 optimizer groups, all 30 inherited groups,
all frozen weights/moments, and the three new learned tensors and moments.
The source checksum and sealed TEST status remained intact. **Unpromoted.**
The strict gate failed; no automatic 9,000-update extension or independent
TEST inference was authorized.

The architectural conclusion is specific: TRAIN-only pair-address transitions
are valuable against the R7.54 parent, but their predictive information here
largely duplicates exact-word transitions. A stronger geometry route must add
information not already expressible by the lexical address and must win a
matched-lexical ablation before promotion.
