# NaNCM v166 — Lexical Birth from a Trainable Signed Concept Cloud

Author: **Justin Howard-Stanley** · Independent Researcher · Santa Cruz, California, USA.

**STATUS: EXPERIMENTAL. NOT PROMOTED.** Active development baseline remains v158, protected fallback v148; v164 is the previous strongest measured opt-in research successor. v165's rejected objective and all inherited model weights/optimizer files are preserved unchanged.

## Why v166 is substantively different

v165 learned an energy reranker over at most 16 words proposed by the v164 context table. Its independent DEV candidate branch recovered only **70/480 = 14.58%** actual words. No perfect reranker can recover a word absent from its support.

v166 trains a **new 12,000-word lexical-birth head** from the *TRAIN-only* `v164_native_distribution.tsv` next-word count events. Given the preceding 1–3 *completed* words, the head computes:

- A 32D word-specific causal state from learned context-word embeddings and recency 0.2/0.3/0.5.
- An independently addressable 128-lower × 128-upper signed geometric field with separate excitatory S+ and inhibitory S− 8D edge tensors, inherited from trained v165, with its AdamW moments transferred. Input/OOV words still access the field by stable FNV lower address; unique output embeddings disambiguate upper-hash collisions.
- A normalized full 12,000-word categorical distribution from learned lexical dot scores, signed-field energy and trainable output priors. It can propose words absent from all v164 context candidates. It is a **fixed learned vocabulary** with the inherited open-vocabulary parent as fallback, NOT an unrestricted spelling generator.

The native C++ integration makes a *single normalized whole-word mixture*:

`P166(w|h) = (1 - alpha166) P164(w|h) + alpha166 Q166(w|h)`.

It exposes the exact 27-symbol prefix/EOS conditionals derived from that **complete-word mixture**, with an immutable vocabulary trie for O(prefix length) probability queries. No suffix of the target word is passed to `wf148_begin` or the generated beam. `V166_LEXICAL_BIRTH=0` yields byte-identical v164 prefix distributions. The v164 causal book-domain gate suppresses v166 on all checked documentation fixtures.

## Training, preservation and limits

`train_birth166.py` trains on the *lossy* original TRAIN-only 500-book conditional index: 199,956 keyed contexts, 733,069 source candidate rows, 199,080 usable groups, **500 AdamW updates, batch 128, 12,000 output words, 32D**. Unlike v165's re-ranking objective, the training target is an actual count-weighted next-word label, and the loss uses **full 12,000-word softmax normalization** (rather than restricting negatives to the inherited top-eight candidates). The optimizer state for all four inherited signed tensors is resumed; lexical parameters and their AdamW moments are added. One *new* v166 experiment checkpoint is saved, without replacing any historical weights or optimizers. The `context_split_val_nll` metric is a diagnostic on held-out TRAIN-index keys, **not book-disjoint**.

This is *not* end-to-end training on the full source bytes of 500 books. Higher-order histories are compressed to only 1–3 completed words by the index. Near-duplicate screening and full overlap with the inherited neural model's training history remain unresolved. The model still needs raw-book next-word training, longer causal context, an actual compositional OOV spell head, and full 256-byte BPB +2/+3 evaluations.

## Measured results (historical DEV and reused TEST samples)

All figures: 480 deterministic complete words from 24 distinct books per split, native 96-beam length ≤20, target never enters the conditioning history. The TEST sample was reused in prior v164 research and is **not a pristine final holdout**.

| Split/metric | v164 α=0 | v166 α=.05 conservative | v166 α=.50 loss-first |
|---|---:|---:|---:|
| DEV spelling/EOS bits/symbol ↓ | 2.880214 | 2.624520 | **2.505830** |
| DEV strict word top-1 | 45/480 | 45/480 | 40/480 |
| DEV strict word top-5 | 84/480 | 86/480 | 91/480 |
| DEV strict word top-32 | 146/480 | 146/480 | 153/480 |
| TEST spelling/EOS bits/symbol ↓ | 2.924645 | 2.635121 | **2.486754** |
| TEST strict word top-1 | 45/480 | 45/480 | 46/480 |
| TEST strict word top-5 | 99/480 | 100/480 | 104/480 |
| TEST strict word top-32 | 155/480 | 159/480 | **175/480** |

The TEST spelling/EOS loss reductions are **9.90% (.05)** and **14.97% (.50)** relative to v164. They do **not** imply whole-byte BPB improvements, and they are not a strict top-1 breakthrough. The v166 DEV top-1 regression at α=.50 blocks production promotion.

### Context and geometry attribution

On all 480 DEV samples, the v166 generator's **in-vocabulary next-word NLL**, measured only when the target is in its output vocabulary, is 6.7371 nats with true history, 7.2942 with shuffled unrelated histories, and 6.8064 with the signed-field term disabled (all samples scored against the same 12,000-word output). Thus learned history and signed geometry make separately measurable contributions, although the frequency prior still matters heavily. Generator top-32 candidate recovery rises to 180/480 vs 70/480 from v164's table-only branch. The union of the v166 top-32 proposals and v164 full-decoder top-32 recovers 202/480. These are **proposal coverage metrics**, not strict full-model top-1.

### Native regression gates

- 686 byte-exact v164 rollback probability vectors with `V166_LEXICAL_BIRTH=0`.
- 4,545 bit-identical v164 documentation-fixture distributions with v166 fully enabled and the causal domain gate active; 0 active generator proposals there.
- Invalid strength and absent sidecar reject model open, not silent fallback.
- Native C++ build, normalization, causal prefix-only predictions; 480-word DEV/TEST native teacher-forced spelling/EOS and strict beam metrics.
- Historical v164/fallback checkpoints and all six protected optimizer/model fingerprint files unchanged (verify inherited `SHA256SUMS` plus the current archive's SHA256SUMS).

## Native build from the complete recovery archive

```sh
make -B v166
OPENBLAS_NUM_THREADS=1 make test-v166
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_doc_guard166.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/eval_birth166.py
OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/beam_native166.py --split dev --strength .05
```

For opt-in inference, first configure the existing **v164** native model, index, word head, open-vocabulary speller, and book-domain environment exactly as in `README_v164.md`; then:

```sh
export V166_LEXICAL_BIRTH=.05 # conservative DEV top-1 preserved
export V166_BIRTH_MODEL="$PWD/checkpoints/v166_lexical_birth.bin"
# load bin/libwordfield166.so; unchanged wf148_* C ABI
# exact v164 rollback: export V166_LEXICAL_BIRTH=0
```

Use `.50` only for a loss-first research ablation; its DEV strict top-1 worsened. Do not modify the v158 production/development default. Files under GitHub `experiments/nancm/v166` are source/diagnostics, not the entire checkpoint archive, which includes cumulative native sources and trained weights.

## Next decisive work

Train this same lexical birth head from *raw* next-word events over hundreds of book-disjoint TRAIN books rather than the top-8 index; learn a **confidence-controlled, counterfactual first-choice gate** on TRAIN-only calibration books, then demand better **strict top-1** on independent DEV before promotion. Restore 256-byte full distribution BPB and byte next/+2/+3 metrics, OOV spelling generation, Android/Termux and new pristine book-disjoint TEST after eliminating near-duplicate overlap. **80% strict next-word top-1 has not been achieved.**