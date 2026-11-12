# PAL R7.11 continuation — hard-negative directed rank-64 memory (research, unpromoted)

The R7.10 accepted 14,416-step checkpoint remains the sole complete checkpoint (`checkpoint/pal_active.pt`, SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`). New R7.11 experimental directed key→value writer was trained for **320 attempted auxiliary steps**, selected at **update 80** on dev and preserved in a tiny exact-resume overlay (`adaptation/r711_experimental_delta.pt`, 173,477 bytes). The joint-LM counter remains 14,416; all five inherited optimizer histories are bitwise unchanged. No additional full checkpoint or separate model exists in this archive.

**Results:** Historical 512-query 32-way directed retrieval CE dev 4.0781 → **3.3053** (same new architecture before→after), test at selected update **3.3483**; top-1 20.31% dev and 16.80% test (3.125% chance). Correct-vs-unrelated read CE gap +0.587 dev/+0.554 test; reversing word order harms recall. Alternate query seeds corroborate within the same repeatedly consulted legacy corpora. **The gate is zero**, since experimental blends worsened historical 128-position dev next-token NLL, so there is no demonstrated improvement in language prediction or generalization. Legacy corpora have no doc-start metadata: no book-level confidence or independent-source claims can be made.

To reproduce, run `PYTHONPATH=source python source/test_r711.py` and `PYTHONPATH=source python source/r711_validation.py --count 512`. Read `research/R7p11_DIRECTED_SELECTIVE_MEMORY_AND_RESTORE.md` for exact protocol, limitations and training command.

---

# PAL R7.7 — Unified Learned Associative Retrieval, accepted step 14,416

**A genuinely continued R7.6 checkpoint, not a newly initialized model.** This research release contains one active, recoverable PAL model (`checkpoint/pal_active.pt`), preserving all **five Adam/optimizer state histories**. The R7.6 step-14,000 checkpoint (SHA-256 `b48b405018024d5c3883351597b2ba8db4a50cd14df2ad1638fe584f6740f419`) is the parent. We added a low-rank *learned similarity-based retrieval* path and then performed **416 further joint updates** to step **14,416**, preserving the original R6/R7 trunk, rank-64 physical mirrored matrix, exact one-word/two-word retrieval, character/subword pipeline, RNG, and optimizer continuity. This is the PAL R7.6→R7.7 lineage; distinct R10/R11/R12 lineages are not merged or overwritten.

**Current checkpoint SHA-256:** `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`

## What changed

- `source/r77_semantic.py`: `SoftAssociation` uses trainable rank-16 residual query/key transforms over the existing shared embedding table, a second-previous-word similarity term, bounded recency, causal top-k=32 values, and a small learned NOW-conditioned mixture gate. Values always come from the older 6,144-token FIFO, **strictly excluding** the current target and the recent 2,048-token context. Candidates for the new path must have keys *different* from the present previous word. The original exact one/two-word paths remain in the same model.
- The new 12,454 learned parameters are attached to the **existing fifth retrieval optimizer** as an additional parameter group, with original histories preserved; there is no sixth optimizer, independent model, or separately trained checkpoint. Total model parameters: approximately 11,336,996.
- `source/eval_r77_fast.py`: matched causal ablations compute the same base trunk once, testing learned-semantic-off, pair-off, exact-only, all-retrieval-off, and outer-matrix-off.
- `source/eval_r77_new_books.py`: external *whole-book* evaluation scaffold. Whole-document content SHA splitting, screening against the original training corpus with 16-token hashes, strict **8,192-word within-document warmup** to prevent accidental cross-book memory, fixed original tokenizer and read-only checkpoint. Input provenance is **not** automatically verified. New books must be supplied before an independent validation claim can be made.
- `source/infer_r77.py`: stochastic sampling from the actual single-model hierarchical/exact/semantic probability mixture, never from a substitute model. Exploratory samples are not fluent.
- `source/r77_outer_aux.py` and `source/probe_outer_assoc.py`: an optional causal contrastive outer-memory objective and correct-state/shuffled-state diagnostic. The attempted continuation using this objective was **rejected**, and the accepted checkpoint is rolled back to 14,416. See `reports/R7p7_outer_assoc_*.json`.

## Validated historical results (4,096 positions in each split)

All NLL values are **nats per whole-word token**; lower is better. These test/development positions have been consulted throughout PAL's development. They are **not independent unseen Gutenberg books**.

| Same checkpoint ablation | Historical dev NLL | Historical test NLL |
|---|---:|---:|
| R7.7 complete, step 14,416 | **5.138900** | **5.774114** |
| New semantic retrieval disabled | 5.139365 | 5.777994 |
| Two-word exact retrieval disabled (semantic retained) | 5.147315 | 5.782342 |
| One-word exact retrieval only | 5.147775 | 5.786192 |
| All retrieval disabled | 5.231042 | 5.888772 |
| Outer matrix disabled (other paths retained) | 5.139003 | 5.774221 |

Against R7.6 step 14,000's historic 4,096-position NLL (dev 5.144823/test 5.785444), the complete R7.7 checkpoint improves by approximately **0.005923 nats dev** and **0.011330 nats test**. The semantic path's incremental gain at final checkpoint is **0.000464 nats dev** and **0.003880 nats test**. The outer matrix is still almost neutral: **0.000103 dev** and **0.000107 test**. These changes are *within a repeatedly consulted benchmark* and do not establish independent generalization or statistical significance.

## Critical finding: initial rank-64 older memory is permutation-invariant inside 64-token blocks

`source/r74_stream.py` lines defining `CrossWindowMirror.initial()` group the older 6,144 tokens into **96 blocks of 64** and replace each block with a masked average of its embeddings before sequentially updating the rank-64 physical matrix. Therefore, any permutation of real tokens within one such block leaves the initializer unchanged. **No learned downstream weights can recover the erased within-block word order from that matrix alone.** This theorem concerns only the *initial* history→matrix map; `CrossWindowMirror.advance` updates on ordered future evictions.

The separate optional rank-64 contrastive associative-recall probe found **negative correct-versus-shuffled state advantage** at step 14,416 (−0.00450 nats). After 128 experimental auxiliary updates (not promoted), it worsened to −0.01319 nats, although raw contrastive loss decreased. This experimental checkpoint was discarded. See `research/R7p7_MEMORY_LIMIT_AND_NEXT_ARCHITECTURE.md`.

## Reproduce / continue exactly this single model

Requires Python 3, PyTorch, NumPy and SentencePiece. All commands are run from the extracted release root; CPU defaults shown. Inference/evaluation are read-only. To continue training, keep **the same output checkpoint**, optimizer histories and corpus and do not modify original token IDs.

```bash
python -m pip install numpy torch sentencepiece
export PYTHONPATH=source OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python source/test_r77.py --root .
python source/eval_r77_fast.py --root . --split test --count 4096 --start 0 --size 1024 --batch 8 --threads 2 --output reports/verify_test_shard0.json
# One further unified 128-update continuation (optional; modifies active checkpoint):
python source/r77_semantic.py --root . --resume checkpoint/pal_active.pt --output checkpoint/pal_active.pt --steps 128 --streams 2 --seq-len 2 --eval 0 --threads 2 --checkpoint-every 64
```

For *whole-book* validation, acquire and verify original NaNROUTER GitHub release archives (texts are in **Releases**, not the ordinary Git repository). The reference batch is `corpora-short-stories-0001` (1,000 Gutenberg short-story books, from 2026-11-10 release metadata):

```bash
# Requires GitHub CLI, network, zstd and sha256sum. These books are not packaged here.
bash source/fetch_nanrouter_books.sh corpora-short-stories-0001 /tmp/pal_books
# Must have at least three different .txt books; each *evaluated* book must contain >8192 words.
python source/eval_r77_new_books.py --root . --books-dir /tmp/pal_books --limit 100 --split test --max-positions 4096 --output reports/R7p7_newbooks_audit.json
```

**No verified independent natural-language book evaluation was completed in this release.** The environment could fetch only GitHub **metadata**, not the release's binary `.tar.zst` content. The included synthetic-book trial is strictly a **pipeline smoke test**, not a real-data or generalization result. To certify an independent result, record the binary release hash, list whole-book IDs and hashes, compare book inventory with any known historical training sources, and ensure no training step touches the evaluation split. Rolling 16-token screening alone cannot guarantee zero contamination.

## Packaging and scientific integrity

- `checkpoint/pal_active.pt` is the **only current checkpoint** in the ZIP and contains all five optimizer histories; experimental step-14,544 states and rolling rollback copies are not packaged.
- The complete original corpus, original tokenizer, original R7.6 code, R7.7 extensions, fixed-window shards, reproducibility hashes and negative-experiment records are retained. No nested historical ZIPs.
- `source/package_r77.py` regenerates the full ZIP and SHA-256 manifest. `RELEASE_NOTES.md`, `CLAIM_LEDGER.md` and `research/` detail claims, limits and next architecture.
- This is an ongoing language-model experiment, **not** evidence of frontier-level performance or fluent generation.

## R7.8 non-promoted CPU experiment

This cumulative research archive includes R7.8 executable order-sensitive writer/causal event code, chronological CPU training logs, all R7.8 validation reports and 23 structural/regression checks. **No R7.8 model was accepted.** `checkpoint/pal_active.pt` is still the last accepted R7.7 checkpoint (joint step 14,416; five optimizer histories). Reproduction instructions and the negative test result are in `research/R7p8_ORDER_SENSITIVE_MEMORY_EXPERIMENT.md`.

## PAL R7.9 cumulative research continuation — no checkpoint promotion

The cumulative R7.9 release adds the CPU training and experiments described in `research/R7p9_NORMALIZED_READER_AND_CAUSAL_RECALL.md`. The sole accepted checkpoint remains R7.7 step **14,416**; the source tensor SHA-256 is unchanged and **all five optimizer histories** remain in `checkpoint/pal_active.pt`. An ordinary +512-step continuation to 14,928 and a separate zero-gated normalized rank64 memory experiment of +128 steps to 14,544 **both worsened** the matched historically consulted 512-position development and test NLL. Identifiable old-history retrieval tests further showed correct-memory versus unrelated-memory generalization failures on test. Neither trial was promoted; all candidate weights are excluded from the cumulative ZIP. R7.7, R7.8, R7.9 source, tests, reports, and corpus are retained for reproducibility, and distinct PAL R10/R11/R12 models remain separate.

Run `python source/test_r79.py` in addition to earlier tests; this verifies exact migration and original optimizer tensors. See the research note for the scientific next gate and limitations of the unavailable independent NaNROUTER Gutenberg book release assets.

## PAL R7.10 reproducible checkpoint / semantic hard-negative evaluation

R7.10 adds **read-only** causal 32-way key→successor recall evaluation and a
transactional checkpoint integrity audit. The accepted checkpoint is STILL
`checkpoint/pal_active.pt`, inherited **step 14,416**, SHA-256
`54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`.
All five optimizer histories and all RNG state histories are unchanged.
No training, candidate promotion or NaNROUTER new-book validation occurred.

Principal 512 historical-query semantic-hard recall: development correct CE
3.867020 vs unrelated 3.866386; test correct CE 3.739855 vs unrelated
3.739821. Test top-1 0.7813% versus chance 3.125%; performance remains below
the uniform recall baseline (CE ln 32=3.465736). This is a **negative memory
recall result**; see `research/R7p10_CHECKPOINT_AND_HARD_NEGATIVE_MEMORY.md`
and the three `reports/R7p10_hard_negative_*.json` files.

Run `python source/test_r710.py`, `python source/r710_checkpoint_integrity.py`
and `python source/r710_hard_negative.py --count 512`. R7.7–R7.9 tests are
unchanged (45/45 inherited/new structural checks pass). The checkpoint
roundtrip candidate is intentionally NOT included: it duplicated the accepted
weights without training and exists only to prove durable save/load integrity.

## PAL R7.12 — causal streaming and gated historical LM research (unpromoted)

Accepted language model unchanged at **14,416 joint updates**, SHA `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`. All 5 inherited optimizer histories unchanged. R7.11's single selected 80-update directed writer overlay is retained; R7.12 adds exact nonzero-gate streaming using one persistent 64×64 matrix plus a rolling 768-token historical ring. Long-stream causal recurrence tests and zero-gate equivalence pass. A new tiny causal sparse retrieval LM-gate overlay, stored for reproducibility, gave an initial historical NLL improvement but **failed correct-vs-wrong memory specificity on the larger development audit**. A second reliability-conditioned calibration failed its development promotion criteria. **No new language-model checkpoint is promoted**. Frozen-writer context expansion beyond 768 words worsened historical hard-negative CE. See `research/R7p12_STREAMING_BRIDGE_AND_FRONTIER_GATES.md` and `reports/R7p12_*.json`; do not treat these as independent books or frontier benchmarks.
