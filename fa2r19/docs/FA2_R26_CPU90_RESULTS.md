# FA2 R26 — one unified PAL model, long-phrase context, corrected step 90

Date: 2026-11-12 (US Pacific). **Status: experimental / not promoted.**
Source of truth: original PAL R19 accepted ancestor + this latest R26 recovery ZIP.
The same unified model contains R18, R21, R22, R23, sparse signed-mirror MoE,
R25 adaptive 0–3gram cache and the new R26 high-order phrase index.
**No separate replacement model and no new GitHub branch.**

## Checkpoint lineage

- Accepted immutable ancestor R18 SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- R25 step-120 immediate predecessor SHA-256: `fc5c4b815607c168aa2d86fc217d62a7cf749e570f0c10c87ed129c16a93fe74`
- Selected R26 step-90 complete candidate SHA-256: `8cb6b5524fafcbdc6427e175000d490451a800287964ee509bf2b44aaea376fc`
- One AdamW optimizer with **11 groups / 137 states**; bitwise preserved
  **172 inherited model tensors, all 130 original optimizer states (390 fields),
  and 12 training-stream snapshots**.
- Additional R26 adapter: **1,692 trainable parameters** in seven tensors.
  Legacy model and optimizer states are frozen but intact. A separate R25
  checkpoint binary is **not required** for R26 recovery or resume.

## Real-book CPU training gate

100 attempted R26 training updates, **step 90 selected by development BPB**.
Only eight source training books were used in the R26 training phase;
the inherited R25 model had already trained on all 12 original training books.
R26 training uses exact saved R25 causal predictions and high-order ngram
features rather than backpropagating the 12M-parameter backbone. The donor
regularizer has the corrected sign: it favors correct-document likelihood.
The numerical results below use the **corrected rerun only**; earlier numbers
from the preliminary sign-error trial are superseded.

| Fixed 4K/8K windows | Inherited R25 BPB ↓ | R26 BPB ↓ | BPB improvement | Books better |
|---|---:|---:|---:|---:|
| Selection development | 2.401927 | **2.396518** | 0.005410 | 3/6 |
| Secondary existing development | 2.451812 | **2.420815** | 0.030997 | 5/6 |
| R25-unconsulted historical-test subset* | 2.350135 | **2.334611** | 0.015524 | 3/4 |

*This historical subset was consulted during the preliminary sign-error pilot,
then reused for the corrected audit. It is **not an independent blind test**.
Likewise, the development books have been used in the broader PAL program.
Selection used only the first six development books. The secondary group was
not used for selection in this round. Historical-test files are not used for
training or checkpoint selection. The per-book bootstrap is illustrative and
uncorrected for repeated researcher access or selection.

- Exact development bytes: 2787; secondary dev bytes: 2550; historical bytes: 2009.
- Correct book history beats unigram-frequency-matched wrong-book donors:
  development 6/6, reserved 6/6, historical 4/4.
- Higher-order candidate coverage in windowed queries: development
  26.2% and secondary 27.7%.

## Genuine long-prefix probes (same corrected checkpoint)

Each query replays the PAL recurrent state causally through the entire book
prefix; no skipping 33K/65K/132K worth of recurrent forward passes.
The higher-order matcher uses 4-, 5-, and 6-token keys, with 8,192 and
131,072-token memory horizons. Memory lookup itself remains CPU-based and
exact-key lexical, not paraphrase-semantic.

| Book ID | Query position | R25 BPB ↓ | R26 BPB ↓ | Wrong-book BPB ↓ | Farthest matching key |
|---|---:|---:|---:|---:|---:|
| 970 | 33,076 | 2.435296 | 2.356972 | 2.900598 | 31,353 |
| 970 | 65,047 | 2.186005 | 2.074883 | 2.665980 | 64,157 |
| 1444 | 33,137 | 2.562169 | 2.553256 | 2.886938 | 32,056 |
| 1444 | 65,092 | 2.222333 | 2.148711 | 2.515595 | 63,907 |
| 1444 | 132,027 | 2.418974 | 2.422897 | 2.563310 | 130,880 |

**Counterexample:** Book 1444 near 132K regresses from R25 BPB
2.418974 to R26 2.422897; wrong history is still
worse. Thus retrieval capacity exceeding 128K has been demonstrated, but a
net causal benefit from the 128K tier is **not** established at this position.
Having a far-back matching key does not prove the model meaningfully used it.

The recurrent **gradient/BPTT horizon remains two consecutive 12-word chunks**
from inherited R24. R26 extends memory **forward access**, not 128K training
backpropagation. Do not claim semantic QA, frontier reasoning, or state-of-art
performance from these results.

## Exact recovery and CPU continuation

1. Unpack the separate original **PAL R19 full release** into a clean directory.
   It contains the immutable R18 ancestor and NaNROUTER book corpus.
2. Overlay **the single R26 recovery ZIP** into that same directory.
   Its checkpoint is `reports/fa2_r26_unified_candidate.pth`.
   It supplies R21/R22 small optimizer initializer assets and all required source.
   Neither R24 nor R25 candidate binaries are required.
3. In that directory run:

```bash
PYTHONPATH=source python source/r26_restore_unified.py --root .
PYTHONPATH=source python source/test_r26_causal.py
```

4. For a new gated CPU continuation of this **same** R26 candidate,
   preserving all prior optimizer moments and without creating another branch:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=source \
python source/r26_resume_cpu.py --root . --steps 110 \
    --train-books 12 --dev-books 6 --offsets 4096,8192
```

`--steps` is a cumulative R26 update target, strictly greater than 90.
Development NLL/BPB plus the correct-vs-unrelated-history gate selects a new
candidate. The resume driver preserves all prior Adam moments, source
weights and document-stream snapshots.

## Immediate next experimental tracks (same architecture)

1. **Trust calibration**: explicit far-tier ablation plus independent-new-author
   testing before increasing memory beyond 128K.
2. **Capacity**: expand recurrent hidden/FFN/attention with parity-preserving
   initialization and migrate Adam moments, without discarding R18–R26.
3. **Causal training**: lengthen recurrent truncated BPTT beyond 24-word
   chunks, then measure capacity scaling on predeclared held-out authors.
4. **Semantic transfer**: test paraphrase, hard negatives, multi-hop QA and
   fresh-author held-out generation, which the existing lexical benchmarks
   do not establish.

No old backup ZIPs or large ancestor binaries are duplicated in the latest
R26 archive; every member has a SHA-256 hash in the manifest.
## Verified final archive and CI

Latest recovery ZIP: `FA2_R26_UNIFIED_LATEST_RECOVERY.zip` (**80,174,176 bytes; 90 members**).  
ZIP SHA-256: `fcecf8f27d91e287090dc68ad0bebc657ac12dc4b98d4e3c50f1a5adb11cf243`.  
Checkpoint SHA-256: `8cb6b5524fafcbdc6427e175000d490451a800287964ee509bf2b44aaea376fc`.

- The ZIP passed all member SHA-256 checks, was extracted atop the original immutable R19 release, and recovered R26 without any separate R24/R25 candidate files.
- [GitHub Actions R26 causal checks passed](https://github.com/shemshallah/nanlm_experimental/actions/runs/43353828036).
- [GitHub Actions FA2 legacy source gates passed](https://github.com/shemshallah/nanlm_experimental/actions/runs/43353802064).
- The exact measured binary lives in the conversation's release ZIP; it is intentionally not pushed into Git history.
