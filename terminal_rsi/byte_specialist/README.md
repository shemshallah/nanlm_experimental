# Optional v10.2 byte specialist — Terminal-RSI remains the host

This module **transfers algorithms and verified source from** the user-supplied
`nancm_v102_rightbytes.zip` **into** standalone Terminal-RSI. It does not
replace the agent, its task policy, critic, or optimizer. It never imports
Poincaré, FA1, or FA2.

## Exact donor provenance

| Item | SHA-256 |
|---|---|
| Supplied archive | `01d31f320e26375347a1587a9d1c3f3185606be866a7d57aaeb2a4e47f55bf67` |
| Byte-specialist source (`nancm_v102.py`) | `de7e68475541af2b234c1272133d8f1ce23c5023fb40a52f33b025637829758b` |

The supplied ZIP contains **source code and architecture notes, no trained
weights or optimizer checkpoint**. We therefore train the C byte-context
mixer from scratch on NaNROUTER books. Its previously reported 1.3794 BPC
was measured on unrelated system documents, **not this corpus**.

Transferable features: 32 context hashes including word-history and
line/column structure, bytewise mixer and adaptive probability maps,
two streaming tokenization-consistent bigram LMs (BPE-subword and whole-word),
and byte-grounded bit/accuracy metrics. Non-transferable: trained weights
(absent), cross-corpus numerical results, and any claim of frontier competence.

## Strict real-book comparison

`controller.py` performs its own content-deduplicated, complete-book-disjoint
partition. No material from DEV or TEST enters the byte training stream, BPE
merges, or word/subword LM tables. The chosen training budget is a prefix of
the TRAIN stream only. DEV and TEST each sample *one window per distinct book*.
Each 512-byte window warms up on 32 observed bytes and scores 480 future
byte positions. All feature arms use exactly the same window bytes.

```bash
# From the repository root, after extracting an authentic NaNROUTER book release:
python -m terminal_rsi.byte_specialist.controller prep \
  --corpus /path/to/nanrouter/books --work /path/to/local-work --maxb 1000000
python -m terminal_rsi.byte_specialist.controller search \
  --work /path/to/local-work
```

The agent's bounded proposal sequence tests no-LM, subword+word LMs,
v102 boundary contexts, 5-gram context, and full 11-bit structure contexts.
Only a verified DEV frozen-BPC improvement changes its selected configuration.
The independent TEST split is opened only after that decision and measures
the selected candidate against the no-LM baseline.

**Important limitation:** each C specialist variant is deterministically
retrained from the original TRAIN data. This is recursive *configuration*
selection under the agent's verifier—not persistent neural weight/Adam
continuation and not an end-to-end Terminal-RSI theorem. The original
standalone RSR/RPR actor weights are untouched; no auto-promotion occurs.

Report metrics (`frozen` and causal `online`) are actual *bits per character
(byte)*, top-1 byte accuracy and top-5 byte accuracy on full-book-disjoint
NaNROUTER text. They are not interchangeable with Terminal-Bench strict
success, bits per token, or spelled-word top-5. The optional donor's spelled
token/subword/word metrics are *teacher-forced per-byte spelling*, not direct
token-vocabulary rankings.

After a search, run `python -m terminal_rsi.byte_specialist.controller audit
--work /path/to/local-work` to reconstruct every reported BPC/top-1/top-5
from the C evaluator's individual prediction records and compute a 2,000-draw
paired whole-book confidence interval on the sealed TEST comparison.

## Deeper host-owned recursive learning (second round)

The optional `residual.py` instruments the **unchanged**, checksum-verified
v102 C model to emit its full frozen 256-byte probability distribution.
Those predictions enter a small **trainable correction layer owned by the
standalone Terminal-RSI host**. Its 256 residual weights, Adam first/second
moments and step, and space/non-space critic survive four iterative phases.
Only an independently measured DEV2 BPC improvement can select the learned
correction. The resulting TEST2 metrics are reported even if disappointing.

```bash
python -m terminal_rsi.byte_specialist.residual prepare \
  --corpus /path/to/nanrouter/books --work /path/to/local-work
for split in calibration dev2 test2; do
  python -m terminal_rsi.byte_specialist.residual tap \
    --work /path/to/local-work --split "$split"
done
python -m terminal_rsi.byte_specialist.residual train \
  --work /path/to/local-work
```

- The 48 training-calibration books were excluded from the original million-
  byte v102 train stream. The 24 fresh DEV2 and 24 fresh TEST2 books were never
  sampled in the earlier search (complete source books, not just positions).
- Each 512-byte window scores 480 next bytes. Full distributions and per-byte
  C loss records must agree. Incorrect shape or probability normalization fails.
- These phases use verified gradient updates and critic-adjusted weighting,
  but should **not** be confused with complete terminal-task RSR/RPR or proof
  of recursive self-improvement.
- The *host correction* now has a persisted optimizer checkpoint, but the
  donor's large C model still retrains from scratch for each experiment.

## Audited first results

Two small, source-controlled reports are kept under `reports/`:

- `nanrouter_v102_1m.json`: first-round, 48 test books, 23,040 bytes; frozen
  BPC 2.3056 -> 2.2609, top-1 54.63% -> 55.28%, top-5 83.93% -> 84.41%.
- `nanrouter_v102_residual_288.json`: 24 **additional** test books, 11,520
  bytes; BPC 2.17819 -> 2.17283, top-1 54.71% -> 54.91%, top-5 84.89%
  -> 84.87%. Labels-shuffled equal-budget control: 2.17432 BPC.
- `residual_checkpoint_288.npz`: **small host correction only**, checkpointed
  with its *own* continuous Adam history. It cannot reconstruct the large
  donor model without rerunning the checksum-pinned training and BPE setup.

These are diagnostic window measurements, not full-book perplexities. Both
rounds leave the parent accepted model untouched. Candidate promotion requires
further validation and no top-5 regression.

## Causal context expansion (third exploratory round)

`contextual.py` extends the **same saved step-288 correction** with seven
additional conditional byte classes based solely on the *previous observed*
character. The new 7 x 256 residual table starts at zero, preserving every
parent prediction exactly. The original global 256-vector and Adam moments
are inherited unchanged, and the new context table has its own Adam parameter
group in the same model/optimizer timeline.

```bash
python -m terminal_rsi.byte_specialist.contextual prepare \
  --corpus /path/to/nanrouter/books --work /path/to/local-work
for split in dev3 test3; do
  python -m terminal_rsi.byte_specialist.residual tap \
    --work /path/to/local-work --split "$split"
done
python -m terminal_rsi.byte_specialist.contextual train \
  --work /path/to/local-work
```

The expanded correction trains on the same permitted TRAIN calibration books,
then compares stages on 24 **additional** DEV books. Strictly unseen TEST3
contains the last eight suitable independent TEST books in this specific
release. These eight test-book results are exploratory, not promotion-grade.
The DEV gate simultaneously requires improving BPC and no drop in either
top-1 or top-5. The saved step-288 parent remains available unchanged.

### Deterministic accuracy-preserving inference gate

The conditional correction's raw likelihood can improve while moving a true
byte out of the top-5. `score_guarded` remedies that *without consulting true
targets*: it uses the learned probabilities only at positions where the
corrected model's argmax and the **membership** of its top-5 predictions both
match the inherited model. Else it uses the inherited probabilities. Thus
top-1 and top-5 match the parent exactly for **any** possible future target;
only the calibrated likelihood may change. The code asserts this invariant.

The guard was designed after inspecting exploratory TEST3, so TEST3 guarded
metrics must be called **post hoc**. The extra 8 previously untouched DEV4
books are reserved for assessment through `prepare-guard`, `tap --split dev4`
and `eval-guard`, with a whole-book paired bootstrap. On the initial local
attempts, v102 was killed three times by container memory pressure before
those probabilities were produced; DEV4 therefore has **no measured result**.
Another larger independent corpus is needed before any promotion claim.

## Resource-safe replay of the original v102 donor

The default v102 `-tshift 0` context allocation reserves approximately
2.2 GiB of `Slot` tables, before sidecars, history, and the host correction.
That caused repeated Linux SIGKILL (-9) failures while replaying the frozen
donor on a shared CPU worker. This is an execution-resource failure, not a
prediction-quality result.

The Terminal-RSI instrumentation now exposes an **opt-in** `--diskctx` switch:

```bash
python -m terminal_rsi.byte_specialist.residual tap \
  --work /path/to/local-work --split dev4 --diskctx
```

It allocates the donor's original, exact-sized context arrays through
`MAP_SHARED` over a securely created, immediately unlinked temporary file in
the specified work directory. The same 32 tables retain identical
hash-addressing, update equations, bit probabilities, and table limits. No
changes are made to the checksum-pinned `nancm_v102.py`, and the C donor is
still retrained from its known 1 MiB source prefix. A deterministic regression
test trains on the same byte stream with anonymous and disk-backed storage
and requires exact binary equality of all frozen records and full 256-way
prediction distributions. This is a CPU/RAM-recovery mechanism, not an
architectural accuracy claim or a persistent full-donor checkpoint.

**Operational caution:** the work directory must live on a disk filesystem,
not a RAM-backed `/tmp` (`tmpfs`), or the memory pressure returns. The
unlinked backing file remains allocated until the C process exits; the
operating system releases it automatically at exit. Disk-backed training can
be slower than RAM-backed training. No full-donor model or massive table file
is included in the small source release.

### One donor replay, multiple independent evaluations

`tap-batch` trains the unchanged donor once and then produces separate frozen
record and full-distribution files for up to 16 explicitly named splits.
It never updates the donor from any evaluation byte. This avoids paying
the very expensive 1 MiB v102 training cost again for every held-out set.

```bash
python -m terminal_rsi.byte_specialist.residual tap-batch \
  --work /path/to/local-work --splits dev4 holdout4 --diskctx
```

The batch path is tested against two fully independent single-evaluation
replays: **both** binary record files and both float32 probability files
must match byte-for-byte. This is a training-efficiency and auditability
change, not a new language-model architecture or an accuracy claim. The
optional GitHub CPU workflow now uses one replay per training stage rather
than one replay per split, and records the two truly unseen assessments only
after their manifests have been prepared.

## Independent train-remainder holdout (fourth assessment)

`holdout.py` prepares an additional set of 64 book-disjoint predictions from
the **377 remaining eligible TRAIN-partition books that were never used** by
the donor's 1 MiB training prefix or the residual's 48-book calibration.
Although they were originally *assigned* to TRAIN, these books contributed
no bytes to any fitted parameters, BPE merges, or token LM tables. The
manifest recomputes and checks the original complete donor stream SHA-256,
the frozen 48-book calibration selection, and the disjoint source-book sets.
This is an auxiliary never-trained-on holdout, not a relabeling of the
original sealed TEST split.

```bash
python -m terminal_rsi.byte_specialist.holdout prepare \
  --corpus /path/to/nanrouter/books --work /path/to/local-work --books 64
python -m terminal_rsi.byte_specialist.residual tap \
  --work /path/to/local-work --split holdout4 --diskctx
python -m terminal_rsi.byte_specialist.holdout evaluate \
  --work /path/to/local-work
```

Evaluation reads the saved step-288 inherited residual, the saved step-480
contextual candidate, and frozen donor distributions; it performs **zero
training updates**. It reports parent/raw/guarded BPC and top-1/top-5 plus
a 2,000-resample paired whole-book BPC confidence interval. No candidate can
be promoted merely by doing well on this auxiliary holdout. Full-book
evaluation, additional external books, and TerminalBench are still separate
requirements.

## Confidence-conditioned recursive trust (fifth exploratory round)

`trustgate.py` extends the **same saved contextual candidate** with ten
additional trainable Adam parameters: one bias, donor-distribution entropy,
the donor's top-two probability margin, and seven previous-observed-byte
class indicators. The contextual logit correction is multiplied by
`1 + tanh(features @ gate_weights)`. At the zero initialization this is
exactly one, preserving the inherited candidate's predictions. The new gate
has its own Adam moments; the inherited 480 residual and 192 contextual
Adam steps and every original tensor remain intact in the **single combined
checkpoint**.

```bash
python -m terminal_rsi.byte_specialist.trustgate train \
  --work /path/to/local-work
python -m terminal_rsi.byte_specialist.trustgate evaluate \
  --work /path/to/local-work --split test3
```

Only the previously designated 48 TRAIN calibration books update the ten
new gate parameters. Staged proposals at 128, 256, and 512 total gate Adam
updates were compared using DEV3; step 128 was selected using the predefined
per-stage improvement threshold. Relative to the *guarded* step-480
contextual candidate, DEV3 BPC improved **2.254258 to 2.252067** on 24
books. Because DEV3 had already selected the contextual architecture,
this gain is a development-only signal.

A separately recorded, read-only TEST3 comparison scored the same 3,840
positions from eight books and reported:

| TEST3 | Guarded context | Guarded adaptive trust |
| --- | ---: | ---: |
| BPC (lower) | 2.290856 | **2.289669** |
| Top-1 | 53.7760% | 53.7760% |
| Top-5 | 83.6979% | 83.6979% |

The paired whole-book bootstrap gain interval was `[0.000090,0.002516]`
BPC; five of eight books improved. These TEST3 results are **post hoc / not
independent for promotion**, since the earlier contextual top-k guard was
designed after inspecting TEST3. The trust gate is not promoted. The frozen
DEV4 replay and auxiliary 64-book holdout must be completed under a reliable
v102 training state before any independent effect is claimed.
