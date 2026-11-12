# z3 — 30-minute Colab v3 smoke run (T4)

Open the notebook:

https://colab.research.google.com/github/shemshallah/nanlm_experimental/blob/master/z3/v3_colab.ipynb

**Status: runner and instructions only.** This file records no metrics and
claims no successful run. Run the notebook, then record whatever the run
actually produced.

## What it runs

`z3/train_v3.py` for a 30-minute wall-clock smoke on a mixed NaNROUTER corpus.
The notebook is a thin runner: it checks the GPU, installs the two packages the
corpus tools need, updates both repositories, mounts Drive, builds the mixed
corpus, checks for a checkpoint, and invokes the trainer.

The T4 profile requested by the notebook, when the trainer exposes the flags:

| Setting | Value |
|---|---|
| `d_model` | 256 |
| layers | 8 |
| heads | 8 |
| context | 1,024 |
| MoE layers | 4 (last four) |
| routed experts | 32, top-2 |
| shared experts | 2 |
| expert FFN | 1,024 |
| MTP | 2 |
| vocabulary | 32,000 (v3 default) |
| batch size | 4 |
| precision | AMP when the trainer advertises `--amp`/`--fp16` |

That configuration is the 128,619,264-parameter v3-small profile when counted
directly from `V3LM.parameter_count()` with `expert_hidden=1024`; the count is
an architecture fact, not a training result. The 384-wide / 12-layer default
(286,517,376 parameters) is also T4-feasible. The notebook only passes flags the
trainer advertises, so a trainer with a fixed config still runs with its own
defaults.

The notebook passes the committed z1 32k BPE tokenizer
(`z1/build/bpe32k_new.json`) when the trainer accepts `--bpe-model`. If that file
is unavailable, it falls back to `--tokenizer bytes`.

## Corpus

The run uses these releases, all from `shemshallah/NaNROUTER`:

| Release tag | Asset prefix |
|---|---|
| `corpora-paragraphs-0010` | `paragraphs-0010` |
| `corpora-short-stories-0004` | `short_stories-0004` |
| `corpora-sentences-0010` | `sentences-0010` |
| `hw5-gutenberg-c256-g1` | `hw5-gutenberg-c256-g1` (21-book Gutenberg) |

If `z3/train_v3.py --help` advertises a corpus-path flag (`--corpus`,
`--corpus-file`, `--train-corpus`, `--train-data`, or `--data`), the notebook:

1. downloads the three `.tar.zst` releases and their published `SHA256SUMS`;
2. verifies each archive's SHA256 before extracting;
3. downloads and extracts the Gutenberg `.tar.gz`;
4. takes up to 50 MB from each release with `nanlmv3/prepare_corpus.py`
   (about 200 MB total, so all four sources are represented);
5. builds a chunk-sampled file with `nanlmv3/sample_corpus.py`
   (20 chunks, 1 held-out validation chunk written last, so the 5% tail matches
   the trainer's `--train-fraction 0.95`);
6. passes that file to the trainer.

`z3/train_v3.py` takes `--corpus`, so this path runs for the current trainer. If
a future trainer has no corpus-path flag, the cell leaves corpus handling to it
and does not duplicate the download. The training/validation split remains
`train_v3.py`'s decision.

## Exact steps

1. Open the Colab link above and sign in.
2. `Runtime` → `Change runtime type` → **T4 GPU** → `Save`.
3. Run the GPU-check cell. It must print a Tesla T4 and roughly 15–16 GB. The
   cell raises if CUDA is unavailable.
4. Run the `pip install -q tokenizers zstandard` cell.
5. Run the clone/update cell. It clones or fast-forwards
   `/content/NaNROUTER` and `/content/nanlm_experimental`. It stops with a
   clear error if `z3/train_v3.py` is missing.
6. Run the Drive cell and approve the Google Drive authorization popup.
   Artifacts go to `/content/drive/MyDrive/nanlmv3_v3`.
7. Run the checkpoint-check cell. It reports whether `best.pt`, `last.pt`, and
   the other artifacts already exist.
8. Run the corpus cell. It downloads and builds the 200 MB mix for the current
   trainer's `--corpus` flag.
9. Run the 30-minute training cell. Keep the tab open. The trainer is expected
   to enforce the 1,800-second cap.
10. Run the result cell. It prints `result.json` (or `metrics.json` /
    `summary.json`) and lists the artifacts.

The Drive cell handles `Mountpoint must not already contain files` without
deleting `MyDrive`:

- if `/content/drive/MyDrive` already exists, Drive is treated as mounted;
- otherwise every non-Drive entry under `/content/drive` is moved to
  `/content/drive_pre_mount_backup` (with a timestamp if a name collides), and
  the mount is retried.

Nothing under `MyDrive` is moved or deleted.

## Runtime

| Phase | Typical T4 time |
|---|---:|
| pip install + clone/update | 1–2 min |
| Corpus download/extract/mix (`--corpus` is used) | 4–8 min |
| Training cap | **30 min (1,800 s)** |
| Final validation + checkpoint write | 1–3 min |
| **Total wall time** | **36–45 min** |

The 30 minutes are wall-clock, not a fixed step count. T4 throughput varies
with the trainer's batch size, expert loop, and evaluation cadence, so do not
compare step counts across runs without also comparing batch size and context.
The run cell also requests `--steps 20000`, `--eval-every 250`, and
`--log-every 50`; `--max-seconds 1800` is the effective stop unless the step
cap is reached first. `last.pt` is written at each evaluation and at the end,
so an interrupted run can recover from the most recent evaluation. A resumed
run with cached corpus files is normally at the lower end of the range.

## Outputs

Expected under `/content/drive/MyDrive/nanlmv3_v3`:

```text
best.pt          # model weights + metadata after a validation improvement
final.pt         # final model weights
ema.pt           # final EMA weights, when EMA is enabled
last.pt          # full training-state checkpoint (optimizer, EMA, RNG, step)
result.json      # best/final metrics, elapsed time, parameter count
history.json     # per-evaluation history
config.json      # resolved run configuration
```

`train_v3.py` writes that set for the current revision. The notebook only
assumes that `best.pt` and one of `result.json`, `metrics.json`, or
`summary.json` exist; otherwise it prints the directory listing and stops.

Do not commit checkpoints to the repository. They stay in Drive.

## Resume behavior

The run cell inspects `z3/train_v3.py --help` before building the command:

1. If `--resume-best` is supported, it is always passed. The current
   `train_v3.py` does not define it.
2. Otherwise, if `$OUT/best.pt` exists and the trainer supports `--init-from`
   (or `--checkpoint` / `--checkpoint-path`), the cell passes that flag with
   the checkpoint path. It also adds `--eval-before` when supported.
3. Otherwise, if only `$OUT/last.pt` exists and `--resume` is supported, the
   cell passes `--resume last`.
4. Otherwise it prints a warning and starts a fresh run.

Consequences:

- `best.pt` is written only after a validation evaluation. A run that dies
  before the first evaluation has no `best.pt`; `last.pt` is written at each
  evaluation and at the end, so recovery starts from the most recent one.
- `best.pt` contains model weights and metadata, not optimizer or data-RNG
  state. Resuming with `--init-from best.pt` starts a fresh optimizer and step
  counter, which is what the current notebook does for a new 30-minute run.
- `last.pt` is the full training-state checkpoint. `--resume last` restores the
  optimizer, EMA, data RNG, RNG state, step, elapsed time, and history. Because
  the notebook passes an explicit `--max-seconds 1800`, resuming a nearly
  finished `last.pt` can leave little or no time in the budget; use it to
  recover an interrupted run, not to add a second full 30 minutes.
- A resumed checkpoint must match the architecture and tokenizer configuration:
  `d_model`, layers, heads, context, expert count/top-k, MTP, and vocabulary.
  Changing those values makes the checkpoint incompatible.
- To force a fresh run, rename both checkpoints instead of deleting them:

  ```sh
  mv "/content/drive/MyDrive/nanlmv3_v3/best.pt" \
     "/content/drive/MyDrive/nanlmv3_v3/best.pt.bak"
  mv "/content/drive/MyDrive/nanlmv3_v3/last.pt" \
     "/content/drive/MyDrive/nanlmv3_v3/last.pt.bak"
  ```

  If only `best.pt` is renamed, the run cell will fall back to
  `--resume last` when `last.pt` exists. If the trainer supports
  `--no-resume-best`, that also bypasses the preferred checkpoint path.
- A corrupt or mismatched checkpoint should be renamed, not deleted, so its
  provenance is preserved.

## 8k context caveat

`V3Config.max_ctx` defaults to 8192, `ctx` and `rope_base` accept larger
contexts, and `rope_scaling.py` implements standard, NTK, and YaRN RoPE.
`train_v3.py` accepts `--ctx 8192` as a preset, so 8k context is
**architecture-supported** in the sense that the config, attention code, and
trainer accept it.

It is not exercised or validated by this notebook:

- this run uses `ctx=1024`;
- a T4 (16 GB) cannot hold an 8k run of this model with 8–12 layers and 32
  experts at a useful batch size; expect OOM or unusable step times;
- 8k needs an A100 40/80 GB (or larger), a small micro-batch, gradient
  checkpointing, and likely sequence parallelism;
- the YaRN/NTK code path exists in `z3/rope_scaling.py` and `--rope-scaling
  auto` selects it above the 1k original window, but long-context quality has
  not been validated by this run;
- long-context quality needs a held-out long-document evaluation. A 30-minute
  T4 smoke cannot provide that.

Do not describe this run as an 8k run, and do not extrapolate its loss to 8k.

## Troubleshooting

**`missing z3/train_v3.py`** — pull the commit that adds the trainer, or copy
the trainer into the checkout. This notebook is only the runner.

**OOM on a T4** — the notebook already requests `d_model=256`, 8 layers, and
batch 4. Lower the `--batch-size` entry to 2 in the training cell's `config`
list, restart the runtime, and make sure no other notebook is using the GPU.

**`Mountpoint must not already contain files`** — rerun the Drive cell. It
recognizes an existing `MyDrive`; otherwise it moves non-Drive leftovers to
`/content/drive_pre_mount_backup` and retries. It never deletes `MyDrive`.

**SHA256 mismatch** — delete only the bad archive under
`/content/z3_build/release/` and rerun the corpus cell. That directory is
ephemeral; the Drive artifacts are untouched.

**No result JSON** — inspect the training log. The result cell lists whatever
files are in `$OUT` and stops instead of inventing a result.

## Maintaining this notebook

Validate the notebook JSON after edits:

```sh
python -m json.tool z3/v3_colab.ipynb > /dev/null
```

The notebook is intentionally trainer-agnostic about optional flags. When
`train_v3.py` changes its CLI, the run cell adapts if the new flag is one of
the aliases it already checks; otherwise update the alias lists in the run
cell and this document together.
