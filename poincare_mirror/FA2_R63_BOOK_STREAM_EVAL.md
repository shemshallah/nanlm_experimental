# FA2 R63 — Full causal book-stream evaluation harness

Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
2026-11-16 · `master` · FA2 only

## Why R63 exists

R62 made book-local geometric successor history persistent across windows.
Without a *sequential evaluation harness*, existing FA2 random-window tests
still could not measure it without leaking future book text or reusing another
book's memory.

R63 provides the missing exact alignment between:

- The existing trained FA2 model's **causal geometric pair IDs**;
- Every OBSERVED next input byte, advanced in book order;
- R62's per-book persistent causal memory with an explicit reset;
- R55/R58 inherited logits and R59/R60 trainable joint-rail feedback;
- The next, +2, +3 prediction heads and their distinct true targets.

Each byte position is scored once (nonoverlapping query suffixes).
The model operates on overlapping left-context windows, bounded by its
own configured maximum context. Later inputs in the same model window
cannot influence earlier outputs due to the causal attention mask.

## Five matched conditions

One frozen model is queried under five arms using identical inherited
logits and pair IDs, and each arm maintains its **own** book history:

1. Neutral joint rail (no external evidence; inherited R55/R58 remain active).
2. Correct geometric address, observed byte and observed successor.
3. Wrong query-side geometric address; historical storage unchanged.
4. Wrong query-side observed byte; historical storage unchanged.
5. Wrong observed successor content with unchanged historical matches.

The scorer produces next/+2/+3 CE, top-1/top-5, next-byte nonspace/space
CE/accuracies, bits per raw byte and throughput. The comparison script uses
the existing whole-book bootstrap with familywise correction across all
four comparisons. The independent TEST split is never opened.

## Real-checkpoint DEV invocation

Use a **recoverable full model+optimizer** checkpoint containing a trained
joint rail, or `--zero-init-joint` *only* to check exact R58 baseline parity.
Never interpret a zero-init joint diagnostic as trained retrieval performance.

```sh
python poincare_mirror/scripts/eval_r63_book_stream.py \
  --checkpoint /path/to/verified_full_fa2_checkpoint.pt \
  --expected-parent-sha256 FULL_SHA256_HERE \
  --data-dir /path/to/verified_gutenberg38 \
  --context 64 --chunk 16 --history 32768 \
  --dev-books 12 --max-book-positions 512 \
  --output-report /path/to/fa2_r63_dev.json
```

A pilot prefix from each book is not a full-book language-model benchmark.
The same positions and train/dev splits must be matched across every arm.
Do not promote an R63 candidate unless the exact-address history beats
neutral and all three controls, with positive corrected next and nonspace
CE and nonregression for +2/+3 and top-1/top-5. Then test longer contexts
and finally the untouched independent TEST books.

## Gate status

Source/synthetic CPU CI tests verify book isolation, no checkpoint
mutation, identical zero-init predictions, causal target accounting, and
well-defined long and short prediction horizons.

The original FA2 R58 checkpoint is recoverable in ChatGPT Library but
not accessible to the current GitHub Actions runner; the local CPU
executor is unavailable. No R63 real-book BPB or accuracy gain is claimed.
