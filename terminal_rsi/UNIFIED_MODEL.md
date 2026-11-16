# Terminal-RSI: single recursive CPU / T4 / 96GB Blackwell architecture

**Terminal-RSI is the system.** FA1, FA2, M3 and Poincare experiments are independent.
This directory is deliberately isolated from them.

The same `unified.py`, `train.py`, loss, causal three-horizon heads, RSI
feedback loop and checkpoint format run on all three compute targets.
Preset choices change **capacity and batch/context**, not the model's identity.

| Mode | Width | Layers | Heads | Context bytes | RSI passes | Training |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | 256 | 3 | 8 | 256 | 2 | float32; reproducible acceptance gate |
| Tesla T4 (16GB) | 512 | 8 | 8 | 1024 | 2 | float16 autocast + scaling |
| 96GB Blackwell | 2560 | 32 | 40 | 8192 | 3 | bfloat16 autocast / SDPA, 90% GPU cap |

Blackwell is deliberately multi-billion-parameter and expensive. Monitor the
reported parameter count, actual GPU peak, effective tokens/s and OOM autotune.
NVIDIA hardware and software runtime determine which SDPA kernel executes.
32GB *system RAM* is different from 96GB *GPU VRAM*. Real full-size checkpoint
serialization can exceed the ten-minute budget. First establish CPU correctness,
then run a shorter Blackwell pilot and profile memory before long training.

## RSI is the core, not an auxiliary afterthought

At each causal prediction position:

1. Byte NOW state and causal local specialists (4-,16-,64-byte spans) produce
   multiple candidate representations, with trainable specialist routing.
2. Self-attention updates the state.
3. The next-byte head predicts a full 256-byte distribution and a learned
   surprise critic estimates prediction error.
4. **Observed previous-token surprise** and predicted surprise feed the
   *next recursive refinement pass*, routed through the same shared backbone.
   Current-token or future-target errors never enter the past.
5. Train the entire model and critic jointly. The RSI training controller uses
   book-disjoint DEV loss to accept improvements or self-modify training policy
   (e.g. halve learning rate). Every change is logged and checkpointed.
6. Auxiliary direct +2 and +3 byte heads support measured future-horizon loss,
   top-1 and top-5. These are direct predictions, not autoregressive rollouts.

The controller changes *parameters and learning policy*, not Python source files
mid-training. Source mutation would break provenance and reproducibility.
Continued checkpoint changes are recoverable from model, AdamW, controller
history, corpus SHA256 and split manifest.

## Critical honesty about the previous disconnect

The older `Terminal_RSI/hybrid_latest.pt` (~119MB) mixed an independently
trained CPU statistical model with a neural correction head trained on WordNet
glosses. Its selected alpha=1 still depended on CPU probabilities. It did not
contain the original fully trained v103 C predictor (2.071214 BPB baseline),
nor a shared gradient-optimized RSI state. Those checkpoints remain untouched.
The candidate here does NOT falsely treat them as identical architectures or
claim a predictive improvement before matched book-disjoint measurement.

This directory is the beginning of a **corrected CPU-first unified candidate**
and needs future audited legacy-weight import/teacher distillation before full
heritage from earlier Terminal-RSI models can be claimed.

## CPU first

Requires PyTorch 2.6+; no `google.colab` installation required.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python terminal_rsi/test_unified.py
python terminal_rsi/train.py --mode cpu \
  --corpus-dir /path/to/extracted_nanrouter_books \
  --checkpoint /path/to/safe/terminal_rsi_cpu_candidate.pt \
  --minutes 10 --eval-every 20
```

If your NaNROUTER release already has `train/`, `dev/`, `test/`
directories, those are respected. Otherwise the sampler deterministically
assigns entire files (not positions) to splits by filename hash.
DEV is reused for model selection, independent TEST stays sealed.

The CPU acceptance test covers: causal no-leak predictions, nonzero gradient
through critic/router/feedback/attention, three proper horizon objectives,
functional recursion, matched replayable evaluation, exact checkpoint reload,
and real RSI policy changes. Only after CPU CI passes should the GPU settings
be exercised.

## Same source on T4 / Blackwell

```bash
# T4 16GB, 10-minute experiment
python terminal_rsi/train.py --mode t4 --corpus-dir /path/to/books \
  --checkpoint /path/to/drive/Terminal_RSI/unified_t4_candidate.pt \
  --minutes 10 --context-override 512

# 96GB Blackwell; shorten context for the FIRST profiling pilot
python terminal_rsi/train.py --mode blackwell --corpus-dir /path/to/books \
  --checkpoint /path/to/drive/Terminal_RSI/unified_blackwell_candidate.pt \
  --minutes 10 --context-override 1024
```

Use the 8192-byte Blackwell default only after checking GPU peak and
checkpoint writes. The runtime dynamically halves training context if CUDA
reports an OOM, stopping at 256 bytes. Full 3.4B-scale training may exceed a
10-minute notebook startup+save budget; do not treat a short pilot as training
completion.

Resume a checkpoint with `--resume`; it refuses architecture or training-
corpus mismatch. This prevents optimizer history from being silently discarded.

## Promotion gates

- CPU source and numerical tests green; CPU checkpoint reload exact.
- Book-disjoint NaNROUTER DEV with matched before/after positions.
- Next, +2, +3: BPB, top-1, top-5; byte and tokenizer semantics specified.
- C v103 2.071214-BPB reference only when evaluated on exactly matching
  independent books and bytes (different corpora cannot be compared).
- Independent TEST once for accepted candidate, never for repeated tuning.
- No accepted checkpoint is overwritten by a speculative candidate.
