# Terminal-RSI: single recursive CPU / T4 / 96GB Blackwell architecture

> **Historical RSI lineage note (November 16, 2026):** This document records the *new stand-alone neural candidate* only. The canonical implementation that preserves the last actually trained v103 GPU RSI source, original 32-context/word/BPE/match state, two inherited Adam groups and exact source/DEV parity is [VERBATIM_PARITY.md](VERBATIM_PARITY.md) with `train_verbatim.py`. Use it for CPU/T4/Blackwell continuation. Nothing in this older page promotes the stand-alone candidate over the protected v103 checkpoint.

**Terminal-RSI is the system.** FA1, FA2, M3 and Poincare experiments are independent.
This directory is deliberately isolated from them.

The same `unified.py`, loss, causal three-horizon heads, RSI feedback loop
and checkpoint format run on all compute targets. `--mode` selects the
**device/execution profile**, not the model graph: every mode has exactly
**100,537,498 parameters** (2,782,515 original v103 core + 97,754,982 additive
upgrades + 1 residual gate). Blackwell is an **option**, not a larger model.

| Mode | Width | Layers | Heads | Context bytes | RSI passes | Execution |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU | 704 | 12 | 11 | 256 | 2 | float32; reproducible acceptance gate |
| Tesla T4 (16GB) | 704 | 12 | 11 | 256 | 2 | same graph on CUDA; microbatch profile |
| 96GB Blackwell | 704 | 12 | 11 | 256 | 2 | same graph; BF16 upgrade-only autocast, native SDPA, GPU window cache, effective batch 8192 |

The Blackwell card is used through **effective batch and v4 branch search**, not
by inflating the architecture. The v4 controller snapshots the same 100M model,
so multiple branch snapshots are cheap. Monitor reported peak VRAM, actual
trials/s and bytes/s; the v103 byte path, not parameter count, is the limiter
at small batches. 32GB *system RAM* is different from 95GB *GPU VRAM*. First
establish CPU and T4 parity, then run the Blackwell smoke and profile before a
long run.

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
# T4 16GB, 10-minute experiment (same 100M graph)
python terminal_rsi/train.py --mode t4 --corpus-dir /path/to/books \
  --checkpoint /path/to/drive/Terminal_RSI/unified_t4_candidate.pt \
  --minutes 10 --context-override 256

# 96GB Blackwell, same graph; use the dedicated smoke first
python -m terminal_rsi.run_blackwell_smoke \
  --output-dir /marimo/terminal_rsi_unified_v4_10m \
  --seconds 600 --context 256 --batch 8192 --rsi-version v4
```

The unified model context is 256 observed bytes; the original v103 path still
uses exactly its last 32 bytes. Use the Blackwell smoke to measure peak VRAM
and throughput before a long run. The v4 controller is the unified path; v3/off
remain parity baselines. `train_verbatim.py` is the canonical strict
checkpoint/parity entry point; `train.py` is the older standalone candidate
path and uses the same presets.

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

## Verified-action RSI mode (the original agent remains in charge)

The **same neural module** is now wired to the existing Terminal-RSI planner's
verified-action loop in [`unified_agent.py`](unified_agent.py), not forked into
a second agent. Pre-existing `terminal_agent.py` and `planner.py` own the
trusted subprocess check, success/failure receipts, actor critic and on-policy
credit. The neural specialist scores only **manifest-approved argv choices**
and receives its policy-gradient reward strictly *after* independent success
verification. Its language-model/critic loss trains causally on observed
command bytes. A learned gate controls the neural contribution to the actor.

Full actor+critic+receipt, neural+optimizer, gate, and RSI controller checkpoint
in one `.pt`, with atomic replacement. Existing original JSON checkpoints and
byte-model checkpoints are neither read nor overwritten unless explicitly
given as an import source.

```bash
# Real-command CPU gate, then verified command training:
python -m unittest terminal_rsi.test_agent_unified -v
python -m terminal_rsi.unified_agent \
  --plan terminal_rsi/examples/plan_probe.json --root . \
  --checkpoint /tmp/terminal_rsi_unified_agent_candidate.pt \
  --mode cpu --train --max-commands 1
```

The T4 and Blackwell arguments use that *same* `unified_agent` entrypoint.
For a fresh agent use `--byte-pretrain /path/to/same-mode-byte-checkpoint.pt`
to inherit the exact trained byte-model weights **and** matching Adam moments.
After an agent checkpoint exists, use the exact same preset; mismatched
optimizer shapes fail fast. Scaling to a larger preset currently requires a
separate audited architecture migration; it is not silently initialized.

Caution: full Blackwell action checkpoints can be tens of gigabytes each
because the verifier saves after every observed action. Do not use a 10-minute
Blackwell run until CPU tests, data provenance, checkpoint size and I/O
capacity have been measured. More GPU memory does not make the verifier more
reliable; externally checked terminal-task success remains the acceptance
metric.