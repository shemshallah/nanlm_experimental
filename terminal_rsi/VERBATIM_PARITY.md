# Terminal-RSI V103: VERBATIM parent + CPU/T4/96GB Blackwell upgrades

**Canonical source of truth for continuing the LAST ACTUALLY TRAINED RSI:**

- `legacy_v103_reference.py`: byte-for-byte archived 82,877-character Python implementation from the original GPU training notebook. It contains the original `GPUFullSystem`, `V103ParityGPUFullSystem`, online bit state, 32 hashed contexts, 5/12-byte historic matches, BPE+word specialist keys, 12 mixers, five APMs, and persistent terminal actor/critic feedback. No source refactoring.
- `legacy_v103_local_copy.py`: verbatim local 5/12-byte causal-copy audit from the original notebook.
- `reference/nancm_v103_bpe.txt`: the original **600-rank BPE merge table**, SHA-256 `cff04c727f2d91e3ae3c2f2bdcf0adeca1422c3a8403488d96aaa759440e8ff5`.
- `verbatim_v103.py`: `VerbatimV103RSI` *contains* that exact original module, with every old parameter, persistent online buffer and historical AdamW moment. It adds the common recursive CPU/T4/Blackwell `TerminalRSI` specialist and an **exactly zero** residual gate. Original bit logits and target prefix semantics remain strictly identical until the new gate is trained.
- `verbatim_metrics.py`: exact archived 256-candidate scoring mathematics, but wrapper-aware, passing all six causal features and four local-copy fields. This fixes the silent feature omission caused by the old scorer's `isinstance(V103ParityGPUFullSystem)` test.
- `verbatim_c256.py`: frozen C 256-way DEV distribution oracle; reports true C-vs-GPU numeric discrepancies when the original C oracle file is available.
- `train_verbatim.py`: canonical book-disjoint DEV CPU-first entry point for one shared training path and full-cycle sampler, original online state changes **after each Adam step**, original terminal verifier+actor/critic feedback, reproducible checkpoints.
- `migrate_verbatim.py`: explicit capacity growth while preserving original v103 tensors and **both** old AdamW groups. New neural heads, attention, feedback and optimizers have structured partial migrations, with changed-width DEV reacceptance required.

`unified.py` supplies only the **additive, unpromoted capacity upgrade** and its three presets. The former `train.py` and `UNIFIED_MODEL.md` describe an independent candidate and **must not** be mistaken for the original RSI-led full training lineage. FA1, FA2, M3 and Poincare are separate projects.

## Exact historical architecture

The inherited original is not a Mamba state-space model and is not a Transformer top-k MoE. It has 32 adaptive context channels, real word/BPE/match specialists, 12 mixers and five adaptive probability maps, driven by a **recurrent RSI actor, feedback state, critic and terminal verifier**. Its C source-model is distinct from the GPU approximation. **Feature/path parity with C is NOT C numeric parity**; the full original trained C oracle requires its separate saved state and 256-probability reference. The historical verified C baseline was ~2.071214 BPB on its own aligned books, not an inherited GPU-v103 score.

| Setting | Inherited RSI | Additive recursion | Nominal capacity | Status |
| --- | --- | --- | --- | --- |
| CPU | Exact original weights, online buffers, 2 AdamW groups | width 256, 3 layers, 8 heads, 256-byte capacity context | CPU reference | GitHub CPU unit tests |
| Tesla T4 (16 GB) | Same exact parent and state | width 512, 8 layers, 8 heads, 1K context | Pilot; no presumed extra corpus | Requires actual GPU acceptance |
| Blackwell (96 GB GPU, 32 GB RAM) | Same exact parent and state | width 2560, 32 layers, 40 heads, up to 8192 context | Multi-billion-param, checkpoint/VRAM risk | Requires VRAM profiling, streaming and actual GPU acceptance |

IMPORTANT: The *old v103 predictor* still consumes exactly 32 prior observed bytes. The larger neural branch can later read longer causal histories, but current `VerbatimV103RSI.forward` feeds its last 32 bytes only. Do not claim 8K effective inherited context or that all upgrades reached performance parity. F32 CPU↔CUDA can differ numerically; same-device old-vs-zero-gate is exactly equal. No benchmark improvement claimed yet.

## CPU FIRST: sealed original data and notebook state

Do not run the 10-minute GPU round until the CPU read-only gate passes. All source below is on GitHub `master`. From repository root, with real **identical** original data:

```bash
python -m unittest terminal_rsi.test_v103_parity terminal_rsi.test_verbatim_training terminal_rsi.test_migrate_verbatim -v
python -m terminal_rsi.verbatim_v103 --mode cpu \
  --legacy-checkpoint "/path/to/Terminal_RSI/terminal_rsi_gpu_v103_full.pt"
python -m terminal_rsi.train_verbatim \
  --mode cpu \
  --legacy-checkpoint "/path/to/Terminal_RSI/terminal_rsi_gpu_v103_full.pt" \
  --train-stream "/path/to/original_10m_train_stream.bin" \
  --bpe-merges terminal_rsi/reference/nancm_v103_bpe.txt \
  --dev-bin "/path/to/original_win_dev.bin" \
  --output "/path/to/Terminal_RSI/verbatim_cpu_candidate.pt" \
  --seconds 0 --batch 4 --max-dev-positions 8
```

This checks **source SHA, 600 original BPE ranks and six-dimensional causal feature SHA, original sampler cursor/stride, independent DEV digest, original v103 online buffers, grouped optimizer moments and exact old-vs-zero-gate logits**, before it writes a new candidate. The strict original `win_dev.bin` can be restored using the historical notebook's SHA-pinned source cell. It does not pull random substitutes from NaNROUTER.

For real CPU continuation, after the read-only acceptance:

```bash
python -m terminal_rsi.train_verbatim --mode cpu \
  --legacy-checkpoint "/path/to/Terminal_RSI/terminal_rsi_gpu_v103_full.pt" \
  --train-stream "/path/to/original_10m_train_stream.bin" \
  --bpe-merges terminal_rsi/reference/nancm_v103_bpe.txt \
  --dev-bin "/path/to/original_win_dev.bin" \
  --output "/path/to/Terminal_RSI/verbatim_cpu_candidate.pt" \
  --resume "/path/to/Terminal_RSI/verbatim_cpu_candidate.pt" \
  --seconds 60 --batch 4 --max-dev-positions 16
```

The inherited `terminal_rsi_gpu_v103_full.pt` file is never overwritten. Exact unified resume restores old+new weights, two old Adam groups plus the upgrade group, original training-source fingerprint, BPE feature and DEV SHA, full-cycle sampler cursor, feedback state, and RNG. Independent TEST remains sealed.

## T4 and Blackwell mode changes

After the *real CPU checkpoint and independent DEV* pass:

```bash
python -m terminal_rsi.migrate_verbatim \
  --source "/path/to/Terminal_RSI/verbatim_cpu_candidate.pt" \
  --mode t4 \
  --output "/path/to/Terminal_RSI/verbatim_t4_candidate.pt"
```

Then use `python -m terminal_rsi.train_verbatim --mode t4 --resume ...` with the **same legacy checkpoint/train-stream/BPE/DEV** flags; first `--seconds 0`, then a bounded pilot. For Blackwell, migrate `verbatim_t4_candidate.pt` with `--mode blackwell` to a new candidate and again run `--seconds 0` for DEV reacceptance first. *Capacity growth is NOT function-preserving once the learned residual gate is nonzero.* Do not automatically promote it on a new GPU.

The 96 GB is VRAM, not the 32 GB host budget. A multi-billion-parameter AdamW model can exhaust host RAM or checkpoint I/O during migration. The current large Blackwell model is an ambitious candidate, not a demonstrated 96 GB fully saturated trainer; profile actual peak GPU, host RAM, time-to-checkpoint and byte/sec before raising context/batch. Use bf16 SDPA/checkpointing only on the additive branch where numerically safe.

## Acceptance gates that cannot be skipped

1. Historical 2-group Adam states remap **by historical parameter order**, not naive registration order; `v103_emit` is registered first but grouped with `v103_*` at training time.
2. Real original CPU checkpoint and all source SHA-256s match.
3. Exact original v103 forward equals new unified forward at gate zero, on identical CPU features.
4. Proper normalized full 256-way byte likelihood, top-1/top-5 and BPB, including causal BPE and local-copy fields.
5. Trained RSI updates online counts after loss/optimizer; terminal verifier updates recurrent actor and critic after the actual shell-free command result.
6. Exact restart of all model weights, RNG, optimizer moments, sampler and verified events.
7. Per-mode matched book-disjoint DEV gate, unrelated/corrupted-history controls and sealed TEST. For C numerical parity provide the original full-256 oracle and call `--reference-full256`.

**Promotion status:** the source-level CPU gates are automated; no original Drive checkpoint-backed CPU/real-book match, GPU parity, or 2.071214-BPB beating run is yet claimed.
