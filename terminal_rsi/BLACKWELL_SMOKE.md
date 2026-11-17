# Terminal-RSI Blackwell 10-minute smoke

This is a **hardware, parity and hierarchical-metrics smoke** for the unified
v4 path, not a quality/frontier result. CPU, T4 and Blackwell all use the same
**100,537,498-parameter graph**; Blackwell is an execution option, not a
different model.

## The unified model

| Component | Parameters |
|---|---:|
| Original v103 byte core | 2,782,515 |
| Additive v4 upgrade branch | 97,754,982 |
| Residual mixing gate | 1 |
| **Total, every mode** | **100,537,498** |

- Graph: width **704**, **12** layers, **11** heads, context **256**, 2
  recursive passes.
- At `residual_gate=0` the original v103 logits are unchanged.
- The gate is **RSI-owned** in v4 (`freeze_residual_gate=True`): direct
  gradient descent must not open it before a verified branch. This was the
  cause of the early BPB explosion at 100M.
- Blackwell uses BF16 autocast only for the upgrade branch; the v103 path,
  RSI actor/critic and inherited AdamW groups stay FP32.

## Hardware / stack

- NVIDIA RTX PRO 6000 Blackwell Server Edition, **95.0 GiB** usable VRAM
- PyTorch **2.11.0+cu130**, CUDA **13.0**, BF16 supported
- 20 CPU threads, 32 GB system RAM
- `terminal_rsi.plan_blackwell --probe-gpu` passed

## Preflight

| Item | Value |
|---|---:|
| Total parameters | **100,537,498** |
| FP32 weights + grads + AdamW state | **1.50 GiB** |
| Two-layer segment activation floor | **16.5 GiB** |
| Estimated recompute peak | **44.0 GiB** |
| Provisional headroom | **25.4 GiB** |

## Effective batch 8192, not 16K

The Blackwell run uses **effective batch 8192** with **microbatch 1024/512**
and gradient accumulation. The v4 controller's legacy batch action no longer
overrides the runtime batch (`fixed_batch=True`); the progress label shows
`batch=fixed(8192)`.

- 16 × 512 microbatches = effective batch 8192.
- Microbatch 2048 OOMs; 1024 fits at ~58 GiB allocated; 512 is the safe
  v4-smoke setting with room for branch snapshots.
- The optimizer updates once per effective batch.

## Completed 10-minute run

Receipt: [`reports/blackwell_smoke_100m_10m/`](reports/blackwell_smoke_100m_10m/).

| Metric | Value |
|---|---:|
| Kept updates | **23** |
| Training loop | **605.5 s** |
| Updates/s | **0.038** (effective batch 8192) |
| Peak allocated | **33.46 GiB** |
| Peak reserved | **91.46 GiB** of 95.0 GiB |
| DEV BPB, 8 probes | 8.188456 → **4.279430** |
| v4 cycles | **2** |
| v4 proposals / accepted / rejected | **2 / 2 / 0** |
| v4 branch updates | **4** |
| Gate-zero original v103 parity | **exact** |
| Sealed TEST | untouched |

### Hierarchical held-out metrics

4 disjoint DEV books, 2,048 scored bytes, 1,970 characters.

| Level | Top-1 | Top-5 | Bits/unit | Loss/unit | Units |
|---|---:|---:|---:|---:|---:|
| Byte | **0.2407** | **0.5132** | **4.9356** | **3.4211** | 2,048 |
| Subword | **0.0562** | **0.2969** | **8.3607** | **5.7952** | 1,209 |
| Token | **0.0562** | **0.2969** | **8.3607** | **5.7952** | 1,209 |
| Word | **0.0000** | **0.1072** | **24.0184** | **16.6483** | 373 |
| Sentence | **0.0000** | **0.0000** | **476.0913** | **330.0013** | 21 |

Byte **BPB = 4.9356**, **BPC = 5.1310**, byte **loss = 3.4211** nats/byte.
Token and subword are the same v103 BPE units; the distinction is reported
explicitly rather than implied. Unit top-1/top-5 require every byte in the unit
to be correct.

## Important: `train_bpb` in PROGRESS is not held-out BPB

`PROGRESS train_bpb=...` is raw bit-BCE on the online-state train stream. At
effective batch 8192 the online v103 state is updated with large batches, so
this display is miscalibrated (38 at the end of the run) even while the exact
256-way held-out byte BPB is **4.94** and the DEV BPB is **4.28**. Use
`receipt.before/after`, `receipt.hierarchical`, and the v4 branch BPB for
quality claims. This is a metric-display issue, not a held-out regression.

## Parity gates

- CPU/T4/Blackwell tests pass on the live Molab Blackwell card.
- All modes expose the same parameter names, shapes and **100,537,498** count.
- CPU↔Blackwell gate-zero parity is tolerance-checked; on-device gate-zero is
  exactly identical.
- CUDA eval accepts CPU-assembled auxiliary probes; regression test exists.
- v3 and v4 share the exact same model/training graph; v4 changes the
  controller and gate ownership, not the architecture.
- GPU sliding-window cache is enabled for Blackwell; a parity test covers the
  CPU context builder.

## Files

- `blackwell_smoke_metrics.json` — full receipt, GPU peak, v4 ledger,
  hierarchical metrics
- `blackwell_smoke_summary.json` — command, preflight, corpus, receipt
- `RUN_SUMMARY.md` — compact human-readable summary
- `blackwell_smoke_console.log` — complete stdout/stderr (Molab scratch)
