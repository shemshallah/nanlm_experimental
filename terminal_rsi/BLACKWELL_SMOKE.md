# Terminal-RSI Blackwell 10-minute smoke

This is a **hardware, parity and throughput smoke** for the unified v4 path,
not a quality or frontier result. CPU, T4 and Blackwell all use the **same
6,495,162-parameter graph**; Blackwell is an execution option, not a larger
model.

## The unified model

| Component | Parameters |
|---|---:|
| Original v103 byte core | 2,782,515 |
| Additive upgrade branch | 3,712,646 |
| Residual mixing gate | 1 |
| **Total, every mode** | **6,495,162** |

At `residual_gate=0` the original v103 logits are unchanged. The upgrade branch
is BF16-autocast only on Blackwell; the archived v103 path, RSI actor/critic and
inherited AdamW groups stay in their original FP32 execution path.

## Hardware / stack

- NVIDIA RTX PRO 6000 Blackwell Server Edition, **95.0 GiB** usable VRAM
- PyTorch **2.11.0+cu130**, CUDA **13.0**, BF16 supported
- 20 CPU threads, 32 GB system RAM
- `terminal_rsi.plan_blackwell --probe-gpu` passed

## Preflight

| Item | Value |
|---|---:|
| Total parameters | **6,495,162** |
| FP32 weights + grads + AdamW state | **0.097 GiB** |
| BF16 activation floor (batch 512) | **0.375 GiB** |
| Planned headroom | **86.9 GiB** |
| Gradient checkpointing | off (3 layers) |

## Run it

```bash
cd /marimo/nanlm_experimental
python -m terminal_rsi.run_blackwell_smoke \
  --output-dir /marimo/terminal_rsi_unified_v4_10m \
  --seconds 600 --context 256 --batch 8192 \
  --max-dev-positions 8 --topk-probes 1 \
  --rsi-version v4 --rsi-v4-cycle-updates 500 \
  --rsi-v4-horizon 5 --rsi-v4-candidates 2 \
  --rsi-v4-shards 4 --rsi-v4-shard-positions 128 \
  --rsi-v4-critic-steps 16 --rsi-v4-policy-steps 8
```

`--rsi-version v3` or `off` are the parity baselines. `v4` is the unified
counterfactual path: real branch training, paired shard scoring, critic/world
model, proposal policy and replay.

## 30-second scaling probes (v4, cycle 500)

| Batch | Updates/30 s | Updates/s | Peak allocated | Peak reserved | v4 cycles / accepted |
|---:|---:|---:|---:|---:|---:|
| 512 | 1,020 | 34.0 | 1.503 GiB | 2.904 GiB | 2 / 1 |
| 4,096 | 1,041 | 34.7 | 1.557 GiB | 21.629 GiB | 2 / 1 |
| 8,192 | 1,046 | 34.9 | 1.564 GiB | 44.0 GiB | 2 / 1 |

The per-update rate is flat because the v103 byte path is the limiter, not
window assembly. **Bytes/s scales with batch**: batch 8,192 processes roughly
8× the training bytes of batch 1,024 at the same update rate. Batch 8,192 is
the chosen smoke setting; batch 16,384 approaches the allocator reservation
limit (44 GiB at 8,192) and was not used blindly.

The GPU window cache (zero-padded sliding view, GPU gather of contexts and
auxiliary features) is enabled by default for Blackwell. It removes the
per-step CPU window construction but does not change the v103 forward cost.

## Completed 10-minute unified-v4 smoke

Receipt: [`reports/blackwell_smoke_10m/`](reports/blackwell_smoke_10m/).

| Metric | Value |
|---|---:|
| Kept updates | **21,040** |
| Training loop | **600.0 s** |
| Updates/s | **35.07** |
| Peak allocated | **1.564 GiB** |
| Peak reserved | **44.0 GiB** of 95.0 GiB |
| v4 cycles | **42** |
| v4 proposals / accepted / rejected | **84 / 1 / 83** |
| v4 branch updates | **630** |
| DEV BPB, 8 probes | 8.188456 → **3.854028** |
| Bit accuracy, 8 probes | 0.359375 → **0.843750** |
| Gate-zero original v103 parity | **exact** |
| Sealed TEST | untouched |

The v4 controller is deliberately conservative here: 1 accepted branch in 42
cycles. This is a throughput/hardware/parity smoke, not a quality claim; the
DEV probe is only 8 positions. The full receipt and v4 event ledger are in
`blackwell_smoke_metrics.json` and `blackwell_smoke_summary.json`.

## Parity gates

- 14/14 CPU/T4/Blackwell tests pass on the live Molab Blackwell card.
- All modes expose the same parameter names, shapes and **6,495,162** count.
- CPU↔Blackwell gate-zero parity: max abs logit error below `2e-3` (FP32 CPU
  vs FP32 v103 path on CUDA).
- On-device `old_v103_logits_identical_at_gate_zero` is exactly true.
- CUDA eval accepts CPU-assembled auxiliary probes; regression test added.
- v3 and v4 share the exact same model/training graph; v4 changes the
  controller, not the architecture.

## What the smoke deliberately does not do

- No tens-of-GiB optimizer checkpoint or champion is written.
- It is not a quality claim; DEV probes are small and the model is fresh-start.
- It does not replace the sealed TEST or the v3 CPU acceptance path.
- It does not use v4 branch snapshots larger than the card can hold; the 6.5M
  graph makes full model/optimizer snapshots cheap.

## Outputs

- `blackwell_smoke_metrics.json` — full receipt, including `gpu` peak memory
  and `rsi_v4` ledger
- `blackwell_smoke_summary.json` — command, preflight, corpus, receipt
- `RUN_SUMMARY.md` — compact human-readable summary
- `blackwell_smoke_console.log` — complete stdout/stderr
