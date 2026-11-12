# v3 — frontier-style sparse hybrid prototype

`v3_model.py` is the next architecture step after the 84.86M v2 build. It is a
prototype, not a frontier-scale trained model.

## What changed over v2

| Feature | v2 | v3 |
|---|---|---|
| Attention | full causal | **sliding-window local + global every 4th layer** |
| Q/K stability | none | **QK-RMSNorm per head** |
| Position | RoPE base 10k | **configurable RoPE base** (for long-context scaling) |
| MoE | 128 mirror pairs, top-8 | **32 routed experts, top-2, +2 shared experts** |
| MoE aux | load balance | **load balance + router z-loss** |
| Prediction | next token + S− | **multi-token prediction (MTP)** |
| Dense FFN | SwiGLU 1024 | **SwiGLU 1536** |
| Context | 512 | **1024 config default** |

Still present: 255-head geometry, 85 triheads, fields, rails, curvature,
32 learnable hyperbolic branches, and the hashed n-gram memory path.

## Default configuration

```text
vocab       32,000
ctx         1,024
d_model     384
layers      12
heads       12
SwiGLU      1,536
MoE layers  4 (last four)
experts     32 routed, top-2, 2 shared
expert FFN  1,536
MTP         2 tokens
params      286,517,376 total
active      roughly 70–80M per token
```

The sparse ratio is the point: ~286M total capacity, ~25% active compute.

## Scaling ladder

| Profile | Total params | Active/token | Tokens | Hardware | Rough GPU time |
|---|---:|---:|---:|---|---:|
| v3-small | 286M | ~75M | 20B | 8×A100 | ~10–20 days |
| v3-medium | 1.5B | ~300M | 100B | 64×A100 | ~2–4 months |
| v3-large | 10B | ~1.5B | 1T | 512×H100 | ~4–8 months |
| frontier | 70B+ | ~10B | 10T+ | 2,000–10,000 GPUs | $50M–$500M+ |

These are planning numbers, not a promise. Compute is roughly
`6 × active_params × tokens`; real utilization, failures, data, and serving
dominate.

## What “frontier territory” actually requires

- **Data:** 1T–15T deduplicated, quality-filtered tokens with a document-disjoint
  evaluation set.
- **Compute:** tens of thousands of GPU-days; a T4/Colab cannot train this.
- **Training:** distributed MoE dispatch, checkpoint sharding, FP8/BF16,
  muP/WSD schedules, expert capacity and router stability.
- **Evaluation:** bits/byte, top-1, top-5, long-context probes, routing
  utilization, zero-shot tasks, and held-out document splits.
- **Serving:** KV cache, MoE expert parallelism, quantization, retrieval.

## Next code steps

1. Add a v3 training harness with MTP loss and fixed validation.
2. Add YaRN/NTK RoPE scaling for 4k–8k context.
3. Add memory cross-attention to the hashed n-gram store.
4. Add distributed expert dispatch; the current expert loop is a correctness
   prototype, not a throughput kernel.
5. Run a 30-minute single-GPU smoke before any larger run.

## Smoke check

```sh
cd nanlm_experimental/z3
python - <<'PY'
import torch
from v3_model import V3Config, V3LM
cfg = V3Config(vocab_size=100, ctx=32, d_model=64, n_layers=4, n_heads=4,
               swiglu=128, window=16, global_every=2, moe_layers=2,
               n_experts=8, topk=2, shared_experts=1, expert_hidden=128, mtp=2)
m = V3LM(cfg)
logits, mtp, aux = m(torch.randint(0, 100, (2, 32)))
print(logits.shape, [x.shape for x in mtp], m.parameter_count())
PY
```
