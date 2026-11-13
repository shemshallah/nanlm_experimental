# v3 — frontier-style sparse hybrid prototype

`v3_model.py` is the next architecture step after the 84.86M v2 build. It is a
prototype, not a frontier-scale trained model.

## What changed over v2

| Feature | v2 | v3 |
|---|---|---|
| Attention | full causal | **sliding-window local + global every 4th layer** |
| Attention heads | full MHA | **optional GQA (`n_kv_heads`)** |
| Q/K stability | none | **QK-RMSNorm per head** |
| Position | RoPE base 10k | **YaRN/NTK RoPE, 8k position cache** |
| MoE | 128 mirror pairs, top-8 | **32 routed experts, top-2, +2 shared experts** |
| MoE dispatch | Python expert loop | **grouped/chunked top-k dispatch** |
| Memory | hashed n-gram path | **optional learned cross-attention + sparse n-gram bias** |
| MoE aux | load balance | **load balance + router z-loss** |
| Prediction | next token + S− | **multi-token prediction (MTP)** |
| LR schedule | constant / cosine | **step-based cosine by default; explicit time mode** |
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
kv heads    12 (GQA optional; 0 = MHA)
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

## Integrated next steps (Nov 12)

1. **Training harness:** `train_v3.py` implements MTP loss, fixed validation,
   time-based cosine LR, EMA, AMP, resume, and best/final/EMA artifacts.
2. **Long context:** `rope_scaling.py` implements standard, NTK, and YaRN RoPE
   with an 8,192-position cache. `V3Config.rope_scaling="auto"` selects
   standard RoPE at 1k and YaRN above the original 1k window.
3. **Memory:** `memory_attention.py` adds 1M-slot hashed 4-gram retrieval plus
   learned memory cross-attention. Enable with `--use-memory`; the store is
   built from the training split with `--memory-max-tokens`.
4. **Expert dispatch:** `moe_vectorized.py` replaces the per-expert loop with
   grouped, row-chunked top-k dispatch. `v3_model.SparseMoE` wraps it. The
   grouped path bounds peak memory at the T4 profile instead of gathering
   `(tokens × topk, d_model, expert_hidden)` weights.
5. **Evaluation:** `eval_v3.py` reports bits/token, top-1/top-5, MTP accuracy,
   router utilization, and per-corpus metrics.
6. **Single-GPU smoke:** `v3_colab.ipynb` / `COLAB_V3.md` run the mixed-corpus
   v3 smoke on a T4. An 8k full-model run needs more GPU memory than a T4.

## Remaining scale work

- Distributed expert dispatch / expert parallelism for multi-GPU runs.
- Checkpoint sharding and FP8/BF16 training at the 286M+ profile.
- A held-out document-disjoint evaluation suite, not just random windows.
- A real 20B-token run; the current evidence is short smoke runs only.

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
