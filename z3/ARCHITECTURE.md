# v3 — frontier-style sparse hybrid prototype

`v3_model.py` is the next architecture step after the 84.86M v2 build. It is a
prototype, not a frontier-scale trained model.

## What changed over v2

| Feature | v2 | v3 |
|---|---|---|
| Attention | full causal | **sliding-window local + global every 4th layer** |
| Attention heads | full MHA | **optional GQA (`n_kv_heads`)** |
| Attention output | plain | **optional sigmoid gate (`attn_gate`)** |
| Q/K stability | none | **QK-RMSNorm per head** |
| Position | RoPE base 10k | **YaRN/NTK RoPE, 8k position cache** |
| MoE | 128 mirror pairs, top-8 | **32 routed experts, top-2, +2 shared experts** |
| MoE dispatch | Python expert loop | **grouped/chunked top-k dispatch** |
| Memory | hashed n-gram path | **optional learned cross-attention + sparse n-gram bias** |
| MoE aux | load balance | **load balance + router z-loss** |
| Prediction | next token + S− | **multi-token prediction (MTP)** |
| LR schedule | constant / cosine | **step-based cosine default; WSD or explicit time opt-in** |
| Dense FFN | SwiGLU 1024 | **SwiGLU 1536** |
| Context | 512 | **1024 config default** |

## Code truth: v2 geometry is not in v3

The 255-head/Poincaré head geometry, the 85 triheads, the hyperbolic fields
and rails, the learnable curvature, and the 32 hyperbolic branches belong to
the **v2/z1/z2** lineage.  `z3/v3_model.py` does not import `head_geometry`,
`hybrid_poincare`, or `poincare_geometry`, and has no curvature parameter.
v3's “branches” are hierarchical MoE routing groups, not hyperbolic branches.

The 255-head geometry is preserved in `z1/` only as historical code; that
directory's geometry modules are not present in this repository, so z1 is an
archive, not a runnable build.  z3 is the self-contained path.

## Default configuration

```text
vocab       64,060 in the trainer (V3Config dataclass: 64,000)
ctx         1,024
d_model     384
layers      12
heads       12
kv heads    4 (GQA; 0 = MHA)
SwiGLU      1,536
MoE layers  4 (last four)
experts     32 routed, top-2, 2 shared
branches    8 branches x 4 experts
expert FFN  1,536
MTP         max lag 3 -> two heads, t+2 and t+3
params      ~316.4M (trainer and V3Config; 64,060 vs 64,000 vocab)
active      roughly 70–80M per token (estimate, not a hardware counter)
```

The sparse ratio is the point: ~316M total capacity, ~25% active compute.  The
concept loop is one shared module invoked at every block; it is not a separate
12× memory.  The hashed n-gram path is **off by default** in both `V3Config`
and `train_v3.py`; enable it explicitly with `--ngram-memory`.

The reproducible **115M** model is the `115m` preset (115,238,237 params):

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/z3_domains_115m/train.bin \
  --val-corpus data/z3_domains_115m/valid.bin \
  --tokenizer bin --out runs/z3_115m
```

The same preset is used on T4, A100, and Blackwell; only `--batch-size` and
activation checkpointing should change with hardware.

## Coherent feedback bus

Every feedback path is gated with `tanh(gate)` and every gate starts at zero.
That makes the feedback architecture numerically a no-op at step zero, while
still allowing gradients to open each path.  The loop is:

```text
attention summary ─┐
hidden summary ────┼─► route state ─► next attention input
concept/object ────┘                 └─► next MoE router
        ▲                                      │
        └──────── token concept residual ◄─────┘

object state ─► next block's object retrieval
memory summary ─► MTP context + final residual
MTP lag k branch summary ─► MTP lag k+1 context
```

Concretely:

- `feedback_token_residual` gates `loop_to_state(token_ctx)` into the residual
  stream, not only the router.
- `feedback_route_concept` gates the pooled concept/object state and the hidden
  summary into the next route state.
- `feedback_object_carry` carries token-level object state across blocks.
- `feedback_memory_mtp` feeds the post-trunk memory summary into the MTP
  context and a gated final residual.
- `feedback_mtp_carry` feeds each MTP branch summary into the next future lag.
- `aux["feedback"]` reports every gate and the route-state norm.
- `train_v3.py --feedback-gate-warmup-steps N` linearly opens the bus over the
  first N steps; `eval_v3.py` always restores progress to 1.0.

The default is all five paths on, but closed at initialization.  Ablate them
individually with `--no-feedback-*`.

## Object pointer semantics

The pair/triple pointer CE is a **self-consistency** objective: the targets are
the model's own detached hard selections.  There is no external object label in
the corpus.  `aux["object_memory"]` therefore reports pair/triple entropy,
unique ratio, top-share, pointer confidence, and gate means, and
`--object-balance-coef` adds a differentiable Switch-style balance pressure over
the 256 vertices.  A high pointer accuracy with low usage entropy means the
object memory has collapsed; do not read it as grounding.

## Measured 1,000-step profiles

All on the same 16-domain corpus, 1,024 context, four batches per step, one
T4, and a 1,000-step budget:

| Profile | Params | Bits/token | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| MHA + cosine | 112,375,040 | 3.3723 | 38.75% | 68.41% |
| GQA-2 + cosine | 111,588,608 | 3.5041 | 36.06% | 67.19% |
| MHA + attention gate + WSD | 112,899,328 | **2.9823** | **45.30%** | **73.41%** |

The gate+WSD profile is the current short-run best. This is smoke-scale
evidence, not a frontier-scale result.

## Scaling ladder

| Profile | Total params | Active/token | Tokens | Hardware | Rough GPU time |
|---|---:|---:|---:|---|---:|
| v3-small | 316M | ~75M | 20B | 8×A100 | ~10–20 days |
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
   step-based cosine/WSD LR, EMA, AMP, resume, and best/final/EMA artifacts.
2. **Attention:** optional GQA (`--n-kv-heads`) and a sigmoid attention-output
   gate (`--attn-gate`), plus QK-RMSNorm and local/global attention.
3. **Long context:** `rope_scaling.py` implements standard, NTK, and YaRN RoPE
   with an 8,192-position cache. `V3Config.rope_scaling="auto"` selects
   standard RoPE at 1k and YaRN above the original 1k window.  The 8k cache is
   headroom, not a demonstrated 8k capability: local layers still build a dense
   T×T mask, GQA expands K/V in the training path, and `generate()` has no KV
   cache.
4. **Memory:** a learned 256-slot cross-attention path is on by default.  The
   hashed n-gram store is **opt-in** (`--ngram-memory`) and is built from the
   training split with `--memory-max-tokens`.
5. **Expert dispatch:** `moe_vectorized.py` replaces the per-expert loop with
   grouped, row-chunked top-k dispatch. `v3_model.SparseMoE` wraps it. The
   grouped path bounds peak memory at the T4 profile instead of gathering
   `(tokens × topk, d_model, expert_hidden)` weights.
6. **Evaluation:** `eval_v3.py` reports bits/token, top-1/top-5, MTP accuracy,
   router utilization, and per-corpus metrics.
7. **Single-GPU smoke:** `v3_colab.ipynb` / `COLAB_V3.md` run the mixed-corpus
   v3 smoke on a T4. An 8k full-model run needs more GPU memory than a T4.

## Remaining scale work

- Distributed expert dispatch / expert parallelism for multi-GPU runs.
- Checkpoint sharding and FP8/BF16 training at the 316M+ profile.
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
