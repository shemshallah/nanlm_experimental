# 12h checkpoint: full per-domain table on GPU (2026-11-22)

Full 1M-token-per-group evaluation of the 175,764,096-token causal 84M
checkpoint, run on a T4 through the single-bundle mount workaround.

## Protocol

- Checkpoint: `step 30002`, 84,455,112 params, `causal_concept=True`
- Device: T4, bf16 AMP
- Cap: **1,000,000 tokens per group**, `test.bin` slices
- Chunk-level held-out, same source; optimistic relative to external tests
- PTB remains the clean OOD benchmark

## Results

| Domain | Tokens | BPB | Top-1 | Top-5 | 6h BPB | Δ BPB |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb-Edu | 997,983 | **1.3064** | 29.72% | 49.63% | 1.3382 | −0.0318 |
| Code | 706,202 | **1.0458** | 50.09% | 70.41% | 1.0865 | −0.0407 |
| Math | 997,983 | **1.0454** | 50.16% | 71.48% | 1.0842 | −0.0388 |
| NaNROUTER news | 40,880 | **1.4858** | 25.28% | 44.46% | 1.5261 | −0.0402 |
| Cosmopedia | 997,983 | **0.9669** | 35.00% | 54.27% | 0.9998 | −0.0329 |
| arXiv | 137,970 | **1.2212** | 34.01% | 53.41% | 1.2572 | −0.0360 |
| PubMed | 29,127 | **1.3644** | 32.22% | 49.52% | 1.3990 | −0.0346 |
| AG News | 52,122 | **1.4533** | 30.01% | 46.65% | 1.5026 | −0.0493 |
| TinyStories | 52,122 | **0.9400** | 45.20% | 70.81% | 0.9880 | −0.0480 |
| FineWeb-70M | 147,168 | **1.4918** | 26.77% | 45.99% | 1.5220 | −0.0302 |
| Law | 22,995 | **1.4072** | 29.98% | 48.71% | 1.4393 | −0.0321 |
| Patents | 18,907 | **1.1140** | 34.89% | 53.56% | 1.1491 | −0.0351 |
| Finance | 21,973 | **1.1257** | 31.91% | 52.19% | 1.1580 | −0.0323 |
| Security | 15,330 | **1.3289** | 35.73% | 54.44% | 1.3789 | −0.0500 |
| Standards | 9,709 | **1.5042** | 28.71% | 46.81% | 1.5360 | −0.0318 |
| Chemistry | 15,841 | **0.9376** | 41.15% | 62.09% | 0.9871 | −0.0495 |
| Recipes | 21,462 | **1.2931** | 35.11% | 56.16% | 1.3480 | −0.0549 |
| QA | 21,462 | **1.3306** | 28.75% | 48.21% | 1.3591 | −0.0285 |

Every group improved. Typical gain is **−0.03 to −0.055 BPB**, mean about
**−0.039**. The earlier CPU interim table had Code at 0.9102; the full
1M-token pass corrects it to **1.0458**, confirming that the CPU value was a
sample-composition artifact.

## Clean OOD

| PTB | 124.18M | **175.76M** |
|---|---:|---:|
| BPB | 1.7624 | **1.7349** |
| Top-1 | 28.48% | **29.04%** |
| Top-5 | 40.34% | **41.01%** |

## Persistence

- Kernel: `shem42/z4-gpu12h-domain-eval-v2`
- JSON: `per_domain_gpu12h.json` in kernel output (also downloaded locally to
  `z4_domain12v2_out/per_domain_gpu12h.json`)
- Input bundle: `shem42/z4-domain-eval-inputs-12h` (single 363 MB kernel
  source: `final.pt`, `test.bin`, manifest, tokenizer)
