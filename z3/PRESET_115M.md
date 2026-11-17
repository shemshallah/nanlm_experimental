# z3 115M preset

The `115m` preset is the fixed z3 model used on every GPU, including Blackwell.
It is **115,238,237 parameters** and deliberately reuses the 115M-class
Blackwell profile rather than switching to the ~316M default on larger
hardware.

```text
vocab            64,060
ctx              512 (RoPE headroom to 8,192)
d_model          320
layers           10
heads            10 Q / 2 KV
SwiGLU           1,280
MoE layers       3 (last three)
routed experts   12, top-2
shared experts   2
expert hidden    1,280
MoE branches     4 branches x 3 experts (real top-2-of-3)
concept loop     shared 256-vertex C(256,2) / C(256,3) memory
object IDs       32,640 x 64
MTP              t+2 and t+3 (mtp=3 -> two heads)
memory           256 learned slots; hashed n-gram OFF by default
feedback         token residual, route/concept, object carry,
                 memory->MTP, MTP->MTP; all gates closed at init,
                 linearly opened over 500 steps
params           115,238,237
```

The full staged train/test protocol is in
[`TRAIN_TEST_115M.md`](TRAIN_TEST_115M.md).  The preset is a CLI argument
bundle; explicit flags always win.  For the short-run Kaggle profile and the
parameter/compute audit, see
[`KAGGLE_10M_ARCHITECTURE.md`](KAGGLE_10M_ARCHITECTURE.md).

The end-to-end driver is:

```sh
python z3/run_115m_pipeline.py --dry-run \
  --corpus arxiv=corpora/domains/arxiv.txt \
  --corpus code=corpora/domains/code.txt

python z3/run_115m_pipeline.py \
  --stages build,train,eval,validate \
  --corpus arxiv=corpora/domains/arxiv.txt \
  --corpus code=corpora/domains/code.txt \
  --out runs/z3_115m
```

```sh
python z3/train_v3.py --preset 115m ...
```

## Why 115M on Blackwell too

A larger GPU should buy **more tokens and more seeds**, not a different model.
Keeping the 115M preset fixed makes T4, A100, and Blackwell runs comparable.
On a 96GB Blackwell card, raise `--batch-size`; on a 16GB T4, keep batch 4–8
and `--grad-checkpoint`. Do not silently switch to the 316M default.

## Build the document-framed corpus

The builder splits documents before tokenization, adds `<doc>`/`<eos>` frames,
drops exact duplicate documents across domains, and writes a manifest with
hashes and counts.

```sh
python z3/build_domain_bins.py \
  --corpus arxiv=corpora/domains/arxiv.txt \
  --corpus chemistry=corpora/domains/chemistry.txt \
  --corpus code=corpora/domains/code.txt \
  --corpus finance=corpora/domains/finance.txt \
  --corpus humanities=corpora/domains/humanities.txt \
  --corpus law=corpora/domains/law.txt \
  --corpus math=corpora/domains/math.txt \
  --corpus multilingual=corpora/domains/multilingual.txt \
  --corpus news=corpora/domains/news.txt \
  --corpus patents=corpora/domains/patents.txt \
  --corpus pubmed=corpora/domains/pubmed.txt \
  --corpus qa=corpora/domains/qa.txt \
  --corpus recipes=corpora/domains/recipes.txt \
  --corpus security=corpora/domains/security.txt \
  --corpus standards=corpora/domains/standards.txt \
  --out data/z3_domains_115m \
  --max-bytes-per-domain 50000000 \
  --max-tokens-per-domain 12000000 \
  --train-fraction 0.98 --valid-fraction 0.01
```

`wikipedia-en` is intentionally omitted from the training list above so the
WikiText-103 test remains clean. Use its held-out slice as an in-domain test.

If you train from text instead of bins, pass `--document-framing` and a
document-disjoint `--val-corpus`; the trainer now accepts a separate validation
file for text mode and no longer uses the tail of the concatenated stream
silently.

## Train

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/z3_domains_115m/train.bin \
  --val-corpus data/z3_domains_115m/valid.bin \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer bin \
  --vocab-size 64060 \
  --layered-model z3/tokenizers/layered_v1/layered_tokenizer.json \
  --batch-size 8 --grad-checkpoint \
  --out runs/z3_115m
```

The trainer prints two contracts to check before a long run:

```text
[model] params=115,238,237 ...
[preset] parameter count matches 115m: 115,238,237
[moe] layout=[{... 'branches': 4, 'experts_per_branch': 3, 'degenerate': False}]
```

If the parameter count or MoE layout changes, stop and inspect the flags.

## Evaluate

`eval_v3.py` now accepts the sidecar and document framing that training used.
This matters: without a sidecar the concept encoder returns a neutral concept,
so a concept-trained checkpoint would be evaluated off-distribution.

Per-domain held-out text written by the builder is under
`data/z3_domains_115m/test/<domain>.txt` and
`data/z3_domains_115m/valid/<domain>.txt`.  Run the full per-domain test suite:

```sh
python z3/eval_domains.py \
  --checkpoint runs/z3_115m/best.pt \
  --test-dir data/z3_domains_115m/test \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer layered --document-framing --sidecar --require-sidecar \
  --ctx 512 --batch-size 8 --device cuda \
  --out runs/z3_115m/test_domains.json
```

The output includes per-domain BPB, token-weighted and macro BPB, a domain
bootstrap 95% interval, routing diagnostics, and the feedback-bus metrics.

For the external WikiText-103 test, use `eval_v3.py` directly:

```sh
python z3/eval_v3.py \
  --checkpoint runs/z3_115m/best.pt \
  --corpus wt103=data/eval/wiki.test.raw \
  --tokenizer layered \
  --document-framing --sidecar --require-sidecar --full \
  --ctx 512 --batch-size 8 --device cuda \
  --out runs/z3_115m/eval_wt103.json
```

Use `--full` for every target token
and exact byte denominators, not the sampled `--batches` mode.

## Fixed code truths

- `eval_v3.py` builds a `SpanSidecar` for each evaluation window.
- Document framing is explicit in the bin builder, the text loader, and eval.
- Exact duplicate documents are dropped across domains by default
  (`--no-dedupe` to disable), and the manifest records the count.
- Text-mode `--val-corpus` is respected; domain-mixture runs warn if it is
  absent.
- `--ngram-memory` is opt-in; the CLI default now matches `V3Config`.
- 12 experts / 4 branches gives a real top-2-of-3; a degenerate 6 x 2 layout is
  warned about at construction and printed in `config.json`.
- The MTP block layout is 2 branches x 4 experts, not the old degenerate 4 x 2.
- CPU attention uses the deterministic SDPA math kernel; CUDA keeps the fused
  kernel selection.
- The coherent feedback bus is on by default and starts closed:
  `--feedback-token-residual`, `--feedback-route-concept`,
  `--feedback-object-carry`, `--feedback-memory-mtp`,
  `--feedback-mtp-carry`.  `aux["feedback"]` reports every gate.
- Feedback gates are closed at step 0 and linearly opened over the first
  500 steps (`--feedback-gate-warmup-steps`); eval always uses progress 1.0.
- Object pointer diagnostics and a differentiable vertex-balance term are in
  `aux["object_memory"]` / `aux["object_balance"]`.
- The default parameter count in the docs is 316,392,727, not the historical
  286.5M figure.
