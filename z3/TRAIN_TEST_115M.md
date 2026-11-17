# z3 115M train/test protocol

This is the runbook for the fixed 115M model.  It is deliberately staged: do
not jump from the data build to a long run.

## 0. Build and audit the data

```sh
python z3/build_domain_bins.py \
  --corpus arxiv=corpora/domains/arxiv.txt \
  --corpus code=corpora/domains/code.txt \
  --corpus math=corpora/domains/math.txt \
  --out data/z3_domains_115m \
  --max-bytes-per-domain 50000000 \
  --train-fraction 0.98 --valid-fraction 0.01
```

Check `data/z3_domains_115m/manifest.json`:

- `document_framing.enabled = true`
- `dedupe.enabled = true`
- every domain has a nonzero `train` count
- `valid`/`test` are nonzero for domains with enough documents
- `bins.train.sha256` and `bins.valid.sha256` are recorded

## 1. Five-step smoke

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/z3_domains_115m/train.bin \
  --val-corpus data/z3_domains_115m/valid.bin \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer bin --steps 5 --eval-before \
  --batch-size 2 --grad-checkpoint \
  --out runs/z3_115m_smoke
```

Must see:

```text
params=115,238,237
[preset] parameter count matches ...
[moe] ... branches=4 experts_per_branch=3 degenerate=False
[sidecar] ... enabled
```

and zero non-finite skips.

## 2. 300-step stability gate

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/z3_domains_115m/train.bin \
  --val-corpus data/z3_domains_115m/valid.bin \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer bin --steps 300 --eval-every 50 \
  --batch-size 4 --grad-checkpoint \
  --out runs/z3_115m_stability
```

Pass conditions:

- no non-finite loss or gradient,
- `feedback.progress` advances toward 1.0 after step 500 (for a 500-step
  warmup, the 300-step gate is intentionally partial),
- `aux["feedback"]` is finite for every block,
- no dead MoE branch,
- validation BPB is finite and decreasing.

## 3. Feedback ablation screen

Run the 5k-step screen with the same data, seed, and token budget:

```sh
python z3/train_v3.py --preset 115m --corpus ... --val-corpus ... \
  --steps 5000 --out runs/z3_115m_full

python z3/train_v3.py --preset 115m --corpus ... --val-corpus ... \
  --steps 5000 --no-feedback-token-residual --no-feedback-route-concept \
  --no-feedback-object-carry --no-feedback-memory-mtp \
  --no-feedback-mtp-carry --out runs/z3_115m_no_feedback
```

Compare at equal tokens, not equal wall time.  The primary readout is dev BPB;
the secondary readout is `aux["feedback"]` gate usage.  A feedback path that
stays at exactly zero for the whole screen is not being used.

## 4. Main run

```sh
python z3/train_v3.py --preset 115m \
  --corpus data/z3_domains_115m/train.bin \
  --val-corpus data/z3_domains_115m/valid.bin \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer bin --steps 20000 --batch-size 8 --grad-checkpoint \
  --out runs/z3_115m
```

Artifacts:

```text
runs/z3_115m/best.pt
runs/z3_115m/final.pt
runs/z3_115m/ema.pt
runs/z3_115m/config.json
runs/z3_115m/history.json
runs/z3_115m/result.json
runs/z3_115m/last.pt
```

## 5. Test

Per-domain held-out test:

```sh
python z3/eval_domains.py \
  --checkpoint runs/z3_115m/best.pt \
  --test-dir data/z3_domains_115m/test \
  --data-manifest data/z3_domains_115m/manifest.json \
  --tokenizer layered --document-framing --sidecar --require-sidecar \
  --ctx 512 --batch-size 8 --device cuda \
  --out runs/z3_115m/test_domains.json
```

External WikiText-103 test:

```sh
python z3/eval_v3.py \
  --checkpoint runs/z3_115m/best.pt \
  --corpus wt103=data/eval/wiki.test.raw \
  --tokenizer layered \
  --document-framing --sidecar --require-sidecar --full \
  --ctx 512 --batch-size 8 --device cuda \
  --out runs/z3_115m/eval_wt103.json
```

## 6. Validate the completed run

`eval_domains.py` reports per-domain BPB, token-weighted BPB, macro BPB, and a
domain bootstrap 95% interval.  `eval_v3.py --full` scores every target token
with exact UTF-8 byte denominators.

Validate the completed run against the promotion gates:

```sh
python z3/validate_run.py \
  --run runs/z3_115m \
  --test runs/z3_115m/test_domains.json \
  --baseline runs/z3_115m_no_feedback/test_domains.json \
  --out runs/z3_115m/validation.json
```

The validator checks parameter count, non-finite skips, feedback warmup/final
progress, sidecar + document framing, gate usage, object gate, MoE layout, and
baseline/per-domain regressions.  It exits non-zero on error-severity failures.

## 7. GitHub-hosted CPU smoke

The manual workflow
[`z3-115m-github-smoke.yml`](../.github/workflows/z3-115m-github-smoke.yml)
runs on `ubuntu-latest`:

- the full z3 unit suite,
- a tiny document-disjoint corpus build,
- 115M parameter-count + full feedback forward check,
- one 115M optimizer step at `ctx=8`, batch 1,
- `eval_domains.py` and `validate_run.py`,
- artifact upload.

This is a wiring smoke, not a quality run. A GitHub-hosted CPU runner cannot
train the 115M model to convergence; use a GPU/self-hosted runner for the real
10k–20k step run.

## 8. One-command pipeline

Once the corpora are downloaded:
```sh
python z3/run_115m_pipeline.py \
  --stages build,train,eval,wt103,validate \
  --corpus arxiv=corpora/domains/arxiv.txt \
  --corpus code=corpora/domains/code.txt \
  --corpus math=corpora/domains/math.txt \
  --wt103 data/eval/wiki.test.raw \
  --out runs/z3_115m
```

Use `--dry-run` first to print every command.  `--baseline-test` adds the
no-feedback comparison to the final validation.

## 9. Promotion gates

- full feedback vs no-feedback: at least `-0.02` BPB on dev with a 95% CI
  excluding zero, or a documented negative result;
- WikiText-103 not worse than `+0.03` BPB vs the no-feedback control;
- no domain regresses by more than `0.05` BPB;
- router: branch entropy healthy, no dead branch, no expert below 0.5% load;
- object memory: gate mean > 0.05, >100 distinct pairs per 1k tokens,
  top pair < 10% of reads;
- feedback: `progress` reaches 1.0, and at least one of the five paths opens;
- throughput: concept + feedback step time is measured and reported.

If a gate fails, stop and inspect `history.json`, `result.json`, and
`test_domains.json` before scaling.
