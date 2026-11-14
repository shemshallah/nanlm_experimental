# FA2 R37 — isolated causal multi-horizon forecasting

**Scope:** FA2 `poincare_mirror` only, on the existing unified MirrorLM
implementation. This change does **not** modify the separately trained
NaNROUTER language model, FA1 checkpoints, or the Diamond research track.

## Diagnosis

The legacy `rollout` branch of `MirrorLM.future_logits` repeatedly applies
the last trained feed-forward / MoE block to a normalized final hidden state.
A feed-forward block is not a learned time transition. It never conditions
on the unknown next byte, cannot carry that byte through the causal attention
stack, and shares weights with NOW prediction during joint training. This
is a precise structural weakness, not an empirical proof of the cause of
any specific held-out regression.

## Integrated R37 candidate

`future_mode=isolated` with `future_adapter_rank=64` adds a small auxiliary
two-step causal forecaster **inside the same FA2 MirrorLM**:

1. Read the causal NOW hidden state; detach it and the shared output embedding
   from auxiliary gradients.
2. Form the model's entire predicted next-byte probability distribution.
   Marginalize the unknown byte into an expected embedding; never feed the
   ground-truth next byte during forecasting.
3. Apply one shared rank-64 nonlinear residual transition with an independent
   bias at each horizon, producing +2 logits and then +3 logits.
4. Train only this auxiliary module with `--future-only`. Every inherited
   NOW weight and its AdamW state remain untouched.

The new output projection and step offsets are zero-initialized. The existing
NOW logits remain identical at migration. The new +2/+3 forecasts initially
equal NOW logits and need to learn; no forecast gain is assumed. The expected
embedding is a tractable approximation to autoregressive marginalization,
not an exact enumeration of all 257² possible future byte sequences.

At width 256, adapter rank 64 adds **49,984 parameters**; at the original
192D width it adds **37,504**. The exact new parameter count is checked by
`count_parameters_exact` and against the constructed model in unit tests.
Existing unrestricted autoregressive text generation continues to use the
same NOW decoder in `model.forward`; it does not depend on auxiliary logits.

## Preserve the existing FA2 checkpoint

Only a complete, architecture-compatible FA2 checkpoint with its original
weights, optimizer moments, tokenization and data fingerprints may be used.
The source release does **not** assert the latest R36 checkpoint is available
for training. Do not substitute a fresh random initialization for it or
silently replace a later R36 model with an older R30 step-2,801 checkpoint.

Copy the exact FA2 checkpoint into a **separate candidate directory** first.
`grow` atomically replaces the candidate's checkpoint after proving the
NOW logit preservation and mapping every existing Adam moment by name.

```bash
mkdir -p candidate
cp accepted-fa2/latest.pt candidate/latest.pt

python poincare_frontier.py grow \
  --run-dir candidate --future-adapter-rank 64 --future-mode isolated \
  --max-logit-drift 0.00001 --report candidate/migration.json

python poincare_frontier.py train \
  --data-dir verified-fa2-book-corpus --run-dir candidate \
  --steps 250 --batch 4 --train-context 256 \
  --future-only --future2 0.30 --future3 0.30 \
  --eval-every 25 --save-every 25 --cpu --cpu-threads 2
```

Adapt `train-context` to the verified model's saved maximum context. Neither
raw model weights nor large corpus files belong in the source repository.

## Validation before promoting a successor

1. Select candidates on *book-disjoint FA2 validation* positions with
   precisely the same tokenizer, context, seed, and byte targets.
2. NOW CE must not increase by more than 0.00002 nats/byte; NOW top-1 and
   top-5 must not decline. In `--future-only`, exact NOW preservation is
   stronger: all backbone tensors, optimizer moments and generated logits
   are unchanged up to numerical determinism.
3. Both +2 and +3 CE must improve by at least 0.001 nats/byte, with no
   decline in either top-1 or top-5. Evaluate long-context behavior and
   unrestricted generated text independently.
4. Only after selecting a candidate use untouched test books and external
   SHA-256-verified WikiText-103 test positions *once*; no test-set tuning.
5. Export the baseline/candidate reports via `scripts/eval_external_bytes.py`
   and compare aligned positions using
   `scripts/compare_isolated_future.py --before before.json --after after.json`.
   Any failed gate rejects checkpoint promotion.

## Verification status

The local CPU suite checks: causal isolation, exact NOW equality at migration,
preservation of model weights/Adam moments/RNG/step; a real CLI training pass
with the backbone frozen; generation-head invariance; parameter accounting;
and metric/identity comparison gates.

**This is an implemented and unit-tested FA2 source extension, not an
empirically promoted checkpoint.** No FA2 R37 held-out metric or free-generation
improvement is claimed without the matching complete FA2 checkpoint.
