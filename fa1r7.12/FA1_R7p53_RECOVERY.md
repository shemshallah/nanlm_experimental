# FA1 R7.53 — Causal successor recovery directly from R7.49

Status: **source candidate, unpromoted**. Work is confined to `master`,
`fa1r7.12`. FA2, M3 and terminal RSI are separate experiments.

## Recoverability and lineage

The last recoverable complete FA1 parent in ChatGPT Library is
`FA1_R749_CUMULATIVE_UNPROMOTED.zip`. Its protected R7.49 checkpoint SHA-256 is
`b8dabe54e2417a75defd01c776606367a3f3b0f18f479b032f8390b6fc87ddb5`.
The former R7.51/R7.52 transient checkpoints were not archived. Their optimizer
states cannot be reconstructed or claimed as preserved. R7.53 extends the real
R7.49 checkpoint **directly**: exactly 29 inherited groups, one additive group
of four trainable tensors, 30 total. There is one model and one optimizer.

## Architecture

R7.53 adds `r753_successor_history_ids` and its causal mask as a **separate
input** from inherited `diamond_backoff_recent_ids`. Therefore extending the
successor window does not change the R7.49 distribution when new gates are
closed. The last observed token is the query. For every earlier observation
`history[t]`, only its already observed `history[t+1]` may be a successor.
No target or future symbol is used for routing.

Two separately normalized distributions are read:

1. Exact lexical predecessor → observed successor.
2. Same-oriented `C(256,2)` predecessor pair → observed successor,
   excluding lexical matches. Two independent predecessor identities and at
   least two historical hits are required before the geometric gate can open.

The inherited 256-point head lookup and 32D apex descent condition a small
trust gate. No new trunk, tokenizer, language model, or optimizer is created.
A learned bounded gain from each new distribution is incorporated through a
mathematically exact composition with the inherited R7.49 mixture:

```text
Pnew(y) = (1-w-g) PR749(y) + w Qword(y) + g Qgeom(y)
0 <= w+g <= 0.25
```

The implementation returns exactly the inherited `(alpha, q)` objects
when the two gains are zero. All new parameters are initialized with closed
gains. Explicit opening of the small new gains is required **after** parity
verification and before training.

## Repair of the historical sampler

The R7.18 selector required more than 32,768 prior within-book tokens,
which prevented short NaNROUTER documents from participating. R7.53 provides
an explicit `position_selector` hook and opt-in `balanced_causal_positions`
implementation, with per-book first-remainder sampling and causal BOS masks.
All earlier FA1 rounds retain their original sampler by default. The R7.53
caller must explicitly request the balanced sampler.

## Verification and promotion gates

1. Restore the R7.49 parent from the exact SHA-256-verified archive.
2. Sample TRAIN and book-disjoint DEV only; keep independent TEST unopened.
3. Verify attached zero-gain bitwise parity against R7.49 on identical inputs.
4. Train exact word successor first, then selectively open geometry.
5. Evaluate identical DEV positions against inherited R7.49 and word-only.
6. Ablate wrong pair address, wrong successor values, foreign-book history,
   and apex conditioning. Resample by independent book, not individual tokens.
7. Validate saved/reloaded model tensors, all 29 inherited optimizer groups,
   their Adam moments, and appended group 30.
8. Add 512/2048/8192-word context and independent long-context gates.
9. Require improved DEV and all attribution gates before any TEST gate.
10. Do not promote or overwrite the protected checkpoint automatically.

GitHub Actions workflow verifies source-level contracts only. It does not
train or score the missing checkpoint; a runner-accessible, SHA-256-verified
R7.49 archive is needed for checkpoint-backed continuation.
