# FA2 R53 — Exact-pair temporal recall: implementation and negative pilot

**2026-11-16 UTC · FA2 Poincaré Mirror only · `master` source · no promotion**

## Architectural finding

R50–R52 showed that trained pair/triangle address IDs barely influence
predictions, even under forcibly enlarged reciprocal gain. Inspection of the
existing simplex `causal_store` found that it aggregates per-position states
by cumulative average without querying **past values by exact pair address**.
It is a valid causal aggregate, but does not itself perform a keyed episodic
read. This is a mechanistic limitation, not proof that address geometry is
fundamentally useless.

R53 adds a **single opt-in retrieval path inside the existing SymbioticGeometry
and shared LM**. It does not add an independent model, tokenizer, output head,
or optimizer. For each of the 28 active pair IDs at token position `t`, it:

1. Looks back a bounded 16 prior token positions (strictly `u < t`).
2. Matches only the same exact 0–32,639 pair ID.
3. Averages existing causal simplex edge representations at those prior hits.
4. Pools the successful active pair reads and returns a rank-dimensional
   vector to the same geometric feedback before the inherited `to_hidden`.
5. Returns **exactly zero** when no previous match exists.

The path has a runtime `pair_history_gain`, default **0**, and an optional
wrong-address negative control that permutes only the *query* IDs, leaving
stored prior key IDs and underlying geometric tokens intact. Permuting past
and present IDs together would preserve equality and would be an invalid
negative control for associative recall.

All original `state_dict` names, values, parameter counts, and optimizer
moments are untouched. The gain is currently **experimental runtime control,
not saved in the checkpoint or enabled for checkpoint continuation**. Any
future training integration must serialize the control, migrate the same
194-state Adam lineage, and revalidate exact zero-gain parity.

## Verification and provenance

- Parent: FA2 step 2,993 SHA-256
  `74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
- NaNROUTER/Gutenberg38 DEV checksum:
  `0314f5cb62b0e06581a17ad98f3f3e93b7d0f8d285a65253adfc128b481f54cb`.
- The 12 DEV books are independent of 12 TRAIN and 14 sealed TEST books.
- Unit tests cover exact-match recall, strictly previous-position masking,
  no future-token leakage under suffix mutations, bounded history, wrong
  query addressing, gradient flow to existing past pair values, exact
  zero-gain parity, and restoration on exceptions.
- The pilot evaluates six fixed arms on the **same** two sampled 64-position
  windows per DEV book: 1,536 positions per arm. Bootstrap resamples whole
  books 400 times with five-way Bonferroni correction, so precision is
  exploratory, not sufficient for promotion.

## Measured DEV pilot

| Arm | Next CE ↓ | Non-space next CE ↓ | +2 CE ↓ | +3 CE ↓ | Next-byte top-1 ↑ |
|---|---:|---:|---:|---:|---:|
| Protected inherited (gain 0) | 2.318347 | 2.573604 | 2.883837 | 3.087467 | 37.435% |
| Exact pair history +0.03 | 2.318112 | 2.573185 | 2.883627 | 3.087983 | 37.500% |
| Wrong-address history +0.03 | 2.318033 | 2.573166 | 2.883936 | 3.087777 | 37.370% |
| Exact pair history −0.03 | 2.318467 | 2.574076 | 2.884105 | 3.087784 | 37.500% |
| Exact pair history +0.10 | 2.318926 | 2.574127 | 2.884199 | 3.087889 | 37.370% |
| Wrong-address history +0.10 | 2.317857 | 2.572870 | 2.883983 | 3.087839 | 37.435% |

Positive +0.03 exact-history gain is only **0.000236 nats/target** over the
protected baseline; the corrected book-cluster interval includes zero.
Wrong-address +0.03 is *slightly better* on overall CE, and at +0.10 wrong
addresses clearly outperform exact ones on the point estimate. This small
pilot does **not** establish real address-dependent predictive value.

Direct paired exact-over-wrong address gains were −0.00007854 nats/target
at +0.03 (95% corrected interval [−0.00078898, +0.00057966]) and
−0.00106873 at +0.10 (interval [−0.00198943, +0.00006813]). **Both intervals
cross zero**, so the fixed-gain path fails the address-specificity gate.

Detailed +1/+2/+3 top-1/top-5, bits per literal byte, non-space accuracy,
throughput, and direct paired exact-versus-wrong intervals are recorded in
`results/FA2_R53_pair_history_DEV_pilot_20261116.json`.

**Decision: no promotion, no training, TEST sealed.** The next architectural
target is *content-aware episodic values* and a serialized, zero-initialized,
learnable local trust gate trained using TRAIN-only true/negative history,
followed by corrected independent DEV and long-context gates. Merely adding
an exact-key mean-read with a fixed gain is not enough.

## Reproduce

```bash
cd poincare_mirror
python scripts/probe_r53_pair_history.py \
  --checkpoint /path/to/step2993.pt \
  --data-dir /path/to/verified_gutenberg38_bytes \
  --context 64 --windows-per-book 2 --batch 2 \
  --bootstrap 400 --seed 2026 --threads 2 \
  --output results/FA2_R53_pair_history_DEV_pilot_20261116.json
python -m unittest discover -s tests -p 'test_r53_causal_pair_history.py' -v
```

This record concerns FA2 only. No FA1, M3, or Terminal-RSI weights were used.
