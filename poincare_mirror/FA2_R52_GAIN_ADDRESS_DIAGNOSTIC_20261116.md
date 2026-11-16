# FA2 R52 — Reciprocal-gain bottleneck versus address information

**2026-11-16 UTC · `master` · FA2 only · read-only DEV · no checkpoint promotion**

## Question and fixed experiment

R50 found almost no sensitivity to shuffling learned pair addresses,
triangle-hash addresses, or oriented incidence. The inherited checkpoint has
`reciprocal.gate ≈ -0.00036708`, corresponding to a very small effective
`tanh(8*gate) ≈ -0.0029366`. R51's eight-update TRAIN-only attempt did not
establish DEV improvement and was not promoted. It remained separate from
this GitHub source investigation.

R52 tests two explanations without changing trained weights:

1. Does an almost-closed gate mask a potentially useful address representation?
2. If the gate is opened as a *controlled intervention*, does correct geometric
   address alignment begin to matter before language loss deteriorates?

Use one immutable step-2,993 checkpoint with SHA-256
`74478d2f77b3c2da7978458f3660dffe8aad788938ce4a7ab6eae7cff2de0817`.
The correct and fully scrambled address conditions share the same model,
initializations, input, and selected causal windows. Five predeclared signed
effective gain settings are `[-0.10, -0.03, 0, +0.03, +0.10]`.

Alongside those 10 conditions, evaluate the exact trained gate, the scrambled
trained gate, and reciprocal disabled: **13 fixed conditions total**. The
scramble is the exact R50 joint intervention; it does not randomly replace
whole hidden states or vary the inherited backbone. A gain of zero is a useful
negative control, not an alternative trained model.

## Guardrails

- Checkpoint, 194 inherited Adam states, causal tokenizer, and model weights
  remain untouched: the probe never instantiates an optimizer or backpropagates.
- Every cold-gate intervention restores the original parameter bit-for-bit,
  even on an exception. R50's address hooks/buffers also restore on exceptions.
- Replay the originally trained configuration after **all** conditions and
  require exactly matching per-book metrics. Compare the entire `state_dict`
  with its original saved in-memory tensors.
- Read only the whole-book NaNROUTER/Gutenberg38 DEV split, with SHA-256
  `0314f5cb62b0e06581a17ad98f3f3e93b7d0f8d285a65253adfc128b481f54cb`.
  Do not open the 14 sealed TEST books, train on DEV, or select the gain from DEV.
- Score next/+2/+3 CE, top-1/top-5, non-space CE/accuracy/top-5, and proper
  raw-byte BPB denominators. Preserve each metric, target count, and throughput
  in a machine-readable report.
- Resample **12 whole DEV books** 2,000 times, with simultaneous Bonferroni
  correction over 12 comparisons to the inherited exact checkpoint and six
  correct-address versus scrambled-address contrasts.

The address-specific `exact_advantage` is `CE_scrambled − CE_exact` at the
*same forced gain*. Positive indicates genuine dependence on correct address
alignment; a positive corrected lower confidence bound is needed before
claiming statistical evidence. This control alone cannot establish semantic
interpretation or deployment benefit.

## Results

Full matched DEV results are in
`results/FA2_R52_gain_address_DEV_20261116.json`. The 12-book protocol uses
eight 64-position windows per book, or 6,144 target positions **per arm**.

| Gain setting | Exact next-byte NLL ↓ | Non-space NLL ↓ | Exact alignment advantage vs scrambled ↑ | Adjusted 95% address interval |
|---|---:|---:|---:|---|
| Inherited −0.002937 | 2.357199291 | 2.617378646 | −0.000032385 | [−0.00013544, +0.00000659] |
| −0.10 | 2.370225151 | 2.626955182 | +0.000006259 | [−0.00000038, +0.00001792] |
| −0.03 | 2.357836366 | 2.618215035 | +0.000000854 | [−0.00000049, +0.00000244] |
| 0.00 | 2.356905798 | 2.616755319 | +0.000037720 | [−0.00000618, +0.00011629] |
| +0.03 | 2.357555290 | 2.615380005 | −0.000002901 | [−0.00002431, +0.00001686] |
| +0.10 | 2.377475669 | 2.623381606 | −0.000002017 | [−0.00014665, +0.00010910] |

All six corrected intervals include zero: **opening the global gate does not
uncover robust address-specific predictive information**. The very large
gain magnitudes −0.10 and +0.10 are significantly worse than the inherited
checkpoint for *overall* next-byte CE: the Bonferroni intervals for their
candidate-favoring gain are [−0.0223448, −0.0003518] and
[−0.0308079, −0.0044054], respectively. A gain of zero reduces mean CE
very slightly, but its corrected comparison to the inherited checkpoint
still includes zero. The exact 0-gain and reciprocal-disabled paths return
identical mean metrics on this dataset.

The inherited baseline is **3.400720 bits per literal byte**, 35.5143%
next-byte top-1, 70.4915% top-5, +2 CE 2.889248 and +3 CE 3.070085.
All other horizon/accuracy results, throughput and raw-byte denominators
are retained in the JSON. All altered modes passed complete bit-exact
restoration checks.

**Diagnosis:** this is evidence against treating the nearly closed global
feedback gate as the *only* bottleneck. The address-to-content signal
remains vanishingly small even when feedback is forcibly amplified, while
global amplification can damage the inherited LM. Next focus should be
local, causally trained pair/triangle address retrieval with a separately
calibrated residual path, zero-initialized for checkpoint parity, and
direct correct-versus-shuffled controls during TRAIN. Do not globally
amplify the current reciprocal output as a substitute for useful memory.

This is a mechanistic probe. Even a positive correct-address margin at a
forced gain would need **non-space next-byte improvement over the inherited
exact model**, longer causal contexts, train-only replication, and a complete
recoverable candidate checkpoint before any successor could be considered.

## Reproduce

```bash
cd poincare_mirror
python scripts/probe_r52_gain_address.py \
  --checkpoint /path/to/step2993.pt \
  --data-dir /path/to/verified_gutenberg38_bytes \
  --context 64 --windows-per-book 8 --batch 4 \
  --bootstrap 2000 --seed 2026 --threads 2 \
  --output results/FA2_R52_gain_address_DEV_20261116.json
python -m unittest discover -s tests -p 'test_r52_gain_address.py' -v
```

This work remains strictly inside FA2's `poincare_mirror/` directory.
FA1 and Terminal-RSI retain their separate weights, metrics, and optimizers.
