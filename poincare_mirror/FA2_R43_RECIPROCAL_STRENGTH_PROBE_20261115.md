# FA2 R43 — Does stronger reciprocal feedback actually help?

**2026-11-15 · read-only DEV diagnostic · no training and no promotion**

R42 taught the existing eight-weight causal gate to classify when the
reciprocal bridge helped. This changed its parameters but did not improve
independent DEV accuracy. The actual learned reciprocal coupling in the
step-3,025 checkpoint is extremely small:

* Stored `reciprocal.gate`: **−0.0008993984**.
* Effective `tanh(8*gate)`: **−0.0071950631**.

To distinguish a weak but promising path from an unhelpful direction, R43
tests a **read-only, per-model reciprocal strength sweep**. It changes only
the scalar gate in memory, measures the same DEV windows, restores the
original gate, and never calls an optimizer or writes a checkpoint. Negative
multipliers reverse the learned sign. A zero multiplier turns its feedback
gain off. An explicit `reciprocal.enabled=False` control checks the entire
bridge independently.

The test uses the R42 step-3,025 candidate, the same NaNROUTER allocation,
**12 independent DEV books × 4 fixed windows × 128 positions = 6,144
predictions per condition**, and 2,000 paired whole-book bootstrap samples.
The 14 independent TEST books remain sealed.
The explicit fully disabled bridge matched the zero-gain control on every
reported metric to floating-point output precision (including identical
overall CE 2.3073618611 and non-space CE 2.5657010305).

| Gate multiplier | Effective gain | +1 CE ↓ | Non-space CE ↓ | Top-1 ↑ | Top-5 ↑ |
|--:|--:|--:|--:|--:|--:|
| 0 | 0 | 2.307362 | 2.565701 | 36.7350% | 71.0286% |
| 1 (learned) | −0.00720 | 2.308245 | 2.566455 | 36.6536% | 70.8496% |
| 4 | −0.02878 | 2.311987 | 2.569118 | 36.4583% | 70.7194% |
| 16 | −0.11512 | 2.337419 | 2.590247 | 36.4258% | 70.4915% |
| 64 | −0.46048 | 2.728224 | 2.897422 | 32.2917% | 66.7155% |
| −4 (sign reversed) | +0.02878 | 2.308456 | 2.564968 | 36.8490% | 71.1914% |
| −16 (sign reversed) | +0.11512 | 2.330193 | 2.573041 | 36.8978% | 71.3704% |

Compared with zero feedback, multiplier 4 *worsened* non-space CE by
**0.003417** nats/byte (95% book-bootstrap interval of **−0.005618 to
−0.001030** for improvement). Multiplier 16 worsened it by **0.024546**
([−0.031171,−0.017866]); multiplier 64 by **0.331721**
([−0.365270,−0.292407]). The sign-reversed multiplier −4 yielded a
non-space improvement of only **0.000733** with CI
**[−0.001750,+0.002969]**, not significant. Its small top-1/top-5
increase does not establish better language likelihood.

**Result:** merely increasing feedback amplitude fails. The bridge currently
needs a more informative *causal geometry representation* and more direct
supervision of its predictive residual, not an arbitrary gain multiplier.
No trained or probed checkpoint is eligible for promotion.

## Repeat without changing learned weights

```bash
python scripts/probe_r43_feedback_strength.py \
  --checkpoint CANDIDATE/latest.pt --data-dir DATA \
  --context 128 --windows-per-book 4 \
  --multipliers '0,1,4,16,64,-4,-16' \
  --bootstrap 2000 --threads 2 --output R43_probe.json
python -m unittest discover -s tests -q
```

The next architecture hypothesis is an opt-in causal residual prediction
objective for the existing point/pair/triangle/apex path, with a detached
backbone baseline and tied byte vocabulary. Its benefit must be assessed
by the bridge-disabled and trust-neutral DEV ablations before any
independent TEST or long-context promotion gate.
