# FA2 R69: user-directed activation of the unified causal future-memory path

The canonical `fa2_active.load_active_fa2` loads the R69 **step 4,465**
checkpoint from the recoverable R69 archive, checks its SHA-256 against
`FA2_ACTIVE.json`, and restores the same `MirrorLM` and one AdamW optimizer.
The 197 inherited AdamW states remain, with the new +2/+3 future-memory
weight and its 198th optimizer state appended in historical optimizer order.

Install the checkpoint from the complete R69 recovery ZIP, then use:

```python
from poincare_mirror.fa2_active import load_active_fa2
model, optimizer, raw = load_active_fa2(
    '/path/to/FA2_R69_STEP4465_CANDIDATE.pt', include_optimizer=True)
assert raw['step'] == 4465 and len(optimizer.state) == 198
```

R69 joins the inherited backbone, multiscale geometric/lexical successor
memory, and strictly causal observed-book +2/+3 successor channels. The
`PairSuccessorTrust.future_weight` is trained but belongs to the same model.
R68 remains the independently recoverable rollback artifact; FA1 and M3
are independent architectures and are not loaded here.

## DEV evidence and limits

Twelve disjoint DEV books, each with 128 warmup and 128 scored positions,
were used for the recorded comparison against neutral history:

| Metric | Neutral | R69 exact | Change |
|---|---:|---:|---:|
| Next CE (nats/byte) | 3.457414 | 3.281490 | -0.175925 |
| +2 CE (nats/byte) | 3.842267 | 3.750476 | -0.091791 |
| +3 CE (nats/byte) | 3.950678 | 3.873201 | -0.077477 |
| +2 top-1 | 17.3177% | 15.8854% | -1.4323 pp |
| +3 top-1 | 16.8620% | 14.4531% | -2.4089 pp |

The +2 and +3 corrected book-level 95% confidence intervals versus neutral
include zero (lower bounds -0.016351 and -0.009465). **The strict DEV gate
failed**, and multi-horizon top-1 regressed. R69 is active only because the
user explicitly directed promotion with these caveats. No independent TEST
books were opened and no state-of-the-art superiority is claimed. Further
training must continue from the complete step-4,465 checkpoint and preserve
its inherited moments. The original R69 recovery ZIP contains the trained
checkpoint and TRAIN report, not the full DEV per-book JSON.
