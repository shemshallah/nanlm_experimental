 (in-vocabulary only)

| Metric | Frozen v174 (12 words) | v175 (48 words + pointer) |
|---|---:|---:|
| 55-book validation NLL ↓ | 6.070959 | **5.939311** |
| Validation top-1 | 1445/13,200 | **1573/13,200** |
| Validation top-5 | 3307/13,200 | **3596/13,200** |
| Reused 49-book TEST NLL ↓ | 6.133046 | **6.004215** |
| TEST top-1 | 1242/11,760 | **1364/11,760** |
| TEST top-5 | 2918/11,760 | **3123/11,760** |

Gain in TEST NLL = **0.128830 nats/word** (~2.10%). TEST top-1 gains **122 words**. Across 3,000 49-book block-bootstrap resamples, conditional 95% interval for NLL gain: **[0.1113, 0.1463] nats**. This is evidence of a positive diagnostic; the reused test and incomplete inherited pretraining provenance prevent pristine generalization certification.

### Causal pointer mechanism and unrelated-history control

In TEST, a prior occurrence of the latest word existed for **3490/11760** contexts; the target was among the previously observed following words in **399** contexts. Turning OFF the copy term at the selected same weights changes strict TEST top-1 from **1364 to 1325** and NLL from **6.004215 to 6.027954**. In 399 directly copyable contexts, top-1 improved from **106 to 190**, but wrong-copy contexts regressed from **225 to 180**. This exposes a concrete next research target: **train a counterfactual trust gate on repeat reliability**, instead of increasing unconditional copy gain.

A causal negative control replaces older words with history from a different book while preserving the last three words. On 1,500 test positions, correct older history has **0.0410 nats** lower word NLL than unrelated older history. This is a *distribution-shift diagnostic*, not a causal proof.
