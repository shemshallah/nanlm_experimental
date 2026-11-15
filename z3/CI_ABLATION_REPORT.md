# Z4 GitHub CI and ablation report

Run: <https://github.com/shemshallah/nanlm_experimental/actions/runs/43742177050>  
Commit: `8fe4b9c`  
Runner: GitHub-hosted `ubuntu-latest`  
Note: no self-hosted `NaNLMind` runner is registered on the repository.

## Test gate

```text
Ran 92 tests in 19.514s
OK
```

The suite includes the text-sidecar/domain-mixture regression that reproduced and
now guards the future-composition logits shadowing bug.

## Ablation matrix

Tiny wiring gate, not a quality benchmark:

- d_model 128, 4 layers, ctx 128, 8 steps, batch 1
- C(256,2) and C(256,3) concept/object memory
- MTP 2, composition head, future structure losses
- generated tagged math corpus

| Row | Steps | Non-finite | Parameters | Final loss | Bits/byte |
|---|---:|---:|---:|---:|---:|
| `baseline_no_concept` | 8 | 0 | 19,907,276 | 10.8468 | 6.8293 |
| `concept_full` | 8 | 0 | 20,371,022 | 10.8202 | 6.8126 |
| `concept_no_mtp_structure` | 8 | 0 | 20,371,022 | 10.8177 | 6.8110 |
| `concept_pair_only` | 8 | 0 | 20,371,022 | 10.8201 | 6.8126 |
| `domain_mixture` | 8 | 0 | 20,369,486 | 10.9293 | 6.8856 |

All rows completed and reported zero non-finite skips. The loss/BPB spread is
tiny and not evidence that any feature is better; this run only proves the
paths execute, evaluate, and remain finite.

## Bugs exposed and fixed

1. MTP block experts `2` exceeded the default MoE group size `4`; the ablation
   base now sets group size `2`.
2. Future-composition logits reused the variable name `logits`, shadowing main
   model logits during evaluation and causing `Target 70 is out of bounds`.
   Renamed to `future_comp_logits`.
3. `LayeredFlatTokenizer.separator` is a scalar, so multi-corpus domain mixing
   produced a 0-D separator array. Separators are now normalized to 1-D.
4. The combinatorial-memory test still assumed the old tensor-only return;
   updated for the `(out, aux)` API.

## Reproduce

```sh
gh workflow run z3-test-ablate.yml \
  -f runner=ubuntu-latest -f steps=8
```

Artifact: `z3-ablation-summary`.
