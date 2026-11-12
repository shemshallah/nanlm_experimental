# PAL R7.8 — order-sensitive rank-64 memory investigation (CPU)

**Decision: NO PROMOTION.** The accepted active checkpoint remains R7.7 at joint step **14,416** (one checkpoint, five optimizer histories). R7.8 exploratory continuations reached step 14,912, but the final candidate did not pass a matched historical test or memory-utility gate. Candidate state is not packaged as an accepted model.

## The structural defect and its repair

R7.7 `CrossWindowMirror.initial` writes masked mean embeddings for each 64-word group. Therefore every within-group permutation of tokens leaves the exact mathematical initial matrix unchanged. R7.8 implements two removable additions to **that same existing rank-64 physical matrix**:

1. `source/r78_ordered.py`: eight-bin chronological moments (monotone time gradient and alternate-bin profile) added to the block event before its original rank-one matrix write. `order_gain=0` at migration reproduces R7.7 state and logits bit-exactly in the regression sample.
2. `source/r78_bigram.py`: an asymmetric directed key(previous token) × value(next token) history sketch, folded into the same rank-64 matrix through `pair_gain=0` at migration. It uses the existing old-memory key and value projections. No second memory, optimizer history, or language model is created.

The new representations distinguish reversed historical order when their gates are active, but **representational possibility is not evidence of useful retrieval**.

## Full continuation sequence (unpromoted)

- R7.7 accepted step 14,416, source checkpoint SHA-256 `54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d`
- Chronological moment writer: 32 + 96 + 96 updates -> 14,640.
- Directed old-token pair writer (ordinary multitask LM objective): 32 + 96 -> 14,768.
- Historical-transition recall + memory-discrimination auxiliary: 48 + 96 -> **14,912**.
- Total R7.8 exploratory updates: **496**. Exact optimizer slots and original model parameters were retained through architecture migrations, not reset. No candidate weights were promoted.

All runs were on CPU with two Torch threads, two streams, preserved checkpoint batch RNG, and no independent new-book corpus. The pair-directed stage uses one target position per stream update. Training logs and JSON reports are under `reports/`.

## Matched historical evaluation: 512 positions per split

Same position indices and same evaluation code for accepted R7.7 and experimental R7.8 at step 14,912:

| Split | R7.7 accepted NLL ↓ | R7.8 experimental NLL ↓ | R7.8 improvement |
|---|---:|---:|---:|
| Development | 4.953608256 | 4.950418789 | +0.003189467 |
| Test | **5.832248628** | 5.834672540 | **-0.002423912** |

The 512-position fixed historical test is not independent book evaluation. Neither result should be compared directly to the earlier 4,096-position R7.7 test 5.77411379 because the sampled positions differ.

## Correct-memory versus unrelated-memory ablation

In a separate fixed 128-position historical probe at candidate step 14,912, the directed pair write improves test next-token NLL by ~`0.00003988` relative to disabling just that write, but worsens development by ~`0.00003830`. These differences are tiny and unstable.

A historical-transition reconstruction probe tested each sampled old key->value positive against 24 old negative values from **the same causal past**. This is a controlled in-history training task and not natural language next-token generalization. At step 14,912:

| Split | Correct-state recall CE ↓ | Unrelated-state recall CE ↓ | Wrong minus correct |
|---|---:|---:|---:|
| Development | 3.309783 | 3.317442 | +0.007659 |
| Test | 3.258588 | **3.250290** | **-0.008298** |

This fails the desired correct-memory advantage on held-out historical test, justifying rejection.

## Tests

`source/test_r77.py`: 12/12 inherited R7.7 regression checks pass.

`source/test_r78.py`: 6/6 migration and time-sensitivity checks pass, including exact unchanged prediction at zero gate and preserved old optimizer moments.

`source/test_r78_bigram.py`: 5/5 tests pass, including old weights unchanged, exact zero-gate prediction, optimizer identity, order sensitivity and propagated gradients.

**Validation limitation:** the NaNROUTER short-story release is public on GitHub, but its release asset was not accessible from this execution environment. The independent whole-book contamination screen and book-bootstrap result were not run. Do not infer independent generalization from the current measurements.

## How to reproduce

All scripts assume `PYTHONPATH=source`, root = archive extraction directory, `torch.set_num_threads(2)`, and the original accepted `checkpoint/pal_active.pt` intact. The exact sequence is:

```sh
# From accepted R7.7 step 14416; write candidate outside the accepted path.
python source/r78_ordered.py --resume checkpoint/pal_active.pt --output checkpoint/pal_r78_candidate.pt --report reports/replay_order_32.json --steps 32 --streams 2 --seq-len 2 --eval 32 --eval-batch 2 --threads 2
python source/r78_ordered.py --resume checkpoint/pal_r78_candidate.pt --output checkpoint/pal_r78_candidate.pt --steps 96 --streams 2 --seq-len 2 --eval 0 --threads 2
python source/r78_ordered.py --resume checkpoint/pal_r78_candidate.pt --output checkpoint/pal_r78_candidate.pt --steps 96 --streams 2 --seq-len 2 --eval 0 --threads 2
# New causal edge gate starts at zero, preserving all old weights and moments.
python source/r78_bigram.py --resume checkpoint/pal_r78_candidate.pt --output checkpoint/pal_r78_event_candidate.pt --steps 32 --streams 2 --eval 32 --eval-batch 4 --threads 2
python source/r78_bigram.py --resume checkpoint/pal_r78_event_candidate.pt --output checkpoint/pal_r78_event_candidate.pt --steps 96 --streams 2 --eval 0 --threads 2
python source/r78_bigram.py --resume checkpoint/pal_r78_event_candidate.pt --output checkpoint/pal_r78_event_candidate.pt --steps 48 --streams 2 --eval 0 --threads 2 --recall-weight .05 --distinguish-weight .20
python source/r78_bigram.py --resume checkpoint/pal_r78_event_candidate.pt --output checkpoint/pal_r78_event_candidate.pt --steps 96 --streams 2 --eval 0 --threads 2 --recall-weight .05 --distinguish-weight .20
```

This recreates exploratory candidate files, not accepted checkpoint promotion. `checkpoint/pal_active.pt` must remain unchanged. The accepted R7.7 archive's recorded training data and original tokenizer must match the stored hashes. The candidate experiment has not been validated on previously unseen books.

## Next scientific gate

1. Import actual NaNROUTER books (the audited release fetch and corpus preparation scripts are already included); screen whole-book overlaps with training data, split by books not adjacent positions.
2. Test order-sensitive queries that cannot be answered by exact retrieval, using cases with identical word bags but different chronological sequence. Measure correct-vs-reversed-state conditional NLL and book-bootstrap confidence intervals.
3. Only then increase memory rank and context length: expanding from rank64 to 128+ or 8192 to more words without predictive-memory sensitivity is not enough.
4. Introduce controlled recurrent/writer supervision aligned with current predictive targets. The current auxiliary increased the gate magnitude but did not yield reliable test benefit.
5. Promote only if both new-book test NLL and memory-ablation utility improve reproducibly, with preserved optimizer history and one active checkpoint.
