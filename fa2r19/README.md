# FA2.1 — PAL R19 unified model

**Code name:** FA2.1  
**Directory:** `fa2r19/`  
**Lineage:** PAL R10 → R11 → R12/R15 → R16 → accepted R18 → R19 semantic audit.

This is **one unified model**, not a fresh model. The R19 semantic-adapter candidates were rejected; the accepted checkpoint is the R18 trained checkpoint and its inherited optimizer history.

## Accepted model identity

- Checkpoint name: `checkpoint/r12_active.pal12`
- Accepted checkpoint SHA-256: `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- Historical R12 optimizer updates: 10,756
- Original R10 / R11 update counts: 2,020 / 130
- One shared model weight set, with the inherited optimizer histories.
- Memory: rank-64 mirror state + local attention/FFN + gated 2,048-position remote lexical retrieval.

## Scientific status

R18's hard-negative **lexical** retrieval reached 95.08% on a historical test and improved held-out historical language BPB. The R19 blind semantic experiments failed promotion: novel paraphrase/relation and two-hop performance remained weak. **FA2.1 is a research language model, not yet a frontier-capable model.**

## Source & artifacts

The reproducibility parent is `PAL_R19_SEMANTIC_FRONTIER_AUDIT_ONE_CHECKPOINT_FULL.zip`, SHA-256 `0761209f51c7eef54e6e0cee6606d01ede4b2231b6dd1222e481e7dc7b20eef9`. It is available in the owner's ChatGPT Library.

The large trained checkpoint and dataset must be byte-verifiable by the checksum above. Do **not** substitute randomly initialized weights and call them FA2.1. GitHub now hosts the byte-verified accepted checkpoint in compressed parts and the archived source modules. Corpus and tokenizer archives remain in the original full reproducibility ZIP.

## Continuation policy

1. Only promote after frozen, author-disjoint held-out dev/test gates and inherited short/long-context regression checks.
2. Train a **single unified model**; preserve all prior weights and Adam moments; use atomic replacement of the accepted checkpoint.
3. Keep rejected experiments as reports and source, never competing accepted checkpoints.
4. Continue semantic/paraphrase and hard two-hop tests; compare with non-neural exact match and equal-compute baselines.
5. For every promotion record the accepted checkpoint SHA-256, optimizer continuity, corpus manifest, benchmark results, and generation steps.

For local verification from the complete archive:

```bash
python -m pip install -r requirements.txt
python source/r18_verify_lineage.py --root .
python source/test_r18.py .
python source/test_r19.py --root .
```


## Verified GitHub recovery (2026-11-12)

The trained accepted checkpoint is now stored here in **36 binary Git blobs** at `checkpoint_parts/` (73,823,829 compressed bytes). All 36 Git blob hashes independently match the locally decoded transfer parts. The full uncompressed checkpoint was reconstructed locally and verified against the original manifest SHA-256. No weights, Adam history, or step counts were reset.

To recover after cloning this repository:

```bash
python fa2r19/restore_fa2_checkpoint.py --root fa2r19
```

This produces `fa2r19/checkpoint/r12_active.pal12` (121,882,910 bytes). The script checks each part's length and SHA-256, the concatenated compressed stream, and the restored checkpoint before atomically installing it. The checkpoint file itself is **not duplicated in Git**.

The 49 archived source modules, the original archive README, the R19 report, requirements, and archived manifest are mirrored in this directory. The original full reproducibility ZIP in ChatGPT Library remains the authoritative source for corpus, tokenizer binaries, and the broader historical reports that are not duplicated here. The GitHub Actions round-trip workflow validates the hosted binary parts.

## FA2.2 R21 adaptive memory candidate — November 12, 2026

A 138-parameter relevance-and-novelty gate now learns when to use R20's causal content-only R18-key retrieval. The existing decoder, learned R18 memory, and Adam histories remain unchanged. The gate initializes to exact R18 logits and activates only at query position >=192 to prevent short-span regression.

A reproducible **1,000-update CPU trial** completed. Development long-context BPB improved by 0.003190 and historical book-disjoint long BPB by 0.000872, relative to the original R18 checkpoint. Short-context results remained bitwise identical by gating policy. These are small, historically reused-split gains, **not fresh book or semantic QA improvements**. Accepted checkpoint remains R18, step 10,756, SHA-256 `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.

See [FA2.2 R21 methods and gates](docs/FA2_2_R21.md), [CPU candidate report](reports/r21_fa2_2_cpu_candidate.json), [relevance gate](source/r21_relevance_novelty.py), and the quarantined base64-encoded 11 KB gate optimizer state at `reports/fa2_2_r21_gate_candidate.pth.b64`. The canonical large checkpoint is not duplicated.

### FA2.2 full 400-chunk frozen-gate audit

The development-selected R21 1,000-update candidate was evaluated without additional training on the original R20 400-chunk windows. Dev BPB was **2.562485** (R20 fixed gain **2.563825**, accepted R18 **2.565997**); historical test BPB was **2.410224** (R20 **2.411362**, R18 **2.413332**). Historical test non-space NLL was **4.519880** (R20 **4.522701**, R18 **4.527414**). This is positive evidence for adaptive relevance gating, not independent semantic transfer, as the historical books were previously consulted. **No promotion.** Detailed frozen comparison: [400-chunk report](reports/r21_400_horizon_probe.json).

## FA2.3 R22 — 32K-addressed document memory (2026-11-12)

**Research candidate; NOT accepted weights.** Built R22 as a 414-parameter, three-tier learned residual reader on the unchanged accepted R18 checkpoint and **frozen** 1,000-step FA2.2/R21 gate. Recent/mid/far slots cover 65–2,048 / 2,048–8,192 / 8,192–32,768 past token positions from the *same document*, with strides 1/8/32 and exact-prefix hash rescue. A causal regression discovered and corrected future-conditioned rescue-set leakage *before the reported results*. Adapter output is bit-exact to R21 at zero new gains. This is 32K-addressed **retrieval**, not 32K recurrent/attention training.

R22 trained **300 + 700 = 1,000 cumulative CPU updates**, restoring the R22 Adam optimizer and broadening to 96 training windows with disk-backed feature caches. Accepted R18 and its optimizer histories, and the R21 gate, were left untouched. **Frozen 400-chunk horizon, eight windows per split:** R18 dev/test BPB `2.565997 / 2.413332`; R21 `2.562485 / 2.410224`; R22 **`2.549748 / 2.401243`**. Historical test non-space NLL `4.527414 → 4.519880 → 4.497821`. Ablations on the same historical test: recent only `2.406611`, middle only `2.407119`, far only `2.407569`, all tiers `2.401243`. R18/R19/R22 tests: **16/16, 12/12, 11/11**. All historical books were used in prior research and thus this is not an untouched new-book test. No semantic QA/multi-hop claim; no promotion.

Full source/reproduction [FA2.3 R22 methods](docs/FA2_3_R22.md), [architecture](source/r22_multiscale_memory.py), [CPU driver](source/r22_cpu_longmemory.py), [400-chunk probe](reports/r22_horizon400.json), [scale ablation](reports/r22_scale_ablation.json), [release manifest](reports/MANIFEST_FA2_3.json), and the small, separately quarantined R22 adapter + Adam checkpoint (base64) at `reports/r22_multiscale_candidate.pth.b64`. The large original R18 checkpoint remains pinned at SHA-256 `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f` (step **10,756**).
