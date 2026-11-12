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

The large trained checkpoint and dataset must be byte-verifiable by the checksum above. Do **not** substitute randomly initialized weights and call them FA2.1. GitHub source files may be mirrored under this folder; a model is considered fully hosted only after the checkpoint bytes are successfully uploaded and independently verified.

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
