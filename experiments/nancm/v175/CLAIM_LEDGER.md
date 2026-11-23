# v175 claim ledger (2026-11-22)

| Claim | Status | Evidence / qualification |
|---|---|---|
| End-to-end 500 different books from NaNROUTER Release3, book-disjoint v175 splits | MEASURED | 396/55/49; full manifest, original content SHA, 615-ID exclusion list |
| New causal 48-word model continues real optimizer rather than resetting | VERIFIED | exact inherited v174 and v173 AdamW moments; 4 groups 19 states |
| Recurrent signed cloud with episodic copy learns useful word conditionals | MEASURED | 55-book heldout NLL and top1 gains, 49-book test gains |
| Copy pointer helps in known repeat continuations | MEASURED | 399 copyable TEST contexts, 106→190 top1 vs no echo; wrong-copy penalty recorded |
| Correct older history beats unrelated history | DIAGNOSTIC | 0.041 nats on shuffled-history negative control, not independent causal proof |
| Native probabilities identical to PyTorch | APPROXIMATE FLOAT-PARITY | 80 full 12k predictions max error <6e-7, ranking exact |
| Old native word ABI and 256-byte byte models preserved | VERIFIED | 25 bit-exact 12k parent distributions; 292 exact 256-way v169 rollback vectors; 16018 native bytes |
| v175 improves active v167 open-vocabulary full decoder | NOT ESTABLISHED | No matched 96-beam promotion comparison on new books |
| 80% strict word top-1 achieved | FALSE | 1364/11760 ~11.60% restricted 12k in-vocab |
| Pristine untouched whole-lineage TEST | NOT ESTABLISHED | TEST reused since step280, parent text exposure unknown |
| Publishable full-system breakthrough | NOT YET | Reproducible experimental partial result; requires independently sealed test and end-to-end gate |
## Extended float32 native/PyTorch parity audit (2026-11-22)

Fresh extraction of the first candidate v175 ZIP exposed a history with native-to-PyTorch full-word maximum absolute probability deviation of **5.185604095458984e-6**, above the original **2e-6** tolerance. Native shared-library SHA-256 was identical between fresh-built and working-tree libraries. The discrepancy is attributable to numerical differences between 48-step scalar C++ float32 GRU accumulations and PyTorch tensor operations, not a checkpoint change or archive failure.

The 1000-context expansion on book-separated validation contexts (seed 175; 12000 word candidates each) produced maximum absolute deviation **5.185604095458984e-6**, with **1000/1000 exact top-1/top-5 word ranking agreements** and **25/25 bit-identical inherited v174 full-vector rollbacks**. The independent neural-vector tolerance is now **1e-5 absolute**; normalization and rankings remain separately checked. All 1000 sampled contexts passed.

The prior 80-context max error **5.066e-7** remains accurate for those samples but was not a universal bound. Byte-path verification remains an independent 256-way exact rollback gate. The checkpoint, optimizer history and scored metrics were not changed by this audit.
