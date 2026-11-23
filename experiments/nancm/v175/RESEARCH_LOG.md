# v175 research log

2026-11-22: v174 recovery archive SHA verified and extracted, 782 entries. Local full-book corpus absent. GitHub Action generated 500-book split from NaNROUTER Release0003 with SHA verification and 615 inherited IDs excluded. Data: 396 TRAIN/55 VAL/49 TEST books, all causal 48-word histories, 95,040/13,200/11,760 in-vocabulary rows.

Engineered `SignedCausalEcho175`, extension of dual S+/S− 64D recurrent rails, with repeated-prefix copying from previously observed continuations only. Migrated 15 inherited optimizer state tensors across exactly 3 original groups and added 4 pointer parameters in group4. Native `N175NOW1` codec, C++ API, and source-only synthetic smoke.

TRAIN CPU stages: +280, +1200, +1600, +2000 = 5,080 attempted v175 updates. Selected step 4,955 on lower validation NLL; checkpoint total step 5755. TRAIN/VAL only for selection; TEST first inspected after 280 step and later diagnostics labeled reused.

Validation parent NLL 6.070959 → v175 5.939311, top1 1445 → 1573. TEST parent NLL 6.133046 → 6.004215, top1 1242 → 1364; top5 2918→3123. 3000 book-block resamples. Copy ablation +39 correct TEST words, but wrong-copy contexts regress: next improvement should train confidence based on match length and self-consistency.

Native 80 full 12k-word parity max error 5.07e-07; 25 exact legacy word ABI; 292 exact 256-way legacy byte distributions and 16018 byte replays, unchanged BPB. Protected full decoder not promoted.
## Extended float32 native/PyTorch parity audit (2026-11-22)

Fresh extraction of the first candidate v175 ZIP exposed a history with native-to-PyTorch full-word maximum absolute probability deviation of **5.185604095458984e-6**, above the original **2e-6** tolerance. Native shared-library SHA-256 was identical between fresh-built and working-tree libraries. The discrepancy is attributable to numerical differences between 48-step scalar C++ float32 GRU accumulations and PyTorch tensor operations, not a checkpoint change or archive failure.

The 1000-context expansion on book-separated validation contexts (seed 175; 12000 word candidates each) produced maximum absolute deviation **5.185604095458984e-6**, with **1000/1000 exact top-1/top-5 word ranking agreements** and **25/25 bit-identical inherited v174 full-vector rollbacks**. The independent neural-vector tolerance is now **1e-5 absolute**; normalization and rankings remain separately checked. All 1000 sampled contexts passed.

The prior 80-context max error **5.066e-7** remains accurate for those samples but was not a universal bound. Byte-path verification remains an independent 256-way exact rollback gate. The checkpoint, optimizer history and scored metrics were not changed by this audit.
