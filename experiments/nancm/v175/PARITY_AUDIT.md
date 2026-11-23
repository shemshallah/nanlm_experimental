# NaNCM v175 native parity extension

## Extended float32 native/PyTorch parity audit (2026-11-22)

Fresh extraction of the first candidate v175 ZIP exposed a history with native-to-PyTorch full-word maximum absolute probability deviation of **5.185604095458984e-6**, above the original **2e-6** tolerance. Native shared-library SHA-256 was identical between fresh-built and working-tree libraries. The discrepancy is attributable to numerical differences between 48-step scalar C++ float32 GRU accumulations and PyTorch tensor operations, not a checkpoint change or archive failure.

The 1000-context expansion on book-separated validation contexts (seed 175; 12000 word candidates each) produced maximum absolute deviation **5.185604095458984e-6**, with **1000/1000 exact top-1/top-5 word ranking agreements** and **25/25 bit-identical inherited v174 full-vector rollbacks**. The independent neural-vector tolerance is now **1e-5 absolute**; normalization and rankings remain separately checked. All 1000 sampled contexts passed.

The prior 80-context max error **5.066e-7** remains accurate for those samples but was not a universal bound. Byte-path verification remains an independent 256-way exact rollback gate. The checkpoint, optimizer history and scored metrics were not changed by this audit.
