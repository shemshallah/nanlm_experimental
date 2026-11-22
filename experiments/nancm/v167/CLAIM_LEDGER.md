# NaNCM v167 — claim ledger

| Claim | Evidence | Result |
|---|---|---|
| v167 mode 0/domain1.5 exactly matches v166 | `results/v167_native_regression.json` | PASS: 951 exact prefix vectors |
| v167 domain1.4 preserves historical documentation | `results/v167_native_regression.json` | PASS: 4,545 exact vectors, zero admissions |
| No target-text leak into domain and agreement gate | `wordfield167_api.cpp` | Causal: completed earlier words; precomputed conditional distribution only |
| DEV-gated top-1/5/32 and symbol NLL | `results/v167_dev_gate_agree50.json`, 480 positions/24 books | 46/94/150; 2.558560 bits/symbol |
| Frozen reused TEST metrics | `results/v167_test_gate_frozen_agree50.json`, 480 positions/24 books | 46/104/162; 2.512776 bits/symbol |
| Full neural weights/optimizers conserved | v166 SHA/fingerprint + inherited checkpoint SHA | v167 only changes inference gate; no rewrites |
| v167 margin-based continued training improves TRAIN-index diagnostic | `results/v167_train.json` | 6.234078→6.085483 nats, 16.016→20.964% modal top-1 |
| v167 margin-based continued training improves book top-5 | `results/v167_dev_margin_sweep.json` | NO: 86→83 at .05; rejected |
| New untouched BOOK TEST and 256-byte BPB and +2/+3 | Not measured | OPEN |
| 80% strict word top-1 achieved | 46/480 ≈9.58% | NO |