# NaNCM v180 claim ledger

| Claim | Evidence | Status |
|---|---|---|
| 2,650,489 valid TRAIN causal transitions | `results/v180/train_reconstruction_stats.json`; zero overlap conflicts | PROVEN |
| Inherited 128+128 signed-cloud parent + sparse ngram residual normalized | `experiments/nancm/v180/normalized_mix180.py`, all-13,200-context native parity, source test | PROVEN |
| Strict reused-VAL in-vocab top-1 12.78%→14.34%; top-5 28.98%→31.52% | `results/v180/v180_frozen_word_gate.json`; 55-book pair bootstrap | SUPPORTED on reused VAL only |
| Word NLL 5.817160→5.713694 nats/word | Fully normalized 12k law, same 13,200 targets | SUPPORTED on reused VAL only |
| Real native C++ full-vector parity with Python | `results/v180/native_full_val180.json`: 13,200 full distributions, max error 9.54e-8 | VERIFIED FULL VAL |
| 50% strict whole-word top-1 | No evidence | OPEN / NOT ACHIEVED |
| New byte BPB / open-vocab word accuracy | No evaluation | UNCLAIMED |
| v180 improved pristine unseen-book TEST | TEST kept closed; older teacher pretraining unknown | UNCLAIMED |
| 38-feature tree ranker better than normalized residual | 1792 vs 1893 on reused VAL | REJECTED |
| TRAIN-top1-tuned gamma better than NLL-tuned gamma | 1838 vs 1893 on reused VAL | REJECTED |
| v179 speculative byte P/Q altered | No; inherited v179 source and binary remain unchanged | EXACT ROLLBACK inherited |
