# v166 claim ledger (all uncited numbers are local reproducible outputs)

| Claim | Evidence | Status |
|---|---|---|
| Training objective predicts new next-word labels across 12,000 words, not restricted v164 candidates | train_birth166.py, 500-step checkpoint, TRAIN index provenance | VERIFIED LOCAL |
| S+/S- and all four optimizer moments retained/migrated | train_birth166.py + v166_lexical_birth.pt / source_v165_optimizer | VERIFIED LOCAL; original archives untouched |
| DEV generator NLL 6.737 true vs 7.294 shuffled vs 6.806 signed field off | results/v166_dev_proposal.json | VERIFIED LOCAL; repeated DEV not pristine |
| DEV native spelling/EOS loss 2.880214→2.505830 (.50) | results/v166_native_dev.json | VERIFIED LOCAL, not full-byte BPB |
| TEST native spelling/EOS loss 2.924645→2.486754 (.50) | results/v166_test_FROZEN.json | VERIFIED LOCAL, reused TEST |
| DEV strict top-1 at .50 | 45→40 | REGRESSION, blocks promotion |
| TEST strict top-1 at .50 | 45→46 | SMALL POSITIVE, insufficient to offset DEV regression |
| .05 DEV+TEST top-1 | 45→45 on both | NO CHANGE; top-5/32 improvements |
| Exact rollback and documentation domain gate | 686 exact DEV vectors, 4545 historic prefix vectors | VERIFIED LOCAL |
| Full byte BPB/+2/+3, raw 500-book training, fresh pristine holdout, near-duplicates | not run | OPEN |
| 80% whole-word top-1 | 9.58% max TEST selected arm | NOT ACHIEVED |
| Promote v166 | multiple unresolved gates and DEV top-1 regression | **NO** |