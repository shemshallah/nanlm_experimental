# v173 provenance

- Parent cumulative model: `NaNCM_v172_TRAINED_SIGNED_WORD_CLOUD_RECOVERY_20261122.zip` (689 entries), no independent file loss permitted. SHA and original bytes checked during archiving.
- Trained ancestor checkpoint: `checkpoints/v172_conditional_signed_cloud.pt`, SHA256 17364490f565082ab2b028e0ea5193761cf93fdbbe5b0d92217a40e7e263ca56, untouched.
- Experimental continued checkpoint: `checkpoints/v173_exact_ce_word.pt`, SHA256 d2fe36eb30d53ecd89e5582ef6f8259b229850cb892fa6eb8a47ae7ddb77b037, 6,880 steps, full AdamW state; not activated.
- Experimental native teacher: `checkpoints/v173_native_signed_teacher.bin`, SHA256 e99e5fbf08f90b7648bffc16917a76d7af585b161c9128aa8bda587c339b4da2, same original 128+128 cloud and 12k vocabulary fingerprints.
- Native source: `experiments/nancm/v173/wordfield173_api.cpp`, includes inherited v172 source, default gamma=0. All original native ABI unchanged; new wf173_joint_* optional.
- Training: TRAIN-derived v164 500-book aggregated ngram counts; 657,062 training rows, 35,312 SHA context-key validation rows; NOT book-disjoint.
- Native byte bench: historical DEV and reused TEST book windows, NOT pristine; all 16,018 byte predictions scored causally.
- Strict word bench: 197 TRAIN-book gamma selection and 244 previously exposed DEV contexts; no new TEST run.
- Archive source: all v172 ZIP entries preserved, 2 modified root text files have exact ancestry copies; no nested backup ZIP.