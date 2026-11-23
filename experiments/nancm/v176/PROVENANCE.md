# v176 provenance

- Uploaded source archive `v133_iterate.zip` is reproduced byte-for-byte in `source_reference/v133_uploaded/`; v133 metrics are training-time online reports and do not have attached trained weights.
- Native inherited parent from exact v175 cumulative ZIP. Preserve unmodified full v175 model and optimizer history plus original v174/v173 checkpoints.
- Word histories, train / val / reused TEST corpus: original NaNROUTER `corpora-short-stories-0003` Release snapshot and original `results/v175_release3/manifest175.json` + SHA256SUMS. 396 TRAIN / 55 VAL / 49 reused TEST books, separated at book ID level. Inherited pretraining overlap not ruled out.
- The new counterfactual gate is trained only on 24,000 TRAIN contexts; optimizer moments saved. Firstletter counts and BPE counts come strictly from all 95,040 TRAIN contexts; 1,600 merges trained on TRAIN word labels only.
- Beta .5 and BPE beta .35 calibrated on 55-book VAL. Reused TEST opened after selection.
- All baseline checkpoints and native 256-way byte predictor remain inherited. Default active word v167 and byte v170 are not modified by research ABI.
- Test reports are scripts and JSON in the recovery archive, plus SHA256SUMS; GitHub source-only CI cannot certify trained checkpoint.