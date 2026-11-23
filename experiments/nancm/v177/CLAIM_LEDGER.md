# v177 claim and promotion ledger

- **PASS / architecture**: native private handle, exact v176 normalization P/Q preserved with cache; compiler, 1,219 full 256-way bit-equal validation vectors and 500 seeded sampler equivalence checks.
- **PASS / bounded batch**: up to 512 prefix states verified through one native interface, inputs checked, model immutable. **This does NOT mean parallel execution.**
- **PASS / warm performance**: 1.094× native sampling throughput after proposal-tree warmup, on 32 already-known VAL contexts ×16 attempts. 562 independent vs 400 unique path prefixes is the *shared-trie benefit that v176 also provided within a single call*. v177's new benefit is persistence across batched verification and speculative attempts.
- **PASS / cold performance NEGATIVE**: v177 0.965× v176 on 96 one-draft-per-context cold examples including begin: slight regression, NOT accelerated streaming.
- **PASS / arithmetic coding**: 184 synthetic plus 60 model-grounded lossless roundtrips, full P-factorization identity. 2-bit gain over 60 examples is quantization and framing, not statistically meaningful improvement or full-text BPB.
- **PASS / old ABI**: 48 exact normalized 256-way frozen v170 comparison vectors; full 16,018 byte replay from prior version is inherited only. Separate long attempt timed out with no new result.
- **REJECTED**: TRAIN-calibrated byte-likelihood word-top-five reranker: VAL top-1 tied 80/550 and top-five conditional NLL worsened. Never promote or train on held-out labels.
- **RETAIN**: v176 best standalone signed/BPE word predictor (VAL word top1 1687/13200), v175 signed temporal optimizer, v167 active open-vocabulary word path, v170 original byte law, and all ancestor checkpoints. No weights or optimizer state overwritten.
- **OPEN**: true batched *parallel* causal byte verification, new pristine book evaluation, joint end-to-end speed and BPB, 80% top-one objective not reached.