# NaNCM v180 Results — normalized word fusion

- Training source: 396 TRAIN books, 95,040 labeled windows, recovered 3,115,680 observed positions, 2,650,489 valid 1-word suffix transitions. Overlap conflict count: **0**.
- Train-book heldout: 19,200 word positions; global 1–4 order memory top-1 **2,830/19,200** (14.74%); top-two across four orders candidate recall **4,792/19,200** (24.96%).
- Reused 55-book VAL: global 1–4 order memory top-1 **2,011/13,200** (15.23%); 8-gram-proposal recall **3,323/13,200** (25.17%).
- Frozen full inherited signed-cloud/BPE word model: top-1 **1,687/13,200** (12.78%); top-5 **3,826/13,200** (28.98%); word NLL **5.81715995**.
- v180 normalized word law, gamma_n=.1 each: top-1 **1,893/13,200** (14.34%); top-5 **4,161/13,200** (31.52%); word NLL **5.71369394**. Top-1 gain +206 words and +1.56 percentage points, word NLL −0.10346601 nats.
- Book bootstrap 95% CI: top-1 gain **[0.01212,0.01902]**, NLL gain **[0.09003,0.11677]**; diagnostics on reused VAL.
- Real native **13,200-context full-12000-vector parity: max abs error 9.5367×10⁻⁸**, 13,200/13,200 valid simplex, native top-1 exactly **1,893/13,200**, native NLL **5.71369394**; parent native probability unchanged.
- Learned 38-feature LightGBM ranker (rejected): VAL top-1 **1,792/13,200**, below accepted mixture. TRAIN-top-1-selected normalized mixture (rejected): **1,838/13,200**.
- Causal prequential within-book cache alone was investigated, not fused: union top-64 candidate ceiling **8,370 → 8,541** (63.41% → 64.70%).
- Data usage: no TEST labels, untouched current inherited train optimizer. Still **far below the 50% top-1 goal**. No byte BPB or open-vocab gains claimed.
- Machine-level reproduction: `results/v180/v180_frozen_word_gate.json`, `results/v180/v180_normalized_mixture.json`, `results/v180/native_sample_val180.json` and native full VAL output.
