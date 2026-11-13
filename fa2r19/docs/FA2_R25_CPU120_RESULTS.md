# FA2 R25 — latest unified PAL checkpoint, step 120 (R25 continuation)

Date: 2026-11-12 US Pacific. **One model / one optimizer / `master`; NOT promoted to accepted frontier model.**

## Model lineage

- **Accepted immutable ancestor**: PAL R18 from original PAL R19 full release. SHA-256 `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`.
- **Inherited candidate**: FA2 R24 step 220 (not promoted). Its model tensors and **123 legacy Adam states** were carried forward bitwise, not reset. R24 checkpoint SHA-256 `b6cef54cde36ab35a1eaa1e392de9c3a2b50ffb1324908611b768cfc724e1e4f`.
- **Latest continuation**: FA2 R25 at **120 cumulative R25 adapter updates**: initial 35-update CPU pilot using eight training books, then 85 more updates using all 12 training books and longer 8K-context probes.
- **One optimizer**: the nine R24 AdamW groups stay intact and a tenth group owns seven new tensors / **1,002 trainable parameters**. Full optimizer has **130 states**, consisting of **123 preserved inherited states and seven R25 states**. The underlying recurrent model, MoE, older memory gates, tokenizer and optimizer moments are inherited unchanged; only the new causal-cache mixer was trained in this round.
- The checkpoint also preserves original R24 12 training-document cursors and their detached recurrence/attention state. R25 pilot uses cached **exact R24 forward logits** plus prefix-only ngram frequency features; this is an adaptation of one unified network, not independent models.

## What improved (matched evaluation, identical book-window positions)

| Split | Original R24 BPB | R25 step 120 BPB | Relative change | Source bytes |
|---|---:|---:|---:|---:|
| 6 development selection books | 2.937444 | **2.650436** | **-9.77%** | 4,258 |
| 6 further existing development books | 2.872003 | **2.600584** | **-9.45%** | 4,087 |
| 4 previously examined historical test books | 2.532761 | **2.385299** | **-5.82%** | 2,416 |

These are real NaNROUTER Gutenberg book token predictions at context offsets around **768/4096/8192**, not whole-book or fresh-author benchmark numbers. The additional R25 training improved the same expanded six-book selection evaluation from **2.658343 BPB at step 35** to **2.650436 BPB at step 120**. Correct-book cache beat unrelated-book cache in all 6 selection, all 6 secondary development, and all 4 historical test books (unigram-frequency matching was not used as a donor control in this R25 run). The secondary group was *not used to select the checkpoint in this R25 run*, but the books were used in past PAL research; it is not globally blind. Thus statistical independence and frontier-quality claims are not established.

## Genuine long-context forward-state checks at step 120

| Development book | Token query | Original R24 BPB | R25 BPB | Wrong-book R25 BPB |
|---|---:|---:|---:|---:|
| 970 | 2,454 | 2.831042 | 2.608834 | 2.853685 |
| 970 | 9,073 | 2.886135 | 2.520200 | 2.933749 |
| 970 | 33,173 | 2.674654 | **2.232997** | 2.655258 |
| 1444 | 2,477 | 2.730801 | 2.688057 | 2.756566 |
| 1444 | 9,110 | 2.892593 | 2.675129 | 2.853718 |
| 1444 | 33,137 | 2.829501 | **2.561880** | 2.795387 |

All earlier tokens were processed by the original recurrent PAL **forward** pass, without skipping or resetting; its trained backpropagation length is still limited to two adjacent R24 chunks. R25 is an exact causal ngram cache with a learned mixture, **not** a learned semantic paraphrase retriever or a proof of strong long-form reasoning. It currently relies on access to known prior document tokens, not a standalone prompt-to-answer interface.

## Recovery (no duplicate R24 checkpoint or large R19 parent in this ZIP)

1. Unpack the separate original `PAL_R19_SEMANTIC_FRONTIER_AUDIT_ONE_CHECKPOINT_FULL.zip` into a new directory, containing `checkpoint/r12_active.pal12`, tokenizer/corpus assets and NaNROUTER books.
2. Overlay the **current single** `FA2_R25_UNIFIED_LATEST_RECOVERY.zip` into that same directory, replacing source files with exact locally tested R25 source. The ZIP contains one model checkpoint `reports/fa2_r25_unified_candidate.pth`, two small historical R21/R22 initializer files, all source, audit reports, logs and SHA-256 member manifest. The R24 step-220 binary is **not required** for R25 inference or recovery.
3. Run `PYTHONPATH=source python source/r25_restore_unified.py --root .` and `PYTHONPATH=source python source/test_r25_causal.py`. Original accepted R18 must have the SHA printed above; the restore script checks all 10 optimizer groups and 130 moment states.
4. To **continue** training the same R25 checkpoint rather than initialize a new model, use the `source/r25_unified_cpu.py --root . --resume-r25 --steps 150 [...]` command with book, window and LR flags. The driver supports recovery if the older R24 binary is absent and preserves the inherited optimizer.

The ZIP is the authoritative **exact locally executed R25 source** for numeric reproduction. The compatible GitHub `master` engine and CI may not be byte-identical to this recovered source. Neither repository nor ZIP includes the raw R19 original accepted checkpoint twice.

## Limitations and next frontier gates

To advance beyond a document-local lexical cache, independently sequester unseen authors/books, do matched-frequency wrong-document and semantic-hard-negative tests, evaluate paraphrase and multi-hop questions, grow recurrent/attention/FFN capacity preserving inherited optimizer state, and benchmark generation quality and end-to-end throughput. Do not equate 32K retrieval-address distance with 32K gradient propagation. **The R25 adapter candidate should not replace the accepted R18 anchor without broader validation.**

## Final ZIP verification and GitHub CI

- Final release: `FA2_R25_UNIFIED_LATEST_RECOVERY.zip`
- SHA-256: `1adebb11625ad7cf493d1b509b785efc1cbccc332134da933755e72fca5fed76`
- Compressed bytes: **80,143,415**; archive members: **84** (82 SHA-256-verified data members).
- Latest model: `reports/fa2_r25_unified_candidate.pth`, SHA-256 `fc5c4b815607c168aa2d86fc217d62a7cf749e570f0c10c87ed129c16a93fe74`.
- Cold extraction with the original R19 ancestor and **without** the R24 binary recovered all **130 optimizer states**, **10 groups**, and **12 document stream snapshots**; eight causal tests passed.
- Exact continuation driver and restore code are committed under `fa2r19/source/r25_*.py`, and the R25 output path is integrated into the existing unified R24 forward method.
- GitHub Actions: [R25 causal source gate success](https://github.com/shemshallah/nanlm_experimental/actions/runs/43348320330) and [FA2 original-checkpoint source gate success](https://github.com/shemshallah/nanlm_experimental/actions/runs/43348320271).
- The ZIP binary is intentionally **not** checked into GitHub; use the ChatGPT release archive plus original R19 archive to reproduce local full-weight results.
