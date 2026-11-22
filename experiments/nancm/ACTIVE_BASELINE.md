# NaNCM — active development baseline

**Promoted by explicit user instruction on 2026-11-21:** NaNCM v158 Causal Whole-Word Cloud Pointer is the active **development** baseline. This is not a claim of independent-book certified production.

- Preserve the earlier v148 fallback and all inherited 128+128 signed cloud weights / AdaGrad and AdamW optimizer histories.
- Native v158 historical documentation DEV strict word top-1 21.5924%, top-5 33.7382%; reused TEST top-1 43.4109%, top-5 52.8424%. These historical fixtures were repeatedly used during development and are not independent-book certification.
- v158 active setting: V158_POINTER_THRESHOLD=.8, V158_POINTER_BUDGET=.35, V158_TOPIC_BUDGET=.015.
- v159 experimental extension: second byte-context worldline pointer and document-local spelling adaptation. Default V159_BYTE_BUDGET=0, V159_CHAR_BUDGET=0, providing exact v158 rollback.
- v159 byte-pointer .03/minmatch 12: historical DEV top-1 unchanged, top-5 33.7382% -> 33.8731%; reused TEST top-1/top-5 unchanged, top-32 64.5995% -> 64.7287%. Unpromoted.
- Rejected: aggressive v159 .12 byte-pointer (reused TEST subset top-1 regression) and unrestricted .12 local spelling (historical DEV top-1 regression).
- Required before independent production promotion: verified NaNROUTER book-disjoint DEV/TEST, 256-byte BPB, BPE, strict whole-word gates, rollback and Android native integration.

The complete recoverable v158 and experimental v159 ZIP artifacts are kept in the ChatGPT Library. This status file is not a substitute for those optimizer/checkpoint archives. Maintain experiments in folders on **master**; do not create competing checkpoint lineages.