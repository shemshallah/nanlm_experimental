# FA2 R24 — final CPU close-out and reproducibility decision

Date: 2026-11-12 (America/Los_Angeles). Repo branch: **master**.  
Disposition: **experimental unified candidate; NOT promoted**.

## Lineage and single architecture

FA2 R24 is the **same PAL model lineage**, integrating R18 recurrent/copy memory,
R21 relevance gating, R22 multiscale memory, R23 trainable contextual reader, and
a four-expert sparse signed-mirror MoE. It has **one** unified candidate checkpoint
and **one nine-group AdamW optimizer** with **123 saved optimizer parameter states**.
Selected point: **220 cumulative R24 updates**.
Original accepted R18 model remains unchanged.

- Accepted R18 SHA-256:
  `dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f`
- R24 step-220 candidate SHA-256:
  `b6cef54cde36ab35a1eaa1e392de9c3a2b50ffb1324908611b768cfc724e1e4f`
- **Final cumulative R24 recovery ZIP** SHA-256:
  `33b8e111df522674996f29f8f0e9cb9da5cb260d8c852d99a8e3c249bb415f24`
- ZIP size: **80,149,946 bytes**, **95 members**, ZIP CRC passed, **93 data members** matched per-file hashes. The archive was distributed in the corresponding ChatGPT conversation, **not committed** to GitHub. The original PAL R19 full release containing R18 ancestor, tokenizer and NaNROUTER book corpus is a separate required dependency.

## Selected development and historical measures

| Measure | Step 50 | Step 220 |
| --- | ---: | ---: |
| Selected development BPB (2,227 bytes) ↓ | 3.51203320 | **3.51113994** |
| Nonblind historical test BPB (2,249 bytes) ↓ | 3.12936584 | **3.12919531** |
| Dev correct-history vs unrelated-book nats/token ↑ | 0.00015060 | **0.00133603** |
| Dev correct-history vs shuffled-value nats/token ↑ | 0.00000173 | **0.00006721** |

These are modest autoregressive improvements, not a semantic-reasoning,
state-of-the-art, or frontier capability claim. Historical test is nonblind.

## Genuine long-distance forward-state probes

| Development book ID | Query offset | Step 50 BPB | Step 220 BPB | Furthest selected memory |
| --- | ---: | ---: | ---: | ---: |
| 970 | 9,073 | 2.887120 | **2.886135** | 9,360 tokens |
| 970 | 33,173 | 2.676226 | **2.674654** | 31,554 tokens |
| 1444 | 9,110 | 2.893143 | **2.892593** | 9,301 tokens |
| 1444 | 33,137 | 2.830041 | **2.829501** | 32,758 tokens |

These probes carry genuine PAL *forward* state through book prefixes; **BPTT is
still only two adjacent 12-word chunks**, not 32K long. A 2K probe on book 970
regressed and preferred unrelated-book history, a retained counterexample.

## Final ten-book frequency-matched wrong-history diagnostic

On **ten existing development books**, 24 consecutive 12-word chunks/book,
**18,856 bytes** in total, the wrong-book donor was selected by first-1,024-token
Hellinger-smoothed unigram-frequency similarity.

- Correct-history advantage over frequency-matched unrelated donors:
  **+0.0004739710 nats/token** as an **equal-book mean**.
- Nonparametric **book bootstrap 95% interval** over 10,000 resamples:
  **[+0.0000359762, +0.0010698316] nats/token**.
- Correct-book history won **8/10** books; **2/10 were counterexamples**.
- These are **post-selection, previously consulted dev books**, not a blind
  author- or distribution-held-out validation. Frequency matching at unigram
  level does not control for all semantic or topic information. Small effects
  and the dependent dataset limit inference. The interval does **not** correct
  for checkpoint selection or past use of these books.

## Complete archive recovery (replayed, not claimed)

The **exact final ZIP** was freshly unpacked with the immutable original R19
assets referenced externally. Verification observed:

- Final archive SHA matched `33b8e111df522674996f29f8f0e9cb9da5cb260d8c852d99a8e3c249bb415f24`.
- **95** ZIP members and **93** SHA-256-verified data members.
- **18/18** local structural, causal, optimizer and stream-parity checks passed.
- Rebuilt unified model from accepted R18 ancestor and recovered candidate,
  replaying development **3.5111399434141566 BPB**, plus nonspace loss and both
  counterfactual gains, with absolute metric differences at most **1e-9**.
- Accepted ancestor SHA-256 was unchanged and the candidate remains unpromoted.

The recovery ZIP contains the exact **locally executed source**,
`source/r24_verify_cold_restore.py`,
`source/r24_frequency_matched_audit.py`, per-book data, logs, and
`reports/r24_latest_manifest.json`. Note: the GitHub master implementation
and local recovery code are not byte-for-byte identical; use the **archive**
to reproduce these precise numeric results.

## Decision and gates

**Close R24 as a reproducible memory experiment. Do not promote to accepted
model and do not claim frontier-level performance.**

The next promoted result would require newly sequestered authors and books,
confidence intervals across independent documents, matched donor and
shuffled-value advantages on unseen authors, robust 2K/8K/32K retrieval and
semantic/multi-hop transfer, and a materially stronger baseline. Extend the
**same architecture/checkpoint/optimizer**; never create independent FA2
replacement models merely to explore an architectural option.

The original R18 checkpoint stays the accepted anchor; FA2 R24 step 220 is
the latest **candidate** to continue from.
