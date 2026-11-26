# NaNCM recovery after v37 completion

v37 run 45907255211 succeeded, artifact 15160065639, step 7998. Word
top1 97/360 vs v36 93/360; top5 163 vs 160; spelling/EOS 1.733041 vs
1.744365. 190/192 transactions accepted; 2876/5910 shared group comparisons
conflicted. Calibration selected resonance and lexical input off, token trust
0/0, bridge .25. Matched sampled bridge BPB: stories 2.385291 -> 2.137510,
sentences 2.850016 -> 2.502537, paragraphs 2.755504 -> 2.305472.
No promotion: whole-word calibration, independent full-document and test gates
remain outstanding.

v38/v39 failed while waiting on v37 because installation rate limit was exceeded.
v40 depended on failed v39. v44 failed in gh release download (release not found)
before executing its preflight; public release and asset URL were confirmed.
The consolidated workflow uses public asset URLs with the existing SHA256
checks, fetches the now-completed immutable ancestry once, and avoids gh run
watch. It respects the rate-limit reset, and runs v44, v38, v39, then v40 only if
v39 produces its artifact. Separate stage outputs and failure status preserve
diagnostic evidence. Existing v41–v43 jobs are not canceled. The recovery only
downloads the four releases used by these actual experiments; WikiText and
chemistry assets were unused by the affected scripts.

v44 adds 48 soft retention weights to the existing signed decoder and reuses
v42 folded event marginals. The local standalone Retention test passed; the
actual corpus and causal-basis integration are not yet verified.
