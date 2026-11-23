# NaN Language Model Research

**Open experimental research into causal language models, signed conceptual memory, geometric computation, and efficient native inference.**

**Independent research by Justin Howard-Stanley · [GitHub: shemshallah](https://github.com/shemshallah)**

> **Built from a cellphone. Developed in public. Measured against real evidence.**
>
> An Android phone is my primary development environment. Larger training and verification jobs can use remote CPUs, GPUs, and GitHub Actions. The aim is to explore language-model architectures that are understandable, reproducible, and increasingly practical on constrained hardware—not to suggest that every experiment runs entirely on a phone.

[Explore the code](#research-tracks) · [See measured results](#verified-research-snapshot) · [Build and test](#getting-started) · **[Support the research](#support-independent-research)**

## Why this project exists

Most language-model development assumes substantial local hardware and a conventional research lab. This repository asks what can be built through a different approach: careful experiments, explicit memory and geometry, native CPU implementations, incremental training, and honest promotion gates.

The research investigates several complementary ideas:

- **Signed conceptual geometry:** 128 lower and 128 upper concept points, with positive/negative connections and learned contextual interactions.
- **Causal memory:** episodic, suffix, and longer-context pathways that retrieve information available *before* the prediction target.
- **Multiple levels of language:** bytes, subwords, complete words, and their interactions, with separate evaluation for each.
- **Exact native verification:** preserving probability distributions while investigating word proposals, speculative byte verification, and concurrent inference.
- **Reproducible experimentation:** retained baselines, rejected candidates, source tests, claim ledgers, and documented data provenance.

These are **research directions, not claims of frontier-model performance**. The repository contains multiple *distinct experimental model lineages*; their checkpoints, metrics, and results must not be combined as though they came from one model.

## Current focus — NaNCM v180

The latest documented NaNCM word-prediction experiment, **v180 (23 November 2026)**, fuses an inherited signed-cloud word distribution with a trained, exact-key causal suffix memory. Its word law is explicitly normalized; the inherited byte predictor and previous optimizer/checkpoint lineage remain unchanged.

The new memory was reconstructed **only from TRAIN data**, using 396 books and 2,650,489 valid one-word causal transitions. It was evaluated on 55 different books from a **previously reused validation cohort**.

### Verified research snapshot

| Strict next-word metric | Inherited v176 word model | v180 opt-in word mixture | Change |
| --- | ---: | ---: | ---: |
| Top-1 accuracy | 12.78% (1,687/13,200) | **14.34% (1,893/13,200)** | +1.56 percentage points |
| Top-5 accuracy | 28.98% (3,826/13,200) | **31.52% (4,161/13,200)** | +2.54 percentage points |
| Word negative log-likelihood | 5.817160 nats/word | **5.713694 nats/word** | −0.103466 nats/word |

**What these numbers actually mean:** they cover *in-vocabulary* next-word targets from a fixed 12,000-word lexicon. They are not open-vocabulary accuracy, natural-text bits per byte, or an untouched external-test result. The 55 validation books are disjoint from v180's new TRAIN memory, but the cohort has been examined in prior rounds and the inherited teacher's historic corpus overlap has not been fully audited. The sealed 49-book TEST cohort was not inspected in v180.

Native C++ and Python agree on all **13,200 complete 12,000-way probability vectors** in this evaluation, with maximum absolute error **9.54 × 10⁻⁸**. That validates the implementation of this probability law, **not generalization to unseen books**.

The long-term target is **50% strict next-word top-1** on genuinely independent books. **That target has not been achieved.** Candidate recall, broader generalization, and sealed evaluation are still major open problems.

**Evidence:** [v180 architecture and reproduction](experiments/nancm/v180/README.md) · [full results](RESULTS_v180.md) · [claim ledger](CLAIM_LEDGER_v180.md) · [research log](RESEARCH_LOG_v180.md) · [provenance](PROVENANCE_v180.md)

### Other relevant NaNCM advances

**v179 — parallel inference, not better language predictions.** Four Linux copy-on-write workers achieved a mean **3.385× throughput** versus serial processing across independent validation contexts in three measured replicates. The model's word/byte probabilities were preserved. The result excludes one-time loading, does not demonstrate faster single-stream decoding, and has not been validated on Android. [Read the v179 report](experiments/nancm/v179/README.md).

**v176–v178 — word-to-byte coupling and verification.** The signed-cloud word top-five, first-letter/BPE links, exact byte-verifier probability law, and shared-prefix verification machinery provide the experimental foundation. Their top-five word proposal accuracy is not interchangeable with byte accuracy or end-to-end decoder speed. [v176](experiments/nancm/v176/README.md) · [v178](experiments/nancm/v178/README.md)

## Research tracks

This is a **research workbench**, not a single installable, production-ready language model. Each track has separate code, data, evaluation protocols, and promotion decisions.

| Track | Research focus | Starting point |
| --- | --- | --- |
| **NaNCM** | Signed 128+128 conceptual cloud, causal word memory, native byte/word inference, speculative verification | [Latest v180](experiments/nancm/v180/README.md), [experimental history](experiments/nancm/) |
| **Z3** | Hierarchical tokenization, geometric/MoE model scaling, 115M-parameter preset, GPU/TPU pathways | [115M preset](z3/PRESET_115M.md), [layered tokenizer](z3/LAYERED_TOKENIZER.md) |
| **Poincaré Mirror / R31–R32** | Signed mirror experts, compute-matched baselines, multidomain routing | [R32 experiment log](r32/EXPERIMENTS.md), [R31 log](r31/EXPERIMENTS.md) |
| **FA1** | Long-context associative and geometric memory, unified checkpoint lineage | [FA1 research notes](fa1r7.12/README.md) |
| **FA2** | Long-context retrieval and memory experiments | [FA2 sources](fa2r19/) |
| **M3** | CPU-model and learned-feedback experiments | [M3 work](m3_default/) |
| **Terminal-RSI** | A separate verified recursive-feedback/terminal-agent research track | [Terminal-RSI documentation](terminal_rsi/README.md) |

Older experiments and unsuccessful trials are retained deliberately. A positive result in one track is not automatically evidence that another track improved.

## Getting started

The repository is **source-and-reports first**. Large trained checkpoints, optimizer histories, and corpus Release assets are not automatically included in a Git clone. The native source smoke test below does **not** require the full trained model.

### Linux or other C++17 environment

Requirements for the basic source test: Git, Make, and a C++17 compiler. Python and ML dependencies are required for the separate evaluation/training scripts.

~~~sh
git clone https://github.com/shemshallah/nanlm_experimental.git
cd nanlm_experimental
make v180-source-smoke
~~~

To compile the **opt-in** native v180 module:

~~~sh
make v180
~~~

**Important:** plain `make` currently follows the historical v167 target. Use explicit version targets to avoid mistaking an inherited default for the newest experiment. Passing a source test or compiling a shared library does not reproduce the reported accuracy.

To reproduce the trained-word results, obtain the verified cumulative NaNCM checkpoint/sidecars and the documented NaNROUTER TRAIN/VAL inputs, then follow the [v180 reproduction procedure](experiments/nancm/v180/README.md). Data must retain its split/provenance constraints. See the [NaNROUTER corpus repository](https://github.com/shemshallah/NaNROUTER) and its Releases.

### Android / Termux (experimental)

The project is developed phone-first, but **the v180 full native trained path and v179 process-fork throughput have not been certified on Android/Termux**. An initial local source-build attempt can start with:

~~~sh
pkg update
pkg install git clang make python
git clone https://github.com/shemshallah/nanlm_experimental.git
cd nanlm_experimental
make v180-source-smoke
~~~

This is a starting point, **not a reported Android pass**. Some model-sidecar and Python evaluation steps require additional packages, RAM, disk, or a separate machine. Please report device model, Android/Termux versions, compiler version, and exact error output if you test it.

## How results are evaluated

We use explicit distinctions to prevent misleading comparisons:

- **Top-1 / top-5:** the correct target's position in a specified vocabulary and evaluation population. Candidate coverage is not prediction accuracy.
- **Word NLL vs byte BPB:** different probability spaces and denominators; one cannot be converted into a claimed gain in the other without a valid measurement.
- **Train / validation / test:** improvements on reused validation books are diagnostics, not untouched independent-test certifications.
- **Parity vs quality:** identical probability vectors prove implementation agreement, not higher predictive skill.
- **Throughput vs latency:** batching multiple independent histories can improve aggregate throughput without making one sequential generation faster.
- **Accepted vs experimental:** candidates remain opt-in unless their relevant regression, causality, data-leakage, and generalization gates pass.

Please cite the **specific version, corpus split, metric, denominator, and evaluation protocol** when sharing any result. Negative results and rejected experiments are part of the scientific record, not failures to be concealed.

## Roadmap

The immediate priorities are to improve *actual* predictive quality and accessibility, not just build larger machinery:

1. Increase strict next-word top-1 through stronger causal candidate recall and better word-distribution calibration, preserving normalization.
2. Establish genuinely sealed book-/author-disjoint evaluation and unrelated-history controls.
3. Measure natural contiguous-text byte BPB, next/+2/+3 prediction metrics, latency, and memory alongside whole-word performance.
4. Investigate signed-cloud and geometric pathways with matched baselines and tests that demonstrate predictive value.
5. Improve deterministic native builds, safe checkpoint recovery, and practical Android/low-memory validation.

These are goals, not delivery promises. See individual experiment logs for the latest measured state.

## Contribute

Help is welcome from researchers, C++ and Python developers, people with Android/Termux devices, dataset curators, and anyone interested in reproducible language modeling.

Useful contributions include reproducible bug reports, build fixes, small reviewable patches, benchmark protocols, negative controls, and clear documentation. Please [open an issue](https://github.com/shemshallah/nanlm_experimental/issues/new) before undertaking a large architectural change. Include the model track/version, command, hardware, data provenance, expected behavior, and observed output; do not attach private datasets or credentials.

Research changes should keep distinct architectures distinguishable, protect inherited checkpoints and optimizer histories, and include enough evidence to determine whether an experimental candidate deserves promotion. The repository's NaNCM/FA1 development history uses **`master` with versioned experiment folders**, not a collection of competing model branches.

## Support independent research

**This research is being built from an Android cellphone—not from a funded laboratory.** That constraint has shaped the project: source-first experimentation, creative memory architectures, transparent tests, and selective use of remote compute when the work exceeds what a phone can do.

If this approach resonates with you, **financial support or a research sponsorship would make a practical difference**. It can help fund:

- **Compute:** CPU/GPU training, carefully controlled evaluations, and independent reproducibility runs.
- **Hardware:** device testing, storage, and equipment needed to verify real mobile/low-memory operation.
- **Open research infrastructure:** corpus handling, release storage, checkpoints, and documented benchmarks.
- **Time for engineering and verification:** tests, native portability, bug fixes, and publishing results—including negative ones.

**Interested in donating or sponsoring?** Use the repository's **Sponsor** control **if GitHub displays an active funding option**, or [start a sponsorship inquiry](https://github.com/shemshallah/nanlm_experimental/issues/new?title=Sponsorship%20inquiry) to discuss support. A confirmed direct donation link will be added here when available. Please do not send money to unverified third-party addresses claiming to represent this project.

Support is voluntary and does not purchase favorable benchmark results, exclusive scientific conclusions, or influence over what gets reported. The code and research record remain open for everyone. **Stars, constructive review, issue reports, and sharing the project also help enormously.**

Thank you for supporting the idea that serious open research can start with the tools someone already has.

## License and attribution

The repository's root [LICENSE](LICENSE) is **BSD 3-Clause**. Individual data, downloaded checkpoints, third-party dependencies, and referenced research may have separate licenses and attribution requirements; verify those before redistribution.

**Maintainer:** Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.

*Research software; experimental and provided without production, performance, or fitness guarantees.*
