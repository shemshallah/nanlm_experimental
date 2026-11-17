# Terminal-RSI: 1,000-update CPU pilot on Molab / GitHub Actions

This is the **Terminal-RSI v103 implementation** in `terminal_rsi/` (not FA1, FA2, M3 or Poincare). The `--fresh-start` pilot uses the original v103 feature extraction, v103 Terminal-RSI feedback and unified additive path, but **does not inherit trained weights**. It produces a disposable checkpoint, not a promoted master model.

## Molab: one CPU training command

In a Molab Python notebook with 32 GB RAM, select **CPU** and paste these commands into shell-enabled cells (prefix with `!` if the notebook requires IPython shell escapes):

```bash
git clone --branch master --single-branch https://github.com/shemshallah/nanlm_experimental.git
cd nanlm_experimental
python -m pip install numpy
python -c 'import torch; print("Torch", torch.__version__, "CPU supported", True)'
sudo apt-get update -qq
sudo apt-get install -y -qq zstd
python -m terminal_rsi.run_molab_cpu_1000 --cpu-threads 4 --updates 1000 --output-dir terminal_rsi_cpu_1000_out
```

Use `--cpu-threads 1`, `2` or `4` without changing model architecture, training corpus, optimizer semantics or RSI feedback. The `--updates` setting counts **exact byte optimizer updates** independently of wall-clock time. Default `--batch 4`, 256 causal bytes, 1,000 updates, 64 new DEV books × 1,024 positions. No old BPB reference or C oracle.

If PyTorch is not installed in the notebook, install a compatible CPU or CUDA PyTorch build **before** running. Do not unnecessarily overwrite a working Blackwell-capable PyTorch environment.

The launcher downloads and SHA-256 checks *only* the public NaNROUTER short-story batch 0004, extracts it, builds a 2-million-byte training stream from English-eligible books, excludes Russian/Cyrillic sources and authors, builds a **new** 64-book English-only DEV with disjoint Gutenberg IDs, and saves:

- `terminal_rsi_cpu_1000_out/terminal_rsi_cpu_1000_metrics.json` — new matched DEV before/after BPB, bit accuracy, byte top-1/top-5, thread count, actual update count, update-loop seconds, separate verified RSI feedback seconds.
- `terminal_rsi_cpu_1000_out/terminal_rsi_cpu_1000_candidate.pt` — unpromoted recoverable model plus **all three optimizer parameter groups and RSI state**.
- `terminal_rsi_cpu_1000_out/cpu_pilot.manifest.json`, `win_dev.manifest.json`, `train_book_ids.json` — source hashes, exclusions, book identities and split checks.

Choose a new `--output-dir` for distinct experiments. The launcher intentionally refuses to overwrite a checkpoint or metric receipt. Save these output files to Molab persistent storage before ending a temporary notebook session.

## GitHub Actions

Run the same experiment on `master` with [terminal_rsi_1000_cpu_pilot.yml](../.github/workflows/terminal_rsi_1000_cpu_pilot.yml). The workflow verifies the official release SHA, v103 CPU parity and genuine train/DEV book identity disjointness, performs exactly 1,000 updates, then uploads the complete unpromoted result as a GitHub Actions artifact (30-day retention).

The general [Terminal-RSI CPU acceptance](../.github/workflows/terminal_rsi_cpu.yml) covers 1/2/4 thread throughput and original weight/parity tests.

## Original trained v103 checkpoint path (not fresh-start)

The public CPU pilot is separate from original trained checkpoint `terminal_rsi_gpu_v103_full.pt`. If using your uploaded original checkpoint, use `python -m terminal_rsi.train_verbatim --legacy-checkpoint /path/to/terminal_rsi_gpu_v103_full.pt` **instead** of `--fresh-start`; provide its exact original TRAIN stream and BPE merges and a freshly generated DEV manifest bound to that TRAIN SHA-256. The runner verifies inherited v103 learned tables and Adam state; it refuses a different TRAIN SHA or overwriting the original checkpoint. Do not claim checkpoint continuation merely because the architecture matches.

## Acceptance rules

- Same original v103 forward logits at upgrade gate=0; no disabled legacy features.
- After exactly 1,000 updates, report **new** matched 64-book DEV BPB, bit accuracy, 256-way top-1/top-5 and counts.
- Each training run executes its verified terminal RSI actor/critic feedback actions and preserves full optimizer states.
- Sealed TEST untouched. Do not compare the new DEV against historical scores from a different dataset.
- CPU duration is the time for 1,000 optimizer updates; RSI-feedback and held-out-evaluation durations are separate.
- Blackwell 96 GiB specialization is a **separate numerical and VRAM acceptance phase**. A passing CPU pilot does not prove Blackwell kernel parity.
