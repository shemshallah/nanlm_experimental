# v38 device smoke
Requires the verified v37 checkpoint plus its ancestors and metadata. See
the CPU workflow for exact release downloads and flags. To use T4 or
Blackwell on a machine with the corpus and artifacts already present,
repeat the same command with `--profile t4` or `--profile blackwell`
and `--smoke-seconds 600`. Install a PyTorch build supporting the GPU
architecture; Blackwell generally requires a newer CUDA-compatible build.
The smoke never updates checkpoint weights or Adam moments. It compares
class probabilities, hidden vectors, word/token losses, and all returned
gradient groups against the legacy CPU implementation. It times bounded
forward/backward batches and records peak CUDA memory, not full optimizer
throughput. CPU reference remains NumPy-hosted; GPU receives field tensors
each call, and host readout/optimizer remain on CPU. This is intentionally
a parity-first hybrid, not a claimed speedup. Batch size and runtime cap
are initial pilot settings; CUDA allocations can fail on insufficient VRAM.
The timed loop is bounded at batch boundaries, not a hard process timeout.
