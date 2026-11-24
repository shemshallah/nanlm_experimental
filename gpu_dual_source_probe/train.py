from pathlib import Path
for p in [
    Path('/kaggle/input/notebooks/shem42/z4-causal-gpu12h/runs/causal_gpu12h/final.pt'),
    Path('/kaggle/input/notebooks/shem42/z4-corpus-150m-mirror/corpus/train.bin'),
]:
    print(p, p.is_file(), p.stat().st_size if p.is_file() else None, flush=True)
print('DUAL_SOURCE_DONE', flush=True)
