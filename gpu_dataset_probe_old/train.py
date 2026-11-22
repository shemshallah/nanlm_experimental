import os
from pathlib import Path
p = Path('/kaggle/input/datasets/shem42/z4-causal-gpu6h-weights')
print('exists', p.is_dir(), flush=True)
if p.is_dir():
    for f in sorted(p.iterdir()):
        print(f.name, f.stat().st_size, flush=True)
print('OLD_DATASET_PROBE_DONE', flush=True)
