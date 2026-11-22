from pathlib import Path
import shutil
import numpy as np

SRC = Path('/kaggle/input/datasets/shem42/z4-broad7h-bins')
DST = Path('/kaggle/working/corpus')
DST.mkdir(parents=True, exist_ok=True)
src = SRC / 'train.bin'
mm = np.memmap(src, dtype=np.uint16, mode='r')
n = len(mm)
want = 150_000_000
slices = 30
part = want // slices
step = (n - part) // slices
with open(DST / 'train.bin', 'wb') as f:
    for i in range(slices):
        start = i * step
        mm[start:start + part].tofile(f)
print('subset tokens', part * slices, 'of', n, flush=True)
shutil.copy2(SRC / 'valid.bin', DST / 'valid.bin')
tok = SRC / 'tokenizer'
if tok.is_dir():
    shutil.copytree(tok, DST / 'tokenizer', dirs_exist_ok=True)
for p in sorted(DST.rglob('*')):
    if p.is_file():
        print(p.relative_to(DST), p.stat().st_size, flush=True)
print('CORPUS_150M_DONE', flush=True)
