from pathlib import Path
import shutil
import numpy as np

CORPUS = Path('/kaggle/input/datasets/shem42/z4-broad7h-bins')
CKPT = Path('/kaggle/input/datasets/shem42/z4-causal-gpu12h-weights')
DST = Path('/kaggle/working/inputs')
DST.mkdir(parents=True, exist_ok=True)

shutil.copy2(CKPT / 'final.pt', DST / 'final.pt')
for name in ('config.json', 'eval_ptb.json'):
    if (CKPT / name).is_file():
        shutil.copy2(CKPT / name, DST / name)

mm = np.memmap(CORPUS / 'train.bin', dtype=np.uint16, mode='r')
n = len(mm)
want = 150_000_000
slices = 30
part = want // slices
step = (n - part) // slices
with open(DST / 'train.bin', 'wb') as f:
    for i in range(slices):
        start = i * step
        mm[start:start + part].tofile(f)
shutil.copy2(CORPUS / 'valid.bin', DST / 'valid.bin')
shutil.copytree(CORPUS / 'tokenizer', DST / 'tokenizer', dirs_exist_ok=True)

for p in sorted(DST.rglob('*')):
    if p.is_file():
        print(p.relative_to(DST), p.stat().st_size, flush=True)
print('RUN_INPUTS_18H_DONE', flush=True)
