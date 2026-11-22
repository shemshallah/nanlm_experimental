from pathlib import Path
import shutil

CORPUS = Path('/kaggle/input/datasets/shem42/z4-broad7h-bins')
CKPT = Path('/kaggle/input/datasets/shem42/z4-causal-gpu12h-weights')
DST = Path('/kaggle/working/inputs')
DST.mkdir(parents=True, exist_ok=True)
shutil.copy2(CKPT / 'final.pt', DST / 'final.pt')
shutil.copy2(CORPUS / 'test.bin', DST / 'test.bin')
shutil.copy2(CORPUS / 'manifest.json', DST / 'manifest.json')
shutil.copytree(CORPUS / 'tokenizer', DST / 'tokenizer', dirs_exist_ok=True)
for p in sorted(DST.rglob('*')):
    if p.is_file():
        print(p.relative_to(DST), p.stat().st_size, flush=True)
print('DOMAIN_EVAL_INPUTS_DONE', flush=True)
