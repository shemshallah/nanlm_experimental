from pathlib import Path
root = Path('/kaggle/input/notebooks/shem42/z4-corpus-150m-mirror/corpus')
print('exists', root.is_dir(), flush=True)
for name in ('train.bin','valid.bin','tokenizer/layered_tokenizer.json'):
    p = root / name
    print(name, p.is_file(), p.stat().st_size if p.is_file() else None, flush=True)
if (root/'train.bin').is_file():
    with open(root/'train.bin','rb') as f:
        print('first16', f.read(16).hex(), flush=True)
print('CORPUS_150M_PROBE_DONE', flush=True)
