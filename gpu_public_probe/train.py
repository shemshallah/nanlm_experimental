from pathlib import Path
for base in ('/kaggle/input/datasets/shem42/z4-fineweb-external-test','/kaggle/input/z4-fineweb-external-test'):
    p=Path(base)
    print(base, 'exists', p.is_dir(), flush=True)
    if p.is_dir():
        print(sorted(x.name for x in p.iterdir()), flush=True)
print('PUBLIC_PROBE_DONE', flush=True)
