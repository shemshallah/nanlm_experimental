from pathlib import Path
root = Path('/kaggle/input')
print('input dirs', sorted(p.name for p in root.iterdir()), flush=True)
for p in sorted(root.rglob('final.pt'))[:5]:
    print('FOUND', p, p.stat().st_size, flush=True)
print('KSOURCE_PROBE_DONE', flush=True)
