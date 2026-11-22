from pathlib import Path
import shutil

src = Path("/kaggle/input/datasets/shem42/z4-broad7h-bins")
dst = Path("/kaggle/working/corpus")
dst.mkdir(parents=True, exist_ok=True)
for name in ("train.bin", "valid.bin", "test.bin", "manifest.json"):
    s = src / name
    if s.is_file():
        shutil.copy2(s, dst / name)
        print("copied", name, (dst / name).stat().st_size, flush=True)
tok = src / "tokenizer"
if tok.is_dir():
    shutil.copytree(tok, dst / "tokenizer", dirs_exist_ok=True)
    print("copied tokenizer", sorted(p.name for p in (dst / "tokenizer").iterdir()), flush=True)
print("CORPUS_MIRROR_DONE", flush=True)
