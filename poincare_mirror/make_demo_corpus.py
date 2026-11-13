#!/usr/bin/env python3
"""Rebuild the tiny held-out excerpt demonstration from the supplied v23 file.

These are snippet groups, NOT a Gutenberg book-level generalization benchmark.
"""
import argparse
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', default='poincare_v23_original.py')
parser.add_argument('--out-dir', default='demo_books')
args=parser.parse_args()
s=Path(args.source).read_text()
core=s.split('EMBEDDED_BIG = """',1)[1].split('"""',1)[0]
lines=[line.strip() for line in core.splitlines() if line.strip()]
output=Path(args.out_dir)
output.mkdir(parents=True, exist_ok=True)
for idx in range(0,len(lines),4):
    group=lines[idx:idx+4]
    if len(group)<3:
        continue
    (output/f'book_{idx//4:03d}.txt').write_text('\n'.join(group)+'\n')
(output/'duplicate.txt').write_text((output/'book_000.txt').read_text())
print('Wrote',len(list(output.glob('*.txt'))),'files including an intentional exact duplicate')
