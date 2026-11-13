#!/usr/bin/env python3
"""Reconstruct a reproducible raw UTF-8 test stream from Salesforce WikiText Parquet.

The official Salesforce/wikitext text column stores a newline at the end of
each nonempty source line, and represents blank lines as empty strings. Blank
rows must therefore become one newline each, not vanish or double up.
Use this for external *evaluation*, never for training a test split.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expected-parquet-sha256')
    args = p.parse_args()
    source_bytes = args.parquet.read_bytes()
    source_sha = sha256(source_bytes)
    if args.expected_parquet_sha256 and source_sha != args.expected_parquet_sha256.lower():
        raise ValueError('Source Parquet digest mismatch')
    import pyarrow.parquet as pq
    lines = pq.read_table(args.parquet, columns=['text']).column('text').to_pylist()
    if not lines or not all(isinstance(line, str) and (not line or line.endswith('\n'))
                             for line in lines):
        raise ValueError('Unexpected WikiText line endings or empty source')
    raw = ''.join(line if line else '\n' for line in lines).encode('utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({'source':str(args.parquet),'parquet_sha256':source_sha,
                      'rows':len(lines),'raw_bytes':len(raw),
                      'raw_sha256':sha256(raw),
                      'reconstruction':'nonempty lines include final LF; blank rows become LF'},indent=2))


if __name__ == '__main__':
    main()
