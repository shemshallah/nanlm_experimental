"""Restore SHA-pinned original Terminal-RSI v103 DEV from its archived notebook.

The notebook contains the ORIGINAL zlib+base64 independent-book windows,
not a re-split or substitute dataset. No training code from the notebook
is executed; only its literal embedded reference dictionary is decoded.
Never automatically restore or inspect sealed TEST.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib

PINNED_REFERENCES = {
    "win_dev.bin": "5826929ae3356dda057c1619559afb7e04f534e185d59d45274f401232193465",
    "curve_dev_frozen_10000000.bin":
        "adf21a4490aeec7a1e7f4e204ffffb3a0ede74fca9f50a0fa0ecc73e80167c21",
}


def restore(notebook: Path, destination: Path, *, include_frozen_c: bool = False):
    archived = json.loads(notebook.read_text(encoding="utf-8"))
    references = None
    for cell in archived.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if not isinstance(source, (str, list)):
            raise ValueError("Malformed original notebook cell source")
        code = "".join(source) if isinstance(source, list) else source
        if "ORIGINAL_V103_INDEPENDENT_REFERENCES" not in code:
            continue
        tree = ast.parse(code)
        for assignment in tree.body:
            if not isinstance(assignment, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and
                       t.id == "ORIGINAL_V103_INDEPENDENT_REFERENCES"
                       for t in assignment.targets):
                continue
            if references is not None:
                raise ValueError("Ambiguous repeated original reference dictionary")
            references = ast.literal_eval(assignment.value)
    if not isinstance(references, dict):
        raise ValueError("Original v103 independent reference cell not present")
    wanted = ["win_dev.bin"]
    if include_frozen_c:
        wanted.append("curve_dev_frozen_10000000.bin")
    decoded = {}
    for name in wanted:
        record = references.get(name)
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise ValueError(f"Original notebook missing SHA-pinned {name}")
        compressed, claimed_digest = record
        expected = PINNED_REFERENCES[name]
        if claimed_digest != expected:
            raise ValueError(f"Original notebook's {name} SHA doesn't match published fingerprint")
        raw = zlib.decompress(base64.b64decode(compressed, validate=True))
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Original {name} payload SHA mismatch: {actual}")
        if name == "win_dev.bin":
            if len(raw) < 12:
                raise ValueError("Original win_dev.bin truncated")
            count, skip = struct.unpack_from("<II", raw)
            if count != 16 or skip < 32:
                raise ValueError("Expected 16 independently held-out DEV books")
            offset = 8
            lengths = []
            for _ in range(count):
                if offset + 4 > len(raw):
                    raise ValueError("Truncated DEV book header")
                width = struct.unpack_from("<I", raw, offset)[0]
                offset += 4
                if width <= skip or offset + width > len(raw):
                    raise ValueError("Malformed book-isolated DEV window")
                lengths.append(width - skip)
                offset += width
            if offset != len(raw) or len(set(lengths)) != 1:
                raise ValueError("DEV books not aligned or extra bytes")
        decoded[name] = raw
    destination.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, raw in decoded.items():
        target = destination / name
        digest = hashlib.sha256(raw).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Refusing to overwrite non-identical {target}")
            status = "verified_existing"
        else:
            pending = target.with_name(target.name + ".restore-pending")
            if pending.exists():
                raise FileExistsError(f"Refusing to overwrite unfinished {pending}")
            pending.write_bytes(raw)
            pending.replace(target)
            status = "restored_original"
        receipts[name] = {"path": str(target), "sha256": digest,
                          "bytes": len(raw), "status": status}
    return receipts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook", type=Path, required=True,
                   help="Original Terminal_RSI_GPU_Full_System_60s.ipynb")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-frozen-c", action="store_true",
                   help="Also restore original C DEV Rec stream for matched comparisons")
    args = p.parse_args(argv)
    print(json.dumps(restore(args.notebook, args.output_dir,
                             include_frozen_c=args.include_frozen_c),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
