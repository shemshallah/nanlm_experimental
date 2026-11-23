#!/usr/bin/env python3
"""NaNCM v178 cumulative single-ZIP release with ancestral manifest integrity.

Input: the independently SHA256-verified v177 recovery ZIP.
All inherited entries remain byte-for-byte identical except the top-level
Makefile (original preserved in ancestry/v177/Makefile). Only one new release
ZIP is written, atomically after full SHA/CRC verification. GitHub keeps
source only; this runtime release carries inherited learned weights and moments.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PARENT = Path('/mnt/data/files/NaNCM_v177_PREFIX_DAG_MACRO_VERIFIER_RECOVERY_20261123.zip')
OUT = Path('/mnt/data/NaNCM_v178_SHARED_HUB_EXACT_VERIFIER_RECOVERY_20261123.zip')
TEMP = OUT.with_suffix('.tmp.zip')
PARENT_SHA = '501a360d31be5e63f9da7e480432f640cbbae9e6b5e3b9e0cee39bc76ced5572'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not PARENT.is_file() or file_sha(PARENT) != PARENT_SHA:
        raise RuntimeError('Parent SHA mismatch or missing archive; refuse publication')
    new = {}
    for path in sorted((ROOT/'experiments/nancm/v178').iterdir()):
        if path.is_file() and path.name != 'RELEASE_178.json':
            # RELEASE_178.json records the completed ZIP digest; embedding it
            # would create a circular content hash. Keep it beside the ZIP.
            new[str(path.relative_to(ROOT))] = path.read_bytes()
    for name in ('libwordfield178.so', 'test_source178'):
        p = ROOT/'bin'/name
        if not p.is_file():
            raise RuntimeError('Missing compiled native artifact '+name)
        new[str(p.relative_to(ROOT))] = p.read_bytes()
    for path in sorted((ROOT/'results').glob('v178_*.json')):
        if path.name == 'v178_release_integrity.json':
            # Generated after sealing the ZIP. Exclude on repeated builds.
            continue
        new[str(path.relative_to(ROOT))] = path.read_bytes()
    new['Makefile'] = (ROOT/'Makefile').read_bytes()
    if b'experiments/nancm/v178/Makefile.fragment' not in new['Makefile']:
        raise RuntimeError('Makefile missing v178 include')
    for src, target in (
        ('README.md', 'README_v178.md'),
        ('CLAIM_LEDGER.md', 'CLAIM_LEDGER_v178.md'),
        ('RESEARCH_LOG.md', 'RESEARCH_LOG_v178.md'),
        ('PROVENANCE.md', 'PROVENANCE_v178.md'),
        ('CHAT_LOG.md', 'CHAT_LOG_v178.md'),
        ('RESULTS.json', 'RESULTS_v178.json'),
    ):
        new[target] = (ROOT/'experiments/nancm/v178'/src).read_bytes()
    inherited = {}
    with zipfile.ZipFile(PARENT, 'r') as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != 963:
            raise RuntimeError('Unexpected inherited entry count or duplicates')
        if archive.testzip() is not None:
            raise RuntimeError('Parent ZIP failed CRC verification')
        if 'Makefile' not in names:
            raise RuntimeError('Parent has no root Makefile')
        original_makefile = archive.read('Makefile')
        new['ancestry/v177/Makefile'] = original_makefile
        for name in names:
            inherited[name] = archive.read(name)
    changes = {name for name in inherited if name in new and inherited[name] != new[name]}
    if changes != {'Makefile'}:
        raise RuntimeError(f'Unexpected inherited entry changes: {sorted(changes)}')
    duplicated = set(new) & (set(inherited) - {'Makefile'})
    if duplicated:
        raise RuntimeError(f'Duplicate names collide with ancestry: {duplicated}')
    inventory = dict(inherited)
    inventory.update(new)
    protected = ('MANIFEST_v178.json', 'SHA256SUMS_v178')
    if any(x in inventory for x in protected):
        raise RuntimeError('v178 release marker collision')
    digests = {name: sha(data) for name, data in sorted(inventory.items())}
    manifest = {
        'version': 178,
        'date': '2026-11-23',
        'repository': 'shemshallah/nanlm_experimental',
        'branch': 'master',
        'parent_version': 177,
        'parent_zip_sha256': PARENT_SHA,
        'parent_entries': len(inherited),
        'inherited_entries_byte_unchanged': len(inherited) - len(changes),
        'root_makefile_old_sha256': sha(original_makefile),
        'root_makefile_old_location': 'ancestry/v177/Makefile',
        'trained_weights_and_optimizer_states': 'inherited unchanged, not retrained',
        'probability_law': 'exact frozen v176/v170 P and Q in tested contexts',
        'actual_parallel_neural_inference': False,
        'sha256': digests,
    }
    inventory['MANIFEST_v178.json'] = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
    inventory['SHA256SUMS_v178'] = (''.join(f'{val}  {key}\n' for key, val in digests.items())).encode()
    if TEMP.exists():
        TEMP.unlink()
    with zipfile.ZipFile(TEMP, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, content in inventory.items():
            # Pin ZIP metadata: ZipFile.writestr(name, ...) otherwise inserts
            # wall-clock DOS timestamps, changing the archive hash each build.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=1)
    with zipfile.ZipFile(TEMP, 'r') as archive:
        if archive.testzip() is not None:
            raise RuntimeError('v178 output ZIP failed CRC')
        if len(archive.namelist()) != len(inventory) or len(set(archive.namelist())) != len(inventory):
            raise RuntimeError('v178 ZIP entries duplicated or lost')
        for name, expected in digests.items():
            if sha(archive.read(name)) != expected:
                raise RuntimeError('entry SHA mismatch: '+name)
        for name, data in inherited.items():
            if name != 'Makefile' and archive.read(name) != data:
                raise RuntimeError('Inherited parent mutated: '+name)
    os.replace(TEMP, OUT)
    result = {
        'status': 'PASS', 'zip': str(OUT), 'bytes': OUT.stat().st_size,
        'sha256': file_sha(OUT), 'entries': len(inventory),
        'sha256_checked': len(digests), 'crc': 'PASS',
        'parent_entries': len(inherited),
        'parent_unchanged': len(inherited)-len(changes),
        'old_makefile_saved': 'ancestry/v177/Makefile',
        'inherited_weights_optimizers_changed': False,
        'native_full_library_included': True,
    }
    (ROOT/'results/v178_release_integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
