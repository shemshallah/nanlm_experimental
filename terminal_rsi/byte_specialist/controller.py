"""Book-disjoint byte-specialist training and verified recursive proposal gate.

Terminal-RSI remains the host. The v102 context mixer is a *donor specialist*:
it is retrained from the frozen training stream for every proposal. Its C
weights are NOT persisted between proposals, so this is verified recursive
configuration refinement, not inherited neural/Adam training or proof of RSI.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import struct

from . import nancm_v102 as donor


SOURCE_SHA256 = 'de7e68475541af2b234c1272133d8f1ce23c5023fb40a52f33b025637829758b'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assign_books(root: Path):
    """Stable filename partition and exact-content deduplication before splits."""
    partitions = {'train': [], 'dev': [], 'test': []}
    seen = set()
    for path in sorted(root.rglob('*.txt')):
        data = path.read_bytes().replace(b'\r', b'')
        if len(data) < 2048:
            continue
        fingerprint = digest(data)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        relative = path.relative_to(root).as_posix()
        bucket = int(hashlib.sha256(relative.encode()).hexdigest()[:12], 16) % 100
        split = 'train' if bucket < 70 else 'dev' if bucket < 85 else 'test'
        partitions[split].append((relative, data))
    if any(len(partitions[s]) < 2 for s in partitions):
        raise ValueError('Need at least two unique books in each split')
    return partitions


def select_windows(books, bpe, *, count: int, width: int, skip: int, seed: int):
    if not 1 <= skip < width:
        raise ValueError('Invalid causal warmup/score boundary')
    eligible = [(name, data) for name, data in books if len(data) > width + 256]
    if len(eligible) < count:
        raise ValueError('Need distinct held-out books for every scored window')
    rng = random.Random(seed)
    selected = rng.sample(eligible, count)
    windows = []
    for name, data in selected:
        start = 128 + rng.randrange(len(data) - width - 128)
        windows.append((name, data[start:start + width]))
    return windows


def write_windows(work: Path, label: str, windows, bpe, *, skip: int):
    buffer = bytearray(struct.pack('<II', len(windows), skip))
    flags = bytearray()
    for _, window in windows:
        buffer.extend(struct.pack('<I', len(window)))
        buffer.extend(window)
        flags.extend(donor.unit_flags(window, bpe))
    (work / f'win_{label}.bin').write_bytes(bytes(buffer))
    (work / f'win_{label}.u').write_bytes(bytes(flags))
    return dict(books=len(windows), scored_bytes=sum(len(w) - skip for _, w in windows),
                book_list_sha256=digest('\n'.join(sorted(n for n, _ in windows)).encode()),
                windows_sha256=digest(bytes(buffer)))


def prepare(*, corpus: Path, work: Path, max_bytes=1600000, merges=320,
            words=1024, windows=48, window_width=512, skip=32):
    if not 65536 <= max_bytes <= 32000000 or not 1 <= merges <= 4096 or not 256 <= words <= 8192:
        raise ValueError('Invalid bounded prep parameters')
    if digest(Path(donor.__file__).read_bytes()) != SOURCE_SHA256:
        raise ValueError('Donor source checksum changed; re-audit before executing C code')
    partitions = assign_books(corpus)
    work.mkdir(parents=True, exist_ok=True)
    # Sort by stable book ID hash, rather than machine-dependent absolute paths.
    tr = sorted(partitions['train'], key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())
    stream = b'\n\n'.join(text for _, text in tr)[:max_bytes]
    if len(stream) != max_bytes:
        raise ValueError('Not enough training bytes; never pad or borrow held-out books')
    (work / 'train.bin').write_bytes(stream)
    counts = collections.Counter(donor.pretok(stream))
    bpe = donor.BPE(donor.train_bpe(counts, merges))
    bpe.save(work / 'bpe.txt')
    lmpart = stream[:int(.6 * len(stream))]
    pts = donor.pretok(lmpart)
    ids = [index for pt in pts for index in bpe.enc_pre(pt)]
    donor.write_lm_table(work / 'lm_sub.bin', bpe.vocab,
                         donor.bigram_matrix(ids, len(bpe.vocab)))
    wv = donor.build_word_vocab(counts, words)
    wids = donor.encode_words(pts, wv)
    donor.write_lm_table(work / 'lm_word.bin', wv,
                         donor.bigram_matrix(wids, len(wv)))
    dev = select_windows(partitions['dev'], bpe, count=windows,
                         width=window_width, skip=skip, seed=1001)
    test = select_windows(partitions['test'], bpe, count=windows,
                          width=window_width, skip=skip, seed=2002)
    dev_meta = write_windows(work, 'dev', dev, bpe, skip=skip)
    test_meta = write_windows(work, 'test', test, bpe, skip=skip)
    meta = dict(schema='terminal-rsi-v102-specialist-splits-v1',
                donor_source_sha256=SOURCE_SHA256, max_train_bytes=max_bytes,
                train_sha256=digest(stream),
                split_counts={name:len(group) for name, group in partitions.items()},
                split_ids_sha256={name:digest('\n'.join(sorted(n for n, _ in group)).encode())
                                  for name, group in partitions.items()},
                development=dev_meta, sealed_test=test_meta,
                width=window_width, skip=skip, bpe_vocab=len(bpe.vocab),
                word_vocab=len(wv), lm_train_bytes=len(lmpart),
                note='Disjoint complete source books; window sampling is one per evaluation book')
    (work / 'split_manifest.json').write_text(json.dumps(meta, indent=2, sort_keys=True)+'\n')
    return meta


def evaluate(work: Path, mask: int, subword: bool, word: bool, label: str,
             *, split='dev'):
    if split not in ('dev', 'test'):
        raise ValueError('Unknown split')
    if not 0 <= mask <= 2047:
        raise ValueError('Context-mask outside donor feature space')
    manifest = json.loads((work / 'split_manifest.json').read_text())
    for suffix in ('bin', 'u'):
        shutil.copyfile(work / f'win_{split}.{suffix}', work / f'win.{suffix}')
    a = argparse.Namespace(work=str(work), maxb=manifest['max_train_bytes'],
                           tshift=0, beam=8)
    flags = list(donor.BASE) + donor.lmflags(a, subword, word, gc=0)
    flags += ['-xc', str(mask)]
    result = donor.run_core(a, flags, label=label, out=f'specialist_{label}_{split}')
    if 'err' in result or 'frozen' not in result or 'online' not in result:
        raise RuntimeError(f'Donor training/evaluation failed: {result.get("err", result)}')
    return {key: result[key] for key in ('label', 'flags', 'sec', 'frozen', 'online')}


def recursive_search(*, work: Path, minimum_gain=.0005):
    """Verifier-selected configuration loop with a never-touched TEST split."""
    if minimum_gain < 0:
        raise ValueError('Negative acceptance threshold')
    manifest = json.loads((work / 'split_manifest.json').read_text())
    current = dict(label='BASE-noLM', mask=0, subword=False, word=False)
    initial = evaluate(work, current['mask'], False, False, current['label'])
    best = initial['frozen']['bpc']
    history = [dict(stage='BASE', candidate=current, development=initial, accepted=True)]
    proposals = (
        ('RSR-1', dict(label='subword+word', mask=0, subword=True, word=True)),
        ('RPR-1', dict(label='boundary-127', mask=127, subword=True, word=True)),
        ('RSR-2', dict(label='plus-word5', mask=255, subword=True, word=True)),
        ('RPR-2', dict(label='all-structure-2047', mask=2047, subword=True, word=True)),
    )
    for stage, candidate in proposals:
        measured = evaluate(work, candidate['mask'], candidate['subword'],
                            candidate['word'], candidate['label'])
        delta = round(best - measured['frozen']['bpc'], 6)
        accepted = delta > minimum_gain
        history.append(dict(stage=stage, candidate=candidate, development=measured,
                            improvement_vs_parent_bpc=delta, accepted=accepted))
        if accepted:
            current, best = candidate, measured['frozen']['bpc']
    # Test data is used once, after the proposal policy is frozen. Baseline
    # comparison is a separate process trained from identical training bytes.
    baseline_test = evaluate(work, 0, False, False, 'BASE-noLM', split='test')
    final_test = evaluate(work, current['mask'], current['subword'],
                          current['word'], current['label'], split='test')
    report = dict(schema='terminal-rsi-v102-specialist-search-v1',
                  model_host='standalone Terminal-RSI', donor='NaNCM v10.2 byte specialist',
                  transfer='Source algorithms and context/LM features; no donated trained weights',
                  corpus='NaNROUTER public-domain short stories; full-book-disjoint',
                  manifest=manifest, selection_metric='development frozen bits_per_character',
                  stages=history, selected=current,
                  test_baseline=baseline_test, test_selected=final_test,
                  frozen_test_bpc_gain=round(baseline_test['frozen']['bpc']-
                                            final_test['frozen']['bpc'],6),
                  candidate_promoted=False, benchmark_eligible=False,
                  caveat='C specialist retrains from scratch per proposal; policy selection is recursive, not continual optimizer-state training')
    (work / 'specialist_search.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def audit_records(work: Path, label: str, split: str, mode: str, manifest):
    """Independent reconstruction from *every* 8-byte C score record."""
    path = work / f'specialist_{label}_{split}_{mode}.bin'
    payload = path.read_bytes()
    scope = manifest['development' if split == 'dev' else 'sealed_test']
    expected = scope['scored_bytes']
    if len(payload) != expected * 8:
        raise ValueError(f'Incomplete or duplicated scores in {path.name}')
    nll = []
    top1 = top5 = 0
    for loss, is1, is5, pad1, pad2 in struct.iter_unpack('<fBBBB', payload):
        if not (math.isfinite(loss) and loss >= 0 and is1 in (0,1) and
                is5 in (0,1) and is1 <= is5 and pad1 == 0 and pad2 == 0):
            raise ValueError('Corrupt byte prediction record')
        nll.append(float(loss))
        top1 += is1
        top5 += is5
    positions_per_book = manifest['width'] - manifest['skip']
    if len(nll) % positions_per_book:
        raise ValueError('Book-position denominator mismatch')
    book_bpc = [sum(nll[i:i+positions_per_book]) /
                (positions_per_book * math.log(2))
                for i in range(0, len(nll), positions_per_book)]
    return dict(scored_bytes=expected, books=len(book_bpc),
                bpc=sum(nll)/(len(nll)*math.log(2)),
                top1=top1/len(nll), top5=top5/len(nll),
                book_bpc=book_bpc)


def audit_search(work: Path):
    """Recompute declared metrics and whole-book paired uncertainty; fail closed."""
    report_path = work / 'specialist_search.json'
    report = json.loads(report_path.read_text())
    manifest = report['manifest']
    all_trials = [(s['candidate']['label'], 'dev', s['development'])
                  for s in report['stages']]
    all_trials.extend((r['label'],'test',r)
                      for r in (report['test_baseline'],report['test_selected']))
    checked = []
    for label, split, item in all_trials:
        for mode in ('frozen','online'):
            independently = audit_records(work,label,split,mode,manifest)
            declared = item[mode]
            for key in ('bpc','top1','top5'):
                if abs(independently[key]-declared[key]) > .00015:
                    raise ValueError(f'Incorrect {key} metric in {label}/{split}/{mode}')
            checked.append(f'{label}/{split}/{mode}')
    base = audit_records(work, report['test_baseline']['label'], 'test', 'frozen',manifest)
    selected = audit_records(work, report['test_selected']['label'], 'test', 'frozen',manifest)
    improvements = [a-b for a,b in zip(base['book_bpc'],selected['book_bpc'])]
    rng = random.Random(1984)
    draws = [statistics.mean(rng.choices(improvements,k=len(improvements)))
             for _ in range(2000)]
    draws.sort()
    report['independent_metric_audit'] = dict(
        status='PASS', samples_per_split=base['scored_bytes'],
        independently_checked_outputs=checked,
        paired_test_book_count=len(improvements),
        test_books_better=sum(v > 0 for v in improvements),
        test_frozen_bpc_gain_mean_per_book=round(statistics.mean(improvements),6),
        test_frozen_bpc_gain_ci95=[round(draws[50],6),round(draws[1949],6)],
        bootstrap_unit='complete source book, one window per book',
        online_caveat='Online scores causally adapt across evaluation windows; frozen is the selection gate')
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report['independent_metric_audit']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prep','search','audit'))
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--maxb',type=int,default=1600000)
    parser.add_argument('--merges',type=int,default=320)
    parser.add_argument('--words',type=int,default=1024)
    parser.add_argument('--windows',type=int,default=48)
    args=parser.parse_args()
    if args.command=='prep':
        if args.corpus is None:
            parser.error('--corpus required for prep')
        report=prepare(corpus=args.corpus,work=args.work,max_bytes=args.maxb,
                       merges=args.merges,words=args.words,windows=args.windows)
    elif args.command=='search':
        report=recursive_search(work=args.work)
    else:
        report=audit_search(args.work)
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
