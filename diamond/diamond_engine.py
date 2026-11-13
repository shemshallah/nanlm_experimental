#!/usr/bin/env python3
"""
diamond_engine.py - single-file C(256,2) / C(256,3) diamond engine
====================================================================
Pairs learn, triples / overmind judge.

Architecture (all in this file, no other modules):
  256 bytes  ->  32640 pair experts (MoE)  ->  C(256,3) triple layer (overmind)
  256 points < 32640 < C(256,3) points > C(256,3)'s 256 points

  Three towers, each a stack of microtransformer blocks with 256-expert MoE:
    byte tower:     1024-4096 bytes -> 32-48 dim, 128-512 chunks
    word tower:     128-256 words   -> 48-64 dim, 16-32 chunks
    sentence tower: 32-64 first-words -> 48-64 dim
  The original 24/32/32, 48/64/64, 16-byte-chunk layout is still the default.
  ``--chunk-conv`` replaces mean-pooling inside each byte/word chunk with an
  order-preserving strided convolution, and ``--depth`` stacks more blocks.
  The frontier preset is 2048 bytes / 256 words / 64 sentences, 48/64/64
  dims, 96/128/128 FF, depth 3, and convolutional chunk stems.

  Cross-conditioning, staged:
    Epoch 1: bottom-up only   byte -> word -> sentence
    Epoch 2+: full two-way    bottom-up + top-down sentence -> word -> byte

  Introspection: a 1-dim head predicts byte loss. Self-sentences are kept
  only if BOTH predicted loss < median AND word entropy < median.

NaNROUTER corpus ingestion
--------------------------
This version is wired to ingest the NaNROUTER corpora tree instead of walking
/usr/share/doc. It accepts one or more --corpus-dir roots, the
NANROUTER_CORPORA environment variable, and auto-discovers
``NaNROUTER/corpora`` / ``nanrouter/corpora`` relative to the workspace.
It reads the released short-story, sentence, paragraph and domain text files,
plus the wordlists, and gives prose categories a fair character budget so one
large book cannot crowd out the rest.

Long-run support
----------------
1000-epoch runs need reporting, checkpointing and a clean stop:
  * one human-readable line per epoch, plus JSONL history (flushed to disk)
  * optional step-level reporting every N optimizer steps
  * cosine LR with warmup and gradient clipping
  * resumable NumPy checkpoints (weights + optimizer slots + epoch counter)
  * --time-budget-minutes stops gracefully before a runner timeout and saves

Run locally / on a runner:
  pip install -q tensorflow-cpu==2.16.2 numpy==1.26.4
  python diamond_engine.py --corpus-dir ../NaNROUTER/corpora \
      --epochs 1000 --examples 4000 --batch 64 --eval-every 10 \
      --checkpoint state/diamond_state.npz --history state/diamond_history.jsonl \
      --time-budget-minutes 330 --memory-limit-gb 0 --threads 4

Outputs:
  * stdout: per-epoch byte / word / sentence loss, bits/byte, top-1 / top-5
  * --out JSON: full history and final metrics
  * --history JSONL: append-only per-epoch records
  * --checkpoint NPZ: resumable weights, optimizer state and epoch counter
"""

import os, sys, re, gc, json, math, time, random, argparse, resource, hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

# --- 6GB / CPU defaults: set BEFORE importing TF ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")


def parse_args():
    p = argparse.ArgumentParser(
        description="Diamond C(256,2)/C(256,3) engine - NaNROUTER CPU training"
    )
    p.add_argument("--examples", type=int, default=4000,
                   help="train examples per epoch (max 20000 on 6GB)")
    p.add_argument("--epochs", type=int, default=1000,
                   help="target epochs: 1=bottom-up, 2+=two-way")
    p.add_argument("--batch", type=int, default=64, help="batch, 64 max on 6GB")
    p.add_argument("--byte-win", type=int, default=1024, choices=[512, 1024, 2048, 4096],
                   help="byte window")
    p.add_argument("--word-win", type=int, default=128, help="word window")
    p.add_argument("--sent-win", type=int, default=32, help="sentence window")
    p.add_argument("--depth", type=int, default=2,
                   help="transformer blocks per tower (1-6)")
    p.add_argument("--chunk-conv", action="store_true",
                   help="use an order-preserving strided conv as the byte/word chunk stem "
                        "instead of mean-pooling each chunk")
    p.add_argument("--vocab", type=int, default=2000, help="top-N words")
    p.add_argument("--dim-b", type=int, default=24, help="byte tower width")
    p.add_argument("--dim-w", type=int, default=32, help="word tower width")
    p.add_argument("--dim-s", type=int, default=32, help="sentence tower width")
    p.add_argument("--ff-b", type=int, default=48, help="byte tower FF width")
    p.add_argument("--ff-w", type=int, default=64, help="word tower FF width")
    p.add_argument("--ff-s", type=int, default=64, help="sentence tower FF width")
    p.add_argument("--byte-chunk", type=int, default=16, choices=[8, 16],
                   help="bytes per byte-tower chunk; 8 gives finer byte order")
    p.add_argument("--corpus-chars", type=int, default=2000000,
                   help="corpus character cap (2M is comfortable on CPU)")
    p.add_argument("--corpus-dir", action="append", default=[],
                   help="NaNROUTER corpora root; repeatable")
    p.add_argument("--max-files", type=int, default=2000,
                   help="maximum corpus files considered")
    p.add_argument("--file-read-chars", type=int, default=20000,
                   help="per-file character cap (keeps many books in the mix)")
    p.add_argument("--require-real-corpus", action="store_true",
                   help="fail instead of falling back to synthetic text")
    p.add_argument("--use-wordlists", action="store_true", default=True,
                   help="use NaNROUTER wordlists to extend the vocabulary")
    p.add_argument("--no-wordlists", dest="use_wordlists", action="store_false")
    p.add_argument("--wordlist-words", type=int, default=20000,
                   help="wordlist words considered for vocabulary")
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--threads", type=int, default=4, help="OMP / TF threads")
    p.add_argument("--memory-limit-gb", type=float, default=0.0,
                   help="RLIMIT_AS in GB, 0=disable")
    p.add_argument("--test-size", type=int, default=512, help="held-out examples")
    p.add_argument("--split-mode", choices=["position", "document"], default="position",
                   help="held-out split: contiguous position block or whole documents")
    p.add_argument("--doc-holdout-frac", type=float, default=0.10,
                   help="fraction of documents held out in document mode")
    p.add_argument("--eval-every", type=int, default=10,
                   help="evaluate held-out set every N epochs")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--report-every-steps", type=int, default=100,
                   help="print a training line every N optimizer steps")
    p.add_argument("--save-every", type=int, default=25,
                   help="checkpoint every N epochs")
    p.add_argument("--time-budget-minutes", type=float, default=0.0,
                   help="gracefully stop and checkpoint after N minutes; 0=off")
    p.add_argument("--checkpoint", type=str, default="diamond_state.npz",
                   help="resumable checkpoint path; empty disables")
    p.add_argument("--history", type=str, default="diamond_history.jsonl",
                   help="append-only JSONL reporting path; empty disables")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing checkpoint")
    p.add_argument("--self-loop-every", type=int, default=0,
                   help="run introspection self-loop every N epochs; 0=off")
    p.add_argument("--self-loop-start-epoch", type=int, default=0,
                   help="first epoch for the self-loop schedule (1-based); 0=from start")
    p.add_argument("--self-streams", type=int, default=64)
    p.add_argument("--self-steps", type=int, default=15)
    p.add_argument("--self-mix", action="store_true",
                   help="mix filtered self-text into train_B (original behavior)")
    p.add_argument("--no-self-loop", action="store_true",
                   help="backward-compatible alias for --self-loop-every 0")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 200 examples, 1 epoch, no self-loop")
    p.add_argument("--save-weights", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="label smoothing for training cross-entropy (0-0.2)")
    p.add_argument("--patience", type=int, default=0,
                   help="early-stop after N held-out evals without improvement; 0=off")
    p.add_argument("--best-checkpoint", type=str, default="",
                   help="optional NPZ path for the best held-out checkpoint")
    p.add_argument("--best-weights", type=str, default="",
                   help="optional H5 path for the best held-out weights")
    p.add_argument("--fixed-train", dest="resample_train", action="store_false",
                   help="reuse one fixed training subset instead of resampling each epoch")
    p.set_defaults(resample_train=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=str, default="diamond_metrics.json")
    return p.parse_args()


ARGS = parse_args()
if ARGS.quick:
    ARGS.examples = 200
    ARGS.epochs = 1
    ARGS.test_size = min(ARGS.test_size, 64)
    ARGS.self_loop_every = 0
    ARGS.corpus_chars = min(ARGS.corpus_chars, 400000)
if ARGS.no_self_loop:
    ARGS.self_loop_every = 0
if ARGS.examples > 20000:
    print(f"[warn] examples {ARGS.examples} > 20000 may exceed 6GB, clamping to 20000", flush=True)
    ARGS.examples = 20000
if ARGS.batch > 64:
    print(f"[warn] batch {ARGS.batch} > 64 may exceed 6GB, clamping to 64", flush=True)
    ARGS.batch = 64
if ARGS.batch < 1:
    ARGS.batch = 1
if ARGS.eval_batch > 128:
    ARGS.eval_batch = 128
if ARGS.label_smoothing < 0.0:
    ARGS.label_smoothing = 0.0
if ARGS.label_smoothing > 0.2:
    ARGS.label_smoothing = 0.2
if ARGS.patience < 0:
    ARGS.patience = 0
if ARGS.dropout < 0.0:
    ARGS.dropout = 0.0
if ARGS.dropout > 0.9:
    ARGS.dropout = 0.9
if ARGS.weight_decay < 0.0:
    ARGS.weight_decay = 0.0
if ARGS.self_loop_start_epoch < 0:
    ARGS.self_loop_start_epoch = 0
ARGS.depth = max(1, min(6, int(ARGS.depth)))
if ARGS.doc_holdout_frac < 0.01:
    ARGS.doc_holdout_frac = 0.01
if ARGS.doc_holdout_frac > 0.50:
    ARGS.doc_holdout_frac = 0.50
if ARGS.test_size < 64:
    ARGS.test_size = 64
if ARGS.test_size > 8192:
    ARGS.test_size = 8192
# Tower widths must be divisible by the four attention heads.
for _name in ("dim_b", "dim_w", "dim_s"):
    _v = max(8, min(256, int(getattr(ARGS, _name))))
    setattr(ARGS, _name, (_v // 4) * 4)
for _name in ("ff_b", "ff_w", "ff_s"):
    _v = max(8, min(512, int(getattr(ARGS, _name))))
    setattr(ARGS, _name, (_v // 4) * 4)
if ARGS.byte_win % ARGS.byte_chunk != 0:
    raise SystemExit("[args] --byte-chunk must divide --byte-win")

# threads + memory limit, before TF work
os.environ["OMP_NUM_THREADS"] = str(ARGS.threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(ARGS.threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
if ARGS.memory_limit_gb and ARGS.memory_limit_gb > 0:
    try:
        lim = int(ARGS.memory_limit_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        print(f"[mem] RLIMIT_AS = {ARGS.memory_limit_gb:.1f} GB, threads={ARGS.threads}", flush=True)
    except Exception as e:
        print(f"[mem] could not set RLIMIT_AS: {e}", flush=True)

import numpy as np
import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(ARGS.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

T0 = time.time()
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
tf.random.set_seed(ARGS.seed)
print(
    f"[engine] diamond C(256,2)/C(256,3) | examples={ARGS.examples} "
    f"epochs={ARGS.epochs} batch={ARGS.batch} byte={ARGS.byte_win} "
    f"word={ARGS.word_win} sent={ARGS.sent_win} vocab={ARGS.vocab} "
    f"threads={ARGS.threads} mem={ARGS.memory_limit_gb}GB",
    flush=True,
)


# ---------------------------------------------------------------- corpus
CORPUS_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".github", ".venv", "venv",
    "build", "dist", ".mypy_cache", ".pytest_cache",
}
CORPUS_SKIP_FILES = {
    "manifest.json", "sources.json", "batches.json", "sha256sums",
    "license.txt", "license.md", "readme.md", "readme.txt", "sources.jsonl",
}
CORPUS_TEXT_EXTS = {".txt", ".text", ".md", ".rst", ".jsonl", ".json", ".csv", ".tsv"}
CORPUS_TEXT_FIELDS = (
    "text", "content", "sentence", "paragraph", "abstract", "body", "story",
    "article", "problem", "solution", "prompt", "completion", "code",
    "question", "answer", "title",
)


def _norm(path):
    return os.path.abspath(os.path.expanduser(path))


def discover_corpus_roots(cli_dirs):
    """Find NaNROUTER corpora roots from CLI, env and the workspace."""
    roots = []
    env = os.environ.get("NANROUTER_CORPORA", "")
    if env:
        for part in re.split(r"[:\n]", env):
            if part.strip():
                roots.append(part.strip())
    roots.extend(cli_dirs)
    bases = [
        os.getcwd(),
        os.path.dirname(os.getcwd()),
        os.environ.get("GITHUB_WORKSPACE", ""),
        os.path.expanduser("~"),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for base in bases:
        if not base:
            continue
        for name in ("NaNROUTER/corpora", "nanrouter/corpora", "corpora"):
            roots.append(os.path.join(base, name))

    out = []
    for root in roots:
        root = _norm(root)
        candidates = [root]
        # Accept a repository root as well as the corpora directory itself.
        if os.path.basename(root).lower() not in ("corpora",):
            candidates.append(os.path.join(root, "corpora"))
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in out:
                out.append(candidate)
    return out


def _corpus_category(path):
    p = path.replace("\\", "/").lower()
    if "/short_stories/" in p or "/short-stories/" in p or "/short stories/" in p:
        return "short_stories"
    if "/paragraphs/" in p:
        return "paragraphs"
    if "/sentences/" in p:
        return "sentences"
    if "/domains/" in p:
        return "domains"
    if "/wordlists/" in p:
        return "wordlists"
    return "other"


def _iter_corpus_files(root):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in CORPUS_SKIP_DIRS and not d.startswith(".")
        )
        for fn in sorted(filenames):
            low = fn.lower()
            if low in CORPUS_SKIP_FILES or low.startswith("."):
                continue
            ext = os.path.splitext(low)[1]
            if ext not in CORPUS_TEXT_EXTS:
                continue
            fp = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(fp) < 64:
                    continue
            except OSError:
                continue
            yield fp


def _text_from_obj(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        for key in CORPUS_TEXT_FIELDS:
            val = obj.get(key)
            if isinstance(val, str) and len(val.strip()) >= 16:
                parts.append(val.strip())
        if parts:
            return "\n".join(parts)
        for key in ("data", "documents", "records", "items"):
            val = obj.get(key)
            if isinstance(val, list):
                sub = [_text_from_obj(x) for x in val[:100]]
                sub = [s for s in sub if s]
                if sub:
                    return "\n".join(sub)
    if isinstance(obj, list):
        sub = [_text_from_obj(x) for x in obj[:100]]
        sub = [s for s in sub if s]
        return "\n".join(sub)
    return ""


def _read_corpus_file(fp, max_chars):
    if max_chars <= 0:
        return ""
    ext = os.path.splitext(fp.lower())[1]
    try:
        if ext == ".jsonl":
            parts = []
            used = 0
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    txt = _text_from_obj(obj)
                    if not txt:
                        continue
                    parts.append(txt)
                    used += len(txt) + 1
                    if used >= max_chars:
                        break
            return "\n".join(parts)[:max_chars]
        if ext == ".json":
            if os.path.getsize(fp) > 40_000_000:
                return ""
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            return _text_from_obj(obj)[:max_chars]
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_chars)
        if ext in (".csv", ".tsv"):
            data = re.sub(r"[\t,;|]+", " ", data)
        return data
    except Exception:
        return ""


def _load_wordlist_words(groups, limit):
    words = []
    seen = set()
    for fp in groups.get("wordlists", []):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if len(words) >= limit:
                        break
                    m = re.search(r"[A-Za-z']+", line)
                    if not m:
                        continue
                    w = m.group(0).lower()
                    if len(w) >= 2 and w not in seen:
                        seen.add(w)
                        words.append(w)
        except Exception:
            continue
        if len(words) >= limit:
            break
    return words


def load_corpus(cap):
    roots = discover_corpus_roots(ARGS.corpus_dir)
    if not roots:
        print("[corpus] WARNING: no NaNROUTER corpora root discovered", flush=True)

    files = []
    seen = set()
    for root in roots:
        for fp in _iter_corpus_files(root):
            rp = os.path.realpath(fp)
            if rp in seen:
                continue
            seen.add(rp)
            files.append(fp)
            if len(files) >= ARGS.max_files:
                break
        if len(files) >= ARGS.max_files:
            break

    groups = defaultdict(list)
    for fp in files:
        groups[_corpus_category(fp)].append(fp)
    rng = random.Random(ARGS.seed)
    for key in groups:
        rng.shuffle(groups[key])

    prose_cats = [
        c for c in ("short_stories", "sentences", "paragraphs", "domains", "other")
        if groups.get(c)
    ]
    base_weights = {
        "short_stories": 0.40, "sentences": 0.25, "paragraphs": 0.25,
        "domains": 0.10, "other": 0.10,
    }
    quotas = {}
    if prose_cats:
        total_w = sum(base_weights[c] for c in prose_cats)
        quotas = {c: int(cap * base_weights[c] / total_w) for c in prose_cats}

    texts = []
    stats = Counter()
    used_files = set()
    used_chars = 0
    for cat in prose_cats:
        quota = quotas[cat]
        cat_chars = 0
        for fp in groups[cat]:
            if used_chars >= cap or cat_chars >= quota:
                break
            take = min(cap - used_chars, quota - cat_chars, ARGS.file_read_chars)
            if take <= 0:
                break
            txt = _read_corpus_file(fp, take).replace("\x00", " ").strip()
            if len(txt) < 32:
                continue
            texts.append(txt)
            used_files.add(fp)
            used_chars += len(txt)
            cat_chars += len(txt)
            stats[cat] += 1

    # Fill any unused budget from remaining files, keeping the category mix.
    if used_chars < cap:
        for cat in prose_cats:
            for fp in groups[cat]:
                if fp in used_files:
                    continue
                if used_chars >= cap:
                    break
                take = min(cap - used_chars, ARGS.file_read_chars)
                txt = _read_corpus_file(fp, take).replace("\x00", " ").strip()
                if len(txt) < 32:
                    continue
                texts.append(txt)
                used_files.add(fp)
                used_chars += len(txt)
                stats[cat] += 1
            if used_chars >= cap:
                break

    wordlist_words = _load_wordlist_words(groups, ARGS.wordlist_words) if ARGS.use_wordlists else []
    # Keep document boundaries so the whole-document split can hold out files,
    # not just a contiguous block of positions from the same files.
    doc_ranges = []
    cursor = 0
    for txt in texts:
        if doc_ranges:
            cursor += 2  # "\n\n" separator
        start = cursor
        cursor += len(txt)
        doc_ranges.append([start, cursor])
    real_text = "\n\n".join(texts)[:cap]
    if doc_ranges:
        doc_ranges = [[s, min(e, len(real_text))] for s, e in doc_ranges if s < len(real_text)]

    if len(real_text.strip()) < 1000:
        if ARGS.require_real_corpus:
            raise SystemExit(
                "[corpus] no usable prose found; refusing synthetic fallback "
                "(remove --require-real-corpus for a smoke run)"
            )
        base = (
            "the function returns the value for the given input and the test checks the result for the case. "
            "a story begins with a person who wants something and ends when the want is answered. "
            "the model reads the words and predicts the next byte from the context it has seen. "
        )
        real_text = (base * max(1, cap // len(base)))[:cap]
        doc_ranges = [[0, len(real_text)]]
        print("[corpus] WARNING: using synthetic fallback text", flush=True)

    # A tiny amount of ASCII text keeps the tokenizer useful even for odd corpora.
    if len(re.findall(r"[a-zA-Z']+", real_text.lower())) < 50:
        base = (
            "the story begins with a question and the answer is found in the next sentence. "
            "the model learns from the words and the context around each word. "
        )
        start = len(real_text)
        real_text += "\n\n" + (base * 50)
        doc_ranges.append([start, len(real_text)])
    if not doc_ranges:
        doc_ranges = [[0, len(real_text)]]

    corpus_sha = hashlib.sha256(real_text.encode("utf-8", errors="ignore")).hexdigest()
    print(
        f"[corpus] roots={roots} files={len(files)} used={len(used_files)} "
        f"chars={len(real_text)} docs={len(doc_ranges)} wordlist={len(wordlist_words)} "
        f"categories={dict(stats)} sha256={corpus_sha[:16]} "
        f"time={time.time() - T0:.1f}s",
        flush=True,
    )
    return real_text, wordlist_words, {
        "roots": roots,
        "files": len(files),
        "used_files": len(used_files),
        "chars": len(real_text),
        "documents": len(doc_ranges),
        "wordlist_words": len(wordlist_words),
        "categories": dict(stats),
        "sha256": corpus_sha,
    }, doc_ranges


real_text, wordlist_words, corpus_stats, doc_ranges = load_corpus(ARGS.corpus_chars)
byte_arr = np.frombuffer(real_text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.int32)
raw_words = re.findall(r"[a-zA-Z']+", real_text.lower())
cnt = Counter(raw_words)
for w in wordlist_words:
    cnt[w] = max(cnt.get(w, 0), 1)
vocab_words = [w for w, _ in cnt.most_common(ARGS.vocab)]
if len(vocab_words) < ARGS.vocab:
    existing = set(vocab_words)
    for w in wordlist_words:
        if w not in existing:
            vocab_words.append(w)
            existing.add(w)
            if len(vocab_words) >= ARGS.vocab:
                break
if len(vocab_words) < 8:
    vocab_words += ["the", "and", "of", "to", "in", "a", "is", "that"]
VOCAB = len(vocab_words) + 1
w2i = {w: i + 1 for i, w in enumerate(vocab_words)}
word_stream = np.array([w2i.get(w, 0) for w in raw_words], dtype=np.int32)

# Sentence rows keep their character start so sentence context can be
# restricted to the same document as the target word.
sent_texts = []
sent_starts = []
last = 0
for m in re.finditer(r"[.!?\n]+", real_text):
    seg = real_text[last:m.start()]
    lead = len(seg) - len(seg.lstrip())
    st = seg.strip()
    if st:
        sent_texts.append(st)
        sent_starts.append(last + lead)
    last = m.end()
if last < len(real_text):
    seg = real_text[last:]
    lead = len(seg) - len(seg.lstrip())
    st = seg.strip()
    if st:
        sent_texts.append(st)
        sent_starts.append(last + lead)

sentences = []
sent_chars = []
for st, sc in zip(sent_texts, sent_starts):
    ws = re.findall(r"[a-zA-Z']+", st.lower())
    if len(ws) >= 3:
        sentences.append([w2i.get(w, 0) for w in ws])
        sent_chars.append(sc)
if not sentences and len(word_stream) >= 3:
    for i in range(0, len(word_stream), 8):
        chunk = word_stream[i:i + 8]
        if len(chunk) >= 3:
            sentences.append(chunk.tolist())
            sent_chars.append(0)
print(f"[corpus] words {len(raw_words)} sents {len(sentences)} vocab {VOCAB}", flush=True)

spans = [m for m in re.finditer(r"[a-zA-Z']+", real_text.lower())]
n_words = min(len(word_stream), len(spans))
word_stream = word_stream[:n_words]
word_char_start = np.array([spans[i].start() for i in range(n_words)], dtype=np.int64)

# Map every word and sentence to a source document. This is the basis of the
# whole-document held-out split and of document-local context windows.
doc_char_start = np.array([s for s, _ in doc_ranges], dtype=np.int64)
doc_char_end = np.array([e for _, e in doc_ranges], dtype=np.int64)
nd = len(doc_ranges)
word_doc = np.searchsorted(doc_char_end, word_char_start, side="right")
word_doc = np.clip(word_doc, 0, max(0, nd - 1))
doc_word_start = np.searchsorted(word_doc, np.arange(nd), side="left")
doc_word_end = np.searchsorted(word_doc, np.arange(nd), side="right")

sent_first = [s[0] for s in sentences if s]
if not sent_first:
    sent_first = [0]
word_to_sent = np.zeros(n_words, dtype=np.int32)
if sent_chars:
    sent_chars_arr = np.array(sent_chars, dtype=np.int64)
    word_sent = np.searchsorted(sent_chars_arr, word_char_start, side="right") - 1
    word_sent = np.clip(word_sent, 0, len(sent_chars_arr) - 1)
    word_to_sent[:n_words] = word_sent.astype(np.int32)
    sent_doc = np.searchsorted(doc_char_end, sent_chars_arr, side="right")
    sent_doc = np.clip(sent_doc, 0, max(0, nd - 1)).astype(np.int32)
else:
    sent_doc = np.zeros(1, dtype=np.int32)
doc_sent_start = np.searchsorted(sent_doc, np.arange(nd), side="left")
doc_sent_end = np.searchsorted(sent_doc, np.arange(nd), side="right")

WIN_B = ARGS.byte_win
WIN_W = ARGS.word_win
WIN_S = ARGS.sent_win
BYTE_CHUNK = ARGS.byte_chunk
BYTE_CHUNKS = WIN_B // BYTE_CHUNK
WORD_CHUNKS = WIN_W // 8
if not sent_first:
    sent_first = [0]


def make_example(pos):
    d = int(word_doc[pos])
    dcs = int(doc_char_start[d])
    dce = int(doc_char_end[d])
    char_start = spans[pos].start()
    b_end = min(char_start, dce)
    b_start = max(dcs, b_end - WIN_B)
    bwin = np.zeros(WIN_B, dtype=np.int32)
    seg = byte_arr[b_start:b_end]
    bwin[WIN_B - len(seg):] = seg

    w_start = max(int(doc_word_start[d]), pos - WIN_W)
    wwin = np.zeros(WIN_W, dtype=np.int32)
    segw = word_stream[w_start:pos]
    wwin[WIN_W - len(segw):] = segw

    s_lo = int(doc_sent_start[d])
    s_hi = int(doc_sent_end[d])
    swin = np.zeros(WIN_S, dtype=np.int32)
    if s_hi > s_lo:
        si = int(word_to_sent[pos]) if pos < len(word_to_sent) else s_lo
        si = min(max(si, s_lo), s_hi - 1)
        start = max(s_lo, si - WIN_S)
        seg = sent_first[start:si]
        if len(seg) > 0:
            swin[WIN_S - len(seg):] = np.array(seg, dtype=np.int32)
        ys = int(sent_first[si + 1]) if si + 1 < s_hi else 0
    else:
        ys = 0

    yb = int(byte_arr[min(char_start, len(byte_arr) - 1)])
    yw = int(word_stream[pos])
    return bwin, wwin, swin, yb, yw, ys


def build(pos_list):
    B, W, S, yb, yw, ys = [], [], [], [], [], []
    for pos in pos_list:
        bwin, wwin, swin, target_b, target_w, target_s = make_example(pos)
        B.append(bwin)
        W.append(wwin)
        S.append(swin)
        yb.append(target_b)
        yw.append(target_w)
        ys.append(target_s)
    return (
        np.array(B, dtype=np.int32), np.array(W, dtype=np.int32), np.array(S, dtype=np.int32),
        np.array(yb, dtype=np.int32), np.array(yw, dtype=np.int32), np.array(ys, dtype=np.int32),
    )


if n_words < 200:
    raise SystemExit(f"[data] corpus is too small ({n_words} words); need at least 200")

# Per-document context: keep 150 words of warm-up context inside each
# document, and never use a target at the final word of a document.
positions = np.arange(max(0, n_words - 1))
pos_doc = word_doc[positions]
eligible = (
    (positions - doc_word_start[pos_doc] >= 150)
    & (positions < doc_word_end[pos_doc] - 1)
)
all_pos = positions[eligible]
if len(all_pos) < 200:
    raise SystemExit(f"[data] only {len(all_pos)} eligible positions; need at least 200")
test_size = min(ARGS.test_size, max(64, len(all_pos) // 10))
split_name = "contiguous-block-v2"
test_doc_sha = ""

if ARGS.split_mode == "document":
    # Hold out whole files. The held-out documents are chosen before sampling
    # test positions, so no training window can come from a test document.
    pos_doc_all = word_doc[all_pos]
    doc_counts = np.bincount(pos_doc_all, minlength=nd)
    rng = np.random.default_rng(ARGS.seed * 1000003 + 991)
    order = rng.permutation(nd)
    target_docs = max(1, int(round(nd * ARGS.doc_holdout_frac)))
    test_docs = []
    test_count = 0
    for d in order:
        if doc_counts[d] == 0:
            continue
        test_docs.append(int(d))
        test_count += int(doc_counts[d])
        if len(test_docs) >= target_docs and test_count >= test_size:
            break
    if not test_docs:
        raise SystemExit("[data] document split could not find a held-out document")
    test_docs = sorted(test_docs)
    test_mask = np.isin(pos_doc_all, test_docs)
    test_all = all_pos[test_mask]
    train_pool = all_pos[~test_mask]
    if len(test_all) > test_size:
        rng2 = np.random.default_rng(ARGS.seed * 1000003 + 992)
        test_pos = np.sort(rng2.choice(test_all, size=test_size, replace=False))
    else:
        test_pos = test_all
    test_size = len(test_pos)
    test_start = -1
    gap = 0
    split_name = "document-holdout-v1"
    test_doc_sha = hashlib.sha256(
        ",".join(str(d) for d in test_docs).encode("utf-8")
    ).hexdigest()[:16]
    print(
        f"[data] held out {len(test_docs)}/{nd} documents "
        f"({test_count} positions, sampled {test_size})",
        flush=True,
    )
else:
    # Contiguous position block with a gap larger than both context windows.
    test_start = min(len(all_pos) // 2, len(all_pos) - test_size)
    test_start = max(0, test_start)
    test_pos = all_pos[test_start:test_start + test_size]
    gap = max(WIN_W, WIN_B) + 1
    train_pool = np.concatenate([
        all_pos[:max(0, test_start - gap)],
        all_pos[min(len(all_pos), test_start + test_size + gap):],
    ])

if len(train_pool) < 16:
    raise SystemExit("[data] corpus produced too few training positions")
if len(train_pool) < ARGS.examples:
    print(
        f"[data] only {len(train_pool)} training positions outside the held-out "
        f"split; reducing examples {ARGS.examples} -> {len(train_pool)}",
        flush=True,
    )
    ARGS.examples = len(train_pool)


def sample_train_positions(epoch_no):
    """Draw fresh positions from the full pool so epochs do not replay one subset."""
    n = min(ARGS.examples, len(train_pool))
    if ARGS.resample_train:
        rng = np.random.default_rng(ARGS.seed * 1000003 + int(epoch_no))
        return rng.choice(train_pool, size=n, replace=False)
    return train_pool[:n]


test_B, test_W, test_S, test_yb, test_yw, test_ys = build(test_pos)
# Training arrays are built lazily at the start of every epoch. This keeps
# memory flat and lets each epoch see different positions from the corpus.
train_B = np.empty((0, WIN_B), dtype=np.int32)
train_W = np.empty((0, WIN_W), dtype=np.int32)
train_S = np.empty((0, WIN_S), dtype=np.int32)
train_yb = np.empty((0,), dtype=np.int32)
train_yw = np.empty((0,), dtype=np.int32)
train_ys = np.empty((0,), dtype=np.int32)
print(
    f"[data] split={split_name} train_pool {len(train_pool)} test {test_B.shape[0]} "
    f"test_start {test_start} gap {gap} docs {nd} "
    f"resample={ARGS.resample_train} time {time.time() - T0:.1f}s",
    flush=True,
)
# Free raw text and token lists, but keep spans: make_example uses it for the
# fresh position samples on every epoch.
del real_text, raw_words, sentences
gc.collect()


# ---------------------------------------------------------------- model
DIM_B = ARGS.dim_b
DIM_W = ARGS.dim_w
DIM_S = ARGS.dim_s
FF_B = ARGS.ff_b
FF_W = ARGS.ff_w
FF_S = ARGS.ff_s
DEPTH = ARGS.depth
USE_CHUNK_CONV = bool(ARGS.chunk_conv)


class MoE(tf.keras.layers.Layer):
    """256 experts = C(256,2) pair heads. Top-2 gating."""

    def __init__(self, n_exp, dim, topk=2):
        super().__init__()
        self.n = n_exp
        self.dim = dim
        self.topk = topk

    def build(self, s):
        super().build(s)
        self.router = self.add_weight(shape=(self.dim, self.n), initializer="glorot_uniform")
        self.ew = self.add_weight(shape=(self.n, self.dim, self.dim), initializer="glorot_uniform")
        self.eb = self.add_weight(shape=(self.n, self.dim), initializer="zeros")

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        xf = tf.reshape(x, (-1, self.dim))
        logits = tf.matmul(xf, self.router)
        vals, idx = tf.math.top_k(logits, k=self.topk)
        gates = tf.nn.softmax(vals, axis=-1)
        Ew = tf.gather(self.ew, idx)
        Eb = tf.gather(self.eb, idx)
        xexp = tf.expand_dims(tf.tile(tf.expand_dims(xf, 1), [1, self.topk, 1]), 2)
        out = tf.squeeze(tf.matmul(xexp, Ew), axis=2) + Eb
        return tf.reshape(tf.reduce_sum(out * tf.expand_dims(gates, -1), axis=1), (B, T, self.dim))


class Micro(tf.keras.layers.Layer):
    """One tiny transformer block, 4 heads. Stack DEPTH of these per tower."""

    def __init__(self, dim, heads, ff, drop=0.1):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff, activation="gelu"),
            tf.keras.layers.Dense(dim),
        ])
        self.drop1 = tf.keras.layers.Dropout(drop)
        self.drop2 = tf.keras.layers.Dropout(drop)

    def call(self, x, training=False):
        a = self.attn(x, x)
        a = self.drop1(a, training=training)
        x = self.ln1(x + a)
        f = self.ffn(x)
        f = self.drop2(f, training=training)
        x = self.ln2(x + f)
        return x


class ChunkConv(tf.keras.layers.Layer):
    """Order-preserving chunk stem.

    Mean-pooling a chunk of eight byte/word embeddings throws away the order
    inside the chunk.  A strided convolution with kernel == stride learns a
    separate weight for each position in the chunk, so the transformer sees
    ``the cat`` differently from ``cat the``.
    """

    def __init__(self, dim, chunk):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(
            dim, chunk, strides=chunk, padding="valid", activation="gelu"
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        return self.norm(self.conv(x))


class TripleStaged(tf.keras.Model):
    def __init__(self, vocab):
        super().__init__()
        self.byte_emb = tf.keras.layers.Embedding(256, DIM_B)
        self.byte_pos = tf.keras.layers.Embedding(BYTE_CHUNKS, DIM_B)
        self.byte_moe = MoE(256, DIM_B, 2)
        if USE_CHUNK_CONV:
            self.byte_chunk_stem = ChunkConv(DIM_B, BYTE_CHUNK)
        else:
            self.byte_proj = tf.keras.layers.Dense(DIM_B)
        self.byte_blocks = [Micro(DIM_B, 4, FF_B, ARGS.dropout) for _ in range(DEPTH)]

        self.word_emb = tf.keras.layers.Embedding(vocab, DIM_W)
        self.word_pos = tf.keras.layers.Embedding(WORD_CHUNKS, DIM_W)
        self.word_moe = MoE(256, DIM_W, 2)
        if USE_CHUNK_CONV:
            self.word_chunk_stem = ChunkConv(DIM_W, 8)
        else:
            self.word_proj = tf.keras.layers.Dense(DIM_W)
        self.word_blocks = [Micro(DIM_W, 4, FF_W, ARGS.dropout) for _ in range(DEPTH)]

        self.sent_emb = tf.keras.layers.Embedding(vocab, DIM_S)
        self.sent_pos = tf.keras.layers.Embedding(WIN_S, DIM_S)
        self.sent_moe = MoE(256, DIM_S, 2)
        self.sent_blocks = [Micro(DIM_S, 4, FF_S, ARGS.dropout) for _ in range(DEPTH)]

        self.byte_to_word = tf.keras.layers.Dense(DIM_W)
        self.word_to_sent = tf.keras.layers.Dense(DIM_S)
        self.sent_to_word = tf.keras.layers.Dense(DIM_W)
        self.word_to_byte = tf.keras.layers.Dense(DIM_B)
        self.sent_to_byte = tf.keras.layers.Dense(DIM_B)
        self.byte_head = tf.keras.layers.Dense(256)
        self.word_head = tf.keras.layers.Dense(vocab)
        self.sent_head = tf.keras.layers.Dense(vocab)
        self.intro = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False, two_way=True):
        b_in, w_in, s_in = inputs

        b_r = tf.reshape(b_in, (-1, BYTE_CHUNKS, BYTE_CHUNK))
        b_emb = self.byte_emb(b_r)
        if USE_CHUNK_CONV:
            b_chunk = self.byte_chunk_stem(b_emb)
        else:
            b_chunk = self.byte_proj(tf.reduce_mean(b_emb, axis=2))
        b_chunk = b_chunk + self.byte_pos(tf.range(BYTE_CHUNKS))[None, :, :]
        b_h = self.byte_moe(b_chunk) + b_chunk
        for block in self.byte_blocks:
            b_h = block(b_h, training=training)
        b_last = b_h[:, -1, :]

        w_r = tf.reshape(w_in, (-1, WORD_CHUNKS, 8))
        w_emb = self.word_emb(w_r)
        if USE_CHUNK_CONV:
            w_chunk = self.word_chunk_stem(w_emb)
        else:
            w_chunk = self.word_proj(tf.reduce_mean(w_emb, axis=2))
        w_chunk = w_chunk + self.word_pos(tf.range(WORD_CHUNKS))[None, :, :]
        w_h = self.word_moe(w_chunk) + w_chunk
        for block in self.word_blocks:
            w_h = block(w_h, training=training)
        w_last = w_h[:, -1, :]

        s_emb = self.sent_emb(s_in) + self.sent_pos(tf.range(WIN_S))[None, :, :]
        s_h = self.sent_moe(s_emb) + s_emb
        for block in self.sent_blocks:
            s_h = block(s_h, training=training)
        s_last = s_h[:, -1, :]

        w_up = w_last + self.byte_to_word(b_last)
        s_up = s_last + self.word_to_sent(w_last)
        if two_way:
            w_cond = w_up + self.sent_to_word(s_up)
            b_cond = b_last + self.word_to_byte(w_cond) + self.sent_to_byte(s_up)
        else:
            w_cond = w_up
            b_cond = b_last
        return self.byte_head(b_cond), self.word_head(w_cond), self.sent_head(s_up), self.intro(b_cond)


model = TripleStaged(VOCAB)
_ = model(
    (
        tf.zeros((2, WIN_B), dtype=tf.int32),
        tf.zeros((2, WIN_W), dtype=tf.int32),
        tf.zeros((2, WIN_S), dtype=tf.int32),
    ),
    training=False,
    two_way=True,
)
print(
    f"[model] params {model.count_params()} byte {WIN_B}={BYTE_CHUNKS}x{BYTE_CHUNK} "
    f"word {WIN_W}={WORD_CHUNKS}x8 sent {WIN_S} "
    f"dims {DIM_B}/{DIM_W}/{DIM_S} ff {FF_B}/{FF_W}/{FF_S} "
    f"depth {DEPTH} chunk_conv {int(USE_CHUNK_CONV)}",
    flush=True,
)
opt = tf.keras.optimizers.AdamW(learning_rate=ARGS.lr, weight_decay=ARGS.weight_decay)
# Build every optimizer slot up front. Epoch 1 is bottom-up, so the top-down
# cross layers have no gradients yet; without this, Keras 3 rejects them when
# they become active in epoch 2, and checkpoint resume would miss their slots.
opt.build(model.trainable_variables)

SIGNATURE = json.dumps({
    "vocab": VOCAB,
    "byte_win": WIN_B,
    "word_win": WIN_W,
    "sent_win": WIN_S,
    "byte_chunk": BYTE_CHUNK,
    "dim_b": DIM_B,
    "dim_w": DIM_W,
    "dim_s": DIM_S,
    "ff_b": FF_B,
    "ff_w": FF_W,
    "ff_s": FF_S,
    "depth": DEPTH,
    "chunk_conv": bool(USE_CHUNK_CONV),
    "corpus_sha256": corpus_stats["sha256"],
    "split": split_name,
    "split_mode": ARGS.split_mode,
    "test_start": int(test_start),
    "test_size": int(test_size),
    "test_gap": int(gap),
    "test_docs": test_doc_sha,
    "doc_holdout_frac": float(ARGS.doc_holdout_frac),
    "resample_train": bool(ARGS.resample_train),
    "label_smoothing": float(ARGS.label_smoothing),
}, sort_keys=True)


def load_history(path):
    rows = []
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            print(f"[history] could not read {path}: {e}", flush=True)
    return rows


def append_history(rec):
    if not ARGS.history:
        return
    path = _norm(ARGS.history)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def optimizer_variables():
    # Keras 3 exposes ``optimizer.variables`` as a property; older versions
    # exposed it as a method. Accept both.
    variables = opt.variables
    return variables() if callable(variables) else variables


def save_state_to(path, epochs_done, extra_meta=None):
    if not path:
        return
    path = _norm(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {f"w{i}": v.numpy() for i, v in enumerate(model.trainable_variables)}
    arrays.update({f"o{i}": v.numpy() for i, v in enumerate(optimizer_variables())})
    meta = {
        "signature": SIGNATURE,
        "epochs_done": int(epochs_done),
        "args": vars(ARGS),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    arrays["__meta__"] = np.array(json.dumps(meta))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    print(f"[checkpoint] saved epoch {epochs_done} -> {path}", flush=True)


def save_state(epochs_done):
    save_state_to(ARGS.checkpoint, epochs_done)


def save_weights_to(path):
    if not path:
        return
    path = _norm(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        model.save_weights(path)
        print(f"[weights] saved -> {path}", flush=True)
    except Exception as e:
        # A failed H5 export must not discard a completed/best checkpoint.
        print(f"[warn] could not export weights to {path}: {e}", flush=True)


def load_state():
    if ARGS.no_resume or not ARGS.checkpoint:
        return 0
    path = _norm(ARGS.checkpoint)
    if not os.path.exists(path):
        return 0
    try:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["__meta__"].item()))
    except Exception as e:
        print(f"[resume] could not read {path}: {e}; starting fresh", flush=True)
        return 0
    if meta.get("signature") != SIGNATURE:
        print("[resume] checkpoint signature mismatch (model or corpus changed); starting fresh", flush=True)
        return 0
    try:
        for i, v in enumerate(model.trainable_variables):
            key = f"w{i}"
            if key not in data:
                raise KeyError(key)
            v.assign(data[key])
        for i, v in enumerate(optimizer_variables()):
            key = f"o{i}"
            if key in data:
                v.assign(data[key])
    except Exception as e:
        print(f"[resume] restore failed: {e}; starting fresh", flush=True)
        return 0
    epochs_done = int(meta.get("epochs_done", 0))
    print(f"[resume] restored epoch {epochs_done} from {path}", flush=True)
    return epochs_done


def lr_at(epoch):
    base = ARGS.lr
    if ARGS.lr_schedule == "constant":
        return base
    if ARGS.warmup_epochs > 0 and epoch < ARGS.warmup_epochs:
        return ARGS.min_lr + (base - ARGS.min_lr) * (epoch + 1) / ARGS.warmup_epochs
    progress = (epoch - ARGS.warmup_epochs) / max(1, ARGS.epochs - ARGS.warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return ARGS.min_lr + 0.5 * (base - ARGS.min_lr) * (1.0 + math.cos(math.pi * progress))


def cross_entropy_per_example(y, logits):
    """Cross-entropy with optional label smoothing, per example."""
    nll = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
    eps = float(ARGS.label_smoothing or 0.0)
    if eps > 0.0:
        logp = tf.nn.log_softmax(logits, axis=-1)
        smooth = -tf.reduce_mean(logp, axis=-1)
        nll = (1.0 - eps) * nll + eps * smooth
    return nll


@tf.function
def train_step(b, w, s, yb, yw, ys, two_way):
    with tf.GradientTape() as tape:
        bl, wl, sl, intro = model((b, w, s), training=True, two_way=two_way)
        per_b = cross_entropy_per_example(yb, bl)
        per_w = cross_entropy_per_example(yw, wl)
        per_s = cross_entropy_per_example(ys, sl)
        il = tf.reduce_mean(tf.square(tf.squeeze(intro, -1) - tf.stop_gradient(per_b)))
        wgt = 1.0 + tf.nn.sigmoid(tf.squeeze(intro, -1))
        wgt = wgt / tf.reduce_mean(wgt)
        total = (
            tf.reduce_mean(per_b * wgt)
            + 0.5 * tf.reduce_mean(per_w)
            + 0.3 * tf.reduce_mean(per_s)
            + 0.1 * il
        )
    grads = tape.gradient(total, model.trainable_variables)
    pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if ARGS.grad_clip and ARGS.grad_clip > 0 and pairs:
        gs, _ = tf.clip_by_global_norm([g for g, _ in pairs], ARGS.grad_clip)
        pairs = list(zip(gs, [v for _, v in pairs]))
    opt.apply_gradients(pairs)
    return total


def evaluate(B, W, S, yb, yw, ys, two_way=True):
    n = len(B)
    if n == 0:
        return {k: 0.0 for k in (
            "byte_loss", "byte_bits", "byte_top1", "byte_top5",
            "word_loss", "word_bits", "word_top1", "word_top5",
            "sent_loss", "sent_bits", "sent_top1", "sent_top5",
        )}
    tot_b = tot_w = tot_s = 0.0
    t1b = t5b = t1w = t5w = t1s = t5s = 0
    for i in range(0, n, ARGS.eval_batch):
        j = min(i + ARGS.eval_batch, n)
        bl, wl, sl, _ = model((B[i:j], W[i:j], S[i:j]), training=False, two_way=two_way)
        pb = tf.nn.softmax(bl, axis=-1).numpy()
        pw = tf.nn.softmax(wl, axis=-1).numpy()
        ps = tf.nn.softmax(sl, axis=-1).numpy()
        m = j - i
        yb_b = yb[i:j]
        yw_b = yw[i:j]
        ys_b = ys[i:j]

        tot_b += float(np.sum(-np.log(np.maximum(pb[np.arange(m), yb_b], 1e-12))))
        tot_w += float(np.sum(-np.log(np.maximum(pw[np.arange(m), yw_b], 1e-12))))
        tot_s += float(np.sum(-np.log(np.maximum(ps[np.arange(m), ys_b], 1e-12))))

        top_b = np.argpartition(-pb, kth=5, axis=1)[:, :5]
        top_w = np.argpartition(-pw, kth=5, axis=1)[:, :5]
        top_s = np.argpartition(-ps, kth=5, axis=1)[:, :5]
        t1b += int(np.sum(np.argmax(pb, axis=1) == yb_b))
        t1w += int(np.sum(np.argmax(pw, axis=1) == yw_b))
        t1s += int(np.sum(np.argmax(ps, axis=1) == ys_b))
        t5b += int(np.sum(np.any(top_b == yb_b[:, None], axis=1)))
        t5w += int(np.sum(np.any(top_w == yw_b[:, None], axis=1)))
        t5s += int(np.sum(np.any(top_s == ys_b[:, None], axis=1)))

    return {
        "byte_loss": tot_b / n, "byte_bits": tot_b / n / math.log(2),
        "byte_top1": t1b / n, "byte_top5": t5b / n,
        "word_loss": tot_w / n, "word_bits": tot_w / n / math.log(2),
        "word_top1": t1w / n, "word_top5": t5w / n,
        "sent_loss": tot_s / n, "sent_bits": tot_s / n / math.log(2),
        "sent_top1": t1s / n, "sent_top5": t5s / n,
    }


def self_loop(epoch_no):
    """Introspection-filtered self-loop; diagnostics only unless --self-mix."""
    print(
        f" SELF-LOOP filtered {ARGS.self_streams} streams x {ARGS.self_steps} steps",
        flush=True,
    )
    NS = min(ARGS.self_streams, len(test_B))
    if NS <= 0:
        return
    cur_w = test_W[:NS].copy()
    cur_b = test_B[:NS].copy()
    cur_s = test_S[:NS].copy()
    gen_words = []
    gen_intro = []
    gen_ent = []
    for _ in range(ARGS.self_steps):
        _, wl, _, intro = model((cur_b, cur_w, cur_s), training=False, two_way=True)
        p = tf.nn.softmax(wl / 0.9, axis=-1).numpy()
        ent = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)
        pred = tf.squeeze(intro, -1).numpy()
        gen_intro.append(pred)
        gen_ent.append(ent)
        nxt = np.array(
            [np.random.choice(VOCAB, p=p[i] / p[i].sum()) for i in range(len(cur_w))],
            dtype=np.int32,
        )
        gen_words.append(nxt)
        cur_w = np.concatenate([cur_w[:, 1:], nxt[:, None]], axis=1)
    gen_words = np.array(gen_words)
    avg_intro = np.mean(np.array(gen_intro), axis=0)
    avg_ent = np.mean(np.array(gen_ent), axis=0)
    keep = [
        i for i in range(NS)
        if avg_intro[i] < np.median(avg_intro) and avg_ent[i] < np.median(avg_ent)
    ]
    print(
        f"  keep {len(keep)}/{NS} intro median {np.median(avg_intro):.2f} "
        f"ent median {np.median(avg_ent):.2f} time {time.time() - T0:.1f}s",
        flush=True,
    )
    for si in keep[:3]:
        words = [
            vocab_words[idx - 1] if 1 <= idx <= len(vocab_words) else "<unk>"
            for idx in gen_words[:, si]
        ]
        print(
            f"  kept {si}: {' '.join(words[:8])} . {' '.join(words[8:15])} "
            f". intro {avg_intro[si]:.2f} ent {avg_ent[si]:.2f}",
            flush=True,
        )
    if not ARGS.self_mix or not keep:
        return
    kept_text = " ".join(
        [
            vocab_words[idx - 1] if 1 <= idx <= len(vocab_words) else "unk"
            for idx in gen_words[:, keep].flatten()
        ]
    )
    gen_bytes = np.frombuffer(kept_text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.int32)
    print(f"  kept text {len(kept_text)} chars bytes {len(gen_bytes)}", flush=True)
    if len(gen_bytes) > WIN_B + 100:
        max_self = min(500, len(gen_bytes) - WIN_B)
        self_B = np.array([gen_bytes[i:i + WIN_B] for i in range(max_self)], dtype=np.int32)
        rep = min(500, len(self_B), len(train_B))
        train_B[:rep] = self_B[:rep]
        print(f"  mixed {rep}", flush=True)
    else:
        print(f"  too short for {WIN_B}-byte window, skip mix", flush=True)
    gc.collect()


# ---------------------------------------------------------------- train
start_epoch = load_state()
history = load_history(ARGS.history)
if start_epoch == 0:
    # Fresh run: do not mix with a stale history file from another checkpoint.
    if ARGS.history and os.path.exists(_norm(ARGS.history)):
        try:
            os.replace(_norm(ARGS.history), _norm(ARGS.history) + ".bak")
        except OSError:
            pass
    history = []
else:
    history = [
        h for h in history
        if h.get("stage") in ("init", "final")
        or int(h.get("epoch", 0)) <= start_epoch
    ]

budget_s = ARGS.time_budget_minutes * 60.0 if ARGS.time_budget_minutes and ARGS.time_budget_minutes > 0 else 0.0

if start_epoch == 0:
    print("[eval] init...", flush=True)
    res = evaluate(test_B, test_W, test_S, test_yb, test_yw, test_ys, two_way=True)
    init_rec = {"stage": "init", "epoch": 0, "ts": datetime.now(timezone.utc).isoformat(), **res}
    append_history(init_rec)
    history.append(init_rec)
    print(
        f"INIT byte {res['byte_loss']:.4f} bits {res['byte_bits']:.4f} "
        f"top1 {res['byte_top1']:.3f} top5 {res['byte_top5']:.3f} | "
        f"word {res['word_loss']:.4f} bits {res['word_bits']:.4f} "
        f"top1 {res['word_top1']:.3f} top5 {res['word_top5']:.3f} | "
        f"sent {res['sent_loss']:.4f} bits {res['sent_bits']:.4f} "
        f"top1 {res['sent_top1']:.3f} top5 {res['sent_top5']:.3f} "
        f"time {time.time() - T0:.1f}s",
        flush=True,
    )
else:
    print(f"[resume] continuing at epoch {start_epoch + 1}/{ARGS.epochs}", flush=True)

completed_epochs = start_epoch
best_eval_loss = float("inf")
best_epoch = 0
best_record = None
for h in history:
    if h.get("stage") in ("init", "final"):
        continue
    if all(k in h for k in ("byte_loss", "word_loss", "sent_loss")):
        h_loss = h.get(
            "eval_loss",
            h["byte_loss"] + 0.5 * h["word_loss"] + 0.3 * h["sent_loss"],
        )
        if h_loss < best_eval_loss:
            best_eval_loss = float(h_loss)
            best_epoch = int(h.get("epoch", 0))
            best_record = dict(h)
evals_without_improvement = 0
early_stop_triggered = False
if start_epoch < ARGS.epochs:
    for epoch in range(start_epoch, ARGS.epochs):
        epoch_no = epoch + 1
        two_way = epoch >= 1
        mode = "bottom-up" if not two_way else "two-way"
        lr = lr_at(epoch)
        try:
            opt.learning_rate.assign(lr)
        except Exception:
            pass
        print(f"EPOCH {epoch_no}/{ARGS.epochs} {mode} lr {lr:.6f} time {time.time() - T0:.1f}s", flush=True)
        if len(train_B) == 0 or ARGS.resample_train:
            t_sample = time.time()
            pos = sample_train_positions(epoch_no)
            train_B, train_W, train_S, train_yb, train_yw, train_ys = build(pos)
            print(
                f"  data {len(pos)} positions sampled from {len(train_pool)} "
                f"in {time.time() - t_sample:.1f}s",
                flush=True,
            )
        t_epoch = time.time()
        # Per-epoch RNG makes the shuffle reproducible even if the run is resumed.
        rng = np.random.default_rng(ARGS.seed * 1000003 + 7 + epoch_no)
        perm = rng.permutation(len(train_B))
        Bs, Ws, Ss = train_B[perm], train_W[perm], train_S[perm]
        ybs, yws, yss = train_yb[perm], train_yw[perm], train_ys[perm]
        tot = 0.0
        steps = 0
        for i in range(0, len(Bs), ARGS.batch):
            loss = train_step(
                Bs[i:i + ARGS.batch], Ws[i:i + ARGS.batch], Ss[i:i + ARGS.batch],
                ybs[i:i + ARGS.batch], yws[i:i + ARGS.batch], yss[i:i + ARGS.batch],
                two_way,
            )
            tot += float(loss)
            steps += 1
            if ARGS.report_every_steps > 0 and steps % ARGS.report_every_steps == 0:
                print(
                    f"  step {steps}/{math.ceil(len(Bs) / ARGS.batch)} "
                    f"loss {float(loss):.4f} elapsed {time.time() - t_epoch:.1f}s",
                    flush=True,
                )
        train_loss = tot / max(1, steps)
        completed_epochs = epoch_no

        res = {}
        do_eval = (
            epoch_no == 1
            or epoch_no == ARGS.epochs
            or (ARGS.eval_every > 0 and epoch_no % ARGS.eval_every == 0)
        )
        if do_eval:
            res = evaluate(test_B, test_W, test_S, test_yb, test_yw, test_ys, two_way=True)
            eval_loss = res["byte_loss"] + 0.5 * res["word_loss"] + 0.3 * res["sent_loss"]
            res["eval_loss"] = eval_loss
            if eval_loss < best_eval_loss - 1e-9:
                best_eval_loss = eval_loss
                best_epoch = epoch_no
                best_record = {
                    "stage": "best",
                    "epoch": epoch_no,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "train_loss": train_loss,
                    "lr": lr,
                    **res,
                }
                evals_without_improvement = 0
                save_state_to(
                    ARGS.best_checkpoint,
                    epoch_no,
                    {"best_eval_loss": best_eval_loss, "best": True},
                )
                save_weights_to(ARGS.best_weights)
                print(
                    f"[best] epoch {epoch_no} eval_loss {eval_loss:.4f} "
                    f"byte_bits {res['byte_bits']:.4f} "
                    f"word_bits {res['word_bits']:.4f} "
                    f"sent_bits {res['sent_bits']:.4f}",
                    flush=True,
                )
            else:
                evals_without_improvement += 1

        elapsed = time.time() - T0
        done = max(1, epoch_no - start_epoch)
        eta = elapsed / done * (ARGS.epochs - epoch_no)
        rec = {
            "stage": f"epoch{epoch_no}_{mode}",
            "epoch": epoch_no,
            "ts": datetime.now(timezone.utc).isoformat(),
            "train_loss": train_loss,
            "lr": lr,
            "elapsed_s": elapsed,
            "eta_s": eta,
            **res,
        }
        append_history(rec)
        history.append(rec)
        if res:
            print(
                f"EPOCH {epoch_no}/{ARGS.epochs} {mode} lr {lr:.6f} "
                f"train {train_loss:.4f} eval {res['eval_loss']:.4f} | "
                f"byte {res['byte_loss']:.4f} bits {res['byte_bits']:.4f} "
                f"top1 {res['byte_top1']:.3f} top5 {res['byte_top5']:.3f} | "
                f"word {res['word_loss']:.4f} bits {res['word_bits']:.4f} "
                f"top1 {res['word_top1']:.3f} top5 {res['word_top5']:.3f} | "
                f"sent {res['sent_loss']:.4f} bits {res['sent_bits']:.4f} "
                f"top1 {res['sent_top1']:.3f} top5 {res['sent_top5']:.3f} | "
                f"elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m",
                flush=True,
            )
        else:
            print(
                f"EPOCH {epoch_no}/{ARGS.epochs} {mode} lr {lr:.6f} "
                f"train {train_loss:.4f} | elapsed {elapsed / 60:.1f}m "
                f"eta {eta / 60:.1f}m",
                flush=True,
            )

        if do_eval and ARGS.patience > 0 and evals_without_improvement >= ARGS.patience:
            save_state(epoch_no)
            early_stop_triggered = True
            print(
                f"[early-stop] no held-out improvement for {ARGS.patience} evals; "
                f"best epoch {best_epoch} eval_loss {best_eval_loss:.4f}",
                flush=True,
            )
            break

        if (
            ARGS.self_loop_every > 0
            and epoch < ARGS.epochs - 1
            and epoch_no >= ARGS.self_loop_start_epoch
            and (epoch_no - ARGS.self_loop_start_epoch) % ARGS.self_loop_every == 0
        ):
            self_loop(epoch_no)
        if ARGS.save_every > 0 and epoch_no % ARGS.save_every == 0:
            save_state(epoch_no)
        if budget_s and time.time() - T0 >= budget_s:
            save_state(epoch_no)
            print(
                f"[budget] reached {ARGS.time_budget_minutes:.0f} min at epoch "
                f"{epoch_no}; checkpoint saved, resume with the same command",
                flush=True,
            )
            break
        gc.collect()

if completed_epochs < ARGS.epochs:
    print(
        f"[train] stopped at epoch {completed_epochs}/{ARGS.epochs}; "
        f"checkpoint is resumable",
        flush=True,
    )

print("FINAL full eval (latest weights)", flush=True)
res = evaluate(test_B, test_W, test_S, test_yb, test_yw, test_ys, two_way=True)
res["eval_loss"] = res["byte_loss"] + 0.5 * res["word_loss"] + 0.3 * res["sent_loss"]
final_rec = {
    "stage": "final",
    "epoch": completed_epochs,
    "ts": datetime.now(timezone.utc).isoformat(),
    "finished": completed_epochs >= ARGS.epochs,
    **res,
}
append_history(final_rec)
history.append(final_rec)
print(
    f"FINAL BYTE loss {res['byte_loss']:.4f} bits {res['byte_bits']:.4f} "
    f"top1 {res['byte_top1']:.3f} top5 {res['byte_top5']:.3f} "
    f"ppl {math.exp(res['byte_loss']):.1f}",
    flush=True,
)
print(
    f"FINAL WORD loss {res['word_loss']:.4f} bits {res['word_bits']:.4f} "
    f"top1 {res['word_top1']:.3f} top5 {res['word_top5']:.3f} "
    f"chance {5 / VOCAB:.3f}",
    flush=True,
)
print(
    f"FINAL SENT loss {res['sent_loss']:.4f} bits {res['sent_bits']:.4f} "
    f"top1 {res['sent_top1']:.3f} top5 {res['sent_top5']:.3f}",
    flush=True,
)
print(
    f"params {model.count_params()} byte {WIN_B} word {WIN_W} sent {WIN_S} "
    f"dims {DIM_B}/{DIM_W}/{DIM_S} depth {DEPTH} chunk_conv {int(USE_CHUNK_CONV)} "
    f"byte_chunk {BYTE_CHUNK} split {split_name} "
    f"completed {completed_epochs}/{ARGS.epochs} epochs "
    f"CPU time {time.time() - T0:.1f}s",
    flush=True,
)
if best_record:
    print(
        f"BEST epoch {best_epoch} eval_loss {best_eval_loss:.4f} "
        f"byte bits {best_record['byte_bits']:.4f} "
        f"word bits {best_record['word_bits']:.4f} "
        f"sent bits {best_record['sent_bits']:.4f}",
        flush=True,
    )
if completed_epochs > 0:
    save_state(completed_epochs)
out = {
    "args": vars(ARGS),
    "params": int(model.count_params()),
    "signature": SIGNATURE,
    "corpus": corpus_stats,
    "time_s": time.time() - T0,
    "completed_epochs": completed_epochs,
    "target_epochs": ARGS.epochs,
    "finished": completed_epochs >= ARGS.epochs,
    "early_stopped": bool(early_stop_triggered),
    "best_epoch": best_epoch,
    "best_eval_loss": best_eval_loss if best_record else None,
    "best": best_record,
    "history": history,
    "final": final_rec,
}
if ARGS.out:
    out_path = _norm(ARGS.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[out] wrote {out_path}", flush=True)
if ARGS.save_weights:
    save_weights_to("diamond_weights.weights.h5")
