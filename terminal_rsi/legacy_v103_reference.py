"""Causal NaNCM-v103 feature bridge for the *existing* Terminal-RSI GPU lineage.

These are genuine training-only token-prefix and historic-match features, not
the original C model's learned Slot/APM weights. Everything is keyed by bytes
strictly before the scored target. Numpy preprocessing is a one-time CPU data
loader; the predictive tensors, mixer, and RSI optimizer run on CUDA.
"""
from __future__ import annotations

import hashlib
import struct
import time
from pathlib import Path

import numpy as np

WORD_DOMAIN = 16384
MATCH_LENGTHS = (5, 12)
HASH_BASE = np.uint64(0x100000001B3)


def mix32(a: int, b: int) -> int:
    """Exact C `mix(u32,u32)` arithmetic, including every overflow."""
    mask = 0xFFFFFFFF
    x = ((a * 0x9E3779B1) ^ ((b + 0x7F4A7C15) * 0x85EBCA6B)) & mask
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & mask
    x = ((x ^ (x >> 15)) * 0x846CA68B) & mask
    return (x ^ (x >> 16)) & mask


def word_byte(c: int) -> bool:
    return (65 <= c <= 90 or 97 <= c <= 122 or 48 <= c <= 57 or c >= 128)


def read_bpe_ranks(path: str | Path | None) -> dict[tuple[int, int], int]:
    """Load the original v103 zero-indexed merge ranks from `bpe.txt`."""
    if path is None:
        return {}
    text = Path(path).read_text(encoding="ascii")
    ranks = {}
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        a, b = [int(x) for x in line.split()]
        if not 0 <= a < 4096 or not 0 <= b < 4096:
            raise ValueError("Invalid NaNCM BPE token ID")
        ranks[a, b] = i + 1
    return ranks


def bpe_segment(prefix: bytes, ranks: dict[tuple[int, int], int]) -> list[int]:
    """Faithful v103 `seg`: lowest-rank adjacent merge, max 48 input bytes."""
    tokens = list(prefix[:48])
    while len(tokens) > 1:
        best, best_rank = -1, 65535
        for j in range(len(tokens) - 1):
            rank = ranks.get((tokens[j], tokens[j + 1]), 0)
            if rank and rank < best_rank:
                best, best_rank = j, rank
        if best < 0:
            break
        tokens[best: best + 2] = [256 + best_rank - 1]
    return tokens


def c_bit_state_machine():
    """Direct transcription of original v103 `build_states` finite automaton.

    Its transition table is mathematically exact; stochastic batched updates
    below are *not* guaranteed to replay the same C slot visitation order.
    """
    def allowed(a, b):
        m, m_big = sorted((a, b))
        limits = (40, 20, 10, 6, 5)
        return m < len(limits) and m_big <= limits[m]

    states = [(0, 0, 0)]
    ids = {(0, 0, 0): 0}
    transitions = []
    index = 0
    while index < len(states):
        a, b, _ = states[index]
        destinations = []
        for bit in (0, 1):
            n0, n1 = a, b
            if bit:
                n1 += 1
                if n0 > 2:
                    n0 = 2 + (n0 - 2) // 3
            else:
                n0 += 1
                if n1 > 2:
                    n1 = 2 + (n1 - 2) // 3
            while not allowed(n0, n1):
                if bit:
                    if n1 >= n0:
                        n1 -= 1
                    else:
                        n0 -= 1
                else:
                    if n0 >= n1:
                        n0 -= 1
                    else:
                        n1 -= 1
            last = bit
            if n0 == 0:
                last = int(n1 > 0)
            if n1 == 0:
                last = 0
            state = (n0, n1, last)
            if state not in ids:
                ids[state] = len(states)
                states.append(state)
            destinations.append(ids[state])
        transitions.append(destinations)
        index += 1
        if len(states) > 255:
            raise AssertionError("Original C bit state count exceeded 255")
    n = np.asarray([(a, b) for a, b, _ in states], dtype=np.float32)
    priors = (n[:, 1] + 0.4) / (n[:, 0] + n[:, 1] + 0.8)
    return np.asarray(transitions, dtype=np.int16), priors.astype(np.float32)


def token_keys(raw: bytes, ranks: dict[tuple[int, int], int]) -> np.ndarray:
    """For each target, encode previous word × current prefix and BPE suffix.

    Word and BPE sidecars are *keys* for trainable causal predictors, not a
    claim of numerical equivalence to the original C word-bigram probability.
    """
    out = np.zeros((len(raw), 2), dtype=np.uint16)
    current_hash, preceding_word, current = 0, 0, bytearray()
    spaced = False
    for pos in range(len(raw)):
        # A prediction at pos has seen raw[:pos], never raw[pos].
        c = raw[pos - 1] if pos else 0
        if pos:
            if word_byte(c):
                if not current:
                    spaced = pos > 1 and raw[pos - 2] == 32
                current_hash = mix32(current_hash + 1, c)
                if len(current) < 48:
                    current.append(c)
            else:
                if current:
                    preceding_word = current_hash
                current_hash, current, spaced = 0, bytearray(), False
        word_hash = mix32(preceding_word, current_hash)
        out[pos, 1] = mix32(word_hash, int(bool(current))) & (WORD_DOMAIN - 1)
        if current:
            prefix = (b" " if spaced else b"") + bytes(current)
            toks = bpe_segment(prefix, ranks)
            tail = toks[-1] if toks else 65535
            prior = toks[-2] if len(toks) > 1 else 65535
            out[pos, 0] = mix32(mix32(tail, prior), min(len(toks), 3)) & (WORD_DOMAIN - 1)
        else:
            out[pos, 0] = mix32(c, 0x77) & (WORD_DOMAIN - 1)
    return out


def _preceding_hashes(raw: bytes, width: int) -> np.ndarray:
    """Hash each width-byte history into u64; index denotes *target* byte."""
    count = len(raw)
    keys = np.zeros(count, dtype=np.uint64)
    if count <= width:
        return keys
    chunk = np.zeros(count - width, dtype=np.uint64)
    data = np.frombuffer(raw, dtype=np.uint8)
    for offset in range(width):
        chunk = chunk * HASH_BASE + data[offset: offset + len(chunk)].astype(np.uint64) + 1
    keys[width:] = chunk
    return keys


class HistoricMatches:
    """Immutable train-only index. Candidate target always precedes query.

    This replays two causal historic-match *lookups* (widths 5,12). It is not
    the C engine's 48 MiB continuing-match automaton. In particular, a DEV
    query never contributes keys to the index or sees another DEV book.
    """

    def __init__(self, train: bytes):
        self.train = np.frombuffer(train, dtype=np.uint8)
        self.sorted = {}
        self.train_keys = {}
        for width in MATCH_LENGTHS:
            started = time.perf_counter()
            print(f"v103 historic index {width}-byte: hashing/sorting {len(train):,} positions...",
                  flush=True)
            keys = _preceding_hashes(train, width)[width:]
            order = np.argsort(keys, kind="stable")
            self.sorted[width] = (keys[order], (order + width).astype(np.int32))
            self.train_keys[width] = keys
            print(f"v103 historic index {width}-byte: ready in "
                  f"{time.perf_counter()-started:.1f}s", flush=True)

    def for_train(self) -> np.ndarray:
        result = np.zeros((len(self.train), 4), dtype=np.uint16)
        for i, width in enumerate(MATCH_LENGTHS):
            keys, targets = self.sorted[width]
            equal = keys[1:] == keys[:-1]
            current = targets[1:][equal]
            previous = targets[:-1][equal]
            # Stable sorting puts earlier original positions before later ones.
            assert np.all(previous < current)
            if current.size:
                # Verify collisions before trusting a historic prediction.
                valid = np.ones(len(current), dtype=bool)
                for offset in range(1, width + 1):
                    valid &= self.train[current - offset] == self.train[previous - offset]
                result[current[valid], 2 * i] = self.train[previous[valid]]
                result[current[valid], 2 * i + 1] = width
        return result

    def for_book(self, book: bytes) -> np.ndarray:
        """Train-source lookup for each DEV target; the book is never indexed."""
        data = np.frombuffer(book, dtype=np.uint8)
        result = np.zeros((len(book), 4), dtype=np.uint16)
        for i, width in enumerate(MATCH_LENGTHS):
            if len(book) <= width:
                continue
            query = _preceding_hashes(book, width)[width:]
            keys, targets = self.sorted[width]
            end = np.searchsorted(keys, query, side="right")
            eligible = end > 0
            qpos = np.nonzero(eligible)[0] + width
            cpos = targets[end[eligible] - 1]
            valid = np.ones(len(cpos), dtype=bool)
            for offset in range(1, width + 1):
                valid &= data[qpos - offset] == self.train[cpos - offset]
            result[qpos[valid], 2 * i] = self.train[cpos[valid]]
            result[qpos[valid], 2 * i + 1] = width
        return result


def build_train_features(raw: bytes, ranks: dict[tuple[int, int], int]):
    """Return [target, six causal features], plus reusable train memory."""
    started = time.perf_counter()
    print(f"v103 features 1/3: compiled C causal BPE and word keys on {len(raw):,} bytes...",
          flush=True)
    # This reference module must remain standalone when embedded in Colab.
    # In notebooks, the builder adds token_keys_fast to globals before main.
    # In the normal package, use the same source-backed C accelerator module.
    fast = globals().get("token_keys_fast")
    if fast is None:
        from terminal_rsi.gpu_experimental.v103_features_fast import token_keys_fast as fast
    tokenizer = fast(raw, ranks)
    print(f"v103 features 1/3: completed in {time.perf_counter()-started:.1f}s",
          flush=True)
    print("v103 features 2/3: build immutable train-only historic index...",
          flush=True)
    memory = HistoricMatches(raw)
    print("v103 features 3/3: materialize verified causal match targets...",
          flush=True)
    links = memory.for_train()
    all_features = np.concatenate((tokenizer, links), axis=1)
    print(f"v103 features: complete in {time.perf_counter()-started:.1f}s",
          flush=True)
    return all_features, memory


def build_book_features(book: bytes, ranks, memory: HistoricMatches):
    fast = globals().get("token_keys_fast")
    if fast is None:
        from terminal_rsi.gpu_experimental.v103_features_fast import token_keys_fast as fast
    return np.concatenate((fast(book, ranks), memory.for_book(book)), axis=1)


def feature_manifest(train: bytes, ranks: dict, features: np.ndarray) -> dict:
    return {"schema": "v103-causal-feature-bridge-v1",
            "train_sha256": hashlib.sha256(train).hexdigest(),
            "merge_count": len(ranks),
            "rows": int(features.shape[0]), "columns": int(features.shape[1]),
            "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "no_test_training": True}


"""Compiled, exact NumPy/ctypes front-end for NaNCM v103 token features.

Colab's Python per-byte hashing/BPE loop can take minutes over 10M bytes.
The original feature semantics are preserved with an O3 C implementation,
an independent Python fixture parity check and a pinned full-train feature
digest. Only observed historical bytes enter either implementation.

No compiler flags request nonportable fast-math or change integer behavior.
The compiled accelerator is *CPU preprocessing*, not a C model prediction.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

_FAST_NATIVE_FN = None
_FAST_NATIVE_LIB = None


def _load_fast_native():
    global _FAST_NATIVE_FN, _FAST_NATIVE_LIB
    if _FAST_NATIVE_FN is not None:
        return _FAST_NATIVE_FN
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("C compiler unavailable; install gcc or enable Colab's standard runtime")
    source = globals().get("V103_FAST_C_SOURCE")
    if source is None:
        source = Path(__file__).with_name("v103_token_keys_fast.c").read_text()
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nancm_v103_keys_") as work:
        c = Path(work) / "nancm_v103_keys.c"
        so = Path(work) / "nancm_v103_keys.so"
        c.write_text(source, encoding="ascii")
        subprocess.run([compiler, "-O3", "-std=c11", "-fPIC", "-shared",
                        "-o", str(so), str(c)], check=True,
                       capture_output=True, text=True)
        _FAST_NATIVE_LIB = ctypes.CDLL(str(so))
    fn = _FAST_NATIVE_LIB.nancm_token_keys
    fn.argtypes = (ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    fn.restype = ctypes.c_int
    _FAST_NATIVE_FN = fn
    print(f"v103 compiled exact feature preprocessor in {time.perf_counter()-start:.2f}s",
          flush=True)
    return fn


def token_keys_fast(raw: bytes, ranks: dict[tuple[int, int], int]) -> np.ndarray:
    """Accelerated bit-for-bit equivalent of v103_features.token_keys."""
    fn = _load_fast_native()
    data = np.frombuffer(raw, dtype=np.uint8)
    triples = np.asarray([(a, b, rank) for (a, b), rank in ranks.items()],
                         dtype=np.uint32).reshape(-1, 3)
    out = np.empty((len(raw), 2), dtype=np.uint16)
    rc = fn(ctypes.c_void_p(data.ctypes.data), len(data),
            ctypes.c_void_p(triples.ctypes.data), len(triples),
            ctypes.c_void_p(out.ctypes.data))
    if rc:
        raise RuntimeError(f"Original-v103 C preprocessing accelerator returned {rc}")
    return out

V103_FAST_C_SOURCE = "/* NaNCM v103 causal GPU-bridge token features, bit-for-bit Python equivalent.\n * This is only CPU preprocessing of observed history, not the C CM model.\n * No target byte is consumed before writing its two feature IDs.\n */\n#include <stdint.h>\n#include <stddef.h>\n#include <stdlib.h>\n#include <string.h>\n\ntypedef uint8_t u8;\ntypedef uint16_t u16;\ntypedef uint32_t u32;\n\nstatic u32 cm_mix(u32 a, u32 b) {\n    u32 x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu;\n    x ^= x >> 16;\n    x *= 0x7FEB352Du;\n    x ^= x >> 15;\n    x *= 0x846CA68Bu;\n    x ^= x >> 16;\n    return x;\n}\n\nstatic int cm_word(u8 c) {\n    return (c >= 65 && c <= 90) || (c >= 97 && c <= 122) ||\n           (c >= 48 && c <= 57) || c >= 128;\n}\n\n/* 'ranks' contains triples (left token, right token, positive BPE rank).\n * 'out' is a preallocated [size,2] little-endian uint16 C-order array.\n * Return 0 on success; negative codes mean invalid inputs / allocation.\n */\nint nancm_token_keys(const u8 *raw, size_t size,\n                     const u32 *ranks, size_t nranks, u16 *out) {\n    if (!raw || !out || (nranks && !ranks)) return -1;\n    /* C original uses pr[4096*4096] = uint16 rank; 32 MiB. */\n    u16 *pr = (u16 *)calloc((size_t)4096 * 4096, sizeof(u16));\n    if (!pr) return -2;\n    for (size_t i = 0; i < nranks; ++i) {\n        u32 a = ranks[3*i], b = ranks[3*i+1], rank = ranks[3*i+2];\n        if (a >= 4096 || b >= 4096 || rank == 0 || rank > 3840) {\n            free(pr);\n            return -3;\n        }\n        pr[(size_t)a*4096 + b] = (u16)rank;\n    }\n    u32 current_hash = 0, preceding_word = 0;\n    u8 current[48];\n    size_t current_len = 0;\n    int spaced = 0;\n    for (size_t pos = 0; pos < size; ++pos) {\n        u8 c = pos ? raw[pos-1] : 0;\n        if (pos) {\n            if (cm_word(c)) {\n                if (!current_len) spaced = (pos > 1 && raw[pos-2] == 32);\n                current_hash = cm_mix(current_hash + 1, c);\n                if (current_len < 48) current[current_len++] = c;\n            } else {\n                if (current_len) preceding_word = current_hash;\n                current_hash = 0;\n                current_len = 0;\n                spaced = 0;\n            }\n        }\n        u32 word_hash = cm_mix(preceding_word, current_hash);\n        out[pos*2+1] = (u16)(cm_mix(word_hash, (u32)(current_len != 0)) & 16383);\n        if (current_len) {\n            u16 tokens[48];\n            int n = 0;\n            if (spaced) tokens[n++] = 32;\n            for (size_t k = 0; k < current_len && n < 48; ++k)\n                tokens[n++] = current[k];\n            while (n > 1) {\n                int best = -1;\n                u16 best_rank = 65535;\n                for (int j = 0; j < n-1; ++j) {\n                    u16 rank = pr[(size_t)tokens[j]*4096 + tokens[j+1]];\n                    if (rank && rank < best_rank) {\n                        best = j;\n                        best_rank = rank;\n                    }\n                }\n                if (best < 0) break;\n                tokens[best] = (u16)(256 + best_rank - 1);\n                memmove(tokens+best+1, tokens+best+2,\n                        (size_t)(n-best-2)*sizeof(u16));\n                --n;\n            }\n            u32 tail = tokens[n-1];\n            u32 prior = n > 1 ? tokens[n-2] : 65535;\n            out[pos*2] = (u16)(cm_mix(cm_mix(tail, prior),\n                                       (u32)(n > 3 ? 3 : n)) & 16383);\n        } else {\n            out[pos*2] = (u16)(cm_mix(c, 0x77) & 16383);\n        }\n    }\n    free(pr);\n    return 0;\n}\n"

"""Terminal-RSI *integrated GPU prototype*, not a numerical port of NaNCM v103.

All trainable byte-context experts, bitwise mixer, recursive critic, trust
calibration, terminal actor, and reward critic live in ONE torch module on
CUDA and share ONE AdamW optimizer/checkpoint. Host subprocess execution and
verification necessarily use the CPU. The original v103 C hash tables and
word/BPE bigram sidecars are NOT converted or loaded by this experimental
backend. The original accepted Terminal-RSI state is never overwritten.
"""

import argparse
import hashlib
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "terminal-rsi-integrated-cuda-prototype-v1"
PARITY_SCHEMA = "terminal-rsi-integrated-cuda-v103-pathways-v2"
HEADS = 32
HASH_BUCKETS = 16384
CONTEXT = 8
PARITY_CONTEXT = 32
PARITY_BUCKETS = 8192
C_TBITS = (18, 24, 24, 24, 24, 24, 24, 22, 24, 24, 22, 24,
           24, 24, 22, 22, 22, 22, 22, 22, 22, 23, 23, 22,
           22, 23, 22, 23, 23, 23, 22, 22)
C_ACTIVE_10M = tuple(
    i < 14 or (14 <= i < 19 and bool((7 >> (i - 14)) & 1)) or
    (21 <= i < HEADS and bool((2047 >> (i - 21)) & 1))
    for i in range(HEADS))
C_10M_DEV_RECORD_SHA256 = "adf21a4490aeec7a1e7f4e204ffffb3a0ede74fca9f50a0fa0ecc73e80167c21"
C_10M_DEV_FULL256_SHA256 = "c1a242b5f4431e2e311ed7e6f7b61df80ee8274d08d88f93698648f84c2cdf4d"
V103_FEATURE_CACHE = {}


def co_prime_stride(n: int, seed: int = 2026) -> int:
    """Choose a deterministic full-cycle permutation multiplier."""
    if n <= 1:
        raise ValueError("Training stream too short for permuted sampling")
    stride = ((seed * 9973 + 1_000_003) % (n - 1)) + 1
    while math.gcd(stride, n) != 1:
        stride = stride % n + 1
    return stride


def coverage_indices(cursor: int, batch: int, n: int, stride: int,
                     device: torch.device) -> torch.Tensor:
    """One visit per position per epoch; no costly multi-MB permutation state.

    For every epoch, affine maps i -> a*i+b (mod n) are bijections because
    gcd(a,n)=1. Persisting cursor and a is sufficient for exact replay.
    """
    if cursor < 0 or batch < 1 or math.gcd(stride, n) != 1:
        raise ValueError("Invalid full-coverage sampler")
    sample = torch.arange(cursor, cursor + batch, device=device, dtype=torch.int64)
    epoch, within = torch.div(sample, n, rounding_mode="floor"), sample % n
    return (within * stride + 104_729 * epoch + 17) % n


def heldout_windows(path: str | None, device: torch.device,
                    *, history: int = CONTEXT):
    """Original NaNCM held-out format, preserving book boundaries and skip."""
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("Truncated independent-book window header")
    count, skip = struct.unpack_from("<II", raw)
    if not 1 <= count <= 10000 or skip < history:
        raise ValueError("Malformed window count or warm-up length")
    offset = 8
    books = []
    for _ in range(count):
        if offset + 4 > len(raw):
            raise ValueError("Truncated held-out length")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if not skip < length <= 1_048_576 or offset + length > len(raw):
            raise ValueError("Truncated held-out book or invalid width")
        book = torch.frombuffer(bytearray(raw[offset:offset + length]),
                                dtype=torch.uint8).to(device)
        # First scored target is at `skip`; its 8 prior bytes are observed.
        books.append(book.unfold(0, history + 1, 1)[skip - history:])
        offset += length
    if offset != len(raw):
        raise ValueError("Unexpected extra data in held-out windows")
    if len({int(x.shape[0]) for x in books}) != 1:
        raise ValueError("A fixed number of positions per book is required")
    return torch.stack(books), hashlib.sha256(raw).hexdigest()


def heldout_auxiliary(path, device, ranks, historic_index):
    """Book-isolated, target-free v103 features, with train-only match keys."""
    if "build_book_features" in globals():
        builder = globals()["build_book_features"]
    else:
        from terminal_rsi.gpu_experimental.v103_features import build_book_features as builder
    raw = Path(path).read_bytes()
    count, skip = struct.unpack_from("<II", raw)
    if skip < PARITY_CONTEXT:
        raise ValueError("Independent books lack 32-byte causal warm-up")
    offset, items = 8, []
    for _ in range(count):
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        book = raw[offset:offset + length]
        offset += length
        items.append(torch.from_numpy(builder(book, ranks, historic_index)[skip:].copy()))
    if offset != len(raw) or len({len(x) for x in items}) != 1:
        raise ValueError("Independent feature alignment mismatch")
    # uint16 advanced indexing is not implemented on CUDA. Keep the exact
    # uint16 feature bytes for hashing/manifest, but represent all six
    # nonnegative feature keys as int32 before moving them to the GPU.
    return torch.stack(items).to(device=device, dtype=torch.int32)


class GPUFullSystem(nn.Module):
    """Thirty-two context channels -> bitwise mixer -> recurrent RSI feedback.

    Each bit is predicted using only prior bytes and earlier bits *within*
    the current byte. Text byte prediction, RSI controls, and command actor
    share the same GPU tensor state. No target future is a model input.
    """

    def __init__(self):
        super().__init__()
        self.context_table = nn.Embedding(HEADS * HASH_BUCKETS, 4)
        self.byte_prefix = nn.Embedding(8 * 256, 16)
        self.bit_position = nn.Embedding(8, 8)
        self.mixer = nn.Sequential(nn.Linear(HEADS * 4 + 24, 128),
                                   nn.Tanh(), nn.Linear(128, 2))
        self.rsi_control = nn.Sequential(nn.Linear(4, 16), nn.Tanh(),
                                         nn.Linear(16, 2))
        self.actor_logits = nn.Parameter(torch.zeros(2))
        self.actor_feedback = nn.Linear(4, 2)
        self.reward_critic = nn.Linear(4, 1)
        self.register_buffer("feedback_state", torch.zeros(4))
        self.register_buffer("verified_events", torch.zeros((), dtype=torch.long))
        nn.init.normal_(self.context_table.weight, std=0.015)
        nn.init.zeros_(self.rsi_control[-1].weight)
        nn.init.zeros_(self.rsi_control[-1].bias)

    def _contexts(self, previous: torch.Tensor) -> torch.Tensor:
        if previous.ndim != 2 or previous.shape[1] != CONTEXT:
            raise ValueError("Expected eight prior bytes per target")
        x = previous.long()
        h = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        suffixes = []
        for i in range(CONTEXT):
            h = ((h * 1_315_423_911) ^ (x[:, 7 - i] + 1)) & 0x7FFFFFFF
            suffixes.append(h)
        suffixes = torch.stack(suffixes, dim=1)
        heads = torch.arange(HEADS, device=x.device, dtype=torch.long)
        # Every address is derived solely from the previous eight bytes.
        addresses = (suffixes[:, heads % CONTEXT] ^
                     (heads * 2_654_435_761)) & (HASH_BUCKETS - 1)
        addresses += heads * HASH_BUCKETS
        return self.context_table(addresses).flatten(1)

    def forward(self, previous: torch.Tensor, target: torch.Tensor):
        """Teacher-forced *causal* bit probabilities; targets only form past-bit prefix."""
        n = previous.size(0)
        target = target.long().flatten()
        if target.numel() != n:
            raise ValueError("One next byte required per history")
        bit = torch.arange(8, device=target.device, dtype=torch.long)
        # bit=0: prefix=0; bit=1: prefix=most significant bit; etc.
        prefix = (target[:, None] >> (8 - bit[None, :]))
        token = bit[None, :] * 256 + prefix
        position = self.bit_position(bit)[None].expand(n, -1, -1)
        context = self._contexts(previous)[:, None].expand(-1, 8, -1)
        features = torch.cat((context, self.byte_prefix(token), position), dim=-1)
        pair = self.mixer(features)
        # RSI gate acts on ALL predictions; its critic updates only AFTER
        # prediction/verified outcomes, never on the current true target.
        control = self.rsi_control(self.feedback_state.detach())
        logits = pair[..., 0] + torch.tanh(control[0]) * pair[..., 1]
        logits = logits * torch.exp(torch.clamp(control[1], -0.25, 0.25))
        bits = ((target[:, None] >> (7 - bit[None, :])) & 1).float()
        return logits, bits

    @torch.no_grad()
    def observe_byte_loss(self, bpb: float, bit_accuracy: float):
        surprise = min(float(bpb) / 8, 4)
        evidence = torch.tensor([surprise, 1 - bit_accuracy,
                                 float(self.feedback_state[0]),
                                 1.], device=self.feedback_state.device)
        self.feedback_state.mul_(0.95).add_(evidence, alpha=0.05)

    @torch.no_grad()
    def observe_terminal(self, verified_success: bool):
        new = torch.tensor([float(not verified_success), float(verified_success),
                            float(self.feedback_state[0]), 1.],
                           device=self.feedback_state.device)
        self.feedback_state.mul_(0.95).add_(new, alpha=0.05)
        self.verified_events.add_(1)


def mix32_tensor(a: torch.Tensor, b: torch.Tensor | int) -> torch.Tensor:
    """Unsigned-u32 v103 `mix` with masked overflow in signed-int64 tensors."""
    mask = 0xFFFFFFFF
    x = (a * 0x9E3779B1) ^ ((b + 0x7F4A7C15) * 0x85EBCA6B)
    x &= mask
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & mask
    x = ((x ^ (x >> 15)) * 0x846CA68B) & mask
    return (x ^ (x >> 16)) & mask


def c_slot_hash_tensor(h: torch.Tensor, tree: torch.Tensor,
                       widths: torch.Tensor | int):
    """Exact C `find` u32 address/checksum, before any GPU table compression."""
    x = ((h ^ (tree * 0x2545F491)) * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    return x >> (32 - widths), (x & 255) | 1


class V103ParityGPUFullSystem(GPUFullSystem):
    """Function-preserving GPU v1 extension with five causal v103 pathways.

    Pathways are GPU tensor approximations of v103's adaptive context state,
    exact-BPE-key/word-bigram specialists, historical matches, 12 mixers and
    five APMs, plus persistable online updates. This is *feature/path parity*,
    not an exact emulation of original C Slot replacement or its LM beam.
    """

    def __init__(self):
        super().__init__()
        # Explicit opt-in; old checkpoints remain predictively identical.
        # The flag is checkpoint configuration, not a learned parameter.
        self.causal_copy_enabled = False
        if "c_bit_state_machine" in globals():
            state_builder = globals()["c_bit_state_machine"]
        else:
            from terminal_rsi.gpu_experimental.v103_features import c_bit_state_machine as state_builder
        transitions, prior = state_builder()
        self.v103_sparse = nn.Embedding(HEADS * PARITY_BUCKETS, 2)
        self.v103_bpe_bigram = nn.Embedding(16384, 2)
        self.v103_word_bigram = nn.Embedding(16384, 2)
        self.v103_twelve_mixers = nn.Linear(39, 12)
        self.v103_second_mix = nn.Linear(12, 1)
        # 5 adaptive probability maps, 256 contexts each, 33 interpolation
        # knots. Exact C additionally has large context domains up to 65536.
        self.v103_apm = nn.Embedding(5 * 256 * 33, 1)
        self.v103_emit = nn.Parameter(torch.zeros(1))
        self.register_buffer("v103_slot_counts",
                             torch.zeros(HEADS * PARITY_BUCKETS, 2))
        self.register_buffer("v103_slot_last_bit",
                             torch.zeros(HEADS * PARITY_BUCKETS,
                                         dtype=torch.uint8))
        self.register_buffer("v103_slot_state",
                             torch.zeros(HEADS * PARITY_BUCKETS, dtype=torch.uint8))
        self.register_buffer("v103_state_next", torch.from_numpy(transitions).long())
        self.register_buffer("v103_state_priors", torch.from_numpy(prior))
        self.register_buffer("v103_original_c_tbits",
                             torch.tensor(C_TBITS, dtype=torch.long) - 2)
        # The frozen 10M C reference used -cm 7 -gc 0 -xc 2047.
        # Its disabled contexts must not influence prediction or state.
        self.register_buffer("v103_c_enabled_contexts",
                             torch.tensor(C_ACTIVE_10M, dtype=torch.bool),
                             persistent=False)
        self.register_buffer("v103_online_updates", torch.zeros((), dtype=torch.long))
        nn.init.zeros_(self.v103_sparse.weight)
        nn.init.zeros_(self.v103_bpe_bigram.weight)
        nn.init.zeros_(self.v103_word_bigram.weight)
        nn.init.zeros_(self.v103_apm.weight)

    @staticmethod
    def _default_aux(previous: torch.Tensor):
        """Live command fallback: only observed context, no historic match."""
        last = previous.long()[:, -8:]
        bpe = ((last[:, -1] * 257 + last[:, -2]) ^ (last[:, -3] * 31)) & 16383
        word = ((last[:, -1] * 131 + last[:, -4] * 257) ^ last[:, -2]) & 16383
        return torch.stack((bpe, word, *([torch.zeros_like(bpe)] * 4)), dim=1)

    def _v103_slots(self, previous: torch.Tensor, target: torch.Tensor,
                    aux: torch.Tensor):
        """C byte-order context hashes and exact per-bit c0 mixing.

        The seven low-order C hashes are exact. Higher semantic addresses
        carry causal BPE/word/match keys but are not C's original 25 hashes.
        """
        v = previous.long()
        c1, c2, c3, c4 = [v[:, -i] for i in range(1, 5)]
        recent = c1 | (c2 << 8) | (c3 << 16) | (c4 << 24)
        older = v[:, -5] | (v[:, -6] << 8) | (v[:, -7] << 16) | (v[:, -8] << 24)
        contexts = [c1 + 1,
                    mix32_tensor(recent & 65535, 2),
                    mix32_tensor(recent & 0xFFFFFF, 3),
                    mix32_tensor(recent, 4),
                    mix32_tensor(recent, mix32_tensor(older & 255, 5)),
                    mix32_tensor(recent, mix32_tensor(older & 65535, 6)),
                    mix32_tensor(recent, mix32_tensor(older, 8))]
        # The exact C high semantic hashes also use incremental word/line,
        # BPE token, memory and LM states unavailable in a 32-byte window.
        for i in range(7, HEADS):
            hint = aux[:, (i - 7) % 6].long()
            contexts.append(mix32_tensor(contexts[(i - 7) % 7],
                                         hint ^ (i * 0x9E3779B1)))
        hashes = torch.stack(contexts, 1)
        bitpos = torch.arange(8, device=v.device)
        tree = (1 << bitpos)[None, :] + (target.long()[:, None] >> (8 - bitpos))
        actual_c_address, _checksum = c_slot_hash_tensor(
            hashes[:, None, :], tree[:, :, None],
            self.v103_original_c_tbits[None, None, :])
        # The full original C address (16–22 bits after -tshift 2) is
        # exactly computed. The smaller GPU state array aliases its low 13
        # bits: collision/state memory is NOT yet C-numerically equivalent.
        address = actual_c_address & (PARITY_BUCKETS - 1)
        heads = torch.arange(HEADS, device=v.device)
        address += heads * PARITY_BUCKETS
        return address, tree

    @staticmethod
    def _causal_copy_logits(logits, bits, target, aux, local=None):
        """Exactly factor a normalized 256-byte base+copy distribution.

        P_mix(byte) = (1-g) P_base(byte) + g [byte==candidate].  The bit
        conditionals use only observed prior bits of the current target, its
        prior 32 bytes, and a candidate from causal preceding history. This
        keeps full trainability: gradient flows through P_base to all old
        neural parameters, without introducing a replacement optimizer.
        """
        n = len(target)
        if local is not None and local.shape != (n, 4):
            raise ValueError("Expected four book-local match fields per target")
        if local is None:
            local = torch.zeros((n, 4), device=target.device, dtype=torch.long)
        candidate = torch.zeros(n, device=target.device, dtype=torch.long)
        width = torch.zeros_like(candidate)
        from_local = torch.zeros(n, device=target.device, dtype=torch.bool)
        for source, is_local, column, size in (
            (aux, False, 2, 5), (local, True, 0, 5),
            (aux, False, 4, 12), (local, True, 2, 12),
        ):
            valid = source[:, column + 1] == size
            candidate = torch.where(valid, source[:, column].long(), candidate)
            width = torch.where(valid, torch.full_like(width, size), width)
            from_local = torch.where(valid, torch.full_like(from_local, is_local),
                                     from_local)
        gate = torch.zeros(n, device=target.device, dtype=logits.dtype)
        for size, is_local, strength in (
            (5, False, 0.06), (12, False, 0.12),
            (5, True, 0.10), (12, True, 0.40),
        ):
            selected = (width == size) & (from_local == is_local)
            gate = torch.where(selected, torch.full_like(gate, strength), gate)
        depth = torch.arange(8, device=target.device)
        prefix_agrees = (target.long()[:, None] >> (8 - depth[None, :])) == (
            candidate[:, None] >> (8 - depth[None, :]))
        candidate_bit = ((candidate[:, None] >> (7 - depth[None, :])) & 1).float()
        # The base conditional probability of the *actual* prefix (at bit k)
        # is computable using previous bits only; current/future bits are
        # never fed into the conditional for their own positions.
        log_base_true = F.logsigmoid((bits * 2 - 1) * logits)
        prior_mass = torch.cat((torch.ones((n, 1), device=target.device,
                                            dtype=logits.dtype),
                                log_base_true[:, :-1].cumsum(-1).exp()), dim=1)
        mass_base = (1.0 - gate[:, None]) * prior_mass
        mass_copy = gate[:, None] * prefix_agrees.float()
        numerator = mass_base * logits.sigmoid() + mass_copy * candidate_bit
        denominator = mass_base + mass_copy
        probability_one = (numerator / denominator.clamp_min(1e-20)).clamp(
            1e-7, 1.0 - 1e-7)
        return torch.logit(probability_one)

    def forward(self, previous: torch.Tensor, target: torch.Tensor,
                aux: torch.Tensor | None = None,
                local: torch.Tensor | None = None):
        if previous.ndim != 2 or previous.shape[1] != PARITY_CONTEXT:
            raise ValueError("v103 GPU expects 32 observed bytes per target")
        old_logits, bits = super().forward(previous[:, -CONTEXT:], target)
        if aux is None:
            aux = self._default_aux(previous)
        if aux.shape != (len(previous), 6):
            raise ValueError("Expected 6 causal NaNCM features per target")
        aux = aux.long()
        addresses, tree = self._v103_slots(previous, target, aux)
        counts = self.v103_slot_counts[addresses]
        slot_logit = torch.log((counts[..., 1] + 1) / (counts[..., 0] + 1))
        seen = (counts.sum(-1) > 0).float()
        last_bit_signal = (self.v103_slot_last_bit[addresses].float() * 2 - 1)
        original_c_state_prob = self.v103_state_priors[self.v103_slot_state[addresses].long()]
        state_logit = torch.logit(original_c_state_prob.clamp(1e-5, 1 - 1e-5))
        sparse = (self.v103_sparse(addresses).sum(-1) + slot_logit +
                  0.05 * seen * last_bit_signal + 0.1 * seen * state_logit)
        sparse = sparse * self.v103_c_enabled_contexts[None, None, :]
        bpe = self.v103_bpe_bigram(aux[:, 0])
        word = self.v103_word_bigram(aux[:, 1])
        # A candidate may contribute only while its known prefix agrees with
        # the already-observed bits of the current target byte.
        match_logits = []
        for source_byte, match_length in ((2, 3), (4, 5)):
            candidate = aux[:, source_byte]
            depth = torch.arange(8, device=target.device)
            candidate_tree = (1 << depth)[None, :] + (candidate[:, None] >> (8 - depth))
            eligible = ((candidate_tree == tree) &
                        (aux[:, match_length, None] > 0))
            candidate_bit = ((candidate[:, None] >> (7 - depth)) & 1)
            signal = (candidate_bit * 2 - 1).float()
            match_logits.append(eligible * signal * (aux[:, match_length, None] / 12))
        # Twelve distinct specialist mixers consume all 32 sparse contexts,
        # the BPE and word bigrams, the two match channels and a bias.
        x = torch.cat((sparse, bpe[:, None, :].expand(-1, 8, -1),
                       word[:, None, :].expand(-1, 8, -1),
                       match_logits[0][..., None], match_logits[1][..., None],
                       torch.ones_like(old_logits)[..., None]), dim=-1)
        mixed = self.v103_second_mix(torch.tanh(self.v103_twelve_mixers(x))).squeeze(-1)
        # Five v103-style probability maps with differentiable 33-knot
        # interpolation. These are trainable GPU APM tables.
        p = ((old_logits.detach().clamp(-7.99, 7.99) + 8) * 2)
        low = p.floor().long().clamp(0, 31)
        weight = p - low
        last_byte = previous[:, -1].long()
        maps = []
        for channel in range(5):
            context = (tree + last_byte[:, None] * (channel + 1)) & 255
            base = (channel * 256 + context) * 33 + low
            bottom = self.v103_apm(base).squeeze(-1)
            upper = self.v103_apm(base + 1).squeeze(-1)
            maps.append(bottom * (1 - weight) + upper * weight)
        new_logits = mixed + torch.stack(maps).mean(0)
        # Function-preserving v1 migration: no initial changes to any logit.
        result = old_logits + torch.tanh(self.v103_emit) * new_logits
        if self.causal_copy_enabled:
            result = self._causal_copy_logits(result, bits, target, aux, local)
        return result, bits

    @torch.no_grad()
    def observe_online_bits(self, previous, target, aux):
        """Persistent sparse per-bit adaptation *after* the optimizer update.

        Counts and last bits are updated from training targets, never during
        evaluation. Unlike C `supd`, same-batch slot collisions aggregate.
        """
        addresses, _ = self._v103_slots(previous, target, aux.long())
        depth = torch.arange(8, device=target.device)
        bits = ((target.long()[:, None] >> (7 - depth)) & 1)
        # Match the *active-path set* of the frozen original C CLI. The
        # remaining state is still compressed, batch-aggregated and different.
        enabled = self.v103_c_enabled_contexts
        addr = addresses[..., enabled].reshape(-1)
        bit = bits[:, :, None].expand(-1, -1, int(enabled.sum())).reshape(-1)
        updates = F.one_hot(bit, 2).float()
        self.v103_slot_counts.index_add_(0, addr, updates)
        # The last event for a colliding slot is selected deterministically
        # by within-batch order. C would apply *all* serial transitions;
        # stochastic GPU batches cannot claim that numerical identity.
        last_event = torch.full((HEADS * PARITY_BUCKETS,), -1,
                                device=addr.device, dtype=torch.long)
        last_event.scatter_reduce_(0, addr,
                                   torch.arange(len(addr), device=addr.device),
                                   reduce="amax", include_self=True)
        touched = torch.nonzero(last_event >= 0).flatten()
        observed = bit[last_event[touched]]
        previous_state = self.v103_slot_state[touched].long()
        self.v103_slot_state[touched] = self.v103_state_next[previous_state, observed].to(torch.uint8)
        self.v103_slot_last_bit[touched] = observed.to(torch.uint8)
        self.v103_online_updates.add_(1)


def choose_data(input_file: str | None) -> bytes:
    if input_file:
        raw = Path(input_file).read_bytes()
        if len(raw) < 1024:
            raise ValueError("Upload at least 1024 bytes of training text/data")
        return raw
    # Offline-only pipeline fixture. Never report this as a real LM result.
    fixture = ("Terminal-RSI checks the outcome, observes the failure, "
               "repairs the command, and updates one recursive critic.\n"
               "NaNCM has causal byte contexts, subword and word memory, "
               "a mixer, trust calibration, and verified feedback.\n")
    return (fixture * 256).encode()


def make_windows(raw: bytes, device: torch.device, *, train_all=False,
                 history: int = CONTEXT):
    # Direct byte-buffer transfer avoids materializing millions of Python ints.
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    # The final contiguous 10% is a diagnostic only, NOT book-disjoint DEV.
    split = int(len(data) * 0.9)
    # With genuinely independent DEV/TEST, all 10M source bytes are eligible
    # for TRAIN (the first/last 8 cannot be target positions in a given book).
    train_source = data if train_all else data[:split]
    train = train_source.unfold(0, history + 1, 1)
    dev = data[split - history:].unfold(0, history + 1, 1)
    return train, dev


def byte_metrics(model, windows, n=256, *, topk_n=64, auxiliary=None,
                 local=None):
    """Exact bit NLL on sampled/all windows and exact 256-way byte top-k.

    Does not compute top-k from eight independent bit guesses. For each
    history, enumerates all 256 causal bit paths and checks their probability
    mass, then measures top-1 and top-5 on stratified histories.
    """
    model.eval()
    with torch.no_grad():
        if windows.ndim == 3:
            books, per, columns = windows.shape
            if columns not in (CONTEXT + 1, PARITY_CONTEXT + 1):
                raise ValueError("Invalid held-out book window shape")
            windows = windows.reshape(books * per, columns)
            if auxiliary is not None:
                auxiliary = auxiliary.reshape(books * per, 6)
            if local is not None:
                local = local.reshape(books * per, 4)
        if n is None:
            n = windows.size(0)
        ix = torch.linspace(0, windows.size(0) - 1,
                            min(n, windows.size(0)),
                            device=windows.device).long()
        batch = windows[ix].long()
        probe_aux = auxiliary[ix].long() if auxiliary is not None else None
        probe_local = local[ix].long() if local is not None else None
        total_nll = 0.0
        bit_hits = 0
        for offset in range(0, len(batch), 512):
            part = batch[offset:offset + 512]
            part_aux = (probe_aux[offset:offset + 512]
                        if probe_aux is not None else None)
            part_local = (probe_local[offset:offset + 512]
                          if probe_local is not None else None)
            if isinstance(model, V103ParityGPUFullSystem):
                logits, bits = model(part[:, :-1], part[:, -1], part_aux,
                                     part_local)
            else:
                logits, bits = model(part[:, :-1], part[:, -1])
            total_nll += float(F.binary_cross_entropy_with_logits(
                logits, bits, reduction="sum"))
            bit_hits += int(((logits > 0) == (bits > 0.5)).sum())
        # Enumerate candidate bytes to get exact top-1 and top-5.
        # This is a small, controlled diagnostic, not train/eval leakage.
        top_ix = torch.linspace(0, batch.shape[0] - 1,
                                min(topk_n, batch.shape[0]),
                                device=batch.device).long()
        probe = batch[top_ix]
        top_aux = probe_aux[top_ix] if probe_aux is not None else None
        top_local = probe_local[top_ix] if probe_local is not None else None
        top1_hits, top5_hits = 0, 0
        largest_mass_error = 0.0
        for offset in range(0, len(probe), 16):
            part = probe[offset:offset + 16]
            vals = torch.arange(256, device=part.device).repeat(part.size(0))
            history = part[:, :-1].repeat_interleave(256, dim=0)
            if isinstance(model, V103ParityGPUFullSystem):
                expanded_aux = (top_aux[offset:offset + 16].repeat_interleave(256, dim=0)
                                if top_aux is not None else None)
                expanded_local = (top_local[offset:offset + 16].repeat_interleave(256, dim=0)
                                  if top_local is not None else None)
                all_logits, all_bits = model(history, vals, expanded_aux,
                                             expanded_local)
            else:
                all_logits, all_bits = model(history, vals)
            logp = -F.binary_cross_entropy_with_logits(
                all_logits, all_bits, reduction="none").sum(-1)
            logp = logp.reshape(part.shape[0], 256)
            mass_error = float(torch.logsumexp(logp, dim=-1).abs().max())
            largest_mass_error = max(largest_mass_error, mass_error)
            top5 = logp.topk(5, dim=-1).indices
            # The v103 extension consumes 32 observed bytes, whereas the
            # inherited v1 model consumes 8.  The label is always the final
            # column, regardless of history width.  Indexing CONTEXT here
            # silently scored a history byte as the target in v103 mode.
            targets = part[:, -1]
            top1_hits += int((top5[:, 0] == targets).sum())
            top5_hits += int((top5 == targets[:, None]).any(1).sum())
        if largest_mass_error > 2e-3:
            raise AssertionError("256-byte tree probability mass is not normalized")
        return {"bpb": float(total_nll / (len(batch) * math.log(2))),
                "bit_accuracy": bit_hits / (len(batch) * 8),
                "top1_byte": top1_hits / len(probe),
                "top5_byte": top5_hits / len(probe),
                "maximum_log_probability_mass_error": largest_mass_error,
                "evaluated_bytes": int(batch.size(0)),
                "topk_evaluated_bytes": int(probe.size(0))}


def compare_frozen_v103(model, independent_books, reference_path,
                        measured_metrics=None, *, auxiliary=None, local=None):
    """Matched NaNCM 10M frozen DEV proper-score gate, never TEST tuning.

    The reference holds actual-target negative log likelihoods from the
    *original* C model. It does NOT hold all 256 probabilities, so this gate
    checks predictive BPB and per-book gains, not numerical model identity.
    """
    records = Path(reference_path).read_bytes()
    if hashlib.sha256(records).hexdigest() != C_10M_DEV_RECORD_SHA256:
        raise ValueError("Original v103 frozen 10M DEV record digest mismatch")
    books, per, cols = independent_books.shape
    if cols not in (CONTEXT + 1, PARITY_CONTEXT + 1) or len(records) != books * per * 8:
        raise ValueError("NaNCM reference does not align with DEV book targets")
    c_rows = list(struct.iter_unpack("<fBB2x", records))
    c_nats = torch.tensor([r[0] for r in c_rows],
                          dtype=torch.float64).reshape(books, per)
    if not torch.isfinite(c_nats).all() or (c_nats < 0).any():
        raise ValueError("Corrupt frozen C likelihood reference")
    model.eval()
    gpu_nats = []
    flat = independent_books.reshape(-1, cols)
    flat_aux = auxiliary.reshape(-1, 6) if auxiliary is not None else None
    flat_local = local.reshape(-1, 4) if local is not None else None
    with torch.no_grad():
        for offset in range(0, len(flat), 512):
            data = flat[offset:offset + 512].long()
            part_aux = (flat_aux[offset:offset + 512].long()
                        if flat_aux is not None else None)
            part_local = (flat_local[offset:offset + 512].long()
                          if flat_local is not None else None)
            if isinstance(model, V103ParityGPUFullSystem):
                logits, bits = model(data[:, :-1], data[:, -1], part_aux,
                                     part_local)
            else:
                logits, bits = model(data[:, :-1], data[:, -1])
            nll = F.binary_cross_entropy_with_logits(
                logits, bits, reduction="none").sum(-1)
            gpu_nats.append(nll.cpu().double())
    gpu_nats = torch.cat(gpu_nats).reshape(books, per)
    delta = (gpu_nats - c_nats) / math.log(2)
    accuracy = {}
    if measured_metrics is not None:
        size = int(measured_metrics["topk_evaluated_bytes"])
        sample = torch.linspace(0, books * per - 1, size).long().tolist()
        c_top1 = sum(c_rows[i][1] for i in sample) / size
        c_top5 = sum(c_rows[i][2] for i in sample) / size
        accuracy = {
            "aligned_c_top1": c_top1, "aligned_c_top5": c_top5,
            "aligned_gpu_top1": measured_metrics["top1_byte"],
            "aligned_gpu_top5": measured_metrics["top5_byte"],
            "topk_probe_bytes": size,
        }
    bpb_gate = float(delta.mean()) <= 0
    accuracy_gate = (not accuracy or
                     (accuracy["aligned_gpu_top1"] >= accuracy["aligned_c_top1"] and
                      accuracy["aligned_gpu_top5"] >= accuracy["aligned_c_top5"]))
    # Positive delta = GPU worse. No multiple-comparison claims.
    return {"protocol": "matched-v103-frozen-10m-book-disjoint-DEV",
            "reference_sha256": hashlib.sha256(records).hexdigest(),
            "books": books, "scored_bytes": books * per,
            "c_frozen_bpb": float(c_nats.mean() / math.log(2)),
            "gpu_bpb": float(gpu_nats.mean() / math.log(2)),
            "gpu_minus_c_bpb": float(delta.mean()),
            "gpu_better_books": int((delta.mean(-1) < 0).sum()),
            "per_book_gpu_minus_c_bpb": delta.mean(-1).tolist(),
            "bpb_parity_gate": bpb_gate,
            "quality_parity_gate": bool(bpb_gate and accuracy_gate),
            **accuracy,
            "numeric_parity_proven": False}


def compare_c_full256(model, independent_books, reference_path,
                      *, auxiliary=None, local=None, probe_count=128):
    """Exact C-vs-GPU 256-way distribution gate on aligned DEV histories.

    Unlike historical `Rec` values, this catches different distributions
    even when the correct target has identical NLL. It does not update the
    model, its online state, or the independent-book reference.
    """
    books, per, cols = independent_books.shape
    if cols not in (CONTEXT + 1, PARITY_CONTEXT + 1):
        raise ValueError("Invalid CM distribution DEV history shape")
    raw = Path(reference_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != C_10M_DEV_FULL256_SHA256:
        raise ValueError("Original 10M C full256 numerical oracle digest changed")
    if len(raw) != books * per * 256 * 4:
        raise ValueError("Original C full-distribution records are misaligned")
    original = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(-1, 256)
    if not torch.isfinite(original).all() or (original <= 0).any():
        raise ValueError("Invalid NaNCM C frozen 256-way probability mass")
    sample = torch.linspace(0, books * per - 1,
                            min(probe_count, books * per)).long()
    windows = independent_books.reshape(-1, cols)
    aux = auxiliary.reshape(-1, 6) if auxiliary is not None else None
    local_flat = local.reshape(-1, 4) if local is not None else None
    original = original[sample].to(windows.device)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample), 8):
            indices = sample[start:start + 8].to(windows.device)
            part = windows[indices].long()
            batch = part[:, :-1].repeat_interleave(256, 0)
            candidates = torch.arange(256, device=batch.device).repeat(len(part))
            features = (aux[indices].long().repeat_interleave(256, 0)
                        if aux is not None else None)
            book_matches = (local_flat[indices].long().repeat_interleave(256, 0)
                            if local_flat is not None else None)
            if isinstance(model, V103ParityGPUFullSystem):
                logits, target_bits = model(batch, candidates, features,
                                            book_matches)
            else:
                logits, target_bits = model(batch, candidates)
            logp = -F.binary_cross_entropy_with_logits(
                logits, target_bits, reduction="none").sum(-1)
            predictions.append(logp.reshape(-1, 256).exp())
    gpu = torch.cat(predictions)
    absolute = (gpu - original).abs()
    # Numerical comparison is separate from quality on each target. C's
    # float32 and Torch's reduction order need tight-but-not-bitwise tolerance.
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    c_probs = original / original.sum(-1, keepdim=True)
    gpu_probs = gpu / gpu.sum(-1, keepdim=True)
    mean_kl = float((c_probs * (c_probs.log() - gpu_probs.log())).sum(-1).mean())
    return {"protocol": "C-v103-full256-matched-frozen-DEV",
            "c256_sha256": hashlib.sha256(raw).hexdigest(),
            "probe_histories": int(len(sample)), "candidates_per_history": 256,
            "max_abs_probability_delta": max_abs,
            "mean_abs_probability_delta": mean_abs,
            "mean_kl_c_to_gpu_nats": mean_kl,
            "exact_distribution_parity": bool(max_abs <= 1e-6 and mean_kl <= 1e-7)}


@torch.no_grad()
def command_bpb(model, observed: bytes, candidate: tuple[str, ...]):
    # True likelihood of candidate's text, conditioned on prior observations.
    text = " ".join(candidate).encode("utf-8")
    context = PARITY_CONTEXT if isinstance(model, V103ParityGPUFullSystem) else CONTEXT
    payload = (b"\0" * context + observed[-2048:] + text)
    start = context + min(2048, len(observed))
    seq = torch.tensor(list(payload), dtype=torch.uint8,
                       device=model.feedback_state.device)
    windows = seq.unfold(0, context + 1, 1).long()
    sample = windows[start - context:]
    if sample.shape[0] != len(text):
        raise AssertionError("Command byte alignment mismatch")
    logits, bits = model(sample[:, :-1], sample[:, -1])
    loss = F.binary_cross_entropy_with_logits(logits, bits, reduction="sum")
    return float(loss / (len(text) * math.log(2)))


def terminal_feedback(model, optimizer):
    """Two real, fixed, shell-free commands; verifier drives GPU actor/critic."""
    model.train()
    choices = ((sys.executable, "-c", "print('broken')"),
               (sys.executable, "-c", "print('ready')"))
    goal = b"Execute command and verify exact stdout: ready\n"
    bpb = [command_bpb(model, goal, c) for c in choices]
    scores = torch.tensor(bpb, device=model.feedback_state.device)
    scores = (scores.mean() - scores) / scores.std().clamp_min(0.5)
    events = []
    # Forced exploratory failure, then recovery. This does not establish
    # learned autonomous task selection or superiority over baselines.
    for choice in (0, 1):
        prior = model.feedback_state.detach().clone()
        logits = model.actor_logits + model.actor_feedback(prior) + 0.25 * scores
        logprobs = logits.log_softmax(0)
        subprocess_result = subprocess.run(
            choices[choice], capture_output=True, text=True, shell=False,
            timeout=5, check=False)
        verified = (subprocess_result.returncode == 0 and
                    subprocess_result.stdout == "ready\n")
        reward = torch.tensor(1.0 if verified else -1.0,
                              device=model.feedback_state.device)
        value = model.reward_critic(prior).squeeze(-1)
        advantage = (reward - value).detach()
        actor_loss = -logprobs[choice] * advantage
        critic_loss = 0.5 * (value - reward).square()
        optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        optimizer.step()
        model.observe_terminal(verified)
        events.append({"choice": choice, "verified_success": verified,
                       "command_bpb": bpb[choice],
                       "returncode": subprocess_result.returncode,
                       "gpu_policy_update": True})
    return events


def save_all(path, model, optimizer, steps, seen, events, source_digest,
             *, sampler, evaluations, holdout_digests, feature_manifest=None):
    # ONE atomic checkpoint with every trainable weight and optimizer moment.
    temporary = Path(str(path) + ".tmp")
    state = {"schema": (PARITY_SCHEMA if isinstance(model, V103ParityGPUFullSystem)
                        else SCHEMA), "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "steps": int(steps),
             "train_bytes_sampled": int(seen), "events": list(events),
             "source_sha256": source_digest,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate(),
             "sampler": dict(sampler), "evaluations": list(evaluations),
             "holdout_digests": dict(holdout_digests),
             "v103_feature_manifest": feature_manifest,
             "configuration": {"heads": HEADS, "hash_buckets": HASH_BUCKETS,
                               "byte_history": (PARITY_CONTEXT if isinstance(model, V103ParityGPUFullSystem)
                                                else CONTEXT),
                               "causal_copy_enabled": bool(getattr(model, 'causal_copy_enabled', False))}}
    torch.save(state, temporary)
    temporary.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None,
                   help="Previously uploaded .txt/.bin train stream")
    p.add_argument("--resume", default=None,
                   help="Previously downloaded GPU prototype checkpoint")
    p.add_argument("--output", default="terminal_rsi_gpu_full_smoke.pt")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--holdout-dev", default=None,
                   help="NaNCM book-disjoint win_dev.bin (never used in training)")
    p.add_argument("--holdout-test", default=None,
                   help="NaNCM book-disjoint win_test.bin (once-only final gate)")
    p.add_argument("--reference-dev", default=None,
                   help="Original C 10M frozen DEV Rec stream, matched by target")
    p.add_argument("--reference-full256", default=None,
                   help="Original C complete 256-byte DEV distribution stream")
    p.add_argument("--eval-test", action="store_true",
                   help="Evaluate untouched independent TEST once after 5 minutes")
    p.add_argument("--checkpoint-every", type=float, default=60.0)
    p.add_argument("--allow-cpu-test", action="store_true")
    p.add_argument("--v103-pathways", action="store_true",
                   help="Extend existing GPU-v1 checkpoint with all five v103 pathways")
    p.add_argument("--causal-copy", action="store_true",
                   help="Train the same GPU model with normalized causal 5/12-byte copy probabilities")
    p.add_argument("--bpe-merges", default=None,
                   help="Original NaNCM v103 BPE rank text, required for v103 mode")
    args = p.parse_args(argv)
    if args.causal_copy and not args.v103_pathways:
        p.error("--causal-copy needs --v103-pathways and the same original checkpoint")
    if args.seconds < 0 or args.batch < 4 or args.checkpoint_every < 0:
        p.error("Need nonnegative seconds and batch >= 4")
    if not torch.cuda.is_available() and not args.allow_cpu_test:
        raise RuntimeError("No CUDA GPU found: Colab Runtime > Change runtime type > GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    corpus = choose_data(args.corpus)
    source_digest = hashlib.sha256(corpus).hexdigest()
    history = PARITY_CONTEXT if args.v103_pathways else CONTEXT
    independent_dev = heldout_windows(args.holdout_dev, device, history=history)
    independent_test = heldout_windows(args.holdout_test, device, history=history)
    if args.reference_dev and independent_dev is None:
        p.error("--reference-dev requires --holdout-dev")
    if args.reference_full256 and independent_dev is None:
        p.error("--reference-full256 requires --holdout-dev")
    if args.eval_test and independent_test is None:
        p.error("--eval-test requires --holdout-test")
    train, dev = make_windows(corpus, device, train_all=independent_dev is not None,
                              history=history)
    eval_dev = independent_dev[0] if independent_dev is not None else dev
    train_aux = eval_aux = test_aux = None
    eval_local = test_local = None
    v103_feature_manifest = None
    if args.v103_pathways:
        if not args.bpe_merges:
            p.error("--v103-pathways requires the original --bpe-merges file")
        if "build_train_features" in globals():
            feature_builder = globals()["build_train_features"]
            manifest_builder = globals()["feature_manifest"]
            rank_reader = globals()["read_bpe_ranks"]
        else:
            from terminal_rsi.gpu_experimental import v103_features as source_features
            feature_builder = source_features.build_train_features
            manifest_builder = source_features.feature_manifest
            rank_reader = source_features.read_bpe_ranks
        ranks = rank_reader(args.bpe_merges)
        if not ranks:
            p.error("Refusing empty BPE merge table in v103 parity mode")
        merge_digest = hashlib.sha256(Path(args.bpe_merges).read_bytes()).hexdigest()
        cache_key = (source_digest, merge_digest)
        if cache_key not in V103_FEATURE_CACHE:
            print("Preparing exact causal BPE prefixes + 5/12-byte historic-match index...",
                  flush=True)
            train_features, match_index = feature_builder(corpus, ranks)
            v103_feature_manifest = manifest_builder(corpus, ranks, train_features)
            v103_feature_manifest["bpe_merge_sha256"] = merge_digest
            V103_FEATURE_CACHE.clear()
            V103_FEATURE_CACHE[cache_key] = (train_features, match_index,
                                              v103_feature_manifest)
        train_features, match_index, v103_feature_manifest = V103_FEATURE_CACHE[cache_key]
        limit = len(corpus) if independent_dev is not None else int(len(corpus) * 0.9)
        train_aux = torch.from_numpy(train_features[history:limit].copy()).to(
            device=device, dtype=torch.int32)
        if independent_dev is not None:
            eval_aux = heldout_auxiliary(args.holdout_dev, device, ranks, match_index)
        else:
            eval_aux = torch.from_numpy(train_features[int(len(corpus) * 0.9):].copy()).to(
                device=device, dtype=torch.int32)
        if independent_test is not None and args.eval_test:
            test_aux = heldout_auxiliary(args.holdout_test, device, ranks, match_index)
        if args.causal_copy:
            # Independently held-out book histories are observed as they
            # arrive, but never written to the train index or optimizer.
            if 'book_local_features' in globals():
                builder = globals()['book_local_features']
            else:
                from terminal_rsi.gpu_experimental.v103_local_copy import book_local_features as builder
            if independent_dev is not None:
                local_dev, digest = builder(args.holdout_dev)
                if digest != independent_dev[1]:
                    raise AssertionError('Copy DEV history alignment changed')
                eval_local = torch.from_numpy(local_dev).to(device=device,
                                                           dtype=torch.int32)
            if independent_test is not None and args.eval_test:
                local_test, digest = builder(args.holdout_test,
                                             expected_sha256=None)
                if digest != independent_test[1]:
                    raise AssertionError('Copy TEST history alignment changed')
                test_local = torch.from_numpy(local_test).to(device=device,
                                                            dtype=torch.int32)
        if train_aux.shape != (len(train), 6) or eval_aux.shape[:1] != eval_dev.shape[:1]:
            raise AssertionError("Training/DEV feature-to-target offset mismatch")
        if train_aux.dtype != torch.int32 or eval_aux.dtype != torch.int32 or (
                test_aux is not None and test_aux.dtype != torch.int32):
            raise AssertionError("Causal features must be CUDA-indexable int32")
        print("V103 causal feature SHA:", v103_feature_manifest["feature_sha256"])
    dev_label = ("book-disjoint NaNROUTER DEV" if independent_dev is not None
                 else "INTERNAL TRAIN-STREAM TAIL (not independent)")
    holdout_digests = {
        "dev": independent_dev[1] if independent_dev is not None else None,
        "test": independent_test[1] if independent_test is not None else None,
    }
    if args.reference_dev:
        holdout_digests["c_dev_reference"] = hashlib.sha256(
            Path(args.reference_dev).read_bytes()).hexdigest()
    if args.reference_full256:
        holdout_digests["c_full256_reference"] = hashlib.sha256(
            Path(args.reference_full256).read_bytes()).hexdigest()
    model = (V103ParityGPUFullSystem() if args.v103_pathways else GPUFullSystem()).to(device)
    if args.causal_copy:
        model.causal_copy_enabled = True
    if args.v103_pathways:
        base_parameters = [p for name, p in model.named_parameters()
                           if not name.startswith("v103_")]
        added_parameters = [p for name, p in model.named_parameters()
                            if name.startswith("v103_")]
        optimizer = torch.optim.AdamW([{"params": base_parameters, "lr": 2e-3},
                                       {"params": added_parameters, "lr": 5e-4}])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    step = 0
    seen = 0
    events = []
    evaluations = []
    sampler = {"schema": "full-cycle-affine-v1", "cursor": 0,
               "stride": co_prime_stride(train.size(0), args.seed),
               "train_positions": train.size(0)}
    migrated_from_v1 = False
    if args.resume:
        loaded = torch.load(args.resume, map_location="cpu", weights_only=True)
        schema = loaded.get("schema")
        if schema not in (SCHEMA, PARITY_SCHEMA):
            raise ValueError("Wrong checkpoint lineage; original C donor is not compatible")
        if schema == PARITY_SCHEMA and not args.v103_pathways:
            raise ValueError("v103 checkpoint needs --v103-pathways; cannot remove modules")
        original_copy = bool(loaded.get('configuration', {}).get('causal_copy_enabled', False))
        if original_copy and not args.causal_copy:
            raise ValueError('Cannot disable the checkpointed causal-copy forward path')
        if args.causal_copy and not original_copy:
            # An inherited `.pt` is a protected parent, not a file to
            # overwrite while adding a new predictive function.
            if Path(args.output).resolve() == Path(args.resume).resolve():
                raise ValueError('Write the causal-copy experiment to a NEW .pt; preserve its parent')
            print('Checkpoint-preserving new causal-copy function; original model and Adam unchanged')
        if loaded["source_sha256"] != source_digest:
            raise ValueError("Train stream changed; refuse silent optimizer continuation")
        if schema == SCHEMA and args.v103_pathways:
            migrated_from_v1 = True
            legacy = GPUFullSystem().to(device)
            legacy.load_state_dict(loaded["model"])
            # Load *original* named tensors verbatim, leave only new modules
            # at function-preserving initialization (v103_emit=0).
            missing, unexpected = model.load_state_dict(loaded["model"], strict=False)
            if unexpected or any(not x.startswith("v103_") for x in missing):
                raise AssertionError("Unsafe migration: legacy parameter mismatch")
            old_optim = torch.optim.AdamW(base_parameters, lr=2e-3)
            old_optim.load_state_dict(loaded["optimizer"])
            old_optim.add_param_group({"params": added_parameters, "lr": 5e-4})
            optimizer = old_optim
            anchor = train[:16].long()
            model.eval(); legacy.eval()
            with torch.no_grad():
                baseline_logits = legacy(anchor[:, -CONTEXT - 1:-1], anchor[:, -1])[0]
                # Confirm the old neural forward EXACTLY before intentionally
                # enabling the new causal-copy output distribution.
                copy_flag = model.causal_copy_enabled
                model.causal_copy_enabled = False
                migrated_logits = model(anchor[:, :-1], anchor[:, -1], train_aux[:16])[0]
                model.causal_copy_enabled = copy_flag
            if not torch.equal(baseline_logits, migrated_logits):
                raise AssertionError("v1 -> v103 migration altered pretraining logits")
            print("EXACT v1-to-v103 initial-logit identity: PASS")
        else:
            model.load_state_dict(loaded["model"])
            optimizer.load_state_dict(loaded["optimizer"])
            if schema == PARITY_SCHEMA and loaded.get("v103_feature_manifest") != v103_feature_manifest:
                raise ValueError("v103 causal features changed: cannot replay faithfully")
        # Optimizer moments are migrated back to the CUDA device as required.
        for state in optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device)
        step, seen = int(loaded["steps"]), int(loaded["train_bytes_sampled"])
        events = list(loaded["events"])
        if args.causal_copy and not original_copy:
            events.append({'kind': 'causal-copy-v1-function-migration',
                           'parent_checkpoint': Path(args.resume).name,
                           'preserved_optimizer': True})
        evaluations = list(loaded.get("evaluations", []))
        original_digests = loaded.get("holdout_digests", {})
        for split in ("dev", "test", "c_dev_reference", "c_full256_reference"):
            if (original_digests.get(split) is not None and
                    original_digests[split] != holdout_digests[split]):
                raise ValueError(
                    f"Original checkpoint's {split} benchmark identity must be preserved: "
                    f"expected SHA-256 {original_digests[split]}, "
                    f"found {holdout_digests[split]}. Run the notebook's 'Restore "
                    "SHA-pinned original book-disjoint reference windows' cell; "
                    "do not replace the old checkpoint's split silently.")
        if "sampler" in loaded and not (schema == SCHEMA and args.v103_pathways):
            sampler = dict(loaded["sampler"])
            if (sampler["schema"] != "full-cycle-affine-v1" or
                    sampler["train_positions"] != train.size(0) or
                    sampler["cursor"] < 0 or
                    math.gcd(sampler["stride"], train.size(0)) != 1):
                raise ValueError("Cannot resume a changed coverage sampler")
        else:
            old_sampler = loaded.get("sampler")
            if old_sampler:
                events.append({"kind": "v103-history-width-migration",
                               "original_sampler": dict(old_sampler),
                               "new_sampler": dict(sampler)})
            print("Preserved all original GPU weights/Adam/RNG; new 32-byte "
                  "history requires a new exact-coverage position range")
        torch.set_rng_state(loaded["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(loaded["cuda_rng"])
        random.setstate(loaded["python_rng"])
    if migrated_from_v1:
        # Save the complete function-preserving migration immediately, before
        # long independent-book DEV evaluation or timed training. A user who
        # stops Colab after this point still has a recoverable v103 `.pt`.
        # Never overwrite the accepted v1 donor or an existing candidate.
        destination = Path(args.output).resolve()
        if destination == Path(args.resume).resolve():
            raise ValueError("v103 migration must not overwrite original GPU-v1 checkpoint")
        if destination.exists():
            raise FileExistsError("Refusing to replace an existing v103 candidate: " +
                                  str(destination))
        save_all(destination, model, optimizer, step, seen, events,
                 source_digest, sampler=sampler, evaluations=evaluations,
                 holdout_digests=holdout_digests,
                 feature_manifest=v103_feature_manifest)
        print("RECOVERABLE v103 checkpoint saved BEFORE evaluation/training:",
              destination, flush=True)
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU TEST")
    print("Training corpus bytes:", len(corpus), "SHA-256:", source_digest[:16])
    print("Resume step:", step, "same model + AdamW state")
    print("Evaluation:", dev_label)
    dev_n = None if independent_dev is not None else 256
    before = byte_metrics(model, eval_dev, dev_n, topk_n=128,
                          auxiliary=eval_aux, local=eval_local)
    print("Baseline diagnostic:", before)
    if args.reference_dev:
        parity_before = compare_frozen_v103(
            model, independent_dev[0], args.reference_dev, before,
            auxiliary=eval_aux, local=eval_local)
        print("Matched C-vs-GPU DEV before:", parity_before)
    if args.reference_full256:
        full_before = compare_c_full256(
            model, independent_dev[0], args.reference_full256,
            auxiliary=eval_aux, local=eval_local, probe_count=64)
        print("Matched C-vs-GPU full256 before:", full_before)
    started = time.monotonic()
    model.train()
    local_updates = 0
    last_checkpoint = started
    start_cursor = sampler["cursor"]
    while time.monotonic() - started < args.seconds:
        positions = coverage_indices(sampler["cursor"], args.batch,
                                     train.size(0), sampler["stride"], device)
        batch = train[positions].long()
        if args.v103_pathways:
            auxiliary = train_aux[positions]
            logits, bits = model(batch[:, :-1], batch[:, -1], auxiliary)
        else:
            logits, bits = model(batch[:, :-1], batch[:, -1])
        loss = F.binary_cross_entropy_with_logits(logits, bits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.v103_pathways:
            model.observe_online_bits(batch[:, :-1], batch[:, -1], auxiliary)
        step += 1
        local_updates += 1
        seen += args.batch
        sampler["cursor"] += args.batch
        model.observe_byte_loss(float(8 * loss.detach() / math.log(2)),
                                float(((logits.detach() > 0) == (bits > 0.5)).float().mean()))
        if (args.checkpoint_every and time.monotonic() - last_checkpoint >=
                args.checkpoint_every):
            save_all(Path(args.output), model, optimizer, step, seen, events,
                     source_digest, sampler=sampler, evaluations=evaluations,
                     holdout_digests=holdout_digests,
                     feature_manifest=v103_feature_manifest)
            print("Protected continuation at optimizer step", step,
                  "permuted coverage cursor", sampler["cursor"], flush=True)
            last_checkpoint = time.monotonic()
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.monotonic() - started
    # Two verified terminal policy updates alter the shared RSI feedback
    # state. Score DEV/TEST only *after* them so the final displayed result
    # describes the model that is actually saved and resumed next round.
    terminal_events = terminal_feedback(model, optimizer)
    print("Terminal verifier outcomes:", terminal_events)
    events.extend(terminal_events)
    events.append({"kind": "gpu-training-round", "updates": local_updates})
    after = byte_metrics(model, eval_dev, dev_n, topk_n=512,
                         auxiliary=eval_aux, local=eval_local)
    print("Final diagnostic:", after)
    evaluations.append({"dev_label": dev_label, "step": step, "before": before,
                        "after": after, "round_seconds": duration,
                        "sampled_positions_this_round": sampler["cursor"] - start_cursor})
    if args.reference_dev:
        parity_after = compare_frozen_v103(
            model, independent_dev[0], args.reference_dev, after,
            auxiliary=eval_aux, local=eval_local)
        print("Matched C-vs-GPU DEV after:", parity_after)
        evaluations.append({"step": step, "split": "dev_c_parity",
                            "before": parity_before, "after": parity_after})
    if args.reference_full256:
        full_after = compare_c_full256(
            model, independent_dev[0], args.reference_full256,
            auxiliary=eval_aux, local=eval_local, probe_count=128)
        print("Matched C-vs-GPU full256 after:", full_after)
        evaluations.append({"step": step, "split": "dev_c_full256",
                            "before": full_before, "after": full_after})
    if args.eval_test:
        if independent_test is None:
            raise ValueError("--eval-test requires independent --holdout-test")
        test_metrics = byte_metrics(model, independent_test[0], None, topk_n=512,
                                    auxiliary=test_aux, local=test_local)
        print("INDEPENDENT TEST (final-only):", test_metrics)
        evaluations.append({"split": "test", "step": step,
                            "result": test_metrics})
    save_all(Path(args.output), model, optimizer, step, seen, events,
             source_digest, sampler=sampler, evaluations=evaluations,
             holdout_digests=holdout_digests,
             feature_manifest=v103_feature_manifest)
    # Verify serialized weights, optimizer state and logits are recoverable.
    check = torch.load(args.output, map_location="cpu", weights_only=True)
    assert check["steps"] == step and check["optimizer"]["state"]
    anchor = eval_dev.reshape(-1, history + 1)[:16].long()
    anchor_aux = eval_aux.reshape(-1, 6)[:16] if eval_aux is not None else None
    anchor_local = eval_local.reshape(-1, 4)[:16] if eval_local is not None else None
    model.eval()
    with torch.no_grad():
        before_reload = (model(anchor[:, :-1], anchor[:, -1], anchor_aux,
                               anchor_local)[0]
                         if args.v103_pathways else model(anchor[:, :-1], anchor[:, -1])[0])
    exact_reload = (V103ParityGPUFullSystem() if args.v103_pathways
                    else GPUFullSystem()).to(device)
    exact_reload.load_state_dict(check["model"])
    if args.v103_pathways:
        exact_reload.causal_copy_enabled = bool(check['configuration'].get('causal_copy_enabled', False))
    exact_reload.eval()
    with torch.no_grad():
        after_reload = (exact_reload(anchor[:, :-1], anchor[:, -1], anchor_aux,
                                     anchor_local)[0]
                        if args.v103_pathways else exact_reload(anchor[:, :-1], anchor[:, -1])[0])
    if not torch.equal(before_reload, after_reload):
        raise AssertionError("GPU checkpoint round-trip changed bit logits")
    for name, param in model.state_dict().items():
        if not torch.equal(param.cpu(), check["model"][name].cpu()):
            raise AssertionError("Checkpoint mutated model state: " + name)
    if not torch.equal(check["model"]["verified_events"], model.verified_events.cpu()):
        raise AssertionError("Terminal verification history was not saved")
    print("Saved full-system GPU prototype:", args.output,
          "updates:", step,
          "total target predictions (legacy+coverage):", seen)
    print("EXACT coverage cursor:", sampler["cursor"],
          "positions in training split:", train.size(0),
          "full epochs:", sampler["cursor"] // train.size(0),
          "round target predictions:", sampler["cursor"] - start_cursor)
    print("Measured training seconds:", round(duration, 3),
          "GPU target positions/second:", round((sampler["cursor"] - start_cursor)
                                               / max(duration, 1e-9)))
    print("Verified exact GPU checkpoint prediction round-trip: PASS")
    print("NOT an exact CUDA port of original NaNCM C model; not a 10M-weight conversion.")
    return check


