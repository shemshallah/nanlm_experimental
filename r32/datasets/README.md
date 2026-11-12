# r32 datasets

## Corpus source: NaNROUTER `corpora/domains/`

The new domain-diverse corpus is published as 16 public GitHub releases. Each
release contains one UTF-8 `.txt` (documents separated by blank lines), a
`SOURCES.json`, a `MANIFEST.json`, and `SHA256SUMS`:

- `corpora-wikipedia-en`, `corpora-arxiv`, `corpora-pubmed`, `corpora-law`,
  `corpora-patents`, `corpora-code`, `corpora-math`, `corpora-news`,
  `corpora-recipes`, `corpora-security`, `corpora-qa`,
  `corpora-multilingual`, `corpora-finance`, `corpora-standards`,
  `corpora-humanities`, `corpora-chemistry`
- Index release: `corpora-domains-latest`

Full assets are 0.05–1.5 GB. `code/prepare_domains.py` samples 4 MB per domain
with HTTP range requests; it does not download the whole releases.

## Prebuilt Kaggle datasets (private)

- Data: https://www.kaggle.com/datasets/shem42/r32-multidomain-16domains
- Code: https://www.kaggle.com/datasets/shem42/r32-code

Data layout (flat, read-only mount):

```
train.bin           # uint16 bytes+EOS, domain tokens added per window by the sampler
valid.bin
meta.json           # vocab 273, doc lengths, train/valid domain ids, hashes
split_manifest.json # per-domain counts, release URLs, sampling parameters
```

## Rebuild

```sh
python r32/code/prepare_domains.py \
  --out-dir r32/data --cache-dir r32/cache \
  --per-domain-mb 4 --chunks 4 --valid-fraction 0.10 --seed 2026
```

The builder filters documents below 200 bytes, truncates over 32 KB, drops
exact duplicates, and splits at document boundaries. It records source URLs
and per-domain counts in `split_manifest.json`.

## Licenses

The releases mix licenses: Wikipedia/Stack Exchange CC BY-SA 4.0, arXiv/PubMed
and several others CC0, IETF Trust for standards, Apache 2.0 for patents, etc.
Check each release's `SOURCES.json` before redistributing text. This repo
commits only code and derived metadata, not the corpus text.
