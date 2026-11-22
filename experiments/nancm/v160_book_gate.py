#!/usr/bin/env python3
"""NaNCM v160 book-disjoint association gate.
Independent book split; this is NOT an evaluation of inherited native v158.
"""
import collections, hashlib, json, os, re, sys, tarfile, struct
from pathlib import Path
import zstandard

def digest_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def tokens(text):
    return [x.decode("ascii") for x in re.findall(rb"[a-z]+",text.lower())]

def partition(name):
    key=("NaNCM-BOOK-SPLIT-20261121-v160\0"+name).encode()
    x=int(hashlib.sha256(key).hexdigest()[:12],16)%100
    return "train" if x<80 else ("dev" if x<90 else "test")

def load_books(archive):
    buckets={p:[] for p in ("train","dev","test")}
    seen=set(); duplicates=0
    with open(archive,"rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as zr:
        with tarfile.open(fileobj=zr,mode="r|") as tar:
            for t in tar:
                if not t.isfile() or not t.name.endswith(".txt"):continue
                if t.name.startswith("/") or ".." in Path(t.name).parts:raise ValueError("unsafe archive path")
                data=tar.extractfile(t).read()
                text_hash=hashlib.sha256(re.sub(rb"\s+",b" ",data.lower()).strip()).hexdigest()
                if text_hash in seen:
                    duplicates+=1;continue
                seen.add(text_hash)
                buckets[partition(t.name)].append((t.name,data))
    return buckets,duplicates

def train(books):
    counts=collections.defaultdict(collections.Counter)
    for name,data in books:
        ws=tokens(data)[:3000]
        for i in range(4,len(ws)):
            w=ws[i]
            if len(w)>64:continue
            for k in (3,4):counts[(k," ".join(ws[i-k:i]))][w]+=1
    index={}
    for (k,key),c in counts.items():
        total=sum(c.values());w,n=c.most_common(1)[0]
        if n>=3 and n/total>=0.70:
            index[f"{k}:{key}"]=(w,n,total)
    return index

def evaluate(books,index):
    n=active=correct=0
    for name,data in books:
        ws=tokens(data)[:2500]
        for i in range(16,len(ws)):
            n+=1
            found=None
            for k in (4,3):
                key=f"{k}:"+ " ".join(ws[i-k:i])
                if key in index:
                    found=index[key][0];break
            if found is not None:
                active+=1;correct+=int(found==ws[i])
    return {"words":n,"route_active":active,"route_correct":correct,
            "route_coverage_pct":round(100*active/max(1,n),4),
            "route_precision_pct":round(100*correct/max(1,active),4),
            "correct_per_total_pct":round(100*correct/max(1,n),4)}



def train_backoff(books):
    # Distinct low-order conditional evidence: k=1 and k=2.
    counts=collections.defaultdict(collections.Counter)
    for name,data in books:
        ws=tokens(data)[:3000]
        for i in range(2,len(ws)):
            if len(ws[i])>64:continue
            for k in (1,2):
                counts[(k," ".join(ws[i-k:i]))][ws[i]]+=1
    idx={}
    for (k,key),c in counts.items():
        total=sum(c.values());word,n=c.most_common(1)[0]
        if n >= (4 if k==1 else 2) and n/total >= (.08 if k==1 else .2):
            idx[f"{k}:{key}"]=(word,n,total)
    return idx

def eval_backoff(books,idx):
    # This is only the backoff route's next-word accuracy, not native fusion.
    n=act=correct=0;ord_hits=collections.Counter()
    for name,data in books:
        ws=tokens(data)[:2500]
        for i in range(16,len(ws)):
            n+=1
            for k in (2,1):
                x=idx.get(f"{k}:"+" ".join(ws[i-k:i]))
                if x:
                    act+=1;correct+=int(x[0]==ws[i]);ord_hits[str(k)]+=1;break
    return {"words":n,"active":act,"correct":correct,
       "coverage_pct":round(100*act/max(1,n),4),
       "precision_pct":round(100*correct/max(1,act),4),
       "total_top1_contribution_pct":round(100*correct/max(1,n),4),
       "order_activations":dict(ord_hits)}

def export_native_windows(books,path,seed,count=24):
    # One 384-byte window per independent book; native inference receives
    # only preceding bytes when scoring a target.
    candidates=[(name,raw) for name,raw in books if len(raw)>=512]
    candidates.sort(key=lambda x:hashlib.sha256((seed+x[0]).encode()).hexdigest())
    selected=candidates[:count]
    out=bytearray(struct.pack("<II",len(selected),48))
    manifest=[]
    for name,raw in selected:
        size=384;bound=len(raw)-size
        offset=int(hashlib.sha256((seed+name+"WINDOW").encode()).hexdigest()[:12],16)%(bound+1)
        window=raw[offset:offset+size]
        out.extend(struct.pack("<I",size));out.extend(window)
        manifest.append({"name":name,"offset":offset,"sha256":hashlib.sha256(window).hexdigest()})
    path.write_bytes(out)
    return manifest

def main():
    archive=Path(sys.argv[1]); sha=Path(sys.argv[2]); out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True)
    expected=[]
    for line in sha.read_text().splitlines():
        items=line.split()
        if len(items)>=2 and Path(items[-1].lstrip("*")).name==archive.name:
            expected.append(items[0])
    assert len(expected)==1,"Missing/ambiguous official release checksum"
    actual=digest_file(archive)
    assert actual.lower()==expected[0].lower(),"Release checksum mismatch"
    books,duplicates=load_books(archive)
    assert min(len(x) for x in books.values())>0,"Empty book split"
    index=train(books["train"])
    p=out/"v160_associations.tsv"
    p.write_text("".join(
        k.split(":",1)[0]+"\t"+k.split(":",1)[1]+"\t"+v[0]+"\t"+str(v[1])+"\t"+str(v[2])+"\n"
        for k,v in sorted(index.items())
    ))
    book_unigram=collections.Counter(w for name,data in books["train"] for w in tokens(data)[:3000])
    (out/"book_unigrams.tsv").write_text("".join(w+"\t"+str(n)+"\n" for w,n in sorted(book_unigram.items())))
    backoff=train_backoff(books["train"])
    (out/"v161_backoff.tsv").write_text("".join(k.split(":",1)[0]+"\t"+k.split(":",1)[1]+"\t"+v[0]+"\t"+str(v[1])+"\t"+str(v[2])+"\n" for k,v in sorted(backoff.items())))
    backoff_dev=eval_backoff(books["dev"],backoff)
    dev_windows=export_native_windows(books["dev"],out/"heldout_dev160.bin","dev",24)
    test_windows=export_native_windows(books["test"],out/"heldout_test160.bin","test",24)
    report={
        "source":"NaNROUTER short-stories batch 0004 GitHub Release",
        "archive":archive.name,"sha256":actual,"split_seed":"NaNCM-BOOK-SPLIT-20261121-v160",
        "split":"SHA256(seed+NUL+book path) modulo 100, TRAIN 0-79 DEV 80-89 TEST 90-99",
        "books":{k:len(v) for k,v in books.items()},"exact_duplicates_removed":duplicates,
        "associations":len(index),"index_sha256":digest_file(p),
        "DEV":evaluate(books["dev"],index),"TEST":evaluate(books["test"],index),
        "book_unigram_types":len(book_unigram),"book_unigram_tokens":sum(book_unigram.values()),
        "v161_backoff_entries":len(backoff),"v161_backoff_DEV":backoff_dev,"v161_backoff_TEST":"NOT YET EVALUATED — select on DEV first",
        "native_dev_windows":len(dev_windows),"native_test_windows":len(test_windows),
        "independent_books":True,
        "near_duplicate_screened":False,
        "full_native_v158_compared":False,
        "byte_BPB_measured":False,
        "result_type":"corpus-only route diagnostic; not a model promotion"
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2))
    (out/"WINDOW_MANIFEST.json").write_text(json.dumps({"DEV":dev_windows,"TEST":test_windows},indent=2))
    (out/"SPLITS.json").write_text(json.dumps({k:[name for name,_ in v] for k,v in books.items()},indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
