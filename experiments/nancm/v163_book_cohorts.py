#!/usr/bin/env python3
"""NaNCM v163: disjoint memory-fit / gate-fit cohorts within TRAIN.
Book-disjoint DEV and TEST stay untouched. No learned cloud tensors modified.
"""
import collections,hashlib,json,sys
from pathlib import Path
from v160_book_gate import load_books,partition,digest_file,train,train_backoff,export_native_windows

def write_index(path, models):
    keys={}
    for m in models:
        for k,v in m.items(): keys[k]=v
    path.write_text("".join(
        k.split(":",1)[0]+"\t"+k.split(":",1)[1]+"\t"+v[0]+"\t"+str(v[1])+"\t"+str(v[2])+"\n"
        for k,v in sorted(keys.items())))
    return len(keys)

def run(archive,checksum,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lines=Path(checksum).read_text().splitlines()
    values=[l.split()[0] for l in lines if len(l.split())>=2 and Path(l.split()[-1].lstrip("*")).name==Path(archive).name]
    assert len(values)==1 and values[0].lower()==digest_file(archive).lower()
    books,dup=load_books(archive)
    alltrain=books["train"]
    # Explicit separate cohorts. No calibration book contributes to the candidate index.
    order=sorted(alltrain,key=lambda p:hashlib.sha256(("v163-cohort\0"+p[0]).encode()).digest())
    memory_books=order[:400]
    calibration_books=order[400:]
    assert len(memory_books)==400 and len(calibration_books)==100
    core_n=write_index(out/"v163_core_index.tsv",[train(memory_books),train_backoff(memory_books)])
    deploy_n=write_index(out/"v163_deploy_index.tsv",[train(alltrain),train_backoff(alltrain)])
    calibration_manifest=[]
    for j in range(3):
        # Three non-overlapping portions of each calibration book.
        # Each 384-byte excerpt includes a 48-byte pure-context warmup.
        from v160_book_gate import export_native_windows
        # Original exporter takes 1 window per book; vary seed by pass.
        win=out/f"v163_calibration_{j}.bin"
        rows=export_native_windows(calibration_books,win,f"calibration-v163-{j}",100)
        calibration_manifest.append(rows)
    (out/"COHORTS.json").write_text(json.dumps({
       "memory_books":[x[0] for x in memory_books],
       "calibration_books":[x[0] for x in calibration_books],
       "DEV":[x[0] for x in books["dev"]],
       "TEST":[x[0] for x in books["test"]],
       "calibration_windows":calibration_manifest},indent=2))
    print(json.dumps({
        "source_sha256":digest_file(archive),"exact_dupes_removed":dup,
        "cohort_books":{"memory":len(memory_books),"calibration":len(calibration_books),
                        "DEV":len(books["dev"]),"TEST":len(books["test"])},
        "memory_index_entries":core_n,"deploy_index_entries":deploy_n,
        "core_index_sha256":digest_file(out/"v163_core_index.tsv"),
        "deploy_index_sha256":digest_file(out/"v163_deploy_index.tsv"),
        "calibration_windows":sum(len(a) for a in calibration_manifest),
        "train_target_overlap_with_core_index_books":False,
        "full_native_model_evaluated":False},indent=2))
if __name__=="__main__":
    assert len(sys.argv)==4
    run(*sys.argv[1:])
