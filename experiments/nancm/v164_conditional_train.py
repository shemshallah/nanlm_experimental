#!/usr/bin/env python3
"""v164: train full conditional candidate distributions from NaNROUTER TRAIN books.
DEV evaluates candidate coverage only; TEST untouched until native parameters fixed.
"""
import collections,hashlib,json,sys
from pathlib import Path
from v160_book_gate import digest_file,load_books,tokens
def build(books):
    orders={1:collections.defaultdict(collections.Counter),
            2:collections.defaultdict(collections.Counter),
            3:collections.defaultdict(collections.Counter)}
    for name,raw in books:
        ws=tokens(raw)[:3000]
        for i in range(3,len(ws)):
            word=ws[i]
            if len(word)>20:continue
            for k in (1,2,3):
                ctx=' '.join(ws[i-k:i])
                orders[k][ctx][word]+=1
    return orders
def export(path,orders):
    keys=rows=0
    with path.open('w') as f:
        for k in (1,2,3):
            total_floor={1:12,2:3,3:2}[k]
            for ctx,c in sorted(orders[k].items()):
                total=sum(c.values())
                if total<total_floor:continue
                for word,count in sorted(c.items(),key=lambda x:(-x[1],x[0]))[:8]:
                    f.write(f"{k}\t{ctx}\t{word}\t{count}\t{total}\n")
                    rows+=1
                keys+=1
    return keys,rows
def diagnostic(books,orders):
    stats={k:{'active':0,'top1':0,'top5':0} for k in [1,2,3]}
    positions=0
    for name,raw in books:
        ws=tokens(raw)[:2500]
        for i in range(16,len(ws)):
            positions+=1
            for k in [1,2,3]:
                q=orders[k].get(' '.join(ws[i-k:i]))
                if q and sum(q.values())>={1:12,2:3,3:2}[k]:
                    words=[w for w,n in sorted(q.items(),key=lambda x:(-x[1],x[0]))[:8]]
                    stats[k]['active']+=1
                    stats[k]['top1']+=int(ws[i]==words[0])
                    stats[k]['top5']+=int(ws[i] in words[:5])
    for k in stats:
        stats[k]['coverage_pct']=round(100*stats[k]['active']/positions,3)
        stats[k]['whole_word_top1_pct']=round(100*stats[k]['top1']/positions,3)
        stats[k]['whole_word_top5_pct']=round(100*stats[k]['top5']/positions,3)
    return {'positions':positions,'order':stats}
def main(archive,checksums,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    digest=digest_file(archive)
    official=[r.split()[0] for r in Path(checksums).read_text().splitlines() if len(r.split())>=2 and Path(r.split()[-1].lstrip('*')).name==Path(archive).name]
    assert len(official)==1 and official[0].lower()==digest.lower()
    books,dups=load_books(archive)
    model=build(books['train'])
    outfile=out/'v164_conditional_top8.tsv'
    nkeys,nrows=export(outfile,model)
    report={'source_sha256':digest,'books':{k:len(v) for k,v in books.items()},
            'exact_duplicate_books_removed':dups,'trained_only_on':'500 TRAIN books',
            'context_orders':[1,2,3],'max_candidates_per_context':8,
            'keys':nkeys,'candidate_rows':nrows,'index_sha256':digest_file(outfile),
            'DEV_diagnostic':diagnostic(books['dev'],model),
            'TEST_evaluated':False,'full_native_model_evaluated':False,
            'leakage_note':'full inherited v158 training-corpus provenance remains uncertified'}
    (out/'REPORT164.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main(*sys.argv[1:])
