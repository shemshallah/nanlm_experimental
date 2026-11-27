"""Exact word-shingle overlap audit on evaluated prefixes, not a full corpus deduplication."""
import hashlib
from collections import defaultdict,Counter

def shingles(words,n):
    for i in range(len(words)-n+1):
        yield i,tuple(words[i:i+n])

def audit(reference,cohorts,load,n=16):
    # Only store queried cohort shingles; tuple equality confirms exact text.
    wanted=defaultdict(list);lengths={};hits=defaultdict(set);pairs=Counter()
    for group,files in cohorts.items():
        for bid,p in files:
            key=(group,bid);doc=load(p,1200);lengths[key]=len(doc)
            for pos,sh in shingles(doc,n):wanted[sh].append((key,pos))
    def mark(key,doc):
        for pos,sh in shingles(doc,n):
            for other,j in wanted.get(sh,()):
                if key==other:continue
                hits[(other,key[0])].add(j)
                pairs[(other,key)]+=1
    for bid,p,cap in reference:mark(('reference',bid),load(p,cap))
    for group,files in cohorts.items():
        for bid,p in files:mark((group,bid),load(p,1200))
    result={}
    for group,files in cohorts.items():
        out={}
        for source in ('reference',*cohorts):
            pos=sum(len(hits.get(((group,bid),source),())) for bid,p in files)
            denom=sum(max(0,lengths[(group,bid)]-n+1) for bid,p in files)
            out[source]={'matching_positions':pos,'total_positions':denom,'fraction':pos/max(1,denom)}
        result[group]=out
    top=[{'query':list(a),'donor':list(b),'shared_shingles':v} for (a,b),v in pairs.most_common(30)]
    return {'shingle_words':n,'scope':'reference training prefixes and first 1200 words of evaluation documents','cohorts':result,'top_pairs':top}
