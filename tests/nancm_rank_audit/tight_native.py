"""Native exact case-history rank with per-word lexical upper bounds."""
import heapq,math
import numpy as np
from native_prefix import ALPHA,STOP
from word_bound import CompletionBound

def seed_words(bound,k,cap):
    if bound.kind=='lift':
        ix=np.argsort(-bound.p,kind='stable')[:max(4*k,32)]
        return {bound.words[i] for i in ix if 0<len(bound.words[i])<cap}
    if bound.kind=='blend':return seed_words(bound.parent,k,cap)|seed_words(bound.child,k,cap)
    return set()

def rank_native(word,raw,root,weight,k=5,cap=64,budget=128):
    assert 0<=weight<1 and k>0 and cap>=2 and budget>=0
    lexical=CompletionBound(word)
    seeds=sorted((weight*word.complete(w) for w in seed_words(lexical,k,cap)),reverse=True)
    # Lower bound on kth full score, not an assertion of a winning word.
    lower=max(0.,seeds[k-1]-1e-15) if len(seeds)>=k else 0.
    heap=[(-1.,b'',((b'',1.),))];best=[];queries=expanded=0
    while heap:
        threshold=max(lower,best[0][0] if len(best)>=k else 0.)
        if -heap[0][0]<threshold:break
        _,prefix,variants=heap[0]
        if queries+len(variants)>budget:break
        heapq.heappop(heap);expanded+=1;tables=[]
        for variant,m in variants:
            q=raw(root+variant);queries+=1;tables.append((variant,m,q))
        if prefix:
            p=(1-weight)*math.fsum(m*float(q[STOP].sum()) for _,m,q in tables)+weight*word.complete(prefix.decode('ascii'))
            item=(p,prefix)
            if len(best)<k:heapq.heappush(best,item)
            elif item>best[0]:heapq.heapreplace(best,item)
        if len(prefix)>=cap-1:continue
        threshold=max(lower,best[0][0] if len(best)>=k else 0.)
        for b in ALPHA:
            child=prefix+bytes([b]);vv=tuple((v+bytes([c]),m*float(q[c])) for v,m,q in tables for c in (b,b-32))
            bound=(1-weight)*math.fsum(m for _,m in vv)+weight*lexical(child.decode('ascii'))
            bound=math.nextafter(bound,math.inf)+1e-15
            if bound>=threshold:heapq.heappush(heap,(-bound,child,vv))
    unresolved=max(lower,-heap[0][0] if heap else 0.)
    cert=len(best)>=k and unresolved<best[0][0]
    # 'lower' is not itself a remaining candidate; if it equals a winning
    # score due to rounding, the strict comparison conservatively withholds.
    return dict(ranked=[(w.decode('ascii'),p) for p,w in sorted(best,reverse=True)],certified=bool(cert),queries=queries,expanded=expanded,unresolved_bound=unresolved,lexical_kth_lower=lower)
