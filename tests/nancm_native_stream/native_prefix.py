"""Bounded exact memoization and certified folded ranking for full-history donors."""
from collections import OrderedDict
import heapq, math
import numpy as np

class PrefixMemo:
    def __init__(self, donor, capacity=2048):
        assert capacity > 0
        self.donor=donor; self.capacity=capacity; self.cache=OrderedDict()
        self.hits=0; self.misses=0; self.evictions=0
    def __call__(self, history):
        key=bytes(history)
        if key in self.cache:
            self.hits+=1; self.cache.move_to_end(key); return self.cache[key]
        self.misses+=1
        q=np.asarray(self.donor(key), dtype=np.float64).copy()
        assert q.shape==(256,) and np.isfinite(q).all() and q.min()>0 and abs(q.sum()-1)<1e-9
        q.setflags(write=False); self.cache[key]=q
        if len(self.cache)>self.capacity:
            self.cache.popitem(last=False); self.evictions+=1
        return q
    def clear(self): self.cache.clear()
    def stats(self):
        return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,resident=len(self.cache),capacity=self.capacity)

ALPHA=tuple(range(97,123))
STOP=np.ones(256,bool); STOP[list(ALPHA)+[b-32 for b in ALPHA]]=False

def folded_rank(word,raw,root,weight,k=5,cap=64,budget=128):
    """Exact completed-word scores, excluding cap events. All remaining bounds retained.

    Count logical queries, not cache misses. Expand a folded node only if every
    literal casing can be evaluated. No approximate merging of native histories.
    """
    assert 0<=weight<1 and k>0 and cap>=2 and budget>=0
    heap=[(-1.,b'',((b'',1.),))]; best=[]; queries=0; expanded=0
    while heap:
        if len(best)>=k and -heap[0][0]<best[0][0]: break
        _,prefix,variants=heap[0]
        if queries+len(variants)>budget: break
        heapq.heappop(heap); expanded+=1; tables=[]
        for variant,m in variants:
            q=raw(root+variant); queries+=1; tables.append((variant,m,q))
        if prefix:
            p=(1-weight)*math.fsum(m*float(q[STOP].sum()) for _,m,q in tables)+weight*word.complete(prefix.decode('ascii'))
            item=(p,prefix)
            if len(best)<k: heapq.heappush(best,item)
            elif item>best[0]: heapq.heapreplace(best,item)
        if len(prefix)>=cap-1: continue
        for b in ALPHA:
            child=prefix+bytes([b])
            variants_child=tuple((v+bytes([c]),m*float(q[c])) for v,m,q in tables for c in (b,b-32))
            bound=(1-weight)*math.fsum(m for _,m in variants_child)+weight*word.mass(child.decode('ascii'))
            if bound>0 and (len(best)<k or bound>=best[0][0]): heapq.heappush(heap,(-bound,child,variants_child))
    unresolved=-heap[0][0] if heap else 0.
    cert=not heap or (len(best)>=k and unresolved<best[0][0])
    return dict(ranked=[(w.decode('ascii'),p) for p,w in sorted(best,reverse=True)],certified=bool(cert),queries=queries,expanded=expanded,unresolved_bound=unresolved)
