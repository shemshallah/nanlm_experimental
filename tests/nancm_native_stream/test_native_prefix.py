import itertools,math,hashlib,numpy as np
from native_prefix import PrefixMemo,folded_rank,STOP,ALPHA
class Words:
    d={'a':.31,'an':.23,'at':.19,'be':.15,'z':.12}
    def mass(self,p): return math.fsum(v for w,v in self.d.items() if w.startswith(p))
    def complete(self,p): return self.d.get(p,0.)
def raw(h):
    z=np.frombuffer(hashlib.shake_256(h).digest(512),dtype='<u2').astype(np.float64)+1
    return z/z.sum()
def main():
    memo=PrefixMemo(raw,4);keys=[b'a',b'A',b'a ',b'a',b'a',b'a!',b'a?',b'A',b'root a',b'a']
    for h in keys:
        q=memo(h);assert np.array_equal(q,raw(h)) and not q.flags.writeable
    assert memo.stats()['hits']==2 and memo.stats()['misses']==8 and memo.stats()['resident']==4
    memo.clear();assert memo.stats()['resident']==0
    root=b'Full history ';w=.95;word=Words();oracle=[]
    for n in (1,2):
        for letters in itertools.product(ALPHA,repeat=n):
            p=0.
            for variant in itertools.product(*[(b,b-32) for b in letters]):
                history=root;m=1.
                for b in variant:m*=raw(history)[b];history+=bytes([b])
                p+=m*float(raw(history)[STOP].sum())
            text=bytes(letters).decode('ascii');oracle.append(((1-w)*p+w*word.complete(text),text))
    oracle=sorted(oracle,reverse=True)[:5];memo=PrefixMemo(raw,256)
    result=folded_rank(word,memo,root,w,cap=3,budget=128)
    assert result['certified'] and [x[0] for x in result['ranked']]==[x[1] for x in oracle]
    err=max(abs(x[1]-y[0]) for x,y in zip(result['ranked'],oracle));assert err<1e-15
    limited=folded_rank(word,memo,root,w,cap=3,budget=1)
    assert not limited['certified'] and limited['queries']==1 and limited['unresolved_bound']>=oracle[0][0]
    zero=folded_rank(word,memo,root,w,cap=3,budget=0);assert not zero['certified'] and zero['queries']==0 and zero['unresolved_bound']==1.
    print(dict(exhaustive_words=702,rank_error=err,certified=result,limited=limited,cache=memo.stats()))
if __name__=='__main__':main()
