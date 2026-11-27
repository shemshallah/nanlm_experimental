import itertools,math,hashlib,numpy as np
from tight_native import rank_native
from native_prefix import folded_rank,ALPHA,STOP,PrefixMemo
class Outside:
    def mass(self,p):return 0.
class Words:
    def __init__(self,seed):
        self.words=tuple(sorted(''.join(w) for n in (1,2) for w in itertools.product('abcdefgh',repeat=n)))
        p=np.random.default_rng(seed).lognormal(0,1,len(self.words));self.p=p/p.sum();self.unknown=0.;self.outside=Outside();self.d=dict(zip(self.words,self.p))
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))
    def complete(self,p):return self.d.get(p,0.)
def raw(h):
    z=np.frombuffer(hashlib.shake_256(h).digest(512),dtype='<u2').astype(np.float64)+1
    return z/z.sum()
def main():
    root=b'Full history ';w=.95;old_queries=new_queries=0;err=0.;checks=0
    for seed in range(4):
        word=Words(seed);oracle=[]
        for n in (1,2):
            for letters in itertools.product(ALPHA,repeat=n):
                p=0.
                for variant in itertools.product(*[(b,b-32) for b in letters]):
                    history=root;m=1.
                    for b in variant:m*=raw(history)[b];history+=bytes([b])
                    p+=m*float(raw(history)[STOP].sum())
                text=bytes(letters).decode('ascii');oracle.append(((1-w)*p+w*word.complete(text),text));checks+=1
        oracle=sorted(oracle,reverse=True)[:5]
        old=folded_rank(word,PrefixMemo(raw,2048),root,w,cap=3,budget=4096)
        new=rank_native(word,PrefixMemo(raw,2048),root,w,cap=3,budget=4096)
        assert old['certified'] and new['certified'] and [x[0] for x in new['ranked']]==[x[1] for x in oracle]
        err=max(err,max(abs(x[1]-y[0]) for x,y in zip(new['ranked'],oracle)))
        old_queries+=old['queries'];new_queries+=new['queries']
        for budget in (0,1,5,16,64):
            result=rank_native(word,PrefixMemo(raw,2048),root,w,cap=3,budget=budget);assert result['queries']<=budget
            if result['certified']:assert [x[0] for x in result['ranked']]==[x[1] for x in oracle]
    assert err<1e-15 and new_queries<old_queries
    print(dict(exhaustive_completions=checks,distributions=4,old_queries=old_queries,new_queries=new_queries,max_probability_error=err,budget_checks=20))
if __name__=='__main__':main()
