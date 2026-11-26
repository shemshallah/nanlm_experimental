"""v14: BPE trigram hypothesis marginalization feeding v13 finite PoE word law.
No gradient updates. BPE learned only from actual reference words."""
import base as B, foundation6 as N, hybrid13 as H
import collections as C, functools, heapq, math

class Pieces:
    def __init__(self,docs,merges=192):
        freq=C.Counter(w for doc in docs for w in doc)
        self.tokens=list(B.A);self.eos=len(self.tokens);self.tokens.append("")
        seq={w:tuple(ord(c)-97 for c in w) for w in freq}
        self.rules={}
        for _ in range(merges):
            counts=C.Counter()
            for w,t in seq.items():
                for pair in zip(t,t[1:]):counts[pair]+=freq[w]
            if not counts:break
            pair=max(counts,key=lambda k:(counts[k],k))
            if counts[pair]<2:break
            v=len(self.tokens);self.tokens.append(self.tokens[pair[0]]+self.tokens[pair[1]])
            self.rules[pair]=v
            for w,t in seq.items():
                if len(t)<2:continue
                out=[];i=0
                while i<len(t):
                    if i+1<len(t) and (t[i],t[i+1])==pair:out.append(v);i+=2
                    else:out.append(t[i]);i+=1
                seq[w]=tuple(out)
        self.bi=C.defaultdict(C.Counter);self.tri=C.defaultdict(C.Counter);u=C.Counter()
        for doc in docs:
            a=b=self.eos
            for w in doc:
                for t in (*seq[w],self.eos):
                    self.bi[b][t]+=1;self.tri[a,b][t]+=1;u[t]+=1;a,b=b,t
        V=len(self.tokens);total=sum(u.values())+.5*V
        self.uni=[(u[i]+.5)/total for i in range(V)]
        self.byfirst=C.defaultdict(list)
        for i,t in enumerate(self.tokens):
            if t:self.byfirst[t[0]].append((i,t))
    @functools.lru_cache(maxsize=32768)
    def encode(self,w):
        t=[ord(c)-97 for c in w]
        while len(t)>1:
            options=[(self.rules[t[i],t[i+1]],i) for i in range(len(t)-1) if (t[i],t[i+1]) in self.rules]
            if not options:break
            v,i=min(options);t[i:i+2]=[v]
        return tuple(t)
    @functools.lru_cache(maxsize=32768)
    def row(self,a,b):
        p=list(self.uni)
        for c in (self.bi.get(b),self.tri.get((a,b))):
            if c:
                n=sum(c.values());z=.75*len(c)/n
                p=[v*z for v in p]
                for i,v in c.items():p[i]+=(v-.75)/n
        return tuple(p)
    def start(self,prev):
        return ((self.encode(prev) or (self.eos,))[-1],self.eos)
    @functools.lru_cache(maxsize=65536)
    def probability(self,prev,w):
        if not w:return 0.
        states=[C.defaultdict(float) for _ in range(len(w)+1)]
        a,b=self.start(prev);states[0][a,b]=1.
        for i in range(len(w)):
            for (a,b),mass in states[i].items():
                row=self.row(a,b);z=1-row[self.eos] if i==0 else 1.
                for t,piece in self.byfirst[w[i]]:
                    if w.startswith(piece,i):states[i+len(piece)][b,t]+=mass*row[t]/z
        return math.fsum(mass*self.row(a,b)[self.eos] for (a,b),mass in states[-1].items())
    @functools.lru_cache(maxsize=8192)
    def propose(self,prev):
        a,b=self.start(prev);beam=[(1.,"",a,b)];done=C.Counter()
        # Bounded proposal beam only. Candidate probability is subsequently
        # recomputed by exact segmentation marginal, not the pruned beam.
        for depth in range(5):
            pool=[]
            for p,w,a,b in beam:
                row=self.row(a,b);z=1-row[self.eos] if not w else 1.
                if w:done[w]+=p*row[self.eos]
                for t in heapq.nlargest(10,range(len(row)),key=lambda t:row[t]):
                    if t==self.eos:continue
                    v=w+self.tokens[t]
                    if len(v)<=24:pool.append((p*row[t]/z,v,b,t))
            beam=heapq.nlargest(24,pool)
        return tuple(heapq.nlargest(12,done,key=lambda w:(done[w],w)))
    def check(self):
        # exact DP vs explicit recursive enumeration on small spellings
        for prev in ("","the"):
            for w in ("a","the","and","there"):
                def brute(a,b,prefix):
                    row=self.row(a,b)
                    if prefix==w:return row[self.eos]
                    z=1-row[self.eos] if not prefix else 1.
                    return math.fsum(row[t]/z*brute(b,t,prefix+piece) for t,piece in self.byfirst[w[len(prefix)]] if w.startswith(prefix+piece))
                a,b=self.start(prev)
                assert math.isclose(self.probability(prev,w),brute(a,b,""),rel_tol=1e-12)
                assert abs(sum(self.row(a,b))-1)<1e-12
        return True

class Hybrid(H.Hybrid):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel);self.pieces=Pieces(docs)
        assert self.pieces.check()
        print("BPE",len(self.pieces.tokens),"pieces; exact segmentation marginal verified",flush=True)
    def subword(self,base,h,recent,mirror):
        prev=h[-1] if h else ""
        top=set(heapq.nlargest(32,base.s,key=lambda w:(base.complete(w),w)))
        top.update(base.ref.bg_rank[:16]);top.update(recent[-16:]);top.update(self.pieces.propose(prev))
        cache=C.Counter(recent[-128:]);weights={}
        for w in sorted(top):
            if not w:continue
            generic=math.log(max(self.baseword(w),1e-300))
            pp=math.log(max(self.pieces.probability(prev,w),1e-300))
            ev=max(-2.,min(2.,(pp-generic)/(1+len(w)/4)))
            r=math.log1p(cache[w])/(1+math.log1p(self.uni[w]))
            score=.40*ev+.18*r
            if mirror:
                raw=self.spell.logword(prev,w,True)
                score+=.40*max(-2.,min(2.,(raw-generic)/(1+len(w)/4)))
            weights[w]=score
        return H.Tilt(base,weights)
    def laws(self,h,ep,recent):
        kn,a,s,lam=self.hypotheses(h,ep,recent,1.)
        return (kn,s,self.correction(s,h,recent,True,.55),self.subword(s,h,recent,False),self.subword(s,h,recent,True))

if __name__=="__main__":
    H.Hybrid=Hybrid
    H.MODES=("continuation","signed","mirror_poe","bpe_poe","full_hybrid")
    H.main()
