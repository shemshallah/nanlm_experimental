"""v15: generative signed/mirror/BPE/replay mixture with certified prefix search.
Reference-side calibration only; no gradient training or imported weights."""
import base as B, foundation6 as N, foundation8 as E, hybrid13 as H, hybrid14 as P
import argparse, collections as C, functools, heapq, json, math, pathlib, time
import numpy as np

class PiecePrefix:
    """Forward probabilities over (last two tokens, pending token suffix).
    Unlike canonical BPE scoring, this marginalizes EVERY segmentation."""
    def __init__(self,pieces,prev):
        self.pieces=pieces;self.prev=prev
        a,b=pieces.start(prev)
        self.states={"":{(a,b,""):1.}}
        self.children={}
    def distribution(self,p):
        if p in self.children:return self.children[p]
        states=self.state(p);out={c:C.defaultdict(float) for c in B.A};end=0.
        for (a,b,rest),m in states.items():
            if rest:
                out[rest[0]][a,b,rest[1:]]+=m
            else:
                row=self.pieces.row(a,b);z=1-row[self.pieces.eos] if not p else 1.
                if p:end+=m*row[self.pieces.eos]
                for t,s in enumerate(self.pieces.tokens):
                    if s:out[s[0]][b,t,s[1:]]+=m*row[t]/z
        for c,ss in out.items():self.states[p+c]=ss
        d={c:math.fsum(ss.values()) for c,ss in out.items()}
        if p:d[B.EOS]=end
        total=math.fsum(states.values())
        assert math.isclose(math.fsum(d.values()),total,rel_tol=1e-10,abs_tol=1e-18)
        self.children[p]=d
        return d
    def state(self,p):
        if p not in self.states:self.distribution(p[:-1])
        return self.states[p]
    def mass(self,p):return math.fsum(self.state(p).values())
    def complete(self,w):return self.distribution(w).get(B.EOS,0.)

class MirrorPrefix:
    def __init__(self,spell,prev):self.spell=spell;self.prev=prev;self.cache={"":1.}
    def mass(self,p):
        if p not in self.cache:
            self.cache[p]=self.mass(p[:-1])*float(self.spell.probs(self.prev,p[:-1],True)[B.SYM.index(p[-1])])
        return self.cache[p]
    def complete(self,w):
        return self.mass(w)*float(self.spell.probs(self.prev,w,True)[26]) if w else 0.

class BasePrefix:
    def __init__(self,law):self.law=law;self.ref=law.ref;self.b={"":1.}
    def bp(self,p):
        if p not in self.b:self.b[p]=self.bp(p[:-1])*self.ref.charprob(p[:-1])[p[-1]]
        return self.b[p]
    def mass(self,p):return self.law.mass(p,self.bp(p))
    def complete(self,w):return self.law.complete(w)

class Mixture:
    def __init__(self,branches,weights):
        self.branches=branches;self.weights=weights
        assert abs(math.fsum(weights)-1)<1e-10 and min(weights)>=0
    def mass(self,p):
        return math.fsum(q*l.mass(p) for l,q in zip(self.branches,self.weights) if q)
    def complete(self,w):
        return math.fsum(q*l.complete(w) for l,q in zip(self.branches,self.weights) if q)
    def next(self,p):
        den=self.mass(p);assert den>0
        d={c:self.mass(p+c)/den for c in B.A}
        if p:d[B.EOS]=self.complete(p)/den
        assert abs(math.fsum(d.values())-1)<1e-8
        return d
    def rank(self,k=5,budget=400):
        # Branch-and-bound over the FULL word space; no target-supplied seeds.
        heap=[(-1.,"")];best=[];seen=0
        while heap and seen<budget:
            if len(best)>=k and -heap[0][0]<best[-1][0]:break
            _,p=heapq.heappop(heap);seen+=1
            if p:
                best=heapq.nlargest(k,best+[(self.complete(p),p)])
            for c in B.A:
                t=p+c;m=self.mass(t)
                if m>0 and (len(best)<k or m>=best[-1][0]):heapq.heappush(heap,(-m,t))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[-1][0])
        return [w for _,w in best],cert,seen

NAMES=("signed","pieces","mirror","replay")
class Replay:
    """Long causal document memory, including previous exact suffix occurrences.
    Sparse recurrence distributions are blended with the original full word law."""
    def __init__(self):self.h=[];self.ep=C.defaultdict(C.Counter);self.long=C.defaultdict(C.Counter)
    def observe(self,w):
        for n in range(1,min(8,len(self.h))+1):self.ep[tuple(self.h[-n:])][w]+=1
        for n in (1,2,3,4,8,12,16,24):
            if len(self.h)>=n:self.long[tuple(self.h[-n:])][w]+=1
        self.h=(self.h+[w])[-256:]
    def law(self,base):
        s=dict(base.s);bg=base.bg;strength=0.
        for n in (1,2,3,4,8,12,16,24):
            if len(self.h)<n:continue
            row=self.long.get(tuple(self.h[-n:]))
            if not row:continue
            count=sum(row.values());lam=min(.90,count/(count+(10 if n==1 else 4 if n==2 else 1.5)))
            q=B.norm(row);s={w:(1-lam)*p for w,p in s.items()};B.add(s,q,lam);bg*=1-lam;strength=max(strength,lam)
        return E.FastLattice(base.ref,s,bg),strength

class Hybrid(H.Hybrid):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel);self.pieces=P.Pieces(docs)
        assert self.pieces.check()
        print("BPE",len(self.pieces.tokens),flush=True)
    def branches(self,session):
        h=session.h;prev=h[-1] if h else ""
        kn,a,s,_=self.hypotheses(h[-64:],session.ep,h[-128:],1.)
        replay,strength=session.law(s)
        p=PiecePrefix(self.pieces,prev);m=MirrorPrefix(self.spell,prev)
        count=sum(self.tables[2].get(tuple(h[-2:]),{}).values()) if len(h)>=2 else 0
        key=(min(3,int(math.log2(1+count))),int(strength>.4))
        return (BasePrefix(s),p,m,BasePrefix(replay)),key,s

class Cal:
    def __init__(self):self.rows=C.defaultdict(lambda:[0,[0.]*4])
    def observe(self,key,probs):
        for k in (("all",),key):
            row=self.rows[k];row[0]+=1
            for j,p in enumerate(probs):row[1][j]+=math.log(max(p,1e-300))
    def weights(self,key):
        n,s=self.rows[("all",)];m,v=self.rows.get(key,(0,[0.]*4))
        scores=[(v[j]+32*s[j]/n)/(m+32) for j in range(4)]
        mx=max(scores);q=B.norm({j:math.exp(2*(x-mx)) for j,x in enumerate(scores)})
        return [.04+.84*q[j] for j in range(4)]
    def export(self):return {str(k):v for k,v in sorted(self.rows.items(),key=lambda x:str(x[0]))}

def fit(ref,docs):
    cal=Cal()
    for doc in docs:
        s=Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                branches,key,_=ref.branches(s);cal.observe(key,[l.complete(w) for l in branches])
            s.observe(w)
    return cal

def evaluate(ref,cal,docs,limit):
    modes=("signed","mixture","calibrated","adaptive")
    ms={k:C.Counter() for k in modes};start=time.monotonic();ws=C.Counter()
    for doc in docs:
        state=Replay();logs=[0.]*4
        for i,w in enumerate(doc[:limit+16]):
            branches,key,base=ref.branches(state)
            static=[.65,.15,.10,.10];prior=cal.weights(key)
            q=B.norm({j:prior[j]*math.exp(logs[j]) for j in range(4)})
            adaptive=[.025+.90*q[j] for j in range(4)]
            laws=(Mixture(branches,[1.,0.,0.,0.]),Mixture(branches,static),Mixture(branches,prior),Mixture(branches,adaptive))
            if i>=16:
                for mode,l in zip(modes,laws):
                    m=ms[mode]
                    if mode=="signed":top,cert,n=base.rank()
                    else:top,cert,n=l.rank()
                    m["n"]+=1;m["t1"]+=w==top[0];m["t5"]+=w in top;m["cert"]+=cert;m["expanded"]+=n
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.;a1=a5=True
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c];rank=sorted(d,key=lambda x:(-d[x],x))
                        a1=a1 and c==rank[0];a5=a5 and c in rank[:5]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-7,abs_tol=1e-15)
                    m["spelled1"]+=a1;m["spelled5"]+=a5
                    for name,q in zip(NAMES,l.weights):ws[mode,name]+=q
                # exact segmentation marginal matches the donor-derived DP.
                if i==16:assert math.isclose(branches[1].complete(w),ref.pieces.probability(state.h[-1],w),rel_tol=1e-10)
            probs=[l.complete(w) for l in branches]
            logs=[.94*v+.35*math.log(max(p,1e-300)) for v,p in zip(logs,probs)]
            mx=max(logs);logs=[max(-15.,v-mx) for v in logs];state.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],teacher_forced_spelled_top1=m["spelled1"]/m["n"],teacher_forced_spelled_top5=m["spelled5"]/m["n"],spelling_bps=m["bits"]/m["symbols"],weights={name:ws[k,name]/m["n"] for name in NAMES}) for k,m in ms.items()},"seconds":time.monotonic()-start}

def checks(ref,cal,doc):
    def at(data,k):
        s=Replay()
        for w in data[:k]:s.observe(w)
        branches,key,_=ref.branches(s);l=Mixture(branches,cal.weights(key))
        return l.rank(),[l.complete(w) for w in ("a","the","house","unseenword")]
    assert at(doc,16)==at(doc[:16]+["future","suffix"],16)
    return {"suffix_invariance":True,"mirror":ref.spell.checks(),"bpe_dp":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(calbooks)==12 and len(lex)>1000
    ref=Hybrid([B.load(p,3000) for _,p in tr],lex,N.Native(a.lib))
    cal=fit(ref,[B.load(p,200) for _,p in calbooks]);frozen=cal.export()
    print("CALIBRATION",json.dumps(frozen),flush=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    out={"gradient_updates":0,"reference_ids":[i for i,p in tr],"calibration_ids":[i for i,p in calbooks],"calibration":frozen,"checks":checks(ref,cal,story[0]),"fresh":{}}
    out["stories"]=evaluate(ref,cal,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};ids={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];out["fresh"][k]=evaluate(ref,cal,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert frozen==cal.export()
    out["fresh_ids"]=ids;out["graph"]=dict(ref.diag)
    out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    np.savez_compressed(a.out.with_name("mirror-counts.npz"),plus=ref.spell.plus,minus=ref.spell.minus)
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
