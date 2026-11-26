"""v9: positional context transport and context-conditional prequential trust.
Frozen reference corpus; all online reliability state is document-local."""
import base as B, foundation4 as F, foundation5 as V, foundation6 as N, foundation7 as S, foundation8 as E
import argparse, collections as C, functools, heapq, json, math, pathlib, time

def sample(xs,n):
    if len(xs)<=n:return xs
    return [xs[(2*i+1)*len(xs)//(2*n)] for i in range(n)]

class Transport(E.FastCompetition):
    names=("local","phrase","syntax","topic","episode","analog","legacy_signed","transport","transport_signed")
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel)
        self.rows=[];self.postings=C.defaultdict(list);self.transport_stats=C.Counter()
        for j,doc in enumerate(docs):
            for i,w in enumerate(doc):
                if not i:continue
                h=tuple(reversed(doc[max(0,i-8):i]));r=len(self.rows)
                self.rows.append((j,h,w))
                for k,v in enumerate(h):self.postings[k,v].append(r)
    @functools.lru_cache(maxsize=131072)
    def affinity(self,a,b):
        if a==b:return 1.
        x=self.roles.get(a,{});y=self.roles.get(b,{})
        z=B.dot(x,y) if x and y else 0.
        sa,sb=S.shape(a),S.shape(b)
        return min(.85,.82*z+(.12 if sa==sb and sa!="W" else 0.))
    @functools.lru_cache(maxsize=8192)
    def neighbors(self,w):
        x=self.roles.get(w)
        if not x:return ()
        candidates=set()
        for k in sorted(x,key=lambda k:(len(self.post[k]),-x[k],k))[:8]:
            if len(self.post[k])<=512:candidates.update(self.post[k])
        return tuple(v for sim,v in sorted(((self.affinity(w,v),v) for v in candidates if v!=w),reverse=True)[:8] if sim>=.30)
    @functools.lru_cache(maxsize=8192)
    def alignment(self,history):
        h=tuple(reversed(history[-8:]))
        if not h:return {},0.
        votes=C.Counter()
        # Multiple independent anchors: preserve the last word, while rare
        # earlier anchors recover sequences that differ at one or two slots.
        for lag,w in enumerate(h):
            post=self.postings.get((lag,w),[])
            wt=(1. if lag==0 else .8/(1+.18*lag))/math.sqrt(1+math.log1p(len(post)))
            for r in sample(post,192 if lag==0 else 64):votes[r]+=wt
        for w in self.neighbors(h[0]):
            for r in sample(self.postings.get((0,w),[]),36):votes[r]+=.15*self.affinity(h[0],w)
        ranked=[]
        for r in heapq.nlargest(384,votes,key=lambda r:(votes[r],-r)):
            book,ctx,w=self.rows[r]
            last=self.affinity(h[0],ctx[0])
            if last<.25:continue
            numer=denom=0.;run=0
            for k,a in enumerate(h):
                weight=(.78**k)*(1+.25/(1+math.log1p(self.uni[a])))
                b=ctx[k] if k<len(ctx) else ""
                numer+=weight*self.affinity(a,b);denom+=weight
                if k==run and a==b:run+=1
            score=numer/denom
            # Require last-slot agreement; do not confuse bag-of-words
            # similarity with a valid causal continuation.
            score=.78*score+.22*last
            if score>=.32:ranked.append((score,run,r))
        selected=sorted(ranked,reverse=True)[:64]
        if not selected:return {},0.
        peak=selected[0][0];bookcounts=C.Counter(self.rows[r][0] for _,_,r in selected)
        out=C.Counter()
        for score,run,r in selected:
            book,ctx,w=self.rows[r]
            out[w]+=math.exp(8*(score-peak))*(1+.25*run)/math.sqrt(bookcounts[book])
        nbooks=len(bookcounts)
        # Fixed evidence-strength law, not fitted on DEV.
        trust=min(.82,max(.05,(peak-.25)*1.1)*(.55+.45*min(1,nbooks/5)))
        self.transport_stats["queries"]+=1;self.transport_stats["neighbors"]+=len(selected)
        self.transport_stats["books"]+=nbooks;self.transport_stats["high_alignment"]+=peak>=.75
        return B.norm(out),trust
    def reconcile(self,h,laws):
        old=super().reconcile(h,laws)
        dist,trust=self.alignment(tuple(h[-8:]))
        # Alignment is a joint context law; use the phrase route as a
        # full-support backoff, not a truncated nearest-neighbor vote.
        transport=self.law([(dist,trust)],laws[1])
        if not dist:return (*old,transport,old[-1])
        # Transport injects independent evidence into the same addressed field.
        # Graph-only modulation has exactly conserved conditional mass.
        sources=[];s={};bg=0.
        for name,l,q in zip((*self.names[:6],"transport"),(*laws,transport),(.10,.10,.10,.10,.10,.10,.40)):
            B.add(s,l.s,q);bg+=q*l.bg
            if l.s:sources.append((("transport-source",name,tuple(h[-4:])),B.norm(l.s),q))
        s,bg=self.field(h,sources,s,bg,"signed")
        return (*old,transport,self.lattice(self,s,bg))

MODES={
 "legacy":(tuple(range(7)),False),
 "transport":((7,),False),
 "aligned":(tuple(range(6))+(7,),True),
 "signed_aligned":(tuple(range(6))+(7,8),True),
 "global_aligned":(tuple(range(6))+(7,8),False),
}
class Session(S.Session):
    def __init__(self,ref,mode="signed_aligned"):
        super().__init__(ref,mode);self.buckets={};self.bucket_counts=C.Counter()
    def bucket(self):
        if not self.h:return ("START",)
        f=self.ref.uni[self.h[-1]]
        freq=0 if f==0 else 1 if f<8 else 2 if f<64 else 3
        exact=bool(len(self.h)>=3 and self.ref.idx.get(("local3",tuple(self.h[-3:]))))
        return (freq,exact,S.shape(self.h[-1]))
    def weights_for(self,mode):
        ids,conditional=MODES[mode];bucket=self.bucket()
        local=self.buckets.get(bucket,[0.]*len(self.logs))
        n=self.bucket_counts[bucket];strength=min(.85,n/(n+4))
        scores={i:(1-strength)*self.logs[i]+strength*local[i] if conditional else self.logs[i] for i in ids}
        z=B.norm({i:math.exp(max(-30.,x-max(scores.values()))) for i,x in scores.items()})
        return [(.9*z[i]+.1/len(ids)) if i in z else 0. for i in range(len(self.logs))]
    def mixture(self,mode):
        if self.cache is None:self.cache=self.ref.experts(self.h,self.ep,self.recent)
        ws=self.weights_for(mode);s={};bg=0.
        for l,q in zip(self.cache,ws):
            if q:B.add(s,l.s,q);bg+=q*l.bg
        return self.ref.lattice(self.ref,s,bg),ws
    def predict(self):
        if self.prob is None:self.prob,self.weights=self.mixture(self.mode)
        return self.prob
    def observe(self,w):
        self.predict()
        bucket=self.bucket();n=len(self.logs)
        old=self.buckets.get(bucket,[0.]*n)
        logs=[.88*v+.60*math.log(max(l.complete(w),1e-300)) for v,l in zip(old,self.cache)]
        mx=max(logs);self.buckets[bucket]=[max(-30.,v-mx) for v in logs];self.bucket_counts[bucket]+=1
        super().observe(w)

def evaluate(ref,docs,limit):
    modes=tuple(MODES)
    ms={mode:C.Counter() for mode in modes};ws={mode:C.Counter() for mode in modes};t=time.monotonic()
    for doc in docs:
        session=Session(ref)
        for i,w in enumerate(doc[:limit+16]):
            session.predict()
            if i>=16:
                for mode in modes:
                    l,weights=session.mixture(mode);top,cert,n=l.rank();m=ms[mode]
                    m["n"]+=1;m["top1"]+=top[0]==w;m["top5"]+=w in top;m["certified"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    m["shortlist128"]+=w in heapq.nlargest(128,l.s,key=lambda v:(l.complete(v),v))
                    # Verify the entire word/prefix identity against the same law.
                    p="";b=1.;prod=1.
                    for c in w+B.EOS:
                        d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                        if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                    for name,q in zip(ref.names,weights):ws[mode][name]+=q
            session.observe(w)
    result={}
    for mode,m in ms.items():
        n=m["n"];result[mode]=dict(m,word_top1=m["top1"]/n,word_top5=m["top5"]/n,spelling_bps=m["bits"]/m["symbols"],mean_weights={k:v/n for k,v in ws[mode].items()})
    return result,time.monotonic()-t
class Bytes(S.AdaptiveBytes):
    def __init__(self,ref,raw,mode):
        super().__init__(ref,raw,mode);self.session=Session(ref,mode)

def raw_eval(ref,raw,docs,mode):
    n=one=five=0;bits=err=0.;t=time.monotonic()
    for doc in docs:
        s=Bytes(ref,raw,mode)
        for i,b in enumerate(doc[:768]):
            p=s.probs();assert len(p)==256 and min(p)>=0
            err=max(err,abs(math.fsum(p)-1))
            if i>=64:
                order=heapq.nlargest(5,range(256),key=lambda j:(p[j],-j));n+=1;one+=b==order[0];five+=b in order;bits-=math.log2(max(p[b],1e-300))
            s.observe(b)
    return {"bytes":n,"raw_bpb":bits/n,"byte_top1":one/n,"byte_top5":five/n,"mass_error":err,"seconds":time.monotonic()-t}

def check(ref,raw,doc):
    h=B.words(doc.decode("utf-8","replace"))[:30]
    def at(data,k):
        s=Session(ref)
        for w in data[:k]:s.observe(w)
        l=s.predict();return l.s,l.bg,l.rank(),s.weights
    assert at(h,16)==at(h[:16]+["a","different","future"],16)
    s=Session(ref);a=s.predict();s.observe(h[0]);assert Session(ref).predict().s==a.s
    for n in (8,31,64):
        def bat(data):
            s=Bytes(ref,raw,"signed_aligned")
            for b in data[:n]:s.observe(b)
            return s.probs()
        assert bat(doc)==bat(doc[:n]+b"unrelated suffix")
    return {"word_causal":True,"byte_causal":True,"document_reset":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:24];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Transport(docs,lex,N.Native(a.lib));raw=B.RawByte([p.read_bytes()[:36000] for _,p in tr])
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr],"reference_contexts":len(ref.rows),"position_keys":len(ref.postings),"matched":{},"fresh_dev":{},"seconds":{}}
    def run(k,ds,dest,limit):
        v,sec=evaluate(ref,ds,limit);dest[k]={"retrieval":F.evaluate(ref,ds,"retrieval",limit),"exact":F.evaluate(ref,ds,"exact",limit),**v};out["seconds"][k]=sec
        print(k,json.dumps(dest[k]),flush=True)
    # Full historical story probe; other probes use shorter paired windows to
    # make two-graph comparison affordable. Report actual target counts.
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5]
    run("stories",[B.load(p,300) for _,p in picked],out["matched"],72)
    for k,p in (("wikitext",r/"wiki"/"wiki.valid.raw"),("chemistry",r/"domain"/"chemistry.txt")):
        with p.open(encoding="utf-8",errors="replace") as f:text=f.read(24000)
        run(k,[B.words(text[i:i+8000])[:400] for i in range(0,len(text),8000)],out["matched"],48)
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};fresh={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;fresh[k]=[i for i,w in chosen];run("fresh_"+k,[w for i,w in chosen],out["fresh_dev"],48)
    out["fresh_ids"]=fresh
    rawtests=[p.read_bytes()[:768] for _,p in picked]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["checks"]=check(ref,raw,rawtests[0])
    out["raw"]={mode:raw_eval(ref,raw,rawtests,mode) for mode in ("legacy","signed_aligned","aligned")}
    # Wrong-context control for the standalone transport distribution.
    dev=[B.load(p,150) for _,p in picked];loss=C.Counter();n=0
    for i,doc in enumerate(dev):
        other=dev[(i+1)%len(dev)]
        for j in range(16,40):
            w=doc[j]
            for name,h in (("correct",doc[:j]),("unrelated",other[:j])):
                laws=ref.experts(h,C.defaultdict(C.Counter),h)
                for mode,index in (("legacy",6),("transport",7),("signed_transport",8)):
                    loss[mode+"_"+name]-=math.log2(max(laws[index].complete(w),1e-300))
            n+=1
    out["counterfactual_bits_per_word"]={k:v/n for k,v in loss.items()}
    out["graph"]=dict(ref.diag);out["transport"]=dict(ref.transport_stats);out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
