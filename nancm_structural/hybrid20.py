"""v20 corpus-scale compact contextual memory.
Exact/skip/learned-role context addresses, reference-only fitting, disjoint calibration.
Inherits v18 without modifying signed parameters. No additional gradient training."""
import hybrid18 as U
import hybrid15 as H, base as B, foundation6 as N, foundation8 as E, foundation7 as S
import argparse, collections as C, functools, heapq, json, math, pathlib, time
import numpy as np

# Context schemas are separately indexed. Full uint64 context hash; separate
# uint32 word ID. No truncation of context hashes to pack a word ID.
SCHEMAS=(
 ("one",((1,0),),.70),("two",((2,0),(1,0)),1.),
 ("three",((3,0),(2,0),(1,0)),1.15),("four",((4,0),(3,0),(2,0),(1,0)),1.25),
 ("six",tuple((i,0) for i in range(6,0,-1)),1.35),
 ("skip1",((3,0),(1,0)),.70),("skip2",((4,0),(1,0)),.55),
 ("role_left",((2,1),(1,0)),.65),("role_right",((2,0),(1,1)),.40),
 ("role_triple",((3,1),(2,1),(1,0)),.60),
 ("role_double",((2,1),(1,1)),.25),
 ("role_gap",((4,1),(2,0),(1,0)),.55))
M=0xFFFFFFFFFFFFFFFF
MUL=0x9e3779b185ebca87
START=0xcbf29ce484222325

def scalar_hash(values):
    x=START
    for v in values:x=((x^int(v))*MUL)&M
    return np.uint64(x)
def vector_hash(arrays,n):
    x=np.full(n,START,dtype=np.uint64)
    for v in arrays:x=(x^v.astype(np.uint64))*np.uint64(MUL)
    return x

class Memory:
    def __init__(self,docs):
        t=time.monotonic()
        counts=C.Counter(w for d in docs for w in d)
        self.words=sorted(counts);self.ids={w:i+1 for i,w in enumerate(self.words)}
        self.inverse=[""]+self.words
        seq=[np.asarray([self.ids[w] for w in d],dtype=np.uint32) for d in docs]
        self.nwords=sum(map(len,seq));self.nbooks=len(seq)
        # Learn distributional roles from the occurrence of frequent left/right
        # anchors. The right neighbor is used ONLY in fitting reference roles.
        anchors=[self.ids[w] for w,_ in counts.most_common(32)]
        anchor=np.zeros(len(self.inverse),np.uint8)
        for j,k in enumerate(anchors,1):anchor[k]=j
        hist=np.zeros((len(self.inverse),64),np.uint32)
        for d in seq:
            for col,x,y in ((0,d[1:],anchor[d[:-1]]),(32,d[:-1],anchor[d[1:]])):
                ok=y>0;np.add.at(hist,(x[ok],col+y[ok]-1),1)
        total=hist.sum(axis=0).astype(np.float64)+1
        freq=np.asarray([0]+[counts[w] for w in self.words],dtype=np.float64)
        ratio=(hist+.1)/(freq[:,None]+3)/(total[None,:]/(2*self.nwords+64))
        # Each role is a pair of discriminative frequent context anchors;
        # rare words fall back to observed spelling category.
        left=ratio[:,:32].argmax(axis=1);right=ratio[:,32:].argmax(axis=1)
        self.role=np.zeros(len(self.inverse),np.uint32)
        self.role[:]=1+left*32+right
        shapes=sorted({S.shape(w) for w in self.words}|{"W"})
        self.shapes={k:1025+i for i,k in enumerate(shapes)}
        for i,w in enumerate(self.inverse):
            if freq[i]<5:self.role[i]=self.shapes.get(S.shape(w),self.shapes["W"])
        del hist,ratio
        self.unigram=B.norm({w:float(n) for w,n in counts.most_common(2048)})
        self.tables={};self.entries=0
        for name,schema,_ in SCHEMAS:
            depth=max(lag for lag,r in schema);keys=[];targets=[]
            for d in seq:
                if len(d)<=depth:continue
                n=len(d)-depth
                a=[(self.role[d] if r else d)[depth-lag:len(d)-lag] for lag,r in schema]
                keys.append(vector_hash(a,n));targets.append(d[depth:])
            k=np.concatenate(keys);v=np.concatenate(targets)
            order=np.lexsort((v,k));k=k[order];v=v[order]
            # Aggregate duplicate (context,word) pairs into counts.
            change=np.empty(len(k),dtype=bool);change[0]=True
            change[1:]=(k[1:]!=k[:-1])|(v[1:]!=v[:-1])
            ix=np.flatnonzero(change)
            cnt=np.diff(np.append(ix,len(k))).astype(np.uint32)
            self.tables[name]=(k[ix],v[ix],cnt);self.entries+=len(ix)
            del keys,targets,k,v,order,change,ix,cnt
        self.bytes=sum(a.nbytes for t in self.tables.values() for a in t)
        self.seconds=time.monotonic()-t
    def roleof(self,w):
        i=self.ids.get(w)
        return int(self.role[i]) if i is not None else self.shapes.get(S.shape(w),self.shapes["W"])
    @functools.lru_cache(maxsize=16384)
    def row(self,name,key):
        k,v,c=self.tables[name]
        a=int(np.searchsorted(k,key,"left"));b=int(np.searchsorted(k,key,"right"))
        if a==b:return {},0,0
        counts=c[a:b];total=int(counts.sum());distinct=b-a
        # Bound work, preserve unretained mass through backoff.
        ix=heapq.nlargest(96,range(a,b),key=lambda j:(int(c[j]),int(v[j])))
        return {self.inverse[int(v[j])]:int(c[j]) for j in ix},total,distinct
    def law(self,h,base,mode="all"):
        # Start with a small expanded corpus unigram correction.
        s={w:.85*v for w,v in base.s.items()};B.add(s,self.unigram,.15);bg=.85*base.bg
        q={};trust=0.;exact=[];stats=C.Counter()
        for name,schema,scale in SCHEMAS:
            if len(h)<max(lag for lag,r in schema):continue
            if mode=="exact" and any(r for _,r in schema):continue
            if mode=="exact" and name.startswith("skip"):continue
            values=[self.roleof(h[-lag]) if r else self.ids.get(h[-lag],0) for lag,r in schema]
            if not all(values):continue
            row,total,distinct=self.row(name,scalar_hash(values))
            if not row:continue
            d={w:max(0.,c-.65) for w,c in row.items()}
            support=math.fsum(d.values())
            if not support:continue
            if name in ("one","two","three","four","six"):
                mass=min(.93,support/(total+3.))
                exact.append((B.norm(d),mass))
            else:
                mass=scale*support/(total+8+distinct)
                B.add(q,B.norm(d),mass);trust+=mass
            stats[name]=total
        for d,mass in exact:
            s={w:(1-mass)*v for w,v in s.items()};B.add(s,d,mass);bg*=1-mass
        if q:
            # Correlated role/skip rows share one probability budget.
            mass=min(.50,trust/(2+trust))
            s={w:(1-mass)*v for w,v in s.items()};B.add(s,B.norm(q),mass);bg*=1-mass
        return E.FastLattice(base.ref,s,bg),stats
    def report(self):
        return {"books":self.nbooks,"words":self.nwords,"vocab":len(self.words),"pairs":self.entries,"index_bytes":self.bytes,"build_seconds":self.seconds}

def branches(ref,small,large,state,cal):
    six,key,base=ref.six(state)
    old=H.Mixture(six,cal.weights(key))
    a,_=small.law(state.h,base)
    b,_=large.law(state.h,base,"exact")
    c,stats=large.law(state.h,base)
    return (old,H.BasePrefix(a),H.BasePrefix(b),H.BasePrefix(c)),(key[0],int(bool(stats.get("three"))))
def fit(ref,small,large,cal,docs):
    rows=C.defaultdict(list)
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                laws,k=branches(ref,small,large,state,cal)
                p=[l.complete(w) for l in laws];rows[("all",)].append(p);rows[k].append(p)
            state.observe(w)
    q=U.em(rows[("all",)],[.55,.10,.15,.20],pseudocount=24)
    return q,{k:U.em(v,q,pseudocount=64) for k,v in rows.items() if k!=("all",)}
def evaluate(ref,small,large,cal,prior,docs,limit,warm=16):
    global_q,local=prior
    ms=C.defaultdict(C.Counter);oracle=C.Counter();start=time.monotonic()
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:warm+limit]):
            if i>=warm:
                laws,k=branches(ref,small,large,state,cal)
                union5=set();union32=set()
                for name,l in zip(("v18","small","large_exact","large_role"),laws):
                    if isinstance(l,H.BasePrefix):
                        rank,cert,n=l.law.rank(32,budget=8000)
                    else:rank,cert,n=l.rank(32,budget=2500)
                    oracle[name+"_top32"]+=w in rank;oracle[name+"_cert32"]+=cert
                    union5.update(rank[:5]);union32.update(rank)
                oracle["n"]+=1;oracle["union5"]+=w in union5;oracle["union32"]+=w in union32
                for mode,weights in (("v18",(1,0,0,0)),("small",(0,1,0,0)),("large_exact",(0,0,1,0)),("large_role",(0,0,0,1)),("hybrid",local.get(k,global_q))):
                    law=H.Mixture(laws,weights);rank,cert,n=law.rank(budget=1200);m=ms[mode]
                    m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                    m["bits"]-=math.log2(max(law.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.
                    for c in w+B.EOS:
                        d=law.next(p);prod*=d[c]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,law.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            state.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"oracle":dict(oracle),"seconds":time.monotonic()-start}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7]
    tr=books[:96];calbooks=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(calbooks)==12 and len(extra)>=96
    ids=[i for i,p in tr+extra];assert len(ids)==len(set(ids)) and not set(ids)&{i for i,p in calbooks}
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=U.Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    cdocs=[B.load(p,200) for _,p in calbooks];cal=U.fit(ref,cdocs)
    small=Memory(docs);print("SMALL",json.dumps(small.report()),flush=True)
    large=Memory(docs+[B.load(p,6000) for _,p in extra]);print("LARGE",json.dumps(large.report()),flush=True)
    # Exact address parity on actual reference context arrays.
    sample=np.arange(1,12,dtype=np.uint32)
    assert all(vector_hash([sample,sample+1],len(sample))[i]==scalar_hash([v,v+1]) for i,v in enumerate(sample))
    prior=fit(ref,small,large,cal,cdocs)
    frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5]
    story=[B.load(p,300) for _,p in picked]
    def at(data,k):
        state=H.Replay()
        for w in data[:k]:state.observe(w)
        laws,key=branches(ref,small,large,state,cal);l=H.Mixture(laws,prior[1].get(key,prior[0]))
        return l.rank(),[l.complete(w) for w in ("the","a","unseenword")]
    assert at(story[0],16)==at(story[0][:16]+["future","suffix"],16)
    out={"pilot":pilot,"reference_ids":ids,"calibration_ids":[i for i,p in calbooks],"small":small.report(),"large":large.report(),"gate":json.loads(frozen),"checks":{"hash_parity":True,"suffix_invariance":True},"fresh":{}}
    out["stories"]=evaluate(ref,small,large,cal,prior,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};freshids={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;freshids[k]=[i for i,w in chosen]
        out["fresh"][k]=evaluate(ref,small,large,cal,prior,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    assert not set(ids)&set(used)
    out["signed_unchanged"]=True;out["fresh_ids"]=freshids;out["graph"]=dict(ref.diag)
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
