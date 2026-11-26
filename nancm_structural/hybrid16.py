"""Native bitwise context mixer + signed word/BPE/mirror generative routes.
Raw-byte metrics and teacher-forced spelling are kept separate from word rank.
Corpus-count fitting, no gradient training."""
import base as B, foundation6 as N, hybrid15 as H
import argparse, collections as C, ctypes as CT, heapq, json, math, pathlib, re, time

class NativeCM:
    def __init__(self,path,docs):
        self.dll=CT.CDLL(str(path.resolve()))
        self.dll.cm_new.restype=CT.c_void_p
        self.dll.cm_free.argtypes=[CT.c_void_p]
        self.dll.cm_fit.argtypes=[CT.c_void_p,CT.c_char_p,CT.c_int]
        self.dll.cm_predict.argtypes=[CT.c_void_p,CT.c_char_p,CT.c_int,CT.POINTER(CT.c_double)]
        self.handle=self.dll.cm_new();assert self.handle
        self.count=0
        for doc in docs:self.dll.cm_fit(self.handle,doc,len(doc));self.count+=len(doc)
    def probs(self,tail):
        out=(CT.c_double*256)()
        self.dll.cm_predict(self.handle,tail,len(tail),out)
        d=list(out);assert min(d)>0 and abs(math.fsum(d)-1)<1e-10
        return d
    def close(self):
        if self.handle:self.dll.cm_free(self.handle);self.handle=None

class Stream:
    def __init__(self,ref,raw,cal,mode):
        self.ref=ref;self.raw=raw;self.cal=cal;self.mode=mode
        self.state=H.Replay();self.tail=b"";self.prefix="";self.lat=None
        self.logweights=[0.]*4;self.posterior=.5;self.cached=None
    def begin(self):
        branches,key,_=self.ref.branches(self.state)
        self.branches=branches
        if self.mode=="signed":weights=[1.,0.,0.,0.]
        elif self.mode=="static":weights=[.65,.15,.10,.10]
        else:
            p=self.cal.weights(key)
            q=B.norm({j:p[j]*math.exp(self.logweights[j]) for j in range(4)})
            weights=[.025+.90*q[j] for j in range(4)]
        self.lat=H.Mixture(branches,weights)
    def probs(self):
        if self.cached is not None:return self.cached[0]
        raw=self.raw.probs(self.tail)
        if self.mode=="native_only":self.cached=(raw,raw);return raw
        if self.lat is None:self.begin()
        d=self.lat.next(self.prefix);word=[0.]*256
        non=math.fsum(raw[i] for i in B.NONLETTERS)
        end=d[B.EOS] if self.prefix else non
        for i in B.NONLETTERS:word[i]=end*raw[i]/non
        for c in B.A:
            lo,hi=ord(c),ord(c.upper());z=raw[lo]+raw[hi]
            m=d[c] if self.prefix else (1-end)*d[c]
            word[lo]=m*raw[lo]/z;word[hi]=m*raw[hi]/z
        p=[(1-self.posterior)*a+self.posterior*b for a,b in zip(raw,word)]
        self.cached=(p,word);return p
    def observe(self,b):
        p=self.probs();self.posterior=self.posterior*self.cached[1][b]/p[b]
        if b in B.LETTERS:self.prefix+=chr(b).lower()
        else:
            if self.prefix:
                if self.mode!="native_only":
                    logs=[.94*v+.35*math.log(max(l.complete(self.prefix),1e-300)) for v,l in zip(self.logweights,self.branches)]
                    mx=max(logs);self.logweights=[max(-15.,v-mx) for v in logs]
                self.state.observe(self.prefix)
            self.prefix="";self.lat=None;self.posterior=.5
        self.tail=(self.tail+bytes([b]))[-24:];self.cached=None

def evaluate(ref,raw,cal,docs,mode):
    n=t1=t5=0;bits=err=0.;units=C.Counter();start=time.monotonic()
    for doc in docs:
        s=Stream(ref,raw,cal,mode);ranks=[]
        for i,b in enumerate(doc[:768]):
            p=s.probs();err=max(err,abs(math.fsum(p)-1))
            order=heapq.nlargest(5,range(256),key=lambda j:(p[j],-j))
            ranks.append((b==order[0],b in order))
            if i>=64:n+=1;t1+=ranks[-1][0];t5+=ranks[-1][1];bits-=math.log2(max(p[b],1e-300))
            s.observe(b)
        # Actual raw-byte unit flags; only complete ASCII words fully inside
        # the measured window. No EOS in this donor-style metric.
        for m in re.finditer(rb"[A-Za-z]+",doc[:768]):
            if m.start()<64 or m.end()>=min(768,len(doc)):continue
            if m.start()>0 and doc[m.start()-1] in B.LETTERS:continue
            units["n"]+=1;units["all1"]+=all(v[0] for v in ranks[m.start():m.end()]);units["all5"]+=all(v[1] for v in ranks[m.start():m.end()])
    return {"bytes":n,"raw_bpb":bits/n,"byte_top1":t1/n,"byte_top5":t5/n,"mass_error":err,
            "raw_ascii_units":dict(units),"teacher_forced_raw_word_top1":units["all1"]/units["n"],"teacher_forced_raw_word_top5":units["all5"]/units["n"],"seconds":time.monotonic()-start}

def checks(ref,raw,cal,doc):
    def at(data,k):
        s=Stream(ref,raw,cal,"adaptive")
        for b in data[:k]:s.observe(b)
        return s.probs()
    for k in (16,31,64):
        assert at(doc,k)==at(doc[:k]+b"different future",k)
    x=raw.probs(doc[:24]);assert x==raw.probs(doc[:24]);assert abs(math.fsum(x)-1)<1e-10
    return {"future_invariance":True,"native_determinism":True,"native_mass":True}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--cm",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    books=B.bookfiles(r/"stories");split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    train=[(i,p) for i,p in books if split(i)<7];tr=train[:96];calbooks=train[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(calbooks)==12 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];rawdocs=[p.read_bytes()[:16000] for _,p in tr]
    ref=H.Hybrid(docs,lex,N.Native(a.lib));cal=H.fit(ref,[B.load(p,200) for _,p in calbooks])
    t=time.monotonic();native=NativeCM(a.cm,rawdocs);fit_seconds=time.monotonic()-t
    old=B.RawByte(rawdocs)
    dev=[p.read_bytes()[:768] for i,p in books if split(i) in (7,8)][:5]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:dev.append(f.read(768))
    out={"gradient_updates":0,"reference_words":sum(map(len,docs)),"raw_reference_bytes":native.count,"native_fit_seconds":fit_seconds,"checks":checks(ref,native,cal,dev[0]),"modes":{}}
    class Old:
        def probs(self,tail):return old.probs(tail[-3:])
    for name,raw,mode in (("old_byte_only",Old(),"native_only"),("native_only",native,"native_only"),("old_signed",Old(),"signed"),("native_signed",native,"signed"),("native_static",native,"static"),("native_adaptive",native,"adaptive")):
        out["modes"][name]=evaluate(ref,raw,cal,dev,mode);print(name,json.dumps(out["modes"][name]),flush=True)
    native.close();out["graph"]=dict(ref.diag)
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
