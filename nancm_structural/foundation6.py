"""v6: native fixed-point/adjoint kernel parity and benchmark on real NaNROUTER."""
import base as B, foundation4 as F, foundation5 as V
import argparse, ctypes as T, collections as C, json, math, pathlib, time
class Edge(T.Structure):
    _fields_=[("i",T.c_uint32),("j",T.c_uint32),("v",T.c_double)]
class Native:
    def __init__(self,path):
        self.lib=T.CDLL(str(path.resolve()));self.fn=self.lib.nancm_solve
        self.fn.argtypes=[T.POINTER(Edge),T.c_size_t,T.POINTER(T.c_double),T.POINTER(T.c_double),T.c_uint32,T.c_double,T.c_double,T.c_uint32]
        self.fn.restype=T.c_int
        self.calls=0;self.seconds=0.;self.iterations=C.Counter()
    def solve(self,active,scale,forcing):
        start=time.monotonic()
        # Exact stable address flattening: node * 8 + coordinate.
        # Compact only active nodes to reduce zero-filled vector work.
        nodes=sorted(set(forcing)|{v for _,i,j,_,_,_ in active for v in (i,j)})
        if not nodes:return {},True
        ids={k:i for i,k in enumerate(nodes)}
        es=(Edge*len(active))(*(Edge(ids[i],ids[j],v) for _,i,j,_,_,v in active))
        n=len(nodes);a=(T.c_double*n)(*(forcing.get(k,0.) for k in nodes));out=(T.c_double*n)()
        it=self.fn(es,len(active),a,out,n,scale,1e-12,48)
        if it<0:raise ValueError("native solver rejected input: "+str(it))
        self.calls+=1;self.seconds+=time.monotonic()-start;self.iterations[it]+=1
        return {k:out[i] for i,k in enumerate(nodes)},bool(it)
class NativeFoundation(V.DiffFoundation):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex);self.kernel=kernel
    def solve_linear(self,active,scale,forcing):return self.kernel.solve(active,scale,forcing)
def parity(ref,native,docs):
    maxerr=0.;n=0;cert=0;diffs=0;pytime=ntime=0.
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter)
        for j,w in enumerate(doc[:40]):
            if j>=16:
                t=time.monotonic();a=ref.predict(h,ep,"signed");pytime+=time.monotonic()-t
                t=time.monotonic();b=native.predict(h,ep,"signed");ntime+=time.monotonic()-t
                assert a.bg==b.bg and a.s.keys()==b.s.keys()
                err=max((abs(a.complete(v)-b.complete(v)) for v in set(a.s)|{w}),default=0.)
                maxerr=max(maxerr,err);assert err<1e-10
                ra,ca,_=a.rank();rb,cb,_=b.rank()
                assert ra==rb and ca==cb;cert+=ca
                n+=1
            B.observe(ep,h,w);h=(h+[w])[-64:]
    return {"boundaries":n,"max_probability_error":maxerr,"rank_parity":True,"certified":cert,"python_seconds":pytime,"native_seconds":ntime}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    books=B.bookfiles(r/"stories");split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in books if split(i)<7][:24];picked=[(i,p) for i,p in books if split(i) in (7,8)][:5]
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];dev=[B.load(p,300) for _,p in picked]
    kern=Native(a.lib);ref=V.DiffFoundation(docs,lex);native=NativeFoundation(docs,lex,kern)
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"parity":parity(ref,native,dev)}
    print("PARITY",json.dumps(out["parity"]),flush=True)
    out["gradient_probe"]=V.gradcheck(native,dev)
    out["word"]={mode:F.evaluate(native,dev,mode) for mode in ("retrieval","analog","signed")}
    raw=B.RawByte([p.read_bytes()[:36000] for _,p in tr])
    rawtests=[p.read_bytes()[:768] for _,p in picked]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["bytecheck"]=V.bytecheck(native,raw,rawtests[0])
    out["byte"]={mode:V.byte_eval(native,raw,rawtests,mode) for mode in ("byte_only","signed","bayes")}
    out["native"]={"calls":kern.calls,"solver_seconds_including_marshalling":kern.seconds,"iterations":dict(kern.iterations)}
    out["graph"]=dict(native.diag)
    out["checkpoint_sha256"]=native.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
