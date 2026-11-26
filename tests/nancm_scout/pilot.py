"""v48 actual-corpus candidate discovery, not a replacement LM checkpoint."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'nancm_structural'),str(root/'tests/nancm_graph')]
import base as B
from fit_split import cohorts
from metric_io import write_metrics
from scout import Scout,rails
import argparse,collections as C,hashlib,math
import numpy as np
import torch

class Index:
    def __init__(self,docs,lex):
        counts=C.Counter(w for d in docs for w in d)
        self.words=['']+sorted(set(counts)|set(lex));self.ids={w:i for i,w in enumerate(self.words) if i}
        self.V=len(self.words);assert self.V**3<2**63 and self.V>160
        self.seq=[np.array([self.ids[w] for w in d],np.int64) for d in docs]
        freq=np.array([0]+[counts[w] for w in self.words[1:]],np.float64)
        self.prior=(freq+.1)/(freq.sum()+.1*self.V);self.log_prior=torch.tensor(np.log(self.prior),dtype=torch.float32)
        anchors=np.zeros(self.V,np.int64)
        for i,(w,n) in enumerate(counts.most_common(64),1):anchors[self.ids[w]]=i
        hist=np.zeros((self.V,128),np.float32)
        for d in self.seq:
            for off,word,a in ((0,d[1:],anchors[d[:-1]]),(64,d[:-1],anchors[d[1:]])):
                ok=a>0;np.add.at(hist,(word[ok],off+a[ok]-1),1)
        background=(hist.sum(0,dtype=np.float64)+1)/(sum(map(len,self.seq))+128)
        code=np.log((hist+8*background)/(freq[:,None]+8)/background).clip(-5,5).astype(np.float32)
        # Give unseen wordlist entries a signed spelling address, not target data.
        for i,w in enumerate(self.words[1:],1):
            for k in range(len(w)+1):
                digest=B.addr('scout:'+('^'+w+'$')[k:k+3])
                code[i,digest%128]+=.15*(1 if digest&256 else -1)
        code*=4/np.maximum(np.linalg.norm(code,axis=1,keepdims=True),1e-6);code[0]=0
        self.code=code;self.tables=[]
        for order in (1,2,3):
            keys=[];targets=[]
            for d in self.seq:
                if len(d)<=order:continue
                k=d[:len(d)-order].copy()
                for j in range(1,order):k=k*self.V+d[j:len(d)-order+j]
                keys.append(k);targets.append(d[order:])
            k=np.concatenate(keys);v=np.concatenate(targets);ix=np.lexsort((v,k));k=k[ix];v=v[ix]
            start=np.flatnonzero(np.r_[True,(k[1:]!=k[:-1])|(v[1:]!=v[:-1])])
            self.tables.append((k[start],v[start],np.diff(np.r_[start,len(k)])))
        self.reference_words=int(freq.sum())
    def encode(self,doc):return [self.ids.get(w,0) for w in doc]
    def countlaw(self,h):
        q=self.prior.copy()
        for order,(keys,ids,counts) in enumerate(self.tables,1):
            if len(h)<order or not all(h[-order:]):continue
            k=0
            for a in h[-order:]:k=k*self.V+a
            a=np.searchsorted(keys,k,'left');b=np.searchsorted(keys,k,'right')
            if a==b:continue
            n=float(counts[a:b].sum());trust=n/(n+8+len(counts[a:b]))
            q*=1-trust;q[ids[a:b]]+=trust*counts[a:b]/n
        assert abs(q.sum()-1)<1e-10
        return q

def examples(index,files,seed,limit):
    rng=np.random.default_rng(seed);rows=[]
    for bid,p in files:
        h=index.encode(B.load(p,1200))
        if len(h)<32:continue
        pos=np.sort(rng.choice(np.arange(16,len(h)),min(limit,len(h)-16),replace=False))
        for i in pos:
            i=int(i);past=h[max(0,i-512):i]
            rows.append((bid,past,rails(index.code,past),h[i]))
    assert rows;return rows

def features(rows):
    return torch.tensor(np.stack([r[2] for r in rows])),torch.tensor([r[3] for r in rows],dtype=torch.long)

def rank(x,k):
    a=np.asarray(x).copy();a[0]=-np.inf
    # Stable sort keeps ties deterministic.
    return np.argsort(-a,kind='stable')[:k].tolist()

def union(a,b,k,quota):
    first=a[:quota];seen=set(first)
    for j in b:
        if j not in seen:first.append(j);seen.add(j)
        if len(first)==k:break
    assert len(first)==k and len(set(first))==k
    return first

def evaluate(model,index,rows):
    x,y=features(rows)
    with torch.no_grad():
        scores=torch.cat([model(x[i:i+32])+.25*index.log_prior for i in range(0,len(x),32)]).numpy()
    # Choose a different book for every wrong-context row.
    books=C.defaultdict(list)
    for i,row in enumerate(rows):books[row[0]].append(i)
    names=sorted(books);assert len(names)>=2
    nxt={bid:names[(j+1)%len(names)] for j,bid in enumerate(names)}
    counter=C.Counter();m=C.Counter()
    for i,(bid,past,feat,t) in enumerate(rows):
        a=rank(index.countlaw(past),160);b=rank(scores[i],160)
        fixed=rank(index.code@feat[0]+.25*np.log(index.prior),160)
        other=books[nxt[bid]];wrong=rank(scores[other[counter[bid]%len(other)]],160);counter[bid]+=1
        m['n']+=1;m['oov']+=t==0
        for name,pool in [('count32',a[:32]),('scout32',b[:32]),('fixed32',fixed[:32]),('matched32',union(a,b,32,16)),('wrong32',union(a,wrong,32,16)),('count128',a[:128]),('matched128',union(a,b,128,96)),('wrong128',union(a,wrong,128,96)),('count160',a[:160]),('expanded160',union(a,b,160,128)),('wrong160',union(a,wrong,160,128))]:
            m[name]+=t in pool
        m['added_outside_count128']+=t not in a[:128] and t in union(a,b,160,128)
    n=m['n'];return {'counts':dict(m),'recall':{k:v/n for k,v in m.items() if k not in ('n','oov')},'known_vocab_fraction':1-m['oov']/n}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('scout-metrics.json'));a=ap.parse_args();r=a.data
    torch.set_num_threads(4);torch.manual_seed(4802)
    groups={k:B.bookfiles(r/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];cal=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(cal)==12 and len(extra)==288
    fit,_,sel,manifest=cohorts(books,tr,cal,extra)
    lex=set()
    for p in (r/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]+[B.load(p,6000) for _,p in extra]
    index=Index(docs,lex);del docs
    out={'vocab':index.V-1,'reference_words':index.reference_words,'reference_books':384,'split_integrity':manifest,'domains':{}}
    write_metrics(a.out,out,'index')
    x,y=features(examples(index,fit,4803,96));valid=examples(index,sel,4804,64)
    model=Scout(index.code);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);rng=np.random.default_rng(4805);trace=[]
    out['before']=evaluate(model,index,valid)
    for i in range(256):
        ix=rng.integers(len(y),size=16);opt.zero_grad();z=model(x[ix])+.25*index.log_prior
        loss=torch.nn.functional.cross_entropy(z,y[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if i==0 or (i+1)%32==0:trace.append([i+1,float(loss.detach())])
    out['fit']={'targets':len(y),'oov':int((y==0).sum()),'updates':256,'trace':trace}
    out['selection']=evaluate(model,index,valid)
    write_metrics(a.out,out,'calibration')
    used=set();forbidden={i for i,p in books}
    hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden:continue
            if len(B.load(p,200))<112:continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest not in hashes
            hashes.add(digest);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':evaluate(model,index,examples(index,chosen,4806,96))}
        write_metrics(a.out,out,'domain:'+name)
    torch.save({'words':index.words,'model':model.state_dict(),'optimizer':opt.state_dict(),'updates':256,'split_integrity':manifest},a.out.with_suffix('.pt'))
    write_metrics(a.out,out)
if __name__=='__main__':main()
