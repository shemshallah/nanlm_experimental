"""Frozen R6 exact causal NOW logit adapter, for trainable R7 residual memory.

All original 32K hierarchical paths and rank64 mirror-state weights load from
ONE R6 43.85M checkpoint. R6 tensors are untouched by the R7 adapter pilot.
"""
import pathlib, numpy as np, torch
import torch.nn.functional as F
from dataclasses import dataclass
D=96;R=64;V=32000;PAIRS=32640;LONGF=72

class R6:
    def __init__(self,root,weights_state=None):
        root=pathlib.Path(root);c=root/'checkpoint';p=root/'corpus'
        if weights_state is None:
            x=np.memmap(c/'weights.raw',dtype='<i4',mode='r',shape=(6,));assert list(x)==[0x50414c35,V,D,R,256,43850000],list(x)
            base=np.memmap(c/'weights.raw',dtype='<f4',mode='r',offset=24)
            offset=0
            def take(n,shape):
                nonlocal offset
                result=np.asarray(base[offset:offset+n].reshape(shape));offset+=n;return result
            self.E=take(V*D,(V,D));self.H=take(256*D,(256,D));self.P=take(PAIRS*R,(PAIRS,R));self.M=take(PAIRS*R,(PAIRS,R));self.Q=take(D*R,(D,R));self.NOW=take(D,(D,));self.W=take((V-1)*D,(V-1,D));self.B=take(V-1,(V-1,));assert (24+offset*4)==(c/'weights.raw').stat().st_size
            h=np.memmap(c/'weights.raw.ctx2048',dtype='<i4',mode='r',shape=(6,));assert list(h)==[0x50414c39,V,D,R,LONGF,43850000]
            self.L=np.memmap(c/'weights.raw.ctx2048',dtype='<f4',mode='r',offset=24,shape=(D,LONGF))
            self.T=np.fromfile(c/'weights.raw.future64',dtype='<f4',count=4*R,offset=20).reshape(4,R).copy() if (c/'weights.raw.future64').exists() else np.zeros((4,R),np.float32)
        else:
            for name in ('E','H','P','M','Q','NOW','W','B','L','T'):
                t=weights_state[name]
                setattr(self,name,t.detach().cpu().numpy() if isinstance(t,torch.Tensor) else np.asarray(t,dtype=np.float32))
        self.heads=np.fromfile(p/'token_heads.u16',dtype='<u2').reshape(V,2).astype(np.int32)
        self.children=np.fromfile(p/'tree.i32',dtype='<i4').reshape(V-1,2)
        self.vocab=(p/'vocab.words.txt').read_text().splitlines();assert len(self.vocab)==V
        parent=np.full(V*2-1,-1,dtype=np.int32);bits=np.zeros(V*2-1,dtype=np.float32)
        for j,(left,right) in enumerate(self.children):
            parent[left]=parent[right]=V+j;bits[right]=1
        paths=[];targets=[];maxlen=0
        for word in range(V):
            cur=word;nodes=[];b=[]
            while parent[cur]>=0:
                pnode=parent[cur];nodes.append(pnode-V);b.append(bits[cur]);cur=pnode
            nodes.reverse();b.reverse();paths.append(nodes);targets.append(b);maxlen=max(maxlen,len(nodes))
        self.maxpath=maxlen;self.pnodes=np.zeros((V,maxlen),dtype=np.int64);self.pbits=np.zeros((V,maxlen),dtype=np.float32);self.pmask=np.zeros((V,maxlen),dtype=np.float32)
        for i,(n,b) in enumerate(zip(paths,targets)):
            self.pnodes[i,:len(n)]=n;self.pbits[i,:len(n)]=b;self.pmask[i,:len(n)]=1
        self.Wt=torch.tensor(self.W.copy(),dtype=torch.float32);self.Bt=torch.tensor(self.B.copy(),dtype=torch.float32)
        self.pnodest=torch.tensor(self.pnodes);self.pbitst=torch.tensor(self.pbits);self.pmaskt=torch.tensor(self.pmask)
    def hidden(self,examples,ctx):
        e=np.asarray(examples,dtype=np.int32).reshape(-1,7)
        x0,x1,x2=e[:,0],e[:,1],e[:,2]
        ha=self.heads[x2,0].astype(np.int32).copy();hb=self.heads[x1,1].astype(np.int32).copy();hb=np.where(ha==hb,(hb+1)&255,hb)
        lo=np.minimum(ha,hb);hi=np.maximum(ha,hb)
        pair=(lo*(511-lo)//2+hi-lo-1).astype(np.intp)
        signs=np.where(ha<hb,1.,-1.).astype(np.float32)
        f=(.4*signs[:,None]*(self.P[pair]-.2*self.M[pair])).astype(np.float32)
        with np.errstate(over='raise',invalid='raise'):
            z=np.asarray(self.NOW[None,:]+f@self.Q.T+np.asarray(ctx,dtype=np.float32)@self.L.T/32,dtype=np.float32)
            z+=.14*self.E[x0]+.28*self.E[x1]+.58*self.E[x2]
            z+=.18*(self.H[ha]-self.H[hb])
            z+=.04*(self.H[(ha+1)&255]+self.H[(ha+2)&255]-self.H[(hb+255)&255]-self.H[(hb+254)&255])
        return np.tanh(z).astype(np.float32)
    def nll(self,h,targets):
        if not isinstance(h,torch.Tensor):h=torch.from_numpy(np.asarray(h))
        y=torch.as_tensor(targets,dtype=torch.long,device=h.device)
        ix=self.pnodest[y];bits=self.pbitst[y];mask=self.pmaskt[y]
        logits=(self.Wt[ix]*h[:,None,:]).sum(dim=-1)+self.Bt[ix]
        return (F.binary_cross_entropy_with_logits(logits,bits,reduction='none')*mask).sum(dim=-1)

class Corpus:
    def __init__(self,root,part,corpus_dir=None):
        self.root=pathlib.Path(root);self.part=part;corpus=pathlib.Path(corpus_dir) if corpus_dir else self.root/'corpus'
        if (corpus/(part+'_temporal.i32')).stat().st_size==0:raise ValueError(f'empty {part} corpus split: supply additional whole documents')
        self.data=np.memmap(corpus/(part+'_temporal.i32'),mode='r',dtype='<i4').reshape(-1,7)
        self.long=np.memmap(corpus/(part+'_longctx2048.i8'),mode='r',dtype='i1').reshape(-1,LONGF)
        self.seq=self.data[:,3]
        self.charoffs=np.memmap(corpus/(part+'_char_offsets.u32'),mode='r',dtype='<u4')
        self.charblob=np.memmap(corpus/(part+'_char_blob.bin'),mode='r',dtype='u1')
        assert self.data.shape[0]==self.long.shape[0]==len(self.charoffs)
        self.positions=np.arange(len(self.seq),dtype=np.int64)
        boundaries=corpus/(part+'_doc_start.u32')
        self.doc_start=np.memmap(boundaries,mode='r',dtype='<u4') if boundaries.exists() else None
        if self.doc_start is not None:assert len(self.doc_start)==len(self.seq)
    def get(self,positions,context=2048):
        indices=np.asarray(positions,dtype=np.int64)
        # All requested past tokens come from indices strictly < target index.
        lags=np.arange(context,0,-1,dtype=np.int64)
        idx=indices[:,None]-lags[None,:]
        mask=idx>=0
        if self.doc_start is not None:mask &= idx>=self.doc_start[indices,None]
        ids=self.seq[np.maximum(idx,0)].copy()
        ids[~mask]=1 # BOS before the split, not train/test leakage.
        return ids,idx
    def spelling(self,index,vocab):
        index=int(index);wid=int(self.seq[index]);
        if wid>0:return vocab[wid]
        at=int(self.charoffs[index]);assert at>0
        out=bytearray()
        while self.charblob[at]!=0:out.append(int(self.charblob[at]));at+=1
        return out.decode('utf8')

if __name__=='__main__':
    import argparse,json,time
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--eval',type=int,default=4000);args=p.parse_args()
    t=__import__('time').time();torch.set_num_threads(2)
    m=R6(args.root);results={}
    for part in ['dev','test']:
        c=Corpus(args.root,part);ix=np.arange(args.eval,dtype=np.int64)*(len(c.data)-1)//args.eval
        losses=[]
        for block in np.array_split(ix, max(1,len(ix)//64)):
            h=m.hidden(c.data[block],c.long[block]);losses.extend(m.nll(h,c.data[block,3]).numpy())
        results[part]={'nll':float(np.mean(losses)),'samples':len(losses)}
    print(json.dumps({'result':results,'elapsed':round(__import__('time').time()-t,2),'checkpoint':'43850000'},indent=2))
