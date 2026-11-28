"""v96 signed recurrent response field. Native 256-byte vocabulary, EOT=4.

Training uses human instruction/response records, not the v95 evaluation answers.
The prompt is encoded causally, and its last signed state is retained throughout
the response. Only response bytes and EOT contribute to supervised loss.
"""
import hashlib,json,math,re
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EOT=4
SOURCE='https://huggingface.co/datasets/databricks/databricks-dolly-15k'
LICENSE='CC BY-SA 3.0'
def load_data(path):
    data=Path(path).read_bytes();seen=set();train=[];dev=[]
    for line in data.splitlines():
        a=json.loads(line);instruction=a['instruction'].strip();context=a.get('context','').strip();answer=a['response'].strip()
        if not instruction or not answer:continue
        identity=' '.join(instruction.lower().split())
        key=hashlib.sha256(identity.encode()).hexdigest()
        if key in seen:continue
        seen.add(key)
        prompt='User: '+instruction+('\nContext: '+context if context else '')+'\nAssistant: '
        p=prompt.encode();y=answer.encode()+bytes([EOT])
        # Reject oversize examples rather than fabricate truncated endings.
        if len(p)>320 or len(y)>256 or len(p)+len(y)>512 or EOT in p or EOT in y[:-1]:continue
        row=(p,y,key,a.get('category','unknown'))
        (dev if int(key[:8],16)%10==0 else train).append(row)
    assert len(train)>1000 and len(dev)>100
    assert not {x[2] for x in train}&{x[2] for x in dev}
    return train,dev,{'source':SOURCE,'license':LICENSE,'sha256':hashlib.sha256(data).hexdigest(),'train_records':len(train),'dev_records':len(dev),'train_response_bytes':sum(len(x[1]) for x in train),'dev_response_bytes':sum(len(x[1]) for x in dev),'split':'normalized instruction SHA256 modulo 10; no duplicate instruction keys'}

class Chat96(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(256,96)
        self.gru=nn.GRU(96,256,batch_first=True)
        self.exchange=nn.Linear(256,256,bias=False)
        self.anchor=nn.Linear(256,256,bias=False)
        self.norm=nn.LayerNorm(256)
        self.out=nn.Linear(256,256)
    def read(self,h,anchor):
        z=h+.5*self.anchor(anchor)
        w=self.exchange.weight;w=w/w.abs().sum(-1,keepdim=True).clamp_min(1.)
        p,n=(z+.5*F.linear(z.tanh(),w)).chunk(2,-1)
        state=torch.cat((p.tanh()*torch.sigmoid(p-n),n.tanh()*torch.sigmoid(n-p)),-1)
        return self.out(self.norm(state))
    def forward(self,x,plen):
        h,_=self.gru(self.embed(x))
        anchor=h[torch.arange(len(x),device=x.device),plen-1]
        return self.read(h,anchor[:,None])
    def prefill(self,prompt):
        x=torch.tensor(list(prompt),dtype=torch.long)[None]
        h,state=self.gru(self.embed(x));anchor=h[:,-1]
        return state,anchor,self.read(anchor,anchor)[0].double().softmax(-1).detach().numpy()
    def advance(self,state,anchor,b):
        x=torch.tensor([[b]],dtype=torch.long)
        h,state=self.gru(self.embed(x),state)
        return state,self.read(h[:,-1],anchor)[0].double().softmax(-1).detach().numpy()

def batch(rows):
    plen=np.array([len(p) for p,y,*_ in rows],np.int64)
    lens=[len(p)+len(y)-1 for p,y,*_ in rows];n=max(lens)
    x=np.zeros((len(rows),n),np.int64);target=np.full((len(rows),n),-100,np.int64)
    for i,(p,y,*_) in enumerate(rows):
        seq=np.frombuffer(p+y,np.uint8).astype(np.int64)
        x[i,:len(seq)-1]=seq[:-1];target[i,len(p)-1:len(seq)-1]=seq[len(p):]
    return tuple(torch.from_numpy(v) for v in (x,target,plen))

def loss(model,rows):
    x,y,p=batch(rows);z=model(x,p);mask=y!=-100
    return F.cross_entropy(z[mask],y[mask]),int(mask.sum())

def train96(model,train,dev,steps=768):
    torch.manual_seed(9601);rng=np.random.default_rng(9602)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    probe=[dev[i] for i in np.random.default_rng(9603).choice(len(dev),min(128,len(dev)),replace=False)]
    def measure():
        model.eval();bits=0.;n=0
        with torch.no_grad():
            for j in range(0,len(probe),16):
                v,k=loss(model,probe[j:j+16]);bits+=float(v)*k/math.log(2);n+=k
        model.train();return bits/n
    trace=[[0,measure()]];seen=set();total=0
    for i in range(steps):
        ix=rng.choice(len(train),16,replace=False);seen.update(int(j) for j in ix)
        opt.zero_grad();v,n=loss(model,[train[j] for j in ix]);total+=n
        v.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (i+1)%64==0 or i+1==steps:
            trace.append([i+1,measure()]);print('CHAT96_DEV',json.dumps(trace[-1]),flush=True)
    return opt,{'steps':steps,'unique_records':len(seen),'response_bytes_seen':total,'probe_records':len(probe),'dev_bpb_trace':trace,'params':sum(p.numel() for p in model.parameters())}

def selftest():
    torch.manual_seed(96);m=Chat96();p=b'User: x\nAssistant: ';a=b'yes'+bytes([EOT])
    x,y,n=batch([(p,a,'','')]);assert (y!=-100).sum()==len(a)
    with torch.no_grad():
        full=m(x,n)[0,len(p)-1:].double().softmax(-1).numpy()
        state,anchor,q=m.prefill(p)
        for j,b in enumerate(a):
            assert np.max(np.abs(q-full[j]))<1e-7
            state,q=m.advance(state,anchor,b)
    l,_=loss(m,[(p,a,'','')]);l.backward()
    assert m.gru.weight_hh_l0.grad.abs().sum()>0 and m.exchange.weight.grad.abs().sum()>0 and m.anchor.weight.grad.abs().sum()>0
    print('v96 response-only masking, full/recurrent byte parity and signed/prompt gradients passed',flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
