"""v97: resume v96 with signed prompt-address memory and normalized copying."""
import math,json,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field96 import Chat96,batch,loss,EOT

class Chat97(Chat96):
    def __init__(self):
        super().__init__()
        self.query97=nn.Linear(256,96,bias=False)
        self.key97=nn.Linear(256,96,bias=False)
        self.value97=nn.Linear(256,256,bias=False)
        self.residual97=nn.Linear(256,256,bias=False)
        self.copy97=nn.Linear(512,1)
        nn.init.zeros_(self.residual97.weight)
        nn.init.zeros_(self.copy97.weight);nn.init.constant_(self.copy97.bias,-5.)
        self.scale97=1.
    def memory(self,h,x,plen):
        n=int(plen.max());h=h[:,:n];ids=x[:,:n]
        valid=torch.arange(n,device=x.device)[None]<plen[:,None]
        return self.key97(h),self.value97(h),ids,valid
    def decode(self,h,anchor,memory):
        base=self.read(h,anchor[:,None])
        if self.scale97==0:return base
        k,v,ids,valid=memory
        score=self.query97(h)@k.transpose(-1,-2)/math.sqrt(96.)
        att=score.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
        context=att@v
        # Read mutually inhibited signed prompt evidence rather than a
        # single final prompt vector. Copying shares this same address.
        p,n=context.chunk(2,-1)
        signed=torch.cat((p.tanh()*torch.sigmoid(p-n),n.tanh()*torch.sigmoid(n-p)),-1)
        logits=base+self.scale97*self.residual97(signed)
        mix=.65*self.scale97*self.copy97(torch.cat((h,signed),-1)).sigmoid()
        copy=torch.zeros_like(logits).scatter_add(-1,ids[:,None,:].expand(-1,h.shape[1],-1),att)
        prob=(1-mix)*logits.softmax(-1)+mix*copy
        return prob.clamp_min(1e-30).log()
    def forward(self,x,plen):
        if self.scale97==0:return super().forward(x,plen)
        h,_=self.gru(self.embed(x))
        anchor=h[torch.arange(len(x),device=x.device),plen-1]
        return self.decode(h,anchor,self.memory(h,x,plen))
    def prefill(self,prompt):
        x=torch.tensor(list(prompt),dtype=torch.long)[None]
        h,state=self.gru(self.embed(x));anchor=h[:,-1]
        mem=self.memory(h,x,torch.tensor([len(prompt)]))
        q=self.decode(h[:,-1:],anchor,mem)[0,0].double().softmax(-1).detach().numpy()
        return state,(anchor,mem),q
    def advance(self,state,anchor,b):
        x=torch.tensor([[b]],dtype=torch.long)
        h,state=self.gru(self.embed(x),state)
        a,mem=anchor
        q=self.decode(h,a,mem)[0,0].double().softmax(-1).detach().numpy()
        return state,q

def curriculum(n,seed,heldout=False):
    """Synthetic grounding, not knowledge training. Values are unique across splits."""
    rng=np.random.default_rng(seed);rows=[];seen=set()
    while len(rows)<n:
        value=''.join(rng.choice(list('abcdefghjkmnpqrstuvwxyz'),size=int(rng.integers(4,9))))
        # Disjoint deterministic value split; no answer appears in both sets.
        key=hashlib.sha256(value.encode()).hexdigest()
        if (int(key[:8],16)%10==0)!=heldout or value in seen:continue
        seen.add(value)
        kind=len(rows)%3
        if kind==0:
            prompt=f'User: The label is {value}. Return only the label.\nAssistant: '
        elif kind==1:
            prompt=f'User: Save the code {value}.\nAssistant: Saved.\nUser: Give me the code.\nAssistant: '
        else:
            prompt=f'User: Copy this string exactly: {value}\nAssistant: '
        rows.append((prompt.encode(),value.encode()+bytes([EOT]),key,'synthetic_grounding'))
    return rows

def nll(model,rows,early=False):
    x,y,p=batch(rows);z=model(x,p)
    v=F.cross_entropy(z.transpose(1,2),y,ignore_index=-100,reduction='none')
    mask=y!=-100
    if early:mask=mask&(torch.arange(y.shape[1])[None]<p[:,None]+31)
    return (v*mask).sum(-1)/mask.sum(-1).clamp_min(1),int(mask.sum())

def wrong_rows(rows):
    return [(rows[(i+1)%len(rows)][0],r[1],r[2],r[3]) for i,r in enumerate(rows)]

def train97(model,ck,train,dev,steps=1536):
    # Preserve original parameter order and AdamW moments before new group.
    names=[n for n,_ in Chat96().named_parameters()];params=dict(model.named_parameters())
    opt=torch.optim.AdamW([params[n] for n in names],lr=.002,weight_decay=.01)
    opt.load_state_dict(ck['optimizer96'])
    assert all(int(s['step'])==768 for s in opt.state.values())
    for g in opt.param_groups:g['lr']=.001
    new=[p for n,p in model.named_parameters() if n not in names]
    opt.add_param_group({'params':new,'lr':.002,'weight_decay':.01})
    rng=np.random.default_rng(9702)
    probe=[dev[i] for i in np.random.default_rng(9703).choice(len(dev),128,replace=False)]
    synthetic=curriculum(2048,9704);sprobe=curriculum(64,9705,True)
    assert not {x[2] for x in synthetic}&{x[2] for x in sprobe}
    def measure():
        model.eval();out={}
        with torch.no_grad():
            for label,rows in [('human',probe),('synthetic',sprobe)]:
                good=bad=bits=0.;count=0
                for j in range(0,len(rows),16):
                    r=rows[j:j+16];v,k=loss(model,r);bits+=float(v)*k;count+=k
                    a,_=nll(model,r,True);b,_=nll(model,wrong_rows(r),True)
                    good+=float(a.sum());bad+=float(b.sum())
                out[label]={'bpb':bits/count/math.log(2),'correct_early_nll':good/len(rows),'wrong_early_nll':bad/len(rows),'prompt_advantage':(bad-good)/len(rows)}
        model.train();return out
    trace=[[0,measure()]];seen=set();total=0
    for i in range(steps):
        ix=rng.choice(len(train),12,replace=False);seen.update(int(j) for j in ix)
        human=[train[j] for j in ix];synth=[synthetic[j] for j in rng.choice(len(synthetic),4,replace=False)]
        rows=human+synth
        opt.zero_grad();v,n=loss(model,rows);total+=n
        # Compare within human and synthetic cohorts; no trivial domain cue.
        contrast=0.
        for r in (human,synth):
            good,_=nll(model,r,True);bad,_=nll(model,wrong_rows(r),True)
            contrast=contrast+F.relu(.25+good-bad).mean()/2.
        (v+.2*contrast).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (i+1)%128==0 or i+1==steps:
            trace.append([i+1,measure()]);print('CHAT97_DEV',json.dumps(trace[-1]),flush=True)
    assert all(int(opt.state[params[n]]['step'])==768+steps for n in names)
    assert all(int(opt.state[p]['step'])==steps for p in new)
    return opt,{'steps':steps,'inherited_optimizer_step':768+steps,'new_optimizer_step':steps,'unique_human_records':len(seen),'response_bytes_seen':total,'probe_records':len(probe),'synthetic_train_records':len(synthetic),'synthetic_probe_records':len(sprobe),'trace':trace,'params':sum(p.numel() for p in model.parameters())}

def selftest():
    torch.manual_seed(97);old=Chat96();m=Chat97()
    missing=m.load_state_dict(old.state_dict(),strict=False)
    assert not missing.unexpected_keys and all('97' in k for k in missing.missing_keys)
    p=b'User: code blue\nAssistant: ';a=b'blue'+bytes([EOT]);x,y,n=batch([(p,a,'','')])
    with torch.no_grad():
        m.scale97=0.;assert torch.equal(m(x,n),old(x,n))
        m.scale97=1.
        full=m(x,n)[0,len(p)-1:].double().softmax(-1).numpy()
        state,anchor,q=m.prefill(p)
        for j,b in enumerate(a):
            assert np.max(np.abs(q-full[j]))<1e-7
            assert abs(q.sum()-1)<1e-12
            state,q=m.advance(state,anchor,b)
        # Appending a future response cannot change the initial distribution.
        xx=torch.cat((x,torch.tensor([[97,98,99]])),1)
        assert torch.allclose(m(xx,n)[:,len(p)-1],m(x,n)[:,len(p)-1],atol=2e-6)
    l,_=loss(m,[(p,a,'','')]);l.backward()
    assert m.query97.weight.grad.abs().sum()>0 and m.copy97.weight.grad.abs().sum()>0 and m.residual97.weight.grad.abs().sum()>0
    t=curriculum(100,9704);v=curriculum(20,9705,True)
    assert not {r[2] for r in t}&{r[2] for r in v}
    print('v97 exact disabled inheritance, autoregressive parity, causal prompt attention, copy gradients and disjoint synthetic split passed',flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);selftest()
