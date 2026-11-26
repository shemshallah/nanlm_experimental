"""Constructed causal task; actual v43/v44 operators and v45 deployed event loss."""
import copy, math, numpy as np, torch
from adaptive_retention import Retention
from signed_edge_credit import EdgeCredit
from event_credit import objective, probabilities

class Joint(torch.nn.Module):
    def __init__(self):
        super().__init__();self.retention=Retention();self.credit=EdgeCredit(points=16)
    def forward(self,x,m,c,b,v):
        r,n=self.retention(m,c,b);return self.credit(x,r,v,n)

def sample(n):
    y=torch.randint(4,(n,));good=torch.nn.functional.one_hot(y,4).float()
    bad=torch.nn.functional.one_hot((y+torch.randint(1,4,(n,)))%4,4).float()
    v=torch.zeros(n,4,16);v[:,:,:4]=torch.eye(4)
    x=torch.zeros(n,4,25);m=torch.randn(n,8,4,2,16)*.015;c=torch.ones(n,8,4,2)
    m[:,2,0,0,:4]+=3*bad;c[:,2,0,0]=3
    m[:,2,2,1,:4]+=good
    bg=torch.zeros(n,16);p=torch.full((n,4),.2);base=torch.full((n,),.2)
    surface=torch.where(torch.rand(n)<.5,.02,.5);raw=torch.full((n,),.001);folded=torch.full((n,),.1);length=torch.full((n,),4.)
    return x,m,c,bg,v,p,good,base,surface,raw,folded,length

def evaluate(model,data):
    with torch.no_grad():
        lift=model(*data[:5]);literal,folded=probabilities(lift,*data[5:-1],.6)
        n=data[-1].sum();acc=float((lift.argmax(1)==data[6].argmax(1)).float().mean())
        return dict(raw_bpb=float(-literal.log2().sum()/n),folded_nll=float(-folded.log().mean()),candidate_accuracy=acc)

def main():
    torch.set_num_threads(2);torch.manual_seed(4502)
    tr,va=sample(384),sample(192);initial=Joint();before=evaluate(initial,va);out={}
    for name,spelling in [('folded_only',0.),('literal_joint',1.)]:
        model=copy.deepcopy(initial);opt=torch.optim.Adam(model.parameters(),lr=.04);rng=np.random.default_rng(4503)
        for i in range(192):
            ix=rng.integers(len(tr[0]),size=32);a=tuple(t[ix] for t in tr)
            opt.zero_grad();loss=objective(model(*a[:5]),*a[5:],.6,spelling);loss.backward()
            if i==0:
                norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
                assert all(norms[k]>0 for k in ('retention.age','retention.admission','credit.support','credit.opposition','credit.routing'))
            opt.step()
        out[name]=evaluate(model,va)
        bad=list(va);bad[1]=bad[1].roll(1,0);bad[2]=bad[2].roll(1,0)
        out[name]['wrong_history']=evaluate(model,tuple(bad));out[name]['initial_gradient_norms']=norms
        assert out[name]['raw_bpb']<before['raw_bpb']
    print(dict(train_targets=384,held_out_targets=192,before=before,**out))
if __name__=='__main__':main()
