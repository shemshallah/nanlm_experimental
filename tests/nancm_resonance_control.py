"""Matched-budget synthetic mechanism control, not a corpus benchmark.
Run from repository root: python tests/nancm_resonance_control.py
"""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"nancm_structural"))
import torch
from selective_resonance import allocation,route,read
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(3504)
def data(n):
    a=torch.zeros(n,8,256);x=torch.zeros_like(a)
    choose=torch.randint(0,2,(n,));row=torch.arange(n);a[row,:,choose]=1.;a[row,:,1-choose]=-1.
    signals=torch.randint(0,2,(n,2)).float();x[:,:,:2]=signals[:,None,:]
    return a,x,signals[row,choose].long()
tr=data(2048);dv=data(2048);batches=torch.randint(0,len(tr[2]),(240,64))
for uniform in (True,False):
    s=torch.nn.Parameter(torch.full((8,256),.1));o=torch.nn.Parameter(torch.full_like(s,.1));r=torch.nn.Parameter(torch.zeros_like(s));p=torch.nn.Parameter(torch.zeros(256,128));n=torch.nn.Parameter(torch.zeros_like(p))
    opt=torch.optim.Adam([s,o,r,p,n],lr=.02)
    def pred(a,x):
        g=allocation(a,s,o);return read(x*route(g,r,uniform),g,p,n,uniform)[:,:2]
    for idx in batches:
        opt.zero_grad();loss=F.cross_entropy(pred(tr[0][idx],tr[1][idx]),tr[2][idx]);loss.backward();opt.step()
    with torch.no_grad():
        z=pred(*dv[:2]);print('uniform' if uniform else 'selective','accuracy',(z.argmax(1)==dv[2]).float().mean().item(),'nll',F.cross_entropy(z,dv[2]).item())
