#!/usr/bin/env python3
"""NaNCM v175: continue v174 AdamW, with book-causal 48-word temporal rails + signed sequence-copy.

Parent v174 3-group optimizer moments are loaded *exactly*; a 4th group
trains the novel episodic pointer. Book-disjoint Release3 train/val/test is
prepared with prepare_release_books175.py. Neither old DEV nor reused TEST is
used for checkpoint selection; full-book TEST is evaluated only at completion.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v174'))
from train_temporal_signed174 import load_parent,CloudNCE,load_sign_geometry,DualPolarTemporal

class SignedCausalEcho175(DualPolarTemporal):
    """Episodic next-word copy from *earlier* occurrences of the current prefix.

    A local match at history position i is allowed to predict only the already
    observed history[i+1]. A last-token match can gain support from earlier
    2-word and 3-word matches; no target or future bytes are accessed.
    """
    def __init__(self,teacher,names,gpos,gneg,dim=64):
        super().__init__(teacher,names,gpos,gneg,dim=dim)
        self.copy_log_strength=nn.Parameter(torch.tensor(-3.2))
        self.copy_bigram=nn.Parameter(torch.tensor(0.0))
        self.copy_trigram=nn.Parameter(torch.tensor(0.0))
        self.copy_log_tau=nn.Parameter(torch.tensor(math.log(18.)))
        self.echo_enabled=True

    def echo_logits(self,hist):
        b,l=hist.shape
        res=torch.zeros((b,12000),dtype=self.embed.dtype,device=hist.device)
        if l<4:return res
        query=hist[:,-1:]
        prev=hist[:,:-1]
        targets=hist[:,1:]
        valid=(prev>=0)&(targets>=0)&(query>=0)&(prev==query)
        # Match histories ending at i, never i+1 or beyond current time.
        pair=torch.zeros_like(valid);pair[:,1:]=(hist[:,:-2]==hist[:,-2:-1])&(hist[:,-2:-1]>=0)
        triple=torch.zeros_like(valid);triple[:,2:]=(hist[:,:-3]==hist[:,-3:-2])&(hist[:,-3:-2]>=0)
        triple[:,2:] &= (hist[:,1:-2]==hist[:,-2:-1])&(hist[:,-2:-1]>=0)
        dist=torch.arange(l-1,0,-1,dtype=torch.float32,device=hist.device)[None,:]
        tau=self.copy_log_tau.exp().clamp(1.,128.)
        decay=torch.exp(-dist/tau)
        score=valid.to(self.embed.dtype)*decay * (1.+2.*torch.sigmoid(self.copy_bigram)*pair+4.*torch.sigmoid(self.copy_trigram)*triple)
        score=score/(score.sum(dim=1,keepdim=True).detach()+1.)
        strength=F.softplus(self.copy_log_strength)*8.
        return res.scatter_add(1,targets.clamp_min(0).long(),strength*score)

    def forward(self,hist,return_components=False):
        parent,old,alpha=super().forward(hist,True)
        if self.echo_enabled:
            echo=self.echo_logits(hist)
            out=parent+echo
        else:
            echo=None;out=parent
        return (out,parent,echo) if return_components else out

def load_models():
    names,C,O,B,spatial,gain,fp=load_parent()
    p=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',map_location='cpu',weights_only=False)
    assert tuple(p['fingerprints'])==tuple(fp) and p['word_order']==names
    teacher=CloudNCE(C,O,B,spatial,gain,names)
    teacher.load_state_dict(p['state_dict']);teacher.eval()
    for k in teacher.parameters():k.requires_grad_(False)
    gp,gn=load_sign_geometry(names)
    m=SignedCausalEcho175(teacher,names,gp,gn)
    old=torch.load(ROOT/'checkpoints/v174_temporal_signed_now.pt',map_location='cpu',weights_only=False)
    assert tuple(old['fingerprints'])==tuple(fp) and old['word_order']==names
    got=m.load_state_dict(old['model_state_dict'],strict=False)
    assert set(got.missing_keys)=={'copy_log_strength','copy_bigram','copy_trigram','copy_log_tau'} and not got.unexpected_keys,got
    oldgroups=[{'params':list(m.gru_pos.parameters())+list(m.gru_neg.parameters()),'lr':0.0006},
               {'params':list(m.signed_norm.parameters())+list(m.to_lex.parameters()),'lr':0.0008},
               {'params':[m.trust,m.signed_scale],'lr':0.0003}]
    optimizer=torch.optim.AdamW(oldgroups,weight_decay=.01)
    optimizer.load_state_dict(old['optimizer_state_dict'])
    assert len(optimizer.state)==15 and len(optimizer.param_groups)==3
    optimizer.add_param_group({'params':[m.copy_log_strength,m.copy_bigram,m.copy_trigram,m.copy_log_tau],
                               'lr':0.0015,'weight_decay':0.0})
    return m,optimizer,names,fp,old

def load_split(split,root):
    with np.load(root/(split+'.npz'),allow_pickle=False) as d:
        X=d['X'].astype('int64');Y=d['Y'].astype('int64');books=d['book'].astype('int32')
    assert X.ndim==2 and X.shape[1]==48 and len(Y)==len(X)==len(books)
    assert Y.min()>=0 and Y.max()<12000 and X.min()>=-1 and X.max()<12000
    return X,Y,books

def metric(model,split,root,batch=96,per_book=False):
    X,Y,B=load_split(split,root)
    n=len(Y); totals={'n':n,'loss':0.,'top1':0,'top5':0,'parent_loss':0.,'parent_top1':0,'parent_top5':0}
    books=defaultdict(lambda:{'n':0,'loss':0.,'parent_loss':0.,'top1':0,'parent_top1':0})
    model.eval()
    with torch.no_grad():
        for start in range(0,n,batch):
            x=torch.from_numpy(X[start:start+batch]);y=torch.from_numpy(Y[start:start+batch]);
            z,base,_=model(x,True)
            for arr,label in ((z,''),(base,'parent_')):
                ce=F.cross_entropy(arr,y,reduction='none')
                top=arr.topk(5,dim=1).indices
                for j in range(len(y)):
                    b=int(B[start+j]); book=books[b];book['n']+=int(label=='')
                    book[label+'loss']+=float(ce[j]);book[label+'top1']+=int(top[j,0]==y[j]);
                totals[label+'loss']+=float(ce.sum());totals[label+'top1']+=int((top[:,0]==y).sum());
                totals[label+'top5']+=int((top==y[:,None]).any(1).sum())
    out={k:(v/n if k.endswith('loss') else v) for k,v in totals.items()}
    if per_book:out['book_aggregates']={str(k):v for k,v in books.items()}
    return out

def verify_corpus(root,names):
    m=json.loads((root/'manifest175.json').read_text())
    assert m['disjoint_checked'] and m['source']=='NaNROUTER GitHub Releases'
    vocab_sha=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
    assert m['vocab_sha256']==vocab_sha,(m['vocab_sha256'],vocab_sha)
    for row in (root/'SHA256SUMS').read_text().splitlines():
        checksum,name=row.split(maxsplit=1)
        assert hashlib.sha256((root/name.strip()).read_bytes()).hexdigest()==checksum,name
    split_ids={s:{b['id'] for b in m['books'] if b['split']==s} for s in ('train','val','test')}
    assert all(not split_ids[x].intersection(split_ids[y]) for x in split_ids for y in split_ids if x!=y)
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=ROOT/'results/v175_corpus')
    p.add_argument('--steps',type=int,default=220)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--every',type=int,default=50)
    p.add_argument('--seed',type=int,default=175)
    p.add_argument('--resume',action='store_true',help='Load selected v175 model and optimizer; never reset AdamW history')
    p.add_argument('--skip-test',action='store_true',help='Do not inspect the book-separated TEST cohort during continuation')
    a=p.parse_args()
    torch.set_num_threads(int(os.getenv('V175_THREADS','2')))
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    m,opt,names,fp,old=load_models();man=verify_corpus(a.dataset,names)
    start_step=0
    if a.resume:
        cp=ROOT/'checkpoints/v175_signed_episodic_48.pt'
        prev=torch.load(cp,map_location='cpu',weights_only=False)
        assert prev['word_order']==names and tuple(prev['fingerprints'])==tuple(fp)
        assert prev['v175_data_sha256']==hashlib.sha256((a.dataset/'manifest175.json').read_bytes()).hexdigest()
        m.load_state_dict(prev['model_state_dict'])
        opt.load_state_dict(prev['optimizer_state_dict'])
        assert len(opt.param_groups)==4 and len(opt.state)==19
        start_step=prev['training_step']-old['training_step']
        assert start_step>=0
        print('RESUMED v175',start_step,'ADAM GROUPS',len(opt.param_groups),'MOMENTS',len(opt.state),flush=True)
    X,Y,_=load_split('train',a.dataset)
    print('SOURCE',man['release_tag'],'BOOKS',{k:man['data'][k]['books'] for k in ('train','val','test')},'EX',len(Y),'PARENT_STEP',old['training_step'],'GROUPS',len(opt.param_groups),'PARENT_MOMENTS',len(old['optimizer_state_dict']['state']),flush=True)
    # Never evaluate TEST during selection; only once after frozen best chosen.
    m.echo_enabled=False;initial=metric(m,'val',a.dataset)
    m.echo_enabled=True
    if not a.resume:
        with torch.no_grad():m.copy_log_strength.fill_(-3.2)
    init_new=metric(m,'val',a.dataset)
    print('VAL_PARENT',initial,'VAL_UNTRAINED',init_new,flush=True)
    best={'loss':initial['loss'],'step':0,'state':copy.deepcopy(m.state_dict()),'opt':copy.deepcopy(opt.state_dict()),'val':initial}
    logs=[];start=time.time();rng=np.random.default_rng(a.seed)
    for step in range(1,a.steps+1):
        idx=rng.integers(0,len(Y),size=a.batch)
        xx=torch.as_tensor(X[idx]);yy=torch.as_tensor(Y[idx]);m.train();m.echo_enabled=True
        z=m(xx);ce=F.cross_entropy(z,yy)
        opt.zero_grad(set_to_none=True);ce.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if step%a.every==0 or step==a.steps:
            met=metric(m,'val',a.dataset);row={'step':start_step+step,'train_nll':float(ce.detach()),'val':{k:v for k,v in met.items() if k!='book_aggregates'},'elapsed':round(time.time()-start,1)}
            logs.append(row);print('STEP',json.dumps(row),flush=True)
            if met['loss']<best['loss']:
                best={'loss':met['loss'],'step':start_step+step,'state':copy.deepcopy(m.state_dict()),'opt':copy.deepcopy(opt.state_dict()),'val':met}
    m.load_state_dict(best['state']);m.echo_enabled=True
    bestval=metric(m,'val',a.dataset,per_book=True)
    test=None if a.skip_test else metric(m,'test',a.dataset,per_book=True)
    report={'version':'v175','status':'EXPERIMENTAL_NOT_PROMOTED','architecture':'48-word dual-signed recurrent rails + trainable causal signed episodic next-word pointer',
     'data_source':man['release_tag'],'archive_sha256':man['archive_sha256'],'manifest_sha256':hashlib.sha256((a.dataset/'manifest175.json').read_bytes()).hexdigest(),
     'parent_step':old['training_step'],'optimizer_parent_exact':True,'optimizer_group_count':4,'optimizer_state_entries':len(best['opt']['state']),
     'train_books':man['data']['train']['books'],'val_books':man['data']['val']['books'],'test_books':man['data']['test']['books'],
     'train_rows':len(Y),'val_baseline':initial,'untrained_val':init_new,'selected_step':best['step'],'selected_val':bestval,'test_after_train':test,'test_reused_after_step_280':bool(a.resume) if test is not None else None,
     'history':logs,'resumed_from_step':start_step,'work_s':round(time.time()-start,1),'scope':'v175 book-disjoint, inherited frozen parent may have earlier exposure to some source texts',
     'active_word':'v167','active_byte':'v170','promotion':False}
    ck={'model_state_dict':best['state'],'optimizer_state_dict':best['opt'],
       'parent_checkpoint':'checkpoints/v174_temporal_signed_now.pt','parent_checkpoint_step':old['training_step'],
       'parent_optimizer_state_dict':old['optimizer_state_dict'],'v173_optimizer_state_dict':old['parent_optimizer_state_dict'],
       'fingerprints':fp,'word_order':names,'training_step':old['training_step']+best['step'],
       'architecture':'v175 trainable signed causal episodic word pointer 48-word context',
       'v175_data_sha256':report['manifest_sha256'],'promotion':'EXPERIMENTAL_ONLY'}
    torch.save(ck,ROOT/'checkpoints/v175_signed_episodic_48.pt')
    (ROOT/'results/v175_longbook_train.json').write_text(json.dumps(report,indent=2))
    print('BEST',best['step'],'VAL',bestval['loss'],'BASE',initial['loss'],'TOP1',bestval['top1'],initial['top1'],'TEST',None if test is None else {k:test[k] for k in ('loss','parent_loss','top1','parent_top1')},flush=True)

if __name__=='__main__':main()