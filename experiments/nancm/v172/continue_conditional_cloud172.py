#!/usr/bin/env python3
"""Continue exactly one inherited signed-cloud lexical successor, including AdamW.
Use TRAIN hash-key validation NLL for checkpoint choice; never select on DEV.
"""
import torch,numpy as np,json,time,math,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
from train_conditional_cloud172 import load_parent,load_examples,CloudNCE,eval_full,dev_cohort,K

def main():
 torch.set_num_threads(2);torch.manual_seed(1721)
 names,C,O,B,spatial,gain,fp=load_parent();vocab={w:i for i,w in enumerate(names)}
 X,Y,W,V,freq=load_examples(vocab)
 tr=np.where(~V)[0];va=np.where(V)[0]
 model=CloudNCE(C,O,B,spatial,gain,names)
 ckpt=torch.load(ROOT/'checkpoints/v172_conditional_signed_cloud.pt',map_location='cpu',weights_only=False)
 assert ckpt['word_order']==names and tuple(ckpt['fingerprints'])==tuple(fp)
 model.load_state_dict(ckpt['state_dict'])
 opt=torch.optim.AdamW([{'params':[model.ctx.weight,model.out.weight],'lr':.0004},{'params':[model.bias,model.cloud_projection.weight],'lr':.001},{'params':[model.gain],'lr':.00004}],weight_decay=.002)
 opt.load_state_dict(ckpt['optimizer_state_dict'])
 # Decay inherited optimizer learning rates rather than reinitializing moments.
 for g in opt.param_groups:g['lr']*=.65
 freq=(freq+1)**.75;freq/=freq.sum();noise=torch.tensor(freq,dtype=torch.float32);log_noise=torch.log(noise+1e-30)
 np_rng=np.random.default_rng(1721+int(ckpt['step']));train_prob=W[tr]/W[tr].sum()
 iterations=int(os.environ.get("V172_UPDATES","1840"));batch=256
 draw=np_rng.choice(tr,size=(iterations,batch),p=train_prob).astype('int32')
 negatives=torch.multinomial(noise,iterations*batch*K,replacement=True).reshape(iterations,batch,K)
 D,Dy,dev=dev_cohort(vocab);known=np.where(Dy>=0)[0]
 best={'step':ckpt['step'],'nll':eval_full(model,X[va[:2048]],Y[va[:2048]])['nll_nats'],'state':{k:v.detach().clone() for k,v in model.state_dict().items()},'optimizer':opt.state_dict()};beg=time.perf_counter()
 log=[]
 for k in range(iterations):
  ids=torch.tensor(X[draw[k]],dtype=torch.long);target=torch.tensor(Y[draw[k]],dtype=torch.long);neg=negatives[k]
  q=model.query(ids)
  pos_s=model.logits(q,target)-math.log(K)-log_noise[target]
  neg_s=model.logits(q,neg)-math.log(K)-log_noise[neg]
  loss=torch.nn.functional.softplus(-pos_s).mean()+torch.nn.functional.softplus(neg_s).sum(1).mean()+.0001*model.cloud_projection.weight.square().sum()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
  if (k+1)%460==0:
   step=int(ckpt['step'])+k+1
   met=eval_full(model,X[va[:2048]],Y[va[:2048]])
   log.append({'step':step,'val':met,'train_loss':loss.item()})
   if met['nll_nats']<best['nll']:
    best={'step':step,'nll':met['nll_nats'],'state':{a:b.detach().clone() for a,b in model.state_dict().items()},'optimizer':opt.state_dict()}
   print('STEP',step,'VAL',met,'train',round(loss.item(),4),'seconds',round(time.perf_counter()-beg,2),flush=True)
 model.load_state_dict(best['state'])
 dev_metric=eval_full(model,D[known],Dy[known]);validation=eval_full(model,X[va[:2048]],Y[va[:2048]])
 ckpt.update({'step':best['step'],'state_dict':best['state'],'optimizer_state_dict':best['optimizer'],'continued_from_step':ckpt['step'],'train_updates_added':iterations,'selection':'TRAIN derived context-key holdout NLL, NOT book-disjoint; DEV not selected'})
 torch.save(ckpt,ROOT/'checkpoints/v172_conditional_signed_cloud.pt')
 report=json.loads((ROOT/'results/v172_conditional_cloud_train.json').read_text())
 report.update({'resumed_optimizer':True,'best_step':best['step'],'continued_steps':iterations,'continue_history':log,'selected_validation_2048':validation,'resumed_dev':dev_metric,'status':'TRAINED_RESEARCH_ONLY_NOT_PROMOTED'})
 (ROOT/'results/v172_conditional_cloud_train.json').write_text(json.dumps(report,indent=2))
 print('SELECTED',best['step'],'VAL',validation,'DEV',dev_metric,flush=True)
if __name__=='__main__':main()