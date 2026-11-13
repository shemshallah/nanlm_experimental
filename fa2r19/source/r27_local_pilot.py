#!/usr/bin/env python3
"""Local real-book R27 pilot. Does NOT promote old PAL or overwrite R26."""
import argparse,copy,json,math,pathlib,random,hashlib,time
import numpy as np,torch
from torch.nn import functional as F
from r13_corpus_cache import read_books_cached
from r26_restore_unified import restore
from r26_unified_cpu import extract_split
from r27_trust import CounterfactualTrustGate,near_only_target

def score_row(model,r,wrong=False):
 if wrong:y,py,ft=r['wrong_y'],r['wp_y'],r['wfeatures']
 else:y,py,ft=r['base_y'],r['p_y'],r['features']
 full=model.r26.target_logprob(y,py,ft)
 near=near_only_target(model.r26,y,py,ft)
 trust=model.r27(full,near,ft,position=r['start'])
 return full,near,trust

def metrics(model,rows):
 total={'ref':0.,'near':0.,'new':0.,'wrong':0.,'nonspace':0.,'n':0,'bytes':0,'nonspace_n':0}
 per_book={};per_band={};betas=[]
 with torch.no_grad():
  for r in rows:
   ref,near,new=score_row(model,r);wrong=score_row(model,r,True)[2]
   b=r['byte_count'];n=len(ref)
   total['ref']-=float(ref.sum());total['near']-=float(near.sum());total['new']-=float(new.sum());total['wrong']-=float(wrong.sum())
   total['n']+=n;total['bytes']+=b
   mask=r['targets']!=32;total['nonspace']-=float(new[mask].sum());total['nonspace_n']+=int(mask.sum())
   band='132K' if r['start']>=131072 else ('65K' if r['start']>=65536 else ('33K' if r['start']>=32768 else 'near'))
   for d,key in [(per_book,r['book_id']),(per_band,band)]:
    arr=d.setdefault(key,[0.,0.,0.,0]);arr[0]-=float(ref.sum());arr[1]-=float(new.sum());arr[2]+=b;arr[3]-=float(near.sum())
   betas.extend(model.r27.fallback_fraction(r['features'],position=r['start']).tolist())
 d=math.log(2)
 if not total['bytes']:return {}
 fmt=lambda v:{'r26_bpb':v[0]/v[2]/d,'r27_bpb':v[1]/v[2]/d,'near_bpb':v[3]/v[2]/d,'bytes':v[2]}
 return {'r26_bpb':total['ref']/total['bytes']/d,'near_bpb':total['near']/total['bytes']/d,
 'r27_bpb':total['new']/total['bytes']/d,'gain_bpb':(total['ref']-total['new'])/total['bytes']/d,
 'correct_vs_wrong_nats':(total['wrong']-total['new'])/total['n'],
 'nonspace_nll':total['nonspace']/total['nonspace_n'],
 'books_improved':sum(a[1]<a[0] for a in per_book.values()),'books':len(per_book),
 'mean_fallback':float(np.mean(betas)),
 'by_book':{k:fmt(v) for k,v in per_book.items()},'by_band':{k:fmt(v) for k,v in per_band.items()}}

def main(a):
 torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 root=pathlib.Path(a.root).resolve()
 u,opt,comp,r6,meta,data=restore(root)
 parent=root/'reports/fa2_r26_unified_candidate.pth';base_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
 assert data['selected']['r26_step']==90 and len(opt.state)==137 and len(opt.param_groups)==11
 old={k:v.clone() for k,v in u.state_dict().items()};op=copy.deepcopy(opt.state_dict())
 carry=copy.deepcopy(data['selected']['carry'])
 u.add_module('r27',CounterfactualTrustGate())
 opt.add_param_group({'params':list(u.r27.parameters()),'lr':a.lr,'weight_decay':1e-4})
 for name,p in u.named_parameters():p.requires_grad_(name.startswith('r27.'))
 books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 offsets=[int(x) for x in a.offsets.split(',')]
 print('R27_PILOT',json.dumps({'train':[b.book_id for b in books['train'][:a.train_books]],'dev':[b.book_id for b in books['dev'][:a.dev_books]],'offsets':offsets}),flush=True)
 train=extract_split(u,books['train'][:a.train_books],offsets,a.chunk_span,a.windows,'train')
 dev=extract_split(u,books['dev'][:a.dev_books],offsets,a.chunk_span,a.windows,'dev')
 assert train and dev
 torch.save({'train':train,'dev':dev,'settings':vars(a)},root/'reports/r27_extracted_cpu_cache.pth')
 baseline=metrics(u,dev)
 print('R27_BASE',json.dumps(baseline),flush=True)
 best=(baseline['r27_bpb'],0,copy.deepcopy(u.r27.state_dict()),copy.deepcopy(opt.state_dict()),baseline)
 for step in range(1,a.steps+1):
  u.train();batch=random.sample(train,min(a.batch,len(train)))
  errs=[]
  for r in batch:
   full,near,score=score_row(u,r);wrong=score_row(u,r,True)[2]
   errs.append(-score.mean()+.025*F.softplus(.01+wrong.mean()-score.mean()))
  opt.zero_grad(set_to_none=True);loss=torch.stack(errs).mean();loss.backward()
  grad=float(torch.nn.utils.clip_grad_norm_(list(u.r27.parameters()),2.))
  if not math.isfinite(grad):raise ValueError('nonfinite gradient')
  opt.step()
  if step%a.eval_every==0 or step==a.steps:
   u.eval();m=metrics(u,dev)
   safe_132=('132K' not in baseline['by_band'] or m['by_band']['132K']['r27_bpb'] <= baseline['by_band']['132K']['r26_bpb']+1e-5)
   near_safe=('near' not in baseline['by_band'] or m['by_band']['near']['r27_bpb'] <= baseline['by_band']['near']['r26_bpb']+1e-5)
   eligible=(m['r27_bpb']<best[0]-1e-6 and m['nonspace_nll']<=baseline['nonspace_nll']+1e-4 and m['correct_vs_wrong_nats']>0 and safe_132 and near_safe)
   print('R27_EVAL',json.dumps({'step':step,'r27_bpb':m['r27_bpb'],'gain_bpb':m['gain_bpb'],'fallback':m['mean_fallback'],'grad':grad,'far_safe':safe_132,'near_safe':near_safe,'eligible':eligible,'per_band':m['by_band']}),flush=True)
   if eligible:best=(m['r27_bpb'],step,copy.deepcopy(u.r27.state_dict()),copy.deepcopy(opt.state_dict()),m)
 u.r27.load_state_dict(best[2]);opt.load_state_dict(best[3]);u.eval()
 assert all(torch.equal(v,u.state_dict()[k]) for k,v in old.items()),'inherited weight mutation'
 now=opt.state_dict();assert op['param_groups']==now['param_groups'][:11]
 for k,vals in op['state'].items():
  for field,prev in vals.items():
   nxt=now['state'][k][field]
   assert (torch.equal(prev,nxt) if torch.is_tensor(prev) else prev==nxt),(k,field)
 assert len(data['selected']['carry'])==len(carry)
 assert hashlib.sha256(parent.read_bytes()).hexdigest()==base_sha
 report={'format':'R27_PILOT_REAL_BOOK_V1','parent_sha256':base_sha,'baseline':baseline,'selected_step':best[1],
         'attempted_updates':a.steps,'selected':metrics(u,dev),'inherited_model_tensors':len(old),
         'inherited_optimizer_states':len(op['state']),'status':'candidate' if best[1] else 'rejected',
         'settings':vars(a)}
 dest=root/'reports/r27_realbook_local_pilot.json';dest.write_text(json.dumps(report,indent=2)+'\n')
 if best[1]:
  tiny={'format':'FA2_R27_ADAPTER_OVERLAY_V1','parent_r26_sha256':base_sha,'selected_step':best[1],
        'r27_state':u.r27.state_dict(),'r27_opt':{k:v for k,v in now['state'].items() if k not in op['state']},
        'r27_group':now['param_groups'][-1]}
  torch.save(tiny,root/'reports/r27_local_overlay.pt')
 print('R27_FINAL',json.dumps({'step':best[1],'bpb':report['selected']['r27_bpb'],'report':str(dest),'overlay_saved':bool(best[1])}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.')
 p.add_argument('--steps',type=int,default=40);p.add_argument('--seed',type=int,default=27027)
 p.add_argument('--train-books',type=int,default=2);p.add_argument('--dev-books',type=int,default=3)
 p.add_argument('--offsets',default='4096,8192,33000,65000,132000');p.add_argument('--chunk-span',type=int,default=20)
 p.add_argument('--windows',type=int,default=1);p.add_argument('--batch',type=int,default=6)
 p.add_argument('--eval-every',type=int,default=5);p.add_argument('--lr',type=float,default=.004)
 main(p.parse_args())