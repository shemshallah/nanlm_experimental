#!/usr/bin/env python3
"""1000-step learning gate, followed by 9000 conditional CPU updates on one PAL model.
Loss-only 1000-step gate, legacy word/future regression guard and atomically promoted
single checkpoint. No test evaluation during selection. Intervals of 1000 updates
promote only if dev bits/byte AND non-space NLL both beat last accepted interval.
"""
import argparse,copy,gc,hashlib,json,os,pathlib,time
import numpy as np
import torch
import r14_slim_base as pal
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
from r11_raw import windows,batch
pal.run_window=run_window_fast

def log(**kw):print(json.dumps(kw,sort_keys=True,default=str),flush=True)

def main(a):
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
 root=pathlib.Path(a.root).resolve();ckpt=root/'checkpoint/r12_active.pal12';out=root/'reports/r15_1000_then_9000.json'
 original=pal.sha(ckpt)
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,ckpt,True,a.seed,64,8)
 assert saved['format']=='PAL12_CAUSAL_RESIDUAL_V1' and saved['r12_updates']==756
 assert saved['r10_updates']==2020 and saved['r11_updates']==130
 assert len(opt.param_groups)==4 and len(opt.state)>=20
 docs,manifest=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 pools=windows(docs,24,8000,160);pools['dev']=pools['dev'][:96]
 for split in ('test',):pools[split]=pools[split][:96]
 pool_hash=hashlib.sha256(json.dumps(pools['train']).encode()).hexdigest()
 dev_hash=hashlib.sha256(json.dumps(pools['dev']).encode()).hexdigest()
 def score():return pal.evaluate(model,bridge,adapter,docs['dev'],pools['dev'],24,2)
 baseline=score();old0,fut0=pal.old_anchors(model,root,64,48)
 rep={'initial_checkpoint_sha256':original,'r12_initial_updates':756,'r10_updates':2020,'r11_updates':130,
      'initial_dev':baseline,'initial_word_nll':old0,'initial_future':fut0,
      'train_pool_count':len(pools['train']),'train_pool_sha256':pool_hash,'dev_pool_sha256':dev_hash,
      'book_counts':{k:len(v) for k,v in docs.items()},'gated_stages':[],
      '1000_step_gate_passed':False,'steps_attempted':0,'steps_promoted':0,
      'test_used_for_selection':False,'training_corpus':'historical NaNROUTER 38-book split, no new domains'}
 def write():
  temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(rep,indent=2)+'\n');os.replace(temp,out)
 write(); log(phase='INITIAL',before=baseline,source_sha256=original,train_windows=len(pools['train']))
 best_bits=baseline['bits_per_byte'];best_other=baseline['other_nll'];seen=0;start=time.monotonic()
 for i in range(1,10001):
  model.train();bridge.train();adapter.train()
  picks=[pools['train'][int(j)] for j in rng.integers(len(pools['train']),size=2)]
  e,lc,t,tag,_=batch(docs['train'],picks,24);b,w=e.shape[:2]
  h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(b,w,-1)
  loss,n=pal.run_window(model,bridge,adapter,h,t,tag,training=True)
  if not torch.isfinite(loss):raise ValueError('nonfinite loss at '+str(i))
  opt.zero_grad(set_to_none=True);loss.backward()
  torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters())+list(adapter.parameters()),1.)
  opt.step();seen+=n
  if i%100==0:
   log(phase='PROGRESS',steps=i,r12_candidate=756+i,train_loss=float(loss.detach()),
       seen_tokens=seen,elapsed=round(time.monotonic()-start,1),rate=round(seen/(time.monotonic()-start),1))
  if i%1000:continue
  del h,e,lc,t,tag,loss;gc.collect()
  dev=score();old,fut=pal.old_anchors(model,root,64,48)
  legacy=(old<=old0+.0008 and all(fut[str(j)]['nll']<=fut0[str(j)]['nll']+.00045 for j in (1,2,3)))
  improved=(dev['bits_per_byte']<best_bits-.0001 and dev['other_nll']<best_other-.0001)
  accepted=bool(legacy and improved)
  row={'steps':i,'candidate_r12_updates':756+i,'dev':dev,'word_nll':old,'future':fut,
       'legacy_guard':legacy,'better_than_best_both':improved,'promoted':accepted,
       'cumulative_train_tokens':seen,'elapsed_seconds':time.monotonic()-start}
  if accepted:
   checkpoint=dict(saved)
   checkpoint.update({'model':model.state_dict(),'stream_bridge':bridge.state_dict(),
    'r12_residual':adapter.state_dict(),'stream_optimizer':opt.state_dict(),
    'stream_rng':copy.deepcopy(rng.bit_generator.state),'torch_rng':torch.get_rng_state().clone(),
    'r12_updates':756+i,'parent_sha256':original,'r15_training_windows_sha256':pool_hash,
    'r15_dev_windows_sha256':dev_hash,'r15_steps_completed':i})
   temp=ckpt.with_suffix('.pal12.tmp');torch.save(checkpoint,temp);os.replace(temp,ckpt)
   row['checkpoint_sha256']=pal.sha(ckpt);best_bits=dev['bits_per_byte'];best_other=dev['other_nll']
   rep['steps_promoted']=i;rep['promoted_r12_updates']=756+i;rep['active_checkpoint_sha256']=row['checkpoint_sha256']
  rep['steps_attempted']=i;rep['gated_stages'].append(row)
  if i==1000:
   rep['1000_step_gate_passed']=accepted
   log(phase='GATE_1000',accepted=accepted,dev=dev,baseline=baseline,legacy_guard=legacy)
   write()
   if not accepted:
    rep['stop_reason']='did not improve dev and pass legacy guards at 1000 updates'
    write();return
  else:log(phase='STAGE',steps=i,accepted=accepted,dev=dev,best_bits=best_bits,active_r12_updates=rep['promoted_r12_updates'])
  write()
 rep['completed_all_10000']=True;rep['train_seconds']=time.monotonic()-start
 rep['train_tokens']=seen;rep['tokens_per_second']=seen/rep['train_seconds'];write()
 log(phase='FINAL',steps_attempted=10000,updates_promoted=rep['promoted_r12_updates'],best_dev_bits=best_bits,active_sha256=pal.sha(ckpt))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.')
 p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261111)
 main(p.parse_args())
