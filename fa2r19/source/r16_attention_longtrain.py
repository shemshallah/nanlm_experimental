#!/usr/bin/env python3
"""R16 long-context attention-only continuation from ONE inherited R15 checkpoint.

No new model or output head. Freeze inherited FFN+GRU+word trunk, train only
existing Q/K/V/out and gain with intact AdamW moments. Increase attention
history capacity 64->256 without altering learned tensor shapes. Gate on
short and long dev BPB and non-space NLL; preserve original legacy anchors.
The original archive remains recoverable even if a new candidate is rejected.
"""
import argparse,copy,gc,hashlib,json,math,os,pathlib,time
import numpy as np,torch
import r14_slim_base as pal
from r13_cpu import run_window_fast
from r13_corpus_cache import read_books_cached
from r11_raw import windows,batch
pal.run_window=run_window_fast

def main(args):
 torch.set_num_threads(args.threads)
 root=pathlib.Path(args.root).resolve();checkpoint=root/'checkpoint/r12_active.pal12'
 parent=pal.sha(checkpoint)
 assert parent=='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f'
 saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,checkpoint,True)
 assert saved['r12_updates']==10756 and len(opt.param_groups)==4
 old_capacity=adapter.capacity;assert old_capacity==64
 adapter.capacity=256
 docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
 short_windows=windows(docs,24,8000,160);long_windows=windows(docs,96,args.long_train_windows,650)
 dev_short=short_windows['dev'][:96];dev_long=long_windows['dev'][:args.long_dev_windows]
 train=long_windows['train']
 def measure():
  return {'short':pal.evaluate(model,bridge,adapter,docs['dev'],dev_short,24,2),
          'long':pal.evaluate(model,bridge,adapter,docs['dev'],dev_long,96,2)}
 original=measure()
 print(json.dumps({'phase':'BASE','long_dev':original['long'],'short_dev':original['short'],'train_windows':len(train)}),flush=True)
 # Each original non-attention tensor is preserved and temporarily frozen.
 model.requires_grad_(False);bridge.requires_grad_(False)
 adapter.requires_grad_(False)
 for m in (adapter.query,adapter.key,adapter.value,adapter.attn_out):m.requires_grad_(True)
 adapter.attn_gain.requires_grad_(True)
 trainable=sum(p.numel() for p in adapter.parameters() if p.requires_grad)
 assert trainable>0
 stages=sorted(set([args.first_gate,args.steps]))
 report={'parent_sha256':parent,'r12_parent_updates':saved['r12_updates'],'model_architecture':'R12 unified checkpoint with R16 trained attention memory cap=256',
         'trainable_parameters':trainable,'capacity_before':old_capacity,'capacity_candidate':adapter.capacity,
         'optimizer_groups_preserved':len(opt.param_groups),'tokenizer_unchanged':True,'new_output_head':False,
         'dev_short_hash':hashlib.sha256(json.dumps(dev_short).encode()).hexdigest(),
         'dev_long_hash':hashlib.sha256(json.dumps(dev_long).encode()).hexdigest(),
         'before':original,'stages':[],'promoted':False}
 best=original;start=time.monotonic();processed=0
 for step in range(1,args.steps+1):
  chosen=[train[int(j)] for j in rng.integers(0,len(train),size=args.batch)]
  e,lc,t,tag,_=batch(docs['train'],chosen,96)
  B,W=e.shape[:2]
  with torch.no_grad():h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
  loss,n=run_window_fast(model,bridge,adapter,h,t,tag,training=True)
  if not torch.isfinite(loss):raise ValueError('Nonfinite loss step '+str(step))
  opt.zero_grad(set_to_none=True);loss.backward()
  torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],1.)
  opt.step();processed+=n
  if step%10==0:print(json.dumps({'phase':'TRAIN','step':step,'tokens':processed,'loss':float(loss.detach()),'sec':round(time.monotonic()-start,1)}),flush=True)
  if step not in stages:continue
  with torch.no_grad():after=measure()
  improved=all(after[scope][metric]<original[scope][metric]-.0001 for scope in ('short','long') for metric in ('bits_per_byte','other_nll'))
  stage={'steps':step,'candidate_r12_updates':10756+step,'dev':after,'both_scopes_both_metrics_better':improved,'trained_tokens':processed}
  report['stages'].append(stage)
  print(json.dumps({'phase':'GATE','step':step,'accepted':improved,'short':after['short'],'long':after['long'],'sec':round(time.monotonic()-start,1)}),flush=True)
  if step==args.first_gate and not improved:
   report['stop_reason']='first long-context attention gate did not improve both development scopes and both loss metrics'
   break
  if improved and step==args.steps:
   # Only now promote, with original parameter identity, optimizer groups, RNG histories.
   checkpoint_save=dict(saved)
   r12config=dict(saved.get('r12_config',{}));r12config['capacity']=adapter.capacity
   checkpoint_save.update({'model':model.state_dict(),'stream_bridge':bridge.state_dict(),
     'r12_residual':adapter.state_dict(),'stream_optimizer':opt.state_dict(),
     'r12_config':r12config,'stream_rng':copy.deepcopy(rng.bit_generator.state),
     'torch_rng':torch.get_rng_state().clone(),'r12_updates':10756+step,
     'r16_attention_source_sha256':parent,'r16_attention_updates':step,
     'r16_attention_train_windows_sha256':hashlib.sha256(json.dumps(train).encode()).hexdigest()})
   tmp=checkpoint.with_suffix('.pal12.tmp')
   torch.save(checkpoint_save,tmp);os.replace(tmp,checkpoint)
   stage['promoted_sha256']=pal.sha(checkpoint);report['promoted']=True
 if not report['promoted']:
  assert pal.sha(checkpoint)==parent
 report['parent_unchanged']=not report['promoted']
 report['training_seconds']=time.monotonic()-start
 out=root/'reports/r16_attention_longtrain.json';out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'phase':'FINAL','promoted':report['promoted'],'parent_unchanged':report['parent_unchanged'],'stages':len(report['stages']),'source':parent}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--threads',type=int,default=2);p.add_argument('--batch',type=int,default=1)
 p.add_argument('--first-gate',type=int,default=40);p.add_argument('--steps',type=int,default=120)
 p.add_argument('--long-train-windows',type=int,default=2000);p.add_argument('--long-dev-windows',type=int,default=32)
 main(p.parse_args())
